import os
import wandb
import hydra
import torch
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from datetime import datetime

from models import *
from utils import get_best_threshold_with_dataloader

# make sure dataset splits are reproducible
torch.manual_seed(0)

MODEL_MAP = {
    'disentangled': DisentangledModel,
    'd': DisentangledModel,
    'erm': ERMModel,
    'erm_adv': ERMAdvModel,
    'erm_adv18': ERMAdv18Model,
    'refusion_cascade_rescale': RefusionCascadeRescale,
    'sensitive_attribute': SensitiveAttributeModel,
    'sa': SensitiveAttributeModel
}

def test(cfg):
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = WandbLogger(project='fairness-medai', tags=[cfg.model, cfg.target, cfg.run_mode, *cfg.additional_tags], config=OmegaConf.to_container(cfg, resolve=True))
    trainer = Trainer(max_epochs=cfg.max_epochs, logger=logger, devices=cfg.num_gpus, log_every_n_steps=30)
    if cfg.run_mode == 'cross_validation':
        if 'd' == cfg.model:
            OmegaConf.set_struct(cfg, True)
            OmegaConf.update(cfg, 'd_checkpoint_path', f'{cfg.d_model_cv_root}/fold_{cfg.cv_fold_idx}/model_target_40.pth', force_add=True)
        else:
            OmegaConf.set_struct(cfg, True)
            model_root = f'{cfg.test_model_root}/fold_{cfg.cv_fold_idx}'
            # only one file in the folder
            model_file = os.listdir(model_root)[0]
            print(f'Loading model from {model_root}/{model_file}')
            OmegaConf.update(cfg, 'test_model_path', f'{model_root}/{model_file}', force_add=True)
            if 'refusion' in cfg.model:
                OmegaConf.set_struct(cfg, True)
                OmegaConf.update(cfg, 'd_model_path', f'{cfg.d_model_cv_root}/fold_{cfg.cv_fold_idx}/model_target_40.pth', force_add=True)
    if cfg.model == 'd':
        model = MODEL_MAP[cfg.model](**cfg)
        # logger.watch(model, log_graph=False)
    else:
        model = MODEL_MAP[cfg.model].load_from_checkpoint(cfg.test_model_path, **cfg)
        # logger.watch(model, log_graph=False)

    if hasattr(cfg, 'threshold_dict_path') and os.path.exists(cfg.threshold_dict_path):
        print(f'Loading threshold dict from {cfg.threshold_dict_path}')
        model.threshold_dict = torch.load(cfg.threshold_dict_path)
    
    if torch.all(model.threshold_dict == 0) or torch.all(model.threshold_dict == 0.5) or (hasattr(cfg, 'test_refit_threshold') and cfg.test_refit_threshold):
        model.cuda()
        model.eval()
        if not hasattr(cfg, 'threshold_selection_method'):
            OmegaConf.update(cfg, 'threshold_selection_method', 'min_gap', force_add=True)
        print(f'Finding best thresholds with {cfg.threshold_selection_method}')
        get_best_threshold_with_dataloader(model.train_dataloader(), model, save_fig_path=f'figs/{time}', fig_name='roc_curve', method=cfg.threshold_selection_method)
        print(f'Best thresholds: {model.threshold_dict}')
        if hasattr(cfg, 'threshold_dict_path'):
            print(f'Saving threshold dict to {cfg.threshold_dict_path}')
            torch.save(model.threshold_dict, cfg.threshold_dict_path)
        # cfg.test_refit_threshold = False
    model.eval()
    test_result = trainer.test(model)
    if cfg.run_mode == 'cross_validation':
        logger.log_metrics(test_result[0], step=cfg.cv_fold_idx)
    return test_result

def cross_validation(cfg):
    test_results = []
    for i in range(cfg.cv_fold_total):
        print(f'Cross validation fold {i}')
        cfg.cv_fold_idx = i
        # returns a list of test results dicts
        test_result = test(cfg)[0]
        test_results.append(test_result)
        for key, value in test_result.items():
            wandb.log({f'{key}_cv': value})
        
    # average test results
    avg_test_result = {}
    test_result_count = {}
    for key in test_results[0]:
        for test_result in test_results:
            if key not in test_result:
                continue
            if key in avg_test_result:
                test_result_count[key] += 1
                avg_test_result[key] += test_result[key]
            else:
                test_result_count[key] = 1
                avg_test_result[key] = test_result[key]
        avg_test_result[key] /= test_result_count[key]
    print(avg_test_result)
    for key, value in avg_test_result.items():
        wandb.log({f'{key}_cv_avg': value})

@hydra.main(config_path='configs/', version_base=None)
def main(cfg):
    # if 'run_mode' in cfg and cfg.run_mode == 'sweep':
    #     hydra.initialize(config_path='configs/')
    #     cfg = hydra.compose(config_name='sweep_refusion')
    #     train(cfg)
    wandb.init(project='fairness-medai', tags=[cfg.model, cfg.target, cfg.run_mode, *cfg.additional_tags], config=OmegaConf.to_container(cfg))
    if 'run_mode' in cfg and cfg.run_mode == 'cross_validation':
        cross_validation(cfg)
    else:
        test(cfg)

if __name__ == '__main__':
    hydra.output_subdir = None
    main()