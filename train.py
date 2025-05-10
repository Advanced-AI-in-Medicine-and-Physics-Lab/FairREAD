import wandb
import hydra
import torch
import pandas as pd
import uuid
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from datetime import datetime

from models import *

torch.set_float32_matmul_precision('medium')

# make sure dataset splits are reproducible
torch.manual_seed(0)

MODEL_MAP = {
    'disentangled': DisentangledModel,
    'd': DisentangledModel,
    'erm': ERMModel,
    'refusion_cascade_rescale': RefusionCascadeRescale,
}

def train(cfg):
    if cfg.run_mode == 'sweep':
        cfg = wandb.config
    elif cfg.run_mode == 'cross_validation' and 'refusion' in cfg.model:
        # cfg['d_model_path'] = f'{cfg['d_model_cv_root']}/fold_{cfg.cv_fold_idx}/model_target_40.pth'
        OmegaConf.set_struct(cfg, True)
        OmegaConf.update(cfg, 'd_model_path', f'{cfg.d_model_cv_root}/fold_{cfg.cv_fold_idx}/model_target_40.pth', force_add=True)
        # print(cfg.d_model_path)
    if cfg.run_mode == 'cross_validation':
        OmegaConf.set_struct(cfg, True)
        run_id = uuid.uuid4().hex
        if cfg.cv_fold_idx == 0:
            OmegaConf.update(cfg, 'model_save_root', f'{cfg.model_save_root}/{run_id}', force_add=True)
        OmegaConf.update(cfg, 'model_save_path', f'{cfg.model_save_root}/fold_{cfg.cv_fold_idx}', force_add=True)
    
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath=cfg.model_save_path, filename=cfg.model+'_model-{epoch:02d}-{val_loss:.2f}-'+time, save_top_k=1, mode='min', verbose=True)
    project_name = cfg.project_name if 'project_name' in cfg else 'fairness-medai'
    logger = WandbLogger(project=project_name, tags=[cfg.model, cfg.target, cfg.run_mode, *cfg.additional_tags], config=OmegaConf.to_container(cfg, resolve=True))
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    trainer = Trainer(max_epochs=cfg.max_epochs, logger=logger, callbacks=[checkpoint_callback, early_stop_callback], devices=cfg.num_gpus)
    
    model = MODEL_MAP[cfg.model](**cfg)
    # logger.watch(model, log_graph=False)
    trainer.fit(model)
    model = MODEL_MAP[cfg.model].load_from_checkpoint(checkpoint_callback.best_model_path, **cfg)
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
        test_result = train(cfg)[0]
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
    if 'run_mode' in cfg and cfg.run_mode == 'cross_validation':
        cross_validation(cfg)
    else:
        train(cfg)

if __name__ == '__main__':
    hydra.output_subdir = None
    main()