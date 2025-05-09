import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import AUROC
from torchmetrics.functional.classification import binary_groups_stat_rates
from sklearn.calibration import calibration_curve, CalibrationDisplay
from lightning import LightningModule

from dataset.chexpert_dataset import CheXpertDataset
from dataset.mimic_dataset import MIMICDataset
from dataset.dataset_utils import do_cache, age_to_class, do_subgroup_balance_sample, do_class_balance_sample
from utils import get_best_decision_threshold

class BaseModel(LightningModule):
    def __init__(self, use_attrs, patient_labels_path, patient_info_path, image_root, learning_rate=1e-3, batch_size=32, target='Cardiomegaly', train_split=(0.8, 0.1, 0.1), sensitive_attributes=['PRIMARY_RACE', 'GENDER', 'AGE_AT_CXR'], data_cache_path=None,
                 cv_fold_idx=None, cv_fold_total=None, use_group_specific_threshold=True, subgroup_balance_sample=False, class_balance_sample=False, sub_sample_method='under',
                 plot_calibration=False, log_subgroup_performance=True, num_subgroups=8, plot_roc_path=None, dataset='chexpert', onegroup_filter=False,
                 *args,**kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target = target
        print(f'Loading dataset {dataset}')
        # Load dataset
        self.all_dataset = self._get_dataset(use_attrs, patient_labels_path, patient_info_path, image_root, target, sensitive_attributes, data_cache_path=data_cache_path, dataset=dataset)
        # if onegroup_filter:
        #     self.all_dataset = self.all_dataset.filter_onegroup(sensitive_attributes, use_attrs)
        
        # # print all dataset demographic distribution
        # print(self.all_dataset.patient_labels[sensitive_attributes].value_counts())

        if cv_fold_idx is not None and cv_fold_total is not None and kwargs.get('run_mode') == 'cross_validation':
            other_dataset, self.test_dataset = self.all_dataset.split_for_cross_validation(0, 0.9)
            self.val_dataset, self.train_dataset = other_dataset.split_for_cross_validation(cv_fold_idx / cv_fold_total, (cv_fold_idx + 1) / cv_fold_total)
            if subgroup_balance_sample:
                self.train_dataset = do_subgroup_balance_sample(self.train_dataset, sample_method=sub_sample_method)
            elif class_balance_sample:
                self.train_dataset = do_class_balance_sample(self.train_dataset, target=target)
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = self.all_dataset.split(train_split)
        
        self.train_dataset.sensitive_attributes = sensitive_attributes
        self.val_dataset.sensitive_attributes = sensitive_attributes
        self.test_dataset.sensitive_attributes = sensitive_attributes
        self.train_dataset.use_attrs = use_attrs
        self.val_dataset.use_attrs = use_attrs
        self.test_dataset.use_attrs = use_attrs

        print('onegroup_filter:', onegroup_filter)
        print('sensitive_attributes:', sensitive_attributes)
        print('use_attrs:', use_attrs)
        if onegroup_filter:
            self.train_dataset.filter_onegroup(sensitive_attributes, use_attrs)
            self.val_dataset.filter_onegroup(sensitive_attributes, use_attrs)
            self.test_dataset.filter_onegroup(sensitive_attributes, use_attrs)

        print('Train dataset demographic distribution:')
        print(self.train_dataset.patient_labels[sensitive_attributes].value_counts())

        print('Train dataset size:', len(self.train_dataset))
        print('Validation dataset size:', len(self.val_dataset))
        print('Test dataset size:', len(self.test_dataset))

        self.auroc = AUROC(task='binary').to(self.device)
        self.register_buffer('threshold_dict', torch.zeros(num_subgroups))
        self.use_group_specific_threshold = use_group_specific_threshold
        self.sensitive_attributes = sensitive_attributes
        self.use_attrs = use_attrs
        self.val_fairness_agg = [[], [], []]
        self.test_fairness_agg = [[], [], []]
        self.plot_calibration = plot_calibration
        self.log_subgroup_performance = log_subgroup_performance
        self.plot_roc_path = plot_roc_path

    @do_cache()
    def _get_dataset(self, use_attrs, patient_labels_path, patient_info_path, image_root, target, sensitive_attributes, data_cache_path, dataset='chexpert'):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if dataset == 'chexpert':
            dataset_obj = CheXpertDataset(patient_labels_path, patient_info_path, image_root=image_root, use_attrs=use_attrs, transform=transform, mode='refusion', target=target, sensitive_attributes=sensitive_attributes)
            # dataset_obj.patient_labels_path = dataset_obj.patient_labels_path.replace('/data/xlx9645/', '/home/xlx9645/')
            # dataset_obj.patient_info_path = dataset_obj.patient_info_path.replace('/data/xlx9645/', '/home/xlx9645/')
            # dataset_obj.patient_labels['Path'] = dataset_obj.patient_labels['Path'].apply(lambda x: x.replace('/data/xlx9645/', '/home/xlx9645/'))
        elif dataset == 'MIMIC':
            dataset_obj = MIMICDataset(patient_labels_path, patient_info_path, image_root=image_root, use_attrs=use_attrs, transform=transform, mode='refusion', target=target, sensitive_attributes=sensitive_attributes)

        return dataset_obj
    
    def on_train_epoch_start(self):
        self.train_fairness_agg = [[], [], []]

    def training_step(self, batch, batch_idx):
        images, labels, sensitive_attribute_vals = batch
        if hasattr(self, 'adv_alpha') and self.adv_alpha is not None:
            outputs, disentangled_outputs = self(images, sensitive_attribute_vals)
            outputs = outputs.squeeze(1) if len(outputs.shape) > 1 else outputs
            loss = self.criterion(outputs, labels)
            self.log('train_loss', loss)
            # Append outputs, labels, sensitive attributes
            self.train_fairness_agg[0].append(outputs.detach())
            self.train_fairness_agg[1].append(labels.detach())
            self.train_fairness_agg[2].append(sensitive_attribute_vals.detach())
            return loss, disentangled_outputs
        else:
            outputs = self(images, sensitive_attribute_vals)
            outputs = outputs.squeeze(1) if len(outputs.shape) > 1 else outputs
            loss = self.criterion(outputs, labels)
            self.log('train_loss', loss)
            # Append outputs, labels, sensitive attributes
            self.train_fairness_agg[0].append(outputs.detach())
            self.train_fairness_agg[1].append(labels.detach())
            self.train_fairness_agg[2].append(sensitive_attribute_vals.detach())
            return loss

    def on_train_epoch_end(self):
        fairness_agg_ls = self.train_fairness_agg
        all_outputs = torch.cat(fairness_agg_ls[0])
        all_labels = torch.cat(fairness_agg_ls[1])
        all_sensitive_attribute_vals = torch.cat(fairness_agg_ls[2])
        # joint_idx = all_sensitive_attribute_vals[:, 0] * 4 + all_sensitive_attribute_vals[:, 1] * 2 + all_sensitive_attribute_vals[:, 2]
        joint_idx = torch.zeros(all_sensitive_attribute_vals.shape[0], device=all_sensitive_attribute_vals.device)
        for i in range(all_sensitive_attribute_vals.shape[1]):
            joint_idx += all_sensitive_attribute_vals[:, i] * (2 ** i)
        joint_idx = joint_idx.cpu().numpy().astype(int)
        for j in np.unique(joint_idx):
            j = int(j)
            labels_current_group = all_labels[joint_idx == j]
            outputs_current_group = all_outputs[joint_idx == j]
            # Compute best threshold for this group
            if self.use_group_specific_threshold:
                best_threshold = get_best_decision_threshold(labels_current_group.cpu(), outputs_current_group.cpu(), save_fig_path=self.plot_roc_path, fig_name=f'train_{j}')
            else:
                best_threshold = 0.5
            self.threshold_dict[j] = torch.tensor(best_threshold).to(self.threshold_dict.device)
        print(f'Best thresholds: {self.threshold_dict}')

    def on_validation_epoch_start(self):
        self.val_fairness_agg = [[], [], []]

    def validation_step(self, batch, batch_idx, log_prefix='val', fairness_agg_ls=None):
        if fairness_agg_ls is None:
            fairness_agg_ls = self.val_fairness_agg
        images, labels, sensitive_attribute_vals = batch
        outputs, disentangled_outputs = self(images, sensitive_attribute_vals, get_feature=True)
        outputs = outputs.squeeze(1) if len(outputs.shape) > 1 else outputs
        loss = self.criterion(outputs, labels)
        self.log(f'{log_prefix}_loss', loss)
        self.auroc = self.auroc.to(outputs.device)
        fairness_agg_ls[0].append(outputs)
        fairness_agg_ls[1].append(labels)
        fairness_agg_ls[2].append(sensitive_attribute_vals)
        return loss, disentangled_outputs

    def on_validation_epoch_end(self, log_prefix='val', fairness_agg_ls=None):
        if fairness_agg_ls is None:
            fairness_agg_ls = self.val_fairness_agg
        all_outputs = torch.cat(fairness_agg_ls[0])
        all_labels = torch.cat(fairness_agg_ls[1])
        all_sensitive_attribute_vals = torch.cat(fairness_agg_ls[2])
        # joint_idx, _ = pd.factorize(pd._libs.lib.fast_zip([all_sensitive_attribute_vals[:, i].cpu().numpy() for i in range(all_sensitive_attribute_vals.shape[1])]))
        # joint_idx = all_sensitive_attribute_vals[:, 0] * 4 + all_sensitive_attribute_vals[:, 1] * 2 + all_sensitive_attribute_vals[:, 2]
        joint_idx = torch.zeros(all_sensitive_attribute_vals.shape[0], device=all_sensitive_attribute_vals.device)
        for i in range(all_sensitive_attribute_vals.shape[1]):
            joint_idx += all_sensitive_attribute_vals[:, i] * (2 ** i)
        joint_idx = joint_idx.cpu().numpy().astype(int)
        acc, tpr, tnr, auroc = [], [], [], []
        for j in np.unique(joint_idx):
            j = int(j)
            labels_current_group = all_labels[joint_idx == j]
            outputs_current_group = all_outputs[joint_idx == j]
            # Use thresholds learned during training
            best_threshold = self.threshold_dict[j]
            acc.append((outputs_current_group > best_threshold).eq(labels_current_group).float().mean())
            tpr.append((outputs_current_group[labels_current_group == 1] > best_threshold).float().mean())
            tnr.append((outputs_current_group[labels_current_group == 0] < best_threshold).float().mean())
            auroc.append(self.auroc(outputs_current_group, labels_current_group))
            if self.plot_calibration:
                fraction_of_positives, mean_predicted_value = calibration_curve(labels_current_group.cpu(), outputs_current_group.cpu(), n_bins=10)
                CalibrationDisplay(prob_true=fraction_of_positives, prob_pred=mean_predicted_value).plot()
            if self.log_subgroup_performance:
                self.log(f'{log_prefix}_acc_{j}', acc[-1])
                self.log(f'{log_prefix}_tpr_{j}', tpr[-1])
                self.log(f'{log_prefix}_tnr_{j}', tnr[-1])
                self.log(f'{log_prefix}_auroc_{j}', auroc[-1])
        delta_eo = max(max(tpr) - min(tpr), max(tnr) - min(tnr))
        delta_eopp_1 = max(tpr) - min(tpr)
        delta_eopp_0 = max(tnr) - min(tnr)
        delta_auc = max(auroc) - min(auroc)
        self.log(f'{log_prefix}_delta_eo_joint', delta_eo)
        self.log(f'{log_prefix}_delta_eopp_1_joint', delta_eopp_1)
        self.log(f'{log_prefix}_delta_eopp_0_joint', delta_eopp_0)
        if all([x != 0 for x in auroc]):
            self.log(f'{log_prefix}_delta_auc_joint', delta_auc)
        if self.plot_calibration:
            plt.savefig(f'{log_prefix}_{self.target}_calibration_curve_{wandb.run.id}.png')
        # Compute auroc
        auroc = self.auroc(all_outputs, all_labels)
        self.log(f'{log_prefix}_auroc', auroc)
        # Compute accuracy
        all_thresholds = torch.tensor([self.threshold_dict[joint_idx[i]] for i in range(len(joint_idx))]).to(all_outputs.device)
        correct = (all_outputs > all_thresholds).eq(all_labels).float().mean()
        self.log(f'{log_prefix}_acc', correct)

    def on_test_epoch_start(self):
        self.test_fairness_agg = [[], [], []]

    def test_step(self, batch, batch_idx, log_prefix='test'):
        return self.validation_step(batch, batch_idx, log_prefix=log_prefix, fairness_agg_ls=self.test_fairness_agg)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end(log_prefix='test', fairness_agg_ls=self.test_fairness_agg)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return test_loader