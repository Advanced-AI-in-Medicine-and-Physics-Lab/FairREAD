from multiprocessing import Pool
import os
import re
import pandas as pd
import numpy as np
import torch
import pathlib
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset.dataset_utils import do_subgroup_balance_sample, make_sensitive_attribute_tensor

class CheXpertDataset(Dataset):
    def __init__(self, patient_labels_path, patient_info_path, image_root, use_attrs, transform=None, mode='sa_prediction', target='Cardiomegaly', sensitive_attributes=['PRIMARY_RACE', 'GENDER', 'AGE_AT_CXR'], **kwargs) -> None:
        assert mode in ['sa_prediction', 'refusion', 'diffusion']
        # Data preprocessing
        if not pathlib.Path(patient_labels_path).is_absolute():
            patient_labels_path = os.path.dirname(os.path.abspath(__file__)) + '/' + patient_labels_path
        if not pathlib.Path(patient_info_path).is_absolute():
            patient_info_path = os.path.dirname(os.path.abspath(__file__)) + '/' + patient_info_path
        self.patient_info = pd.read_excel(patient_info_path)
        self.patient_labels = pd.read_csv(patient_labels_path)
        self.patient_labels['Patient ID'] = self.patient_labels['Path'].apply(lambda x: 'patient' + re.search(r'patient(\d+)', x).group(1))
        self.patient_labels = self.patient_labels.merge(self.patient_info, left_on='Patient ID', right_on='PATIENT')

        # Drop rows with no or unknown race labels
        # self.patient_labels = self.patient_labels.dropna(subset=sensitive_attributes + [target])
        # Fill na with 0
        self.patient_labels[target] = self.patient_labels[target].fillna(0)
        self.patient_labels = self.patient_labels[self.patient_labels[target].isin([0, 1])]
        for sensitive_attribute in sensitive_attributes:
            if sensitive_attribute == 'AGE_AT_CXR':
                continue
            if 'Non-White' in use_attrs[sensitive_attribute]:
                self.patient_labels.loc[self.patient_labels[sensitive_attribute] != 'White', sensitive_attribute] = 'Non-White'
            self.patient_labels = self.patient_labels[self.patient_labels[sensitive_attribute].isin(use_attrs[sensitive_attribute])]
        self.patient_labels['age_class'] = self.patient_labels['AGE_AT_CXR'].apply(lambda x: 1 if x >= 60 else 0)
        self.patient_labels['Path'] = self.patient_labels.apply(lambda x: image_root + '/' + x['Path'], axis=1)
        # self.patient_labels = self.patient_labels[self.patient_labels.apply(lambda x: os.path.exists(image_root + '/' + x['Path']), axis=1)]
        self.patient_labels = self.patient_labels.reset_index(drop=True)

        self.patient_labels_path = patient_labels_path
        self.patient_info_path = patient_info_path
        self.transform = transform
        self.mode = mode
        self.target = target
        self.sensitive_attributes = sensitive_attributes
        self.use_attrs = use_attrs
        self.image_root = image_root
        
    def split(self, train_ratio=(0.8, 0.1, 0.1)):
        train_patient_labels, temp_patient_labels = train_test_split(
            self.patient_labels,
            train_size=train_ratio[0],
            # stratify=self.patient_labels['joint_idx'],
            random_state=42
        )
        val_patient_labels, test_patient_labels = train_test_split(
            temp_patient_labels,
            test_size=train_ratio[2] / (train_ratio[1] + train_ratio[2]),
            # stratify=temp_patient_labels['joint_idx'],
            random_state=42
        )
        train_dataset = CheXpertDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, self.mode, self.target, self.sensitive_attributes)
        val_dataset = CheXpertDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, self.mode, self.target, self.sensitive_attributes)
        test_dataset = CheXpertDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, self.mode, self.target, self.sensitive_attributes)
        train_dataset.patient_labels = train_patient_labels
        val_dataset.patient_labels = val_patient_labels
        test_dataset.patient_labels = test_patient_labels
        return train_dataset, val_dataset, test_dataset
    
    def split_for_cross_validation(self, start_fraction, end_fraction):
        start_idx = int(len(self.patient_labels) * start_fraction)
        end_idx = int(len(self.patient_labels) * end_fraction)
        fraction_dataset = CheXpertDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, self.mode, self.target, self.sensitive_attributes)
        fraction_dataset.patient_labels = self.patient_labels.iloc[start_idx:end_idx]
        # fraction_dataset.all_images = self.all_images[start_idx:end_idx]
        other_dataset = CheXpertDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, self.mode, self.target, self.sensitive_attributes)
        other_dataset.patient_labels = pd.concat([self.patient_labels.iloc[:start_idx], self.patient_labels.iloc[end_idx:]])
        # other_dataset.all_images = self.all_images[:start_idx] + self.all_images[end_idx:]
        return fraction_dataset, other_dataset

    def filter_onegroup(self, sensitive_attributes, use_attrs):
        for sensitive_attribute in sensitive_attributes:
            if sensitive_attribute == 'AGE_AT_CXR':
                continue
            self.patient_labels = self.patient_labels[self.patient_labels[sensitive_attribute].isin(use_attrs[sensitive_attribute])]
        
    def _to_fcro_format(self, save_path, split='train', mode='w', fold_idx=0):
        fcro_patient_labels = self.patient_labels.rename(columns={'Patient ID': 'Patient', 'PRIMARY_RACE': 'Race'})
        writer = pd.ExcelWriter(save_path, engine='openpyxl', mode=mode)
        sheet_name = 'test' if split == 'test' else f'{split}_{fold_idx}'
        fcro_patient_labels.to_excel(writer, index=False, sheet_name=sheet_name)
        writer.close()

    def __len__(self):
        return len(self.patient_labels)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_root, self.patient_labels.iloc[idx]['Path'])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.mode == 'sa_prediction':
            # assert len(self.sensitive_attributes) == 1
            label = self.patient_labels.iloc[idx][self.sensitive_attributes].values
            label = make_sensitive_attribute_tensor(self.use_attrs, self.sensitive_attributes, label)
            return image, label, label.unsqueeze(0)
        elif self.mode == 'refusion':
            label = self.patient_labels.iloc[idx][self.target]
            label = torch.tensor(label, dtype=torch.float32)
            sensitive_attributes_values =  self.patient_labels.iloc[idx][self.sensitive_attributes].values
            sensitive_attributes_values = make_sensitive_attribute_tensor(self.use_attrs, self.sensitive_attributes, sensitive_attributes_values)
            return image, label, sensitive_attributes_values
        elif self.mode == 'diffusion':
            return image
