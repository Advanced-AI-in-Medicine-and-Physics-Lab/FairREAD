import os
import re
import ast
import pandas as pd
import numpy as np
import torch
import pathlib
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset.dataset_utils import do_subgroup_balance_sample, make_sensitive_attribute_tensor

def process_patient_labels(patient_labels, patient_info, image_root, use_attrs, target, sensitive_attributes):
    patient_labels = patient_labels.merge(patient_info, left_on='dicom_id', right_on='id')
    patient_labels = patient_labels.drop_duplicates(subset='dicom_id')
        
    # Process image path:
    # The image_path column in MIMIC is stored as a string representation of a list.
    patient_labels['image_path'] = patient_labels['image_path'].apply(lambda x: os.path.join(image_root, x[0]))
        
    if target in patient_labels.columns:
        patient_labels = patient_labels[patient_labels[target] != -1]
        patient_labels[target] = patient_labels[target].fillna(0)
        
    # Process sensitive attributes
    # For non-numeric sensitive attributes, adjust categories if needed (e.g., consolidate non-white races)
    for sensitive_attribute in sensitive_attributes:
        if sensitive_attribute == 'anchor_age':
            continue  # leave numeric attributes as is (optionally you can create a binary age class)
        if 'Non-White' in use_attrs.get(sensitive_attribute, []):
            patient_labels = patient_labels.replace({sensitive_attribute: {'WHITE': 'White'}})
            patient_labels.loc[patient_labels[sensitive_attribute] != 'White', sensitive_attribute] = 'Non-White'
        patient_labels = patient_labels[patient_labels[sensitive_attribute].isin(use_attrs.get(sensitive_attribute, []))]
        
        
        # Optionally, create a binary age group if needed (similar to AGE_AT_CXR in CheXpert)
        patient_labels['age_class'] = patient_labels['anchor_age'].apply(lambda x: 1 if x >= 60 else 0)
    
    return patient_labels

class MIMICDataset(Dataset):
    def __init__(self, patient_labels_path, patient_info_path, image_root, use_attrs, transform=None, mode='sa_prediction', 
                 target='labels', sensitive_attributes=['race', 'gender', 'anchor_age'], **kwargs) -> None:

        assert mode in ['sa_prediction', 'refusion', 'diffusion']
        # Load the CSV file containing both labels and patient info
        self.init_patient_labels = pd.read_csv(patient_labels_path)
        self.patient_info = pd.read_json(patient_info_path)
        # self.patient_info_to_fit_threshold = pd.read_json('/home/xlx9645/fair-rrg/data/mimic_cxr/mimic_annotation_promptmrg_w_da_train.json')
        
        self.patient_labels = process_patient_labels(self.init_patient_labels, self.patient_info, image_root, use_attrs, target, sensitive_attributes)
        # self.patient_labels_to_fit_threshold = process_patient_labels(self.init_patient_labels, self.patient_info_to_fit_threshold, image_root, use_attrs, target, sensitive_attributes)

        # Save parameters and reset index
        self.patient_labels_path = patient_labels_path
        self.patient_info_path = patient_info_path
        self.transform = transform
        self.mode = mode
        self.target = target
        self.sensitive_attributes = sensitive_attributes
        self.use_attrs = use_attrs
        self.image_root = image_root
        self.patient_labels = self.patient_labels.reset_index(drop=True)
    
    def split(self, train_ratio=(0.8, 0.1, 0.1)):
        if train_ratio[0] == 0:
            train_patient_labels = pd.DataFrame(columns=self.patient_labels.columns)
            test_patient_labels = self.patient_labels
        else:
            train_patient_labels, test_patient_labels = train_test_split(
                self.patient_labels,
                train_size=train_ratio[0],
                stratify=self.patient_labels['joint_idx'],
                random_state=42
            )
        train_dataset = MIMICDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, 
                                     self.mode, self.target, self.sensitive_attributes)
        val_dataset = MIMICDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, 
                                   self.mode, self.target, self.sensitive_attributes)
        test_dataset = MIMICDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, 
                                    self.mode, self.target, self.sensitive_attributes)
        train_dataset.patient_labels = train_patient_labels
        val_dataset.patient_labels = pd.DataFrame(columns=self.patient_labels.columns)
        test_dataset.patient_labels = test_patient_labels
        return train_dataset, val_dataset, test_dataset
    
    def split_for_cross_validation(self, start_fraction, end_fraction):
        start_idx = int(len(self.patient_labels) * start_fraction)
        end_idx = int(len(self.patient_labels) * end_fraction)
        fraction_dataset = MIMICDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, self.mode, self.target, self.sensitive_attributes)
        fraction_dataset.patient_labels = self.patient_labels.iloc[start_idx:end_idx]
        # fraction_dataset.all_images = self.all_images[start_idx:end_idx]
        other_dataset = MIMICDataset(self.patient_labels_path, self.patient_info_path, self.image_root, self.use_attrs, self.transform, self.mode, self.target, self.sensitive_attributes)
        other_dataset.patient_labels = pd.concat([self.patient_labels.iloc[:start_idx], self.patient_labels.iloc[end_idx:]])
        # other_dataset.all_images = self.all_images[:start_idx] + self.all_images[end_idx:]
        return fraction_dataset, other_dataset

    def __len__(self):
        return len(self.patient_labels)
    
    def __getitem__(self, idx):
        # Open and transform the image
        image = Image.open(self.patient_labels.iloc[idx]['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'sa_prediction':
            # Return sensitive attribute prediction tuple: (image, sensitive attribute tensor, sensitive attribute tensor unsqueezed)
            sensitive_vals = self.patient_labels.iloc[idx][self.sensitive_attributes].values
            label = make_sensitive_attribute_tensor(self.use_attrs, self.sensitive_attributes, sensitive_vals)
            return image, label, label.unsqueeze(0)
        elif self.mode == 'refusion':
            # For refusion, return (image, target label, sensitive attributes)
            label = self.patient_labels.iloc[idx][self.target]
            # If the target is stored as a list in string form, convert it:
            if isinstance(label, str) and label.startswith('['):
                label = ast.literal_eval(label)
            label = torch.tensor(label, dtype=torch.float32)
            sensitive_vals = self.patient_labels.iloc[idx][self.sensitive_attributes].values
            sensitive_vals = make_sensitive_attribute_tensor(self.use_attrs, self.sensitive_attributes, sensitive_vals, age_attr='anchor_age')
            return image, label, sensitive_vals
        elif self.mode == 'diffusion':
            # For diffusion, simply return the image
            return image