import pathlib
import numpy as np
import pandas as pd
import pickle
import torch
import itertools
from functools import wraps
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def subsample_data(df, sa_name=None, sa_values=None, target_name=None, target_values=None, result_sa_ratios=None, result_class_ratios=None):
    assert result_sa_ratios is not None or result_class_ratios is not None
    if result_sa_ratios is None:
        df = df[df[target_name].isin(target_values)]
        # joint_idx, joint_values = pd.factorize(pd._libs.lib.fast_zip([df[target_name].values]))
        for i, target_value in enumerate(target_values):
            df.loc[df[target_name] == target_value, 'joint_idx'] = i
        joint_ratio = np.array(result_class_ratios)
    elif result_class_ratios is None:
        df = df[df[sa_name].isin(sa_values)]
        # joint_idx, joint_values = pd.factorize(pd._libs.lib.fast_zip([df[sa_name].values]))
        for i, sa_value in enumerate(sa_values):
            df.loc[df[sa_name] == sa_value, 'joint_idx'] = i
        joint_ratio = np.array(result_sa_ratios)
    else:
        df = df[df[sa_name].isin(sa_values) & df[target_name].isin(target_values)]
        # joint_idx, joint_values = pd.factorize(pd._libs.lib.fast_zip([df[target_name].values, df[sa_name].values]))
        for i, (sa_value, target_value) in enumerate(itertools.product(sa_values, target_values)):
            df.loc[(df[sa_name] == sa_value) & (df[target_name] == target_value), 'joint_idx'] = i
        joint_ratio = (np.array([result_sa_ratios]).T * np.array(result_class_ratios)).flatten()
    
    joint_ratio = joint_ratio / joint_ratio.max()
    current_joint_nums = df['joint_idx'].value_counts().sort_index().to_dict()
    keep_amount = [num / ratio for num, ratio in zip(current_joint_nums.values(), joint_ratio)]
    min_amount = np.min(keep_amount)
    joint_ratio = [min_amount * ratio / cur_num for ratio, cur_num in zip(joint_ratio, current_joint_nums.values())]
    for i in range(len(joint_ratio)):
        joint_num = current_joint_nums[i]
        target_joint_num = int(joint_num * joint_ratio[i])
        df = df.drop(df[df['joint_idx'] == i].sample(n=joint_num - target_joint_num).index)
    return df.reset_index(drop=True)

def do_subgroup_balance_sample(df):
    ros = RandomUnderSampler()
    subgroups = df['joint_idx']
    X_resampled, _ = ros.fit_resample(df, subgroups)
    return X_resampled

def age_to_class(age):
    # if age < 60:
    #     return 0
    # else:
    #     return 1
    return (age >= 60).long()

def do_cache():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data_cache_path = kwargs.get('data_cache_path', None)
            if data_cache_path is not None and pathlib.Path(data_cache_path).exists():
                print(f'Loading data from {data_cache_path}')
                return pickle.load(open(data_cache_path, 'rb'))
            print('WARNING: data cache not found, computing from scratch')
            result = func(*args, **kwargs)
            if data_cache_path is not None:
                if not pathlib.Path(data_cache_path).parent.exists():
                    pathlib.Path(data_cache_path).parent.mkdir(parents=True)
                pickle.dump(result, open(data_cache_path, 'wb'))
            return result
        return wrapper
    return decorator

def subsample_data(df, sample_method='under', subgroup_key='joint_idx'):
    if sample_method == 'over':
        ros = RandomOverSampler()
        subgroups = df[subgroup_key]
        X_resampled, _ = ros.fit_resample(df, subgroups)
    else:
        rus = RandomUnderSampler()
        subgroups = df[subgroup_key]
        X_resampled, _ = rus.fit_resample(df, subgroups)
    return X_resampled

def do_subgroup_balance_sample(d, sample_method='under'):
    d.patient_labels['age_class'] = d.patient_labels['AGE_AT_CXR'].apply(lambda x: 1 if x >= 60 else 0)
    d.sensitive_attributes = ['age_class' if sa == 'AGE_AT_CXR' else sa for sa in d.sensitive_attributes]
    joint_idx = pd.factorize(pd._libs.lib.fast_zip([d.patient_labels[sa].values for sa in d.sensitive_attributes]))[0]
    d.patient_labels['joint_idx'] = joint_idx
    d.patient_labels = subsample_data(d.patient_labels, sample_method=sample_method)
    d.sensitive_attributes = ['AGE_AT_CXR' if sa == 'age_class' else sa for sa in d.sensitive_attributes]
    d.patient_labels = d.patient_labels.sample(frac=1, random_state=0).reset_index(drop=True)
    return d

def do_class_balance_sample(d, target):
    d.patient_labels = subsample_data(d.patient_labels, sample_method='over', subgroup_key=target)
    d.patient_labels = d.patient_labels.sample(frac=1, random_state=0).reset_index(drop=True)
    return d

def make_sensitive_attribute_tensor(use_attrs, sensitive_attributes, sensitive_attribute_vals, age_attr='AGE_AT_CXR'):
    for i, sensitive_attribute in enumerate(sensitive_attributes):
        if sensitive_attribute == age_attr:
            sensitive_attribute_vals[i] = 1 if sensitive_attribute_vals[i] >= 60 else 0
            continue
        sensitive_attribute_vals[i] = use_attrs[sensitive_attribute].index(sensitive_attribute_vals[i])
    return torch.tensor(np.float32(sensitive_attribute_vals)) 
