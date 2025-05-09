import argparse
import ast
import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional_tags", type=ast.literal_eval, help="Additional tags")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--conv", type=ast.literal_eval, help="Convolutional layers")
    parser.add_argument("--cv_fold_idx", type=int, help="Cross-validation fold index")
    parser.add_argument("--cv_fold_total", type=int, help="Total number of cross-validation folds")
    parser.add_argument("--d_model_cv_root", type=str, help="Root path for cross-validation results")
    parser.add_argument("--data_cache_path", type=str, help="Path to data cache file")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--image_root", type=str, help="Root directory of images")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--model_save_root", type=str, help="Path to save the trained model")
    parser.add_argument("--num_cascade_blocks", type=int, help="Number of cascade blocks")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs")
    parser.add_argument("--patient_info_path", type=str, help="Path to patient info file")
    parser.add_argument("--patient_labels_path", type=str, help="Path to patient labels file")
    parser.add_argument("--run_mode", type=str, help="Run mode")
    parser.add_argument("--sa_encoder", type=str, help="Self-attention encoder type")
    parser.add_argument("--sensitive_attributes", type=ast.literal_eval, help="List of sensitive attributes")
    parser.add_argument("--subsample_attr", type=str, help="Subsample attribute")
    parser.add_argument("--subsample_ratios", type=ast.literal_eval, help="Subsample ratios")
    parser.add_argument("--subsample_target", type=str, help="Subsample target")
    parser.add_argument("--subsample_target_ratios", type=ast.literal_eval, help="Subsample target ratios")
    parser.add_argument("--subsample_target_values", type=ast.literal_eval, help="Subsample target values")
    parser.add_argument("--subsample_values", type=ast.literal_eval, help="Subsample values")
    parser.add_argument("--target", type=str, help="Target")
    parser.add_argument("--use_attrs", type=ast.literal_eval, help="Attributes to use")
    return parser.parse_args()

def get_best_decision_threshold(y_true, y_score, save_fig_path=None, fig_name=None, method='min_gap'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    if method == 'min_gap':
        gaps = abs(tpr - (1-fpr))
        ix = np.argmin(gaps)
    elif method == 'gmeans':
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
    elif method == 'youden_j':
        youden_j = tpr - fpr
        ix = np.argmax(youden_j)
    elif method == 'none':
        return 0.5
    elif method == 'test_all':
        all_thresholds = {}
        # calculate thresholds for all methods
        for method in ['min_gap', 'gmeans', 'youden_j', 'none']:
            if method == 'min_gap':
                gaps = abs(tpr - (1-fpr))
                ix = np.argmin(gaps)
            elif method == 'gmeans':
                gmeans = np.sqrt(tpr * (1-fpr))
                ix = np.argmax(gmeans)
            elif method == 'youden_j':
                youden_j = tpr - fpr
                ix = np.argmax(youden_j)
            elif method == 'none':
                # find closest to 0.5
                ix = np.argmin(np.abs(thresholds - 0.5))
            all_thresholds[method] = ix
        print({'key': thresholds[ix] for key, ix in all_thresholds.items()})
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.title(f'ROC Curve')
        for method, ix in all_thresholds.items():
            plt.plot(fpr[ix], tpr[ix], 'o', label=f'{method} threshold={thresholds[ix]:.2f}')
        os.makedirs(save_fig_path, exist_ok=True)
        plt.legend()
        plt.savefig(f'{save_fig_path}/{fig_name}.png')
        plt.close()
        return 0.5
    return thresholds[ix]

def get_best_threshold_with_dataloader(dataloader, model, save_fig_path=None, fig_name=None, method='min_gap'):
    fairness_agg_ls = [[], [], []]
    with torch.no_grad():
        for i, (x, y, attrs) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y, attrs = x.cuda(), y.cuda(), attrs.cuda()
            outputs = model(x, attrs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.squeeze(1) if len(outputs.shape) > 1 else outputs
            fairness_agg_ls[0].append(outputs.detach())
            fairness_agg_ls[1].append(y.detach())
            fairness_agg_ls[2].append(attrs.detach())
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
            best_threshold = get_best_decision_threshold(labels_current_group.cpu(), outputs_current_group.cpu(), save_fig_path=save_fig_path, fig_name=fig_name + f'_group_{j}', method=method)
            # best_threshold = 0.5
            model.threshold_dict[j] = torch.tensor(best_threshold).cuda()


