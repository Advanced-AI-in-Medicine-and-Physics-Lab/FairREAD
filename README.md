# FairREAD

This is the code repo for "FairREAD: Re-fusing Demographic Attributes after Disentanglement for Fair Medical Image Classification".

[![arXiv](https://img.shields.io/badge/arXiv-2412.16373-b31b1b.svg)](https://arxiv.org/abs/2412.16373)

## Introduction
Recent advancements in deep learning have shown transformative potential in medical imaging, yet concerns about fairness persist due to performance disparities across demographic subgroups. Existing methods aim to address these biases by mitigating sensitive attributes in image data; however, these attributes often carry clinically relevant information, and their removal can compromise model performance—a highly undesirable outcome. To address this challenge, we propose Fair Re-fusion After Disentanglement (FairREAD), a novel, simple, and efficient framework that mitigates unfairness by re-integrating sensitive demographic attributes into fair image representations. FairREAD employs orthogonality constraints and adversarial training to disentangle demographic information while using a controlled re-fusion mechanism to preserve clinically relevant details. Additionally, subgroup-specific threshold adjustments ensure equitable performance across demographic groups. Comprehensive evaluations on a large-scale clinical X-ray dataset demonstrate that FairREAD significantly reduces unfairness metrics while maintaining diagnostic accuracy, establishing a new benchmark for fairness and performance in medical image classification.

## Run experiments

1. Clone the repository:
```bash
git clone git@github.com:Advanced-AI-in-Medicine-and-Physics-Lab/FairREAD.git
cd FairREAD
```
2. Create a conda environment and install the required packages:
```bash
conda environment create -f environment.yml
conda activate fairread
```

3. Download the CheXpert dataset, which is used in our experiments. You can find the dataset [here](https://stanfordmlgroup.github.io/competitions/chexpert/). We use a curated version of the dataset, where the serialized dataset objects are available under the `data_cache` directory, and will be loaded automatically. Note that the original images are not included in these serialized files, so you need to download them from the link above. 

4. The first-stage training of the fair image encoder is completed based on the [FCRO](https://github.com/ubc-tea/FCRO-Fair-Classification-Orthogonal-Representation?tab=readme-ov-file) repo. The trained model checkpoints are available in the `disentangled` directory. Alternatively, you can also pretrain the model using the FCRO repo following their instructions. If you choose to use our pretrained checkpoints, nothing needs to be changed. 

5. Modify the config files in the `configs` directory according to your needs. Specifically, please make sure to modify `data_root` to the root of the downloaded CheXpert dataset. The config files include the follows:
    - `cross_validation_erm.yaml`: Config file for cross-validation with the ERM model (which means the vanilla model without any optimization for fairness).
   - `cross_validation_refusion_<pathology>.yaml`: Config file for cross-validation with refusion for each pathology, including Cardiomegaly, Pleural Effusion, Pneumonia, and Fracture.

6. This project relies on Weights & Biases (wandb) for logging. Login to your wandb account using the following command:
```bash
wandb login
```
If you don't have a wandb account, you can create one [here](https://wandb.ai/site).

7. Run the training script:
```bash
python train.py -cn "cross_validation_fairread_<pathology>"
```
where `<pathology>` is the pathology you want to train the model for. 

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find this work useful, please consider citing our paper:
```
@misc{gao2024fairread,
      title={FairREAD: Re-fusing Demographic Attributes after Disentanglement for Fair Medical Image Classification}, 
      author={Yicheng Gao and Jinkui Hao and Bo Zhou},
      year={2024},
      eprint={2412.16373},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.16373}, 
}
```