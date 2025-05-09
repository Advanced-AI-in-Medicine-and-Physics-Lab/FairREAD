# FairREAD

This is the code repo for "FairREAD: Re-fusing Demographic Attributes after Disentanglement for Fair Medical Image Classification".

[![arXiv](https://img.shields.io/badge/arXiv-2411.04424v1-b31b1b.svg)](https://arxiv.org/abs/2411.04424v1)

## Introduction
Recent advancements in deep learning have shown transformative potential in medical imaging, yet concerns about fairness persist due to performance disparities across demographic subgroups. Existing methods aim to address these biases by mitigating sensitive attributes in image data; however, these attributes often carry clinically relevant information, and their removal can compromise model performance—a highly undesirable outcome. To address this challenge, we propose Fair Re-fusion After Disentanglement (FairREAD), a novel, simple, and efficient framework that mitigates unfairness by re-integrating sensitive demographic attributes into fair image representations. FairREAD employs orthogonality constraints and adversarial training to disentangle demographic information while using a controlled re-fusion mechanism to preserve clinically relevant details. Additionally, subgroup-specific threshold adjustments ensure equitable performance across demographic groups. Comprehensive evaluations on a large-scale clinical X-ray dataset demonstrate that FairREAD significantly reduces unfairness metrics while maintaining diagnostic accuracy, establishing a new benchmark for fairness and performance in medical image classification.

## Run experiments

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find this work useful, please consider citing our paper:
```

```