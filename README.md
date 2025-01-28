# Code to reproduce "Mapping Conservation Practices via Deep Learning: Improving Performance via Hillshade Imagery, Sampling Design, and Centerline Dice Loss"

#### Authors:          Charles J. Labuzzetta, Zhengyuan Zhu
#### Point of contact: Charles J. Labuzzetta (clabuzzetta@gmail.com)
#### Repository Type:  Python
#### Year of Origin:   2024 (original publication)
#### Year of Version:  2024
#### Version:          1.0.0
#### Digital Object Identifier (DOI): [https://doi.org/10.1080/29979676.2024.2401756](https://doi.org/10.1080/29979676.2024.2401756)

***

_Suggested Citation:_

Labuzzetta, CJ, Zhu, Z.
2024
Code to reproduce "Mapping Conservation Practices via Deep Learning: Improving Performance via Hillshade Imagery, Sampling Design, and Centerline Dice Loss"
Software release. Ames, IA.
[https://github.com/labuzzetta/bmp_unet](https://github.com/labuzzetta/bmp_unet)

_Authors' [ORCID](https://orcid.org) nos.:_

- Charles J. Labuzzetta [0000-0002-6027-0120](https://orcid.org/0000-0002-6027-0120)

***
***

This repository contains the code used to implement the experiments described in Labuzzetta, C. J., & Zhu, Z. (2024). Mapping Conservation Practices via Deep Learning: Improving Performance via Hillshade Imagery, Sampling Design, and Centerline Dice Loss. Statistics and Data Science in Imaging, 1(1). https://doi.org/10.1080/29979676.2024.2401756. These scripts can be used as an example to build image classification pipelines for similar use cases, dive into the methodology, and/or reproduce the analysis itself. Access to the original imagery data may be coordinated on a case-by-case basis.

## Repository organization

The repository contains the following folders and files:

- Files in the main-level of the repository
  * `LICENSE.md`: is the official license.
  * `README.md` is this document.
- `cnn/` contains the code to that was used to train models and evaluate the performance of the models on the test set.
  * Any file beginning with `model_wrr_gw_*` is one of the 5 experimental models (nolidar, lidar, lidar_imagenet, pps, cldice) trained on the grassed waterway dataset.
  * Any file beginning with `model_wrr_pd_*` is one of the 5 experimental models (nolidar, lidar, lidar_imagenet, pps, cldice) trained on the pond dam dataset.
  * Any file beginning with `model_wrr_te_*` is one of the 5 experimental models (nolidar, lidar, lidar_imagenet, pps, cldice) trained on the terrace dataset.
  * Any file beginning with `model_wrr_wa_*` is one of the 5 experimental models (nolidar, lidar, lidar_imagenet, pps, cldice) trained on the WASCOB dataset.
  * Any file beginning with `test_wrr_gw_*` evaluates the corresponding model file above on the grassed waterway test set.
  * Any file beginning with `test_wrr_pd_*` evaluates the corresponding model file above on the pond dam test set.
  * Any file beginning with `test_wrr_te_*` evaluates the corresponding model file above on the terrace test set.
  * Any file beginning with `test_wrr_wa_*` evaluates the corresponding model file above on the WASCOB test set.
  * The `model/` directory contains additional python functions that are shared in common among the scripts above.
    - The `augmentation.py` file contains functions used to perform data augmentation as described in the paper.
    - The `callbacks.py` file contains functions that help with monitoring and adjusting performance during training.
    - The `losses.py` file contains functions that calculate the loss during training.
- `results/` contains the outputs from running the `model_*` and `test_*` scripts of the same file name in `cnn/`. 
  * `testing/` contains the results from running the `test_*` scripts on the respective test sets.
  * `validation/` contains the results from running the `model_*` scripts on the respective validation sets (used/evaluated during training).
- `slurm/` contains the slurm submission scripts for the `model_*` and `test_*` scripts of the same file name in `cnn/`.
  * `train/` contains the slurm submission scripts used to run the `model_*` files in `cnn/`.
  * `test/` contains the slurm submission scripts used to run the `test_*` files in `cnn/`.

# Acknowledgments

This research was funded by the Center for Survey Statistics
and Methodology at Iowa State University, Ames, Iowa.
