# BenthicNet Probes & Testing

## Purpose
The purpose of this repo is to provide code and support for working with BenthicNet - a large dataset of benthic (seafloor) imagery.
The code here may be used for hierarchical multi-label (HML) learning or for one-hot multi-class tasks.
This framework also provides support for testing on frozen (non-trainable) encoders or for fine-tuning/unfrozen encoders.

To demonstrate the use of the code presented to load and train a model (pre-trained self-supervised on BenthicNet, supervised on ImageNet, or any other of supported architectures) in a typical one-hot multi-class classification task, we supply a tutorial notebook "one_hot.ipynb".
The code in this notebook also reflects the main_one_hot.py Python script and can be with some minor alterations be modified to work in a hierarchical multi-label setting.

## Getting Started
To use this repo, we recommend setting up a virtual Python environment and then installing the packages listed in requirements.txt.
```bash
pip install -r requirements.txt
```
Additionally, we recommend using the provided notebook "one_hot.ipynb" as a demonstration of how to prepare a model,
prepare dataloaders, and train the model on a one-hot classification task. This notebook may also be easily adapted
for frozen encoders and hierarchical multi-label learning.

Before running, please ensure that all paths to datasets, models, dataset CSVs, etc are correct.

## Directory Guide (In Alphabetical Order)
1. benthicnet_norms - stats for average/standard deviation on BenthicNet imagery partitions;
2. cfgs - contains JSON configuration files for training models of differing architecture, probes vs fine-tuning/learning, etc;'
3. graph_info - contains CSV for all BenthicNet annotations, which is used to build hierarchical graphs for HML classification;
4. scripts - scripts prepared for use with SLURM scheduling on high performance clusters (HPCs);
5. slurm - auxiliary scripts used for moving and extracting imagery from tarballs during HPC training/limited multi-node support; and
6. utils - contains auxiliary code and inner workings for running the required learning tasks.
7. ap_score-to_threshold_or_not.pdf - exerpt discussing the implications of thresholded outputs vs using sigmoid outputs for calculating average precision score.

## Main Directory Python Files
1. check_for_problem_imgs.py - a Python script for checking if there are any issues with accessing required images based on the dataset CSV;
2. main.py - main Python code which is called when running hierarchical training tasks;
3. main_one_hot.py - main Python code called when running one-hot multi-class classification tasks;
4. one_hot.ipynb - a demonstration notebook showcasing parameters, the parts of configuration files, and how to load and train a model in a one hot multi-label classification setting;
5. test_depth.ipynb - a notebook used for calculating evaluation metrics on test data (may require seaborn package as well); and
6. test_depth_group.ipynb - adaptation of test_depth.ipynb to facilitate testing multiple models at once.

## Information on Loading Models
By default, if a path for an encoder checkpoint (file containing model weights) is provided, 
the [model construction function](https://github.com/DalhousieAI/benthicnet_probes/blob/master/utils/utils.py#L677) (or its [one hot variant](https://github.com/DalhousieAI/benthicnet_probes/blob/master/utils/utils.py#L750)) will automatically
get the weights from the checkpoint and attempt to load them via the ["load_model_state" function](https://github.com/DalhousieAI/benthicnet_probes/blob/master/utils/utils.py#L514).
If no path is provided, the model construction randomly initializes weights.

In the tutorial notebook, the contruction function is called with the path provided as an argument and details on how the model is loaded may not be a concern.
However, while attempts have been made to ensure the state loading function is flexible, there are ultimately a limited number of cases that can be considered.
Most notably, the function assumes that the key to accessing the state dictionary is either "state_dict" or in the case of models from Facebook AI, "model".
Additionally, the key within the state dictionary indicating the encoder weights is assumed to be either "encoder" or "backbone".
If the keys for the state dictionary or the encoder do not fall under our assumptions, some minor editing of the "load_model_state" function may be required.

## Known Issues
1. Multi-node training is currently not fully supported. Specifically, the copy and extract shell script to move data to each node is not implemented.
   If this is not an issue, then by default, the DDP strategy in Lightning should support multi-node computation.

## Contact
For any questions, concerns, or issues in using this repo, please contact [isaac.xu@dal.ca](mailto:isaac.xu@dal.ca).
