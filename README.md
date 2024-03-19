# BenthicNet Probes & Testing

## Purpose
The purpose of this repo is to provide code and support for working with BenthicNet - a large dataset of benthic (seafloor) imagery.
The code here may be used for hierarchical multi-label (HML) learning or for one-hot multi-class tasks.
This framework also provides support for testing on frozen (non-trainable) encoders or for fine-tuning/unfrozen encoders.

## Getting Started
To use this repo, we recommend setting up a virtual Python environment and then installing the packages listed in requirements.txt.
```bash
pip install -r requirements.txt
```
Additionally, we recommend using the provided notebook "one_hot.ipynb" as a demonstration of how to prepare a model, 
prepare dataloaders, and train the model on a one-hot classification task. This notebook may also be easily adapted
for frozen encoders and hierarchical multi-label learning.

## Directory Guide (In Alphabetical Order)
1. benthicnet_norms - stats for average/standard deviation on BenthicNet imagery partitions;
2. cfgs - contains json configuration files for training models of differing architecture, probes vs fine-tuning/learning, etc;' 
3. graph_info - contains csv for all BenthicNet annotations, which is used to build hierarchical graphs for HML classification;
4. scripts - scripts prepared for use with SLURM scheduling on high performance clusters (HPCs);
5. slurm - auxiliary scripts used for moving and extracting imagery from tarballs during HPC training/limited multi-node support; and
6. utils - contains auxiliary code and inner workings for running the required learning tasks.

## Main Directory Python Files
1. check_for_problem_imgs.py - a python script for checking if there are any issues with accessing required images based on the dataset csv;
2. main.py - main python code which is called when running hierarchical training tasks;
3. main_one_hot.py - main python code called when running one-hot multi-class classification tasks;
4. one_hot.ipynb - a demonstration notebook showcasing parameters, the parts of configuration files, and how main_one_hot.py functions;
5. test_depth.ipynb - a notebook used for calculating evaluation metrics on test data; and
6. test_depth_group.ipynb - adaptation of test_depth.ipynb to facilitate testing multiple models at once.

## Known Issues
1. Multi-node training is currently not fully supported. Specifically, the copy & extract to move data to each node is not implemented.
   If this is not an issue, then by default, the DDP strategy in Lightning should support multi-node.

## Contact
For any questions, concerns, or issues in using this repo, please contact isaac.xu@dal.ca. 
