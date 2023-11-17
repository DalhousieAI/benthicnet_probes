#!/bin/bash
sbatch --job-name=hp_img_rn50 hp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/imagenet_models/imagenet_v2_rn50.ckpt hp_img_rn50
sbatch --job-name=hp_mcv2_rn50 hp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/mocov2+-100e_epoch=099.ckpt hp_mcv2_rn50
sbatch --job-name=hp_bt_rn50 hp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/bt-100e_epoch=099.ckpt hp_bt_rn50
sbatch --job-name=hp_byol_rn50 hp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/byol-100e_epoch=099.ckpt hp_byol_rn50
sbatch --job-name=hp_simsiam_rn50 hp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/simsiam-100e_epoch=099.ckpt hp_simsiam_rn50
