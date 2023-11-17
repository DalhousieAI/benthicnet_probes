#!/bin/bash
sbatch --job-name=ohp_hl_rn50 ohp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/hl_models/hl_rn50.ckpt ohp_hl_rn50
sbatch --job-name=ohp_hl_400e_rn50 ohp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/hl_models/hl_rn50_400e.ckpt ohp_hl_400e_rn50
sbatch --job-name=ohp_img_rn50 ohp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/imagenet_models/imagenet_v2_rn50.ckpt ohp_img_rn50
sbatch --job-name=ohp_mcv2_rn50 ohp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/mocov2+-100e_epoch=099.ckpt ohp_mcv2_rn50
sbatch --job-name=ohp_bt_rn50 ohp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/bt-100e_epoch=099.ckpt ohp_bt_rn50
sbatch --job-name=ohp_byol_rn50 ohp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/byol-100e_epoch=099.ckpt ohp_byol_rn50
sbatch --job-name=ohp_simsiam_rn50 ohp_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/ssl_models/simsiam-100e_epoch=099.ckpt ohp_simsiam_rn50
