#!/bin/bash
sbatch --job-name=ohft_ohp_rn50 ohft_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/one_hot_encoders/ohl_rn50_epoch=99-val_loss=0.7047.ckpt ohft_oh_rn50
sbatch --job-name=ohft_hl_rn50 ohft_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/one_hot_encoders/ohp_hl_rn50_epoch=99-val_loss=1.0108.ckpt ohft_hl_rn50
sbatch --job-name=ohft_hl_400e_rn50 ohft_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/one_hot_encoders/ohp_hl_rn50_400e_epoch=99-val_loss=0.9211.ckpt ohft_hl_400e_rn50
sbatch --job-name=ohft_img_rn50 ohft_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/one_hot_encoders/ohp_img_rn50_epoch=99-val_loss=0.8577.ckpt ohft_img_rn50
sbatch --job-name=ohft_mcv2_rn50 ohft_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/one_hot_encoders/ohp_mcv2_rn50_epoch=99-val_loss=0.7799.ckpt ohft_mcv2_rn50
