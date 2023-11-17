#!/bin/bash
sbatch --job-name=ohft_ohp_rn50 ohft_rn50.sh /lustre06/project/6012565/isaacxu/benthicnet_probes/pretrained_encoders/one_hot_models/ohl_rn50_epoch=99-val_loss=0.7047.ckpt ohft_oh_rn50
