# Copy and extract data over to the node
source "../slurm/copy_and_extract_data.sh"

python ../main.py \
    --train_cfg "../cfgs/cnn/resnet50_hp_1024_test.json" \
    --enc_pth "../pretrained_encoders/100K_benthicnet_resnet50_checkpoint_epoch=99-val_loss=0.1433.ckpt" \
    --csv "../data_csv/benthicnet_nn.csv" \
    --nodes 1 \
    --gpus 2 \
    --name "hl_train_test"
