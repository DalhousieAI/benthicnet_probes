python ../main.py ^
--train_cfg "../cfgs/cnn/resnet50_hl.json" ^
--enc_pth "../pretrained_encoders/100K_benthicnet_resnet50_checkpoint_epoch=99-val_loss=0.1433.ckpt" ^
--test_mode true ^
--nodes 1 ^
--gpus 1 ^
--windows true ^
--name "probe_test"
