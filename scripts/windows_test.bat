python ../main.py ^
--train_cfg "../cfgs/cnn/resnet50_hl.json" ^
--csv "../data_csv/benthicnet_nn.csv" ^
--nodes 1 ^
--gpus 1 ^
--windows true ^
--name "probe_test"
