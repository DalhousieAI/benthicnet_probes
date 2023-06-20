python ../main.py ^
--train_cfg "../cfgs/cnn/resnet50_hl.json" ^
--csv "../data_csv/size_10K_benthicnet.csv" ^
--nodes 1 ^
--gpus 1 ^
--windows true ^
--name "probe_test"
