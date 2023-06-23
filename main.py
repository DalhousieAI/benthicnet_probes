import os
from argparse import Namespace
from datetime import datetime
from json import loads

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from utils.benthicnet_dataset import gen_datasets
from utils.utils import (
    construct_dataloaders,
    construct_model,
    gen_R_mat,
    gen_root_graphs,
    get_augs,
    get_df,
    parser,
    process_data_df,
    set_seed,
)


# train_cfg, nodes, gpus, tar_dir, csv, colour_jitter, enc, graph_pth, seed, random_partition, name, windows
def main():
    # Prepare argument parameters
    args = parser()
    set_seed(args.seed)

    # Set up environment variables
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    if args.windows:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    # Set up training configurations
    train_cfg_path = args.train_cfg
    with open(train_cfg_path, "r") as f:
        train_cfg_content = f.read()

    train_cfg = loads(train_cfg_content)
    train_kwargs = OmegaConf.create(train_cfg)

    # Get graphs
    root_graphs, _, _ = gen_root_graphs(args.graph_pth)
    Rs = {root: gen_R_mat(graph) for root, graph in root_graphs.items()}

    # Build model
    model = construct_model(train_kwargs, Rs, args.enc_pth, args.test_mode)

    # Set up data
    data_df = get_df(args.csv)
    data_df = process_data_df(data_df, Rs)

    train_transform, val_transform = get_augs(args.colour_jitter)
    transform = [train_transform, val_transform]

    train_dataset, val_dataset, test_dataset = gen_datasets(
        data_df, args.tar_dir, transform, seed=args.seed
    )

    del data_df  # Save memory after using data_df

    if args.random_partition:
        dataloaders = construct_dataloaders(
            [train_dataset, val_dataset, test_dataset], train_kwargs
        )
    else:
        raise NotImplementedError(
            "Geo-spatially based test partitioning has not yet been completed."
        )

    del train_dataset, val_dataset, test_dataset  # Save memory after using datasets

    train_dataloader = dataloaders[0]
    val_dataloader = dataloaders[1]
    test_dataloader = dataloaders[2]

    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    csv_logger = CSVLogger("logs", name=args.name + "_logs", version=timestamp)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="checkpoint_{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        every_n_epochs=train_kwargs.max_epochs,
        save_weights_only=False,
    )

    # Determine logging rate
    total_steps_per_epoch = len(train_dataloader)
    # Number of times to update logs per epoch (needs to be adjusted if sample size is small and batch size is big)
    num_log_updates_per_epoch = 4

    log_every_n_steps = total_steps_per_epoch // num_log_updates_per_epoch

    # Automatically log learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, lr_monitor]

    trainer_args = Namespace(**train_kwargs)

    if args.test_mode:
        del train_dataloader, val_dataloader
        trainer = Trainer.from_argparse_args(
            trainer_args,
            logger=csv_logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=False),
            accelerator="cuda",
            num_nodes=1,
            devices=[0],
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=False,
        )

        trainer.test(model, dataloaders=test_dataloader)
    else:
        del test_dataloader
        trainer = Trainer.from_argparse_args(
            trainer_args,
            logger=csv_logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=False),
            accelerator="cuda",
            num_nodes=args.nodes,
            devices=args.gpus,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=True,
        )

        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
