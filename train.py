import os
import torch
import random
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    model_checkpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from config import Config
from dataset.dataloader import RubikDataModule
from model.DeepcubeA_module import DeepcubeA
import datetime

torch.set_float32_matmul_precision("medium")


def main():
    # Parse configuration
    config = Config()
    args = config.parse_args()

    # Set random seed
    seed_everything(args.seed, workers=True)

    args.log_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M")
    )
    args.checkpoint_dir = os.path.join(args.log_dir, args.checkpoint_dir)
    args.converged_checkpoint_dir = os.path.join(
        args.log_dir, args.converged_checkpoint_dir
    )

    # Configure accelerator and devices
    if args.devices.lower() == "cpu":
        accelerator = "cpu"
        devices = 1  # CPU uses one process by default
    elif args.devices.lower() == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = "auto"
    else:
        # User specified GPU id(s)
        accelerator = "gpu"
        if "," in args.devices:
            devices = [int(x) for x in args.devices.split(",")]
        else:
            devices = [int(args.devices)]

    # Create required directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.converged_checkpoint_dir, exist_ok=True)

    # Initialize model once and reuse it in later stages
    model = DeepcubeA(args)

    # Set initial K and maximum K
    initial_K = 16
    max_K = args.K  # Can be adjusted as needed

    model_e_checkpoint = "logs/20250818_1819/converged_checkpoints/final_model_K_14.pth"
    model.model_theta_e.load_state_dict(torch.load(model_e_checkpoint))
    model_checkpoint = "logs/20250818_1819/converged_checkpoints/final_model_K_15.pth"
    model.model_theta.load_state_dict(torch.load(model_checkpoint))

    for K in range(initial_K, max_K + 1):
        print(f"\n--- 开始训练 K={K} ---")

        # Update model K
        model.K = K

        # Create new dataset configuration
        args.K = K  # Set current K

        # Initialize new data module
        data_module = RubikDataModule(args)

        # # Set callback function (disabled for now as it seems unnecessary)
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=args.checkpoint_dir,
        #     filename=f'K_{K}_'+'{epoch}-{val_loss:.2f}',
        #     save_top_k=3,
        #     monitor='val_loss',
        #     mode='min'
        # )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        )

        # lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # # Configure logger (use separate log directory for each K)
        # logger = TensorBoardLogger(
        #     save_dir=args.log_dir,
        #     name=f'train_logs_K_{K}'
        # )

        # Initialize trainer; by default validate once per epoch (about 5000 steps)
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            precision="16-mixed",  # Enable mixed precision
            devices=devices,
            logger=False,
            callbacks=[early_stopping_callback],
            deterministic=True,
            enable_progress_bar=True,
            enable_checkpointing=True,
        )

        print(trainer.log_every_n_steps)

        # Train model
        trainer.fit(model, datamodule=data_module)

        print(f"--- 完成训练 K={K} ---\n")


if __name__ == "__main__":
    main()
