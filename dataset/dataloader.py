import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from model.Cube import Cube, TARGET_STATE


class RubikDataset(Dataset):
    def __init__(self, config, num_samples, is_train=True):
        super().__init__()
        self.config = config
        self.num_samples = num_samples
        self.is_train = is_train
        self.cube = Cube()
        self.K = config.K  # Maximum scramble depth
        self.all_actions = list(self.cube.moves.keys())

    def __len__(self):
        return self.num_samples

    def get_neighbors(self, state):
        """
        Get all neighboring states for the given state.
        Args:
            state: Current cube state, np.array.
        Returns:
            A list of all neighboring states.
        """
        return self.cube.get_neibor_state(state)

    def __getitem__(self, idx):
        # Randomly choose scramble depth i ∈ [1, K].
        if (
            np.random.random() < 0.5 and self.is_train
        ):  # Increase probability of depth K during training to speed up convergence
            i = self.K
        else:
            i = np.random.randint(1, self.K + 1)

        # Start from the initial state and apply i random actions
        state = TARGET_STATE.copy()
        # Sample i random actions
        actions = np.random.choice(self.all_actions, size=i, replace=True)

        for action in actions:
            state = self.cube.apply_action(state, action)

        # Get all neighboring states
        neighbor_states = self.get_neighbors(state.copy())

        # Return data wrapped in a dict
        return {
            "state": state,  # 54
            "steps": i,
            "neighbors": neighbor_states,  # 12, 54
        }


class RubikDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.num_train_samples = config.num_train_samples
        self.num_val_samples = config.num_val_samples

    def prepare_data(self):
        # No download needed; the dataset is generated automatically
        pass

    def setup(self, stage=None):
        # Create training and validation datasets
        self.train_dataset = RubikDataset(
            self.config, self.num_train_samples, is_train=True
        )
        self.val_dataset = RubikDataset(
            self.config, self.num_val_samples, is_train=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self._worker_init_fn,
        )

    def _worker_init_fn(self, worker_id):
        # Get worker's initial seed (changes with epoch)
        worker_seed = (self.config.seed + worker_id + torch.initial_seed()) % 2**32

        # Set seeds for numpy, torch, and python random
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
