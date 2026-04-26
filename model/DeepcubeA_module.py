import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from model.DNN import DNN
from model.Cube import TARGET_STATE_ONE_HOT


class RelativeMSELoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(((pred - target) / (target + 1e-8)) ** 2)


# Check whether A contains a tensor equal to b
def row_allclose_mask(A, b, rtol=1e-4, atol=1e-6):
    # Compute element-wise error
    diff = torch.abs(A - b)  # (B, D)
    tol = atol + rtol * torch.abs(b)  # (D,), broadcast automatically expands to (B, D)

    # Element mask that satisfies tolerance criteria
    mask_elements = diff <= tol  # (B, D), bool

    # Check whether all elements in each row satisfy the criteria
    mask_rows = mask_elements.all(dim=1)  # (B,)

    return mask_rows


class DeepcubeA(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.convergence_threshold = config.convergence_threshold
        self.chunk_size = config.chunk_size
        self.converged_checkpoint_dir = config.converged_checkpoint_dir
        self.compile = config.compile

        # Input dimension (54 stickers, each with 6 possible colors, one-hot encoded)
        self.input_dim = 54 * 6

        self.model_theta = DNN(self.input_dim, num_residual_blocks=4)  # Trainable model
        self.model_theta_e = DNN(
            self.input_dim, num_residual_blocks=4
        ).eval()  # Supervision/target model

        self.target_state = torch.tensor(
            TARGET_STATE_ONE_HOT, dtype=torch.float32
        ).reshape(1, -1)

        if self.compile:
            self.model_theta = torch.compile(self.model_theta)
            self.model_theta_e = torch.compile(self.model_theta_e)

        self.K = 1

        # Loss function
        self.criterion = nn.MSELoss()

        # Save hyperparameters
        self.save_hyperparameters(config)

    def transfer_batch_to_tensor(self, batch):
        """
        Convert batch data to tensors and move to the correct device in bulk.
        Args:
            batch: Input batch data.
        Returns:
            Processed batch dictionary containing tensor data.
        """
        batch_dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_dict[key] = value.to(self.device)
            else:
                batch_dict[key] = torch.tensor(value, device=self.device)
        return batch_dict

    def forward(self, x):
        return self.model_theta(x)

    def model_step(self, batch):
        # Extract states and neighbors from batch
        batch_dict = self.transfer_batch_to_tensor(batch)
        states = batch_dict["state"]
        neighbor_states = batch_dict["neighbors"]

        B, N, D = neighbor_states.shape

        states = F.one_hot(states.long(), num_classes=6).float().view(B, -1)
        neighbor_states = (
            F.one_hot(neighbor_states.long(), num_classes=6).float().view(B * N, -1)
        )

        # Predict in chunks to avoid OOM on GPU memory
        num_chunks = (B * N + self.chunk_size - 1) // self.chunk_size
        chunked_neighbors = torch.chunk(neighbor_states, num_chunks, dim=0)

        with torch.no_grad():
            neighbor_costs = []
            for chunk in chunked_neighbors:
                mask = row_allclose_mask(chunk, self.target_state.to(chunk.device))
                cost = self.model_theta_e(chunk)
                cost[mask] = 0.0
                neighbor_costs.append(cost)

            # Aggregate results
            neighbor_costs = torch.cat(neighbor_costs, dim=0)
            neighbor_costs = neighbor_costs.view(B, N)

        # Compute min[J_theta_e(A(x_i, a)) + 1]
        min_neighbor_cost = neighbor_costs.abs().min(dim=1)[0] + 1

        # Predict current state cost using model_theta
        current_cost = self.model_theta(states)

        # Always compute loss
        loss = self.criterion(current_cost.squeeze(), min_neighbor_cost)
        return loss, current_cost

    def training_step(self, batch, batch_idx):
        # Call model_step to get loss
        loss, _ = self.model_step(batch)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # Get validation loss
        val_loss = self.trainer.callback_metrics.get("val_loss")

        if val_loss is not None and val_loss < self.convergence_threshold:
            self.log("converged", True)

            # Save model weights to the dedicated converged checkpoint directory
            import os

            os.makedirs(self.converged_checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.converged_checkpoint_dir, f"converged_model_K_{self.K}.pth"
            )
            torch.save(self.model_theta.state_dict(), checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")

            # If converged, update model_theta_e
            self.model_theta_e.load_state_dict(self.model_theta.state_dict())

            # The original paper does not explicitly state whether to carry over
            # previous-round parameters; here we fully inherit to avoid expensive
            # retraining from scratch.
            # self.model_theta = DNN(self.input_dim, num_residual_blocks=4)
            # if self.compile:
            #     self.model_theta = torch.compile(self.model_theta)

            # Stop training
            self.trainer.should_stop = True

    def on_train_end(self):
        # Check whether training ended normally (not due to early stopping)
        # Save only when training did not stop because of convergence
        if not self.trainer.callback_metrics.get("converged", False):
            # Get validation loss of the last epoch
            val_loss = self.trainer.callback_metrics.get("val_loss")

            # Save model weights to the dedicated converged checkpoint directory
            import os

            os.makedirs(self.converged_checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.converged_checkpoint_dir, f"final_model_K_{self.K}.pth"
            )
            torch.save(self.model_theta.state_dict(), checkpoint_path)
            print(f"训练结束，模型已保存到 {checkpoint_path}")

            # Update model_theta_e
            self.model_theta_e.load_state_dict(self.model_theta.state_dict())

    def validation_step(self, batch, batch_idx):
        # Compute validation loss
        loss, current_cost = self.model_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_cost", current_cost.mean(), on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model_theta.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return {"optimizer": optimizer}

    def load_state_dict_theta_e(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.model_theta_e.load_state_dict(state_dict)
        self.model_theta_e.zero_output = False
