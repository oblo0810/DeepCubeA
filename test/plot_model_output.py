import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.DNN import DNN
from model.Cube import Cube, TARGET_STATE, TARGET_STATE_ONE_HOT
import torch  # Add missing import
from config import Config

plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue


def generate_shuffled_state(cube, num_shuffles):
    """
    Generate a cube state after a specified number of shuffles.
    Args:
        cube: Cube instance.
        num_shuffles: Number of shuffles.
    Returns:
        Shuffled cube state.
    """
    state = TARGET_STATE.copy()
    all_actions = list(cube.moves.keys())
    actions = np.random.choice(all_actions, size=num_shuffles, replace=True)
    for action in actions:
        state = cube.apply_action(state, action)
    return state


def state_to_one_hot(state):
    """
    Convert cube state to one-hot encoding.
    Args:
        state: Cube state with shape (54,).
    Returns:
        One-hot encoded state with shape (54*6,).
    """
    one_hot = np.eye(6)[state]
    return one_hot.flatten()


def main():
    # Model path
    config = Config()
    args = config.parse_args()
    checkpoint_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Load model
    input_dim = (
        54 * 6
    )  # Rubik's cube has 54 stickers, each with 6 possible colors, encoded as one-hot
    model = DNN(input_dim, num_residual_blocks=4)

    # Load model weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Check whether this is a PyTorch Lightning checkpoint
        # Extract model weights
        model_weights = {
            k.replace("_orig_mod.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("_orig_mod.")
        }
        model.load_state_dict(model_weights)
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {str(e)}")

    model = model.to(device)
    model.eval()  # Set evaluation mode
    print("模型加载成功")

    # Create cube instance
    cube = Cube()

    # Prepare data
    num_shuffles_range = range(1, 30)  # 1 to 30 shuffles
    num_samples = 1000  # Number of samples per shuffle count
    avg_outputs = []
    max_outputs = []  # Store maximum output for each shuffle count

    print("开始计算模型输出...")
    for num_shuffles in num_shuffles_range:
        print(f"处理打乱次数: {num_shuffles}")

        # Generate samples in batch
        states = []
        for _ in range(num_samples):
            # Generate shuffled state
            state = generate_shuffled_state(cube, num_shuffles)
            states.append(state)

        # Batch-convert to one-hot encoding
        batch_one_hot = np.array([state_to_one_hot(state) for state in states])

        # Convert to tensor
        input_tensor = torch.tensor(batch_one_hot, dtype=torch.float32).to(device)

        # Batch model prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            # Convert output to NumPy array
            outputs_np = outputs.cpu().numpy().flatten()

        # Compute mean and maximum values
        avg_output = np.mean(outputs_np)
        max_output = np.max(outputs_np)  # Compute maximum value
        avg_outputs.append(avg_output)
        max_outputs.append(max_output)  # Store maximum value
        print(
            f"打乱次数 {num_shuffles} 的平均输出: {avg_output:.4f}, 最大输出: {max_output:.4f}"
        )

    # Plot statistics
    plt.figure(figsize=(12, 6))
    # Plot mean curve
    plt.plot(
        num_shuffles_range,
        avg_outputs,
        marker="o",
        linestyle="-",
        color="b",
        label="Average Value",
    )
    # Plot max curve
    plt.plot(
        num_shuffles_range,
        max_outputs,
        marker="s",
        linestyle="--",
        color="r",
        label="Maximum Value",
    )
    plt.title("Relationship between Cube Shuffles and DNN Output")
    plt.xlabel("Number of Shuffles")
    plt.ylabel("Model Output")
    plt.grid(True)
    plt.xticks(num_shuffles_range)
    plt.legend()  # Add legend
    plt.tight_layout()

    # Save plot image
    save_path = "model_output_vs_shuffles.png"
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存到 {save_path}")


if __name__ == "__main__":
    main()
