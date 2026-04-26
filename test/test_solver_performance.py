import numpy as np
import time
import heapq
from model.Cube import Cube, TARGET_STATE
from config import Config
from model.DNN import DNN
import torch
import os
import random

from solver_utils import *

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_random_state(cube, min_scrambles=1000, max_scrambles=10000):
    """
    Generate a randomly scrambled cube state.
    Args:
        cube: Cube object.
        min_scrambles: Minimum number of scrambles.
        max_scrambles: Maximum number of scrambles.
    Returns:
        Randomly scrambled cube state.
    """
    state = TARGET_STATE.copy()
    num_scrambles = random.randint(min_scrambles, max_scrambles)
    moves = list(cube.moves.keys())

    for _ in range(num_scrambles):
        move = random.choice(moves)
        state = cube.apply_action(state, move)

    return state, num_scrambles


def main():
    config = Config()
    args = config.parse_args()
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Load model
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return

    model = load_model(model_path, device)

    # Create Cube object
    cube = Cube()

    # Generate test set
    num_tests = 200
    min_scrambles = 1000
    max_scrambles = 10000

    total_solve_length = 0
    total_time = 0
    solved_count = 0
    scramble_counts = []

    print(f"开始生成 {num_tests} 个随机魔方状态并求解...")

    for i in range(num_tests):
        # Generate random state
        state, num_scrambles = generate_random_state(cube, min_scrambles, max_scrambles)
        scramble_counts.append(num_scrambles)

        # Solve cube
        start_time = time.time()
        action_path, _ = a_star_search(state, model, cube)
        end_time = time.time()

        # Record results
        solve_time = end_time - start_time

        if action_path is not None:
            solve_length = len(action_path)
            total_solve_length += solve_length
            total_time += solve_time
            solved_count += 1
            print(
                f"测试 {i + 1}/{num_tests}: 打乱次数={num_scrambles}, 解长度={solve_length}, 耗时={solve_time:.2f}秒"
            )
        else:
            print(
                f"测试 {i + 1}/{num_tests}: 打乱次数={num_scrambles}, 求解失败, 耗时={solve_time:.2f}秒"
            )

        # Print running average after every 10 tests
        if (i + 1) % 10 == 0:
            current_avg_length = (
                total_solve_length / solved_count if solved_count > 0 else 0
            )
            current_avg_time = total_time / solved_count if solved_count > 0 else 0
            print(
                f"进度: {i + 1}/{num_tests}, 当前平均解长度: {current_avg_length:.2f}, 当前平均耗时: {current_avg_time:.2f}秒"
            )

    # Compute final statistics
    avg_scrambles = sum(scramble_counts) / num_tests
    avg_solve_length = total_solve_length / solved_count if solved_count > 0 else 0
    avg_solve_time = total_time / solved_count if solved_count > 0 else 0
    success_rate = solved_count / num_tests * 100

    # Output results
    print("\n===== 测试结果 =====")
    print(f"总测试数: {num_tests}")
    print(f"成功求解数: {solved_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"平均打乱次数: {avg_scrambles:.2f}")
    print(f"平均解长度: {avg_solve_length:.2f}")
    print(f"平均求解时间: {avg_solve_time:.2f}秒")
    print("====================")


if __name__ == "__main__":
    main()
