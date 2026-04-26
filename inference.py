import torch
import numpy as np
import heapq
import os
from config import Config
from model.DNN import DNN
from model.Cube import Cube, TARGET_STATE
from vis import generate_rubiks_html
import time

from solver_utils import *


def generate_html(initial_state, solution_path):
    # Color mapping (0=white, 1=red, 2=green, 3=yellow, 4=orange, 5=blue)
    COLOR_MAP = {0: "white", 1: "red", 2: "green", 3: "yellow", 4: "orange", 5: "blue"}

    # Face order and index ranges
    FACE_ORDER = {
        "U": list(range(0, 9)),  # Up face
        "R": list(range(9, 18)),  # Right face
        "F": list(range(18, 27)),  # Front face
        "D": list(range(27, 36)),  # Down face
        "L": list(range(36, 45)),  # Left face
        "B": list(range(45, 54)),  # Back face
    }

    # Convert initial state to the format required by generate_rubiks_html
    initial_state_dict = {}
    for face, indices in FACE_ORDER.items():
        initial_state_dict[face] = [COLOR_MAP[initial_state[i]] for i in indices]

    # Build state for each step
    moves = []
    for state in solution_path:
        move_state = {}
        for face, indices in FACE_ORDER.items():
            move_state[face] = [COLOR_MAP[state[i]] for i in indices]
        moves.append(move_state)

    # Generate the HTML via generate_rubiks_html
    output_file = "rubiks_solution.html"
    generate_rubiks_html(initial_state_dict, FACE_ORDER, moves, output_file)
    print(f"已生成解决方案网页: {output_file}")


def main():
    # Load configuration
    config = Config()
    args = config.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model_path, device=device)

    # Create Cube object
    cube = Cube()

    # Initial state starts at the target state
    initial_state = TARGET_STATE.copy()
    state_path = []

    if args.actions:
        # Use user-provided action sequence
        print(f"使用指定的动作序列: {args.actions}")
        shuffle_actions = args.actions.split()
        # Validate actions
        valid_actions = set(cube.moves.keys())
        invalid_actions = [
            action for action in shuffle_actions if action not in valid_actions
        ]
        if invalid_actions:
            raise ValueError(
                f"无效的动作: {invalid_actions}，有效的动作是: {valid_actions}"
            )
    else:
        # Randomly generate action sequence
        print("随机打乱魔方100次...")
        all_actions = list(cube.moves.keys())
        shuffle_actions = np.random.choice(all_actions, size=100, replace=True)

    # Apply action sequence
    for action in shuffle_actions:
        initial_state = cube.apply_action(initial_state, action)
        state_path.append(initial_state.copy())

    print("开始A*搜索...")

    # Run A* search
    start_time = time.time()
    action_path, solution_state_path = a_star_search(initial_state, model, cube)
    end_time = time.time()
    solving_time = end_time - start_time

    # Save visualization output
    if solution_state_path:
        # Merge scramble and solution paths to show the full process
        # full_state_path = state_path # + solution_state_path[1:]
        generate_html(TARGET_STATE.copy(), state_path)
        # generate_html(initial_state, solution_state_path[1:])
        print(f"找到解决方案，路径长度: {len(solution_state_path)}")
        print(f"求解时间: {solving_time:.4f} 秒")  # Print solve time
        print("打乱路径:", shuffle_actions)
        if action_path:
            print("解决方案路径:", action_path)
    else:
        print("未找到解决方案")


if __name__ == "__main__":
    main()
