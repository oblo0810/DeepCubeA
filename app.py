import flask
from flask import request, jsonify
import torch
import numpy as np
import os
from config import Config
from model.DNN import DNN
from model.Cube import Cube, TARGET_STATE
from solver_utils import *

# Initialize Flask app
app = flask.Flask(__name__, static_folder=None)
app.config["JSON_AS_ASCII"] = False
app.config["DEBUG"] = True

# Load configuration
config = Config()
args = config.parse_args()
# Set default model path
model_path = args.model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and create Cube object
model = load_model(model_path, device)
cube = Cube()


# Initialize state endpoint
@app.route("/initState", methods=["POST"])
def init_state():
    # Set the initial state to the target state
    initial_state = TARGET_STATE.copy()

    # Generate rotation indices and state mappings
    rotateIdxs_old = {}
    rotateIdxs_new = {}
    for move_name in cube.moves.keys():
        # Use the actual move mapping defined in Cube
        move_mapping = cube.moves[move_name]
        # Build the mapping from old to new indices
        rotateIdxs_old[move_name] = move_mapping.tolist()
        rotateIdxs_new[move_name] = list(range(54))

    # Define mappings between state and feature extraction indices
    # Assumes state and feature extraction use the same ordering
    stateToFE = list(range(54))
    FEToState = list(range(54))
    legalMoves = list(cube.moves.keys())

    response = {
        "state": initial_state.tolist(),
        "rotateIdxs_old": rotateIdxs_old,
        "rotateIdxs_new": rotateIdxs_new,
        "stateToFE": stateToFE,
        "FEToState": FEToState,
        "legalMoves": legalMoves,
    }

    return jsonify(response)


# Rubik's cube solving endpoint
@app.route("/solve", methods=["POST"])
def solve():
    try:
        data = request.json
        if not data or "state" not in data:
            return jsonify({"error": "请求参数错误，缺少state字段"}), 400

        state = np.array(data["state"])
        if state.shape != (54,):
            return jsonify({"error": "state参数格式错误，应为长度为54的数组"}), 400

        print("开始求解魔方...")
        action_path, solution_state_path = a_star_search(state, model, cube)

        if action_path is None:
            return jsonify({"error": "未能找到解决方案"}), 404

        # Build reverse action path
        solveMoves_rev = []
        for action in action_path:
            rev_action = action[:]
            # Reverse action direction
            if "inv" in rev_action:
                rev_action = rev_action[0]
            else:
                rev_action += "_inv"
            solveMoves_rev.append(rev_action)

        print(action_path)
        print(solveMoves_rev)

        response = {
            "moves": [action for action in action_path],
            "moves_rev": solveMoves_rev,
            "solve_text": action_path,
        }

        return jsonify(response)
    except Exception as e:
        print(f"求解过程中发生错误: {str(e)}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500


# Static file service
@app.route("/static/<path:path>")
def send_static(path):
    print("Serving static file:", path)
    return flask.send_from_directory("web/deepcube.igb.uci.edu/static", path)


# Home page
@app.route("/")
def home():
    return flask.send_from_directory("web/deepcube.igb.uci.edu", "index.html")


# Handle missing heapq module
import heapq

if __name__ == "__main__":
    # Ensure the checkpoint directory exists
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
        print("创建checkpoint目录，请将模型文件放入该目录")

    # Check whether the model file exists
    if not os.path.exists(model_path):
        print(f"警告：未找到模型文件 {model_path}")
        print("请确保模型文件存在于checkpoint目录中")

    # Start server
    # Listen on localhost only
    app.run(host="127.0.0.1", port=5000)
