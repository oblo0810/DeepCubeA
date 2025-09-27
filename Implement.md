# DeepCubeA Implementation Details

## 1. Rubik’s Cube State Representation

A Rubik’s Cube has 54 stickers in total and 6 colors. Each sticker is assigned a unique index based on the face it belongs to and its position on that face. In the solved state, the face-to-color mapping is fixed as follows:

- U (Up): White  
- F (Front): Green  
- L (Left): Orange  
- R (Right): Red  
- B (Back): Blue  
- D (Down): Yellow  

We represent the cube state as a 54-dimensional vector, where each element corresponds to the color of one sticker. For example, the solved state is represented as:

```python
# Colors represented by integers (0=white, 1=red, 2=green, 3=yellow, 4=orange, 5=blue)
initial_state = [
    0,0,0, 0,0,0, 0,0,0,  # U face
    1,1,1, 1,1,1, 1,1,1,  # R face
    2,2,2, 2,2,2, 2,2,2,  # F face
    3,3,3, 3,3,3, 3,3,3,  # D face
    4,4,4, 4,4,4, 4,4,4,  # L face
    5,5,5, 5,5,5, 5,5,5   # B face
]
```

When input to the DNN model, the state vector is converted into a 54×6 one-hot encoding.

## 2. Action Representation

We use standard face notation to represent cube rotations:

* F, B, L, R, U, D: Rotate the Front, Back, Left, Right, Up, and Down faces.
* A single letter means a clockwise 90° rotation, while an apostrophe (') indicates a counterclockwise 90° rotation.

Examples:

* `R`: Rotate the right face clockwise by 90°.
* `R'`: Rotate the right face counterclockwise by 90°.

## 3. Deep Approximate Value Iteration (DAVI)

We use a DNN to approximate the distance (cost-to-go) $J(s)$ from a state $s$ to the solved state. Training follows a stepwise approach that simulates value iteration:

1. At step $K$, generate training samples by scrambling the solved cube with 1–K random moves.
2. Use the previously trained model $J_{\theta_{K-1}}$ as supervision.
3. Update using:

$$
J_{\theta_K}(s) = \min_a \big[ J_{\theta_{K-1}}(A(s,a)) + 1 \big]
$$

## 4. Training Pseudocode

```python
# Algorithm 1: DAVI
# Input:
#   B: batch size
#   K: maximum scramble depth
#   M: number of training iterations
#   C: convergence check frequency
#   ε: error threshold
# Output:
#   θ: trained neural network parameters

theta = initialize_parameters()
theta_e = theta

for m in 1 to M:
    X = get_scrambled_states(B, K)
    for x_i in X:
        y_i = min_a [g_theta_e(A(x_i, a)) + 1]
    theta = train(X, y, theta)
    if (m mod C == 0) and (loss < ε):
        theta_e = theta

return theta
```

## 5. Inference: BWAS Search Algorithm

We use an A* search variant that maintains two sets of states:

* **OPEN**: states pending expansion
* **CLOSED**: states already expanded

Path length formula:

$$
f(x) = \lambda \cdot g(x) + h(x)
$$

* $h(x)$: heuristic, predicted by the trained DNN $J_\theta$
* $g(x)$: actual path length from the initial state to state $x$
* $\lambda$: weighting parameter, set to 0.6 in the paper

At each step, select the N shortest paths from OPEN (N=10000 in the paper), move them to CLOSED, and expand their neighbors.

## 6. Neural Network Architecture and Implementation Details

### Model Architecture

1. **Hidden Layers**: Two fully connected hidden layers of sizes 5000 and 1000.
2. **Residual Blocks**: 4 residual blocks, each containing two hidden layers of size 1000.
3. **Output Layer**: A single linear unit predicting the cost-to-go.
4. **Normalization & Activation**: Batch normalization and ReLU applied to all hidden layers.

### Training Details

1. **Batch size**: 10,000
2. **Optimizer**: ADAM
3. **Regularization**: None
4. **Maximum scramble depth (K)**: 30
5. **Error threshold (ε)**: 0.20 (our model can't reach threshod 0.05 which adopted in the original paper)
6. **Convergence check**: Every 5,000 iterations
7. **Iterations per epoch**: 1000
8. **Training time**: ~12 hours
9. **Hardware**: Single NVIDIA A100 GPU
10. **VRAM usage**: ~4GB

## 7. Downloading Web Source Code

We used the Google Chrome extension **Resources Saver**.

![web](assets/download.png)


## Debug

1. DNN中的所乘的 K 可以去掉，让模型直接输出cost-to-go估计值即可。
2. 还是用MSE_LOSS感觉比较好一点，先别用relative了感觉❌。还是要用relative，因为如果不用，K=2的时候模型收敛位置的损失就高达0.15，收敛判别的依据感觉并不准确。❌ 最终没有用relative_mse_loss，只用的mse_loss，relative会加剧模型过拟合到K比较小的情况，让模型在K比较大的时候也非常容易收敛，但这并不是我们想要的，我们需要模型更多的去学习K较大的状态。
3. 不要在dataset中一个数据一个数据放到torch上，这样效率太低，最好还是在training_loop中成批量一起转移。
4. 有一个问题，如果说model_theta_e遇到一个自己没见过的状态，实际上的距离是K+1，但是它给出来了一个小于K的值，然后这个时候模型就会低估当前这个状态的距离？在实际的训练结果中，模型的输出确实也相对比较低，比如30次打乱给出的最高分才是12multiple，这种低估不知道是否有问题。
5. 还是因为粗心Cube的旋转操作出错了。
6. 模型的收敛速度特别慢，当K=14时收敛（损失小于0.05）需要30多个epoch，当K值更大（超过20）时或许收敛速度会更快，但考虑到计算资源有限，在K>14不再遵循阈值，而是最大epoch（20个）进行训练，最终也能训练出来功能较为合理的模型。

