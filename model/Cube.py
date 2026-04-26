import numpy as np

# Target State
# Represent colors as integers (0=white, 1=red, 2=green, 3=yellow, 4=orange, 5=blue)
TARGET_STATE = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,  # U face
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,  # R face
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,  # F face
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,  # D face
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,  # L face
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,  # B face
    ],
    dtype=np.int32,
)

TARGET_STATE_ONE_HOT = np.eye(6)[TARGET_STATE]


# Function to print state as a cube unfolded diagram
def print_cube_state(state, title=None):
    print(title)
    # U face
    print(" " * 6 + " ".join(map(str, state[0:3])))
    print(" " * 6 + " ".join(map(str, state[3:6])))
    print(" " * 6 + " ".join(map(str, state[6:9])))
    # L, F, R, B faces
    for i in range(3):
        print(
            " ".join(map(str, state[36 + i * 3 : 39 + i * 3]))
            + " "
            + " ".join(map(str, state[18 + i * 3 : 21 + i * 3]))
            + " "
            + " ".join(map(str, state[9 + i * 3 : 12 + i * 3]))
            + " "
            + " ".join(map(str, state[51 + i * 3 : 48 + i * 3]))
        )
    # D face
    print(" " * 6 + " ".join(map(str, state[27:30])))
    print(" " * 6 + " ".join(map(str, state[30:33])))
    print(" " * 6 + " ".join(map(str, state[33:36])))
    print()


def invert_mapping(mapping):
    """Generate inverse mapping."""
    inv = np.empty(len(mapping), dtype=int)
    inv[mapping] = np.arange(len(mapping))
    return inv


def make_moves():
    """
      Generate sticker index mappings for a 3x3x3 Rubik's cube (clockwise + counterclockwise).
      Sticker index order:
    U: 0-8,  R: 9-17,  F: 18-26,
    D: 27-35, L: 36-44, B: 51-47
    """

    moves = {}

    def cycle(mapping, positions):
        """Update mapping according to cyclic positions."""
        temp = mapping.copy()
        for cycle_pos in positions:
            cycle = np.append(cycle_pos, cycle_pos[0])
            mapping[cycle[:-1]] = temp[cycle[1:]]

    # Initialize identity mapping
    identity = np.arange(54)

    # Define clockwise rotations for each face
    face_cycles = {
        "U": [
            # Rotate upper face itself
            [2, 8, 6, 0],
            [5, 7, 3, 1],
            # Side ring
            [20, 9, 53, 36],
            [19, 10, 52, 37],
            [18, 11, 51, 38],
        ],
        "D": [
            [29, 35, 33, 27],
            [28, 32, 34, 30],
            [24, 44, 45, 17],
            [25, 43, 46, 16],
            [26, 42, 47, 15],
        ],
        "F": [
            [18, 24, 26, 20],
            [19, 21, 25, 23],
            [17, 2, 36, 33],
            [14, 1, 39, 34],
            [11, 0, 42, 35],
        ],
        "B": [
            [51, 45, 47, 53],
            [52, 48, 46, 50],
            [9, 29, 44, 6],
            [12, 28, 41, 7],
            [15, 27, 38, 8],
        ],
        "L": [
            [36, 38, 44, 42],
            [37, 41, 43, 39],
            [33, 18, 6, 47],
            [30, 21, 3, 50],
            [27, 24, 0, 53],
        ],
        "R": [
            [17, 15, 9, 11],
            [16, 12, 10, 14],
            [45, 8, 20, 35],
            [48, 5, 23, 32],
            [51, 2, 26, 29],
        ],
    }

    # Generate clockwise and counterclockwise mappings
    for face, cycles in face_cycles.items():
        mapping = identity.copy()
        cycle(mapping, cycles)
        moves[face] = mapping
        moves[face + "_inv"] = invert_mapping(mapping)

    return moves


class Cube:
    def __init__(self):
        # Initialize move mappings
        self.moves = make_moves()

    def apply_action(self, state, action):
        """
        Get the new state from the given action and state.
        Args:
            state: Current cube state, np.array.
            action: Action to execute, such as 'U', 'R_inv', etc.
        Returns:
            New cube state.
        """
        if action not in self.moves.keys():
            raise ValueError(f"不支持的动作: {action}")
        return state[self.moves[action]]

    def get_neibor_state(self, state):
        """
        Get all neighboring states for the current state.
        Args:
            state: Current cube state, np.array.
        Returns:
            List of neighboring states, np.array.
        """
        neibor_states = []
        for action in self.moves.keys():
            neibor_states.append(self.apply_action(state, action))
        return np.stack(neibor_states, axis=0)

    def is_solved(self, state):
        """
        Check whether current state is solved.
        Args:
            state: Current cube state, np.array.
        Returns:
            Boolean indicating whether it is solved.
        """
        return np.array_equal(state, TARGET_STATE)

    def view_state(self, state):
        pass


if __name__ == "__main__":
    # Initialize the Cube object
    cube = Cube()

    # Get the initial solved state
    initial_state = np.arange(54)

    # Define the rotation action
    action = "F"
    # Print the initial state
    print_cube_state(initial_state, "Initial Cube State:")

    # Print the applied action
    print(f"Applied action: {action}")

    # Apply the rotation action
    new_state = cube.apply_action(initial_state, action)

    # Print the new state
    print_cube_state(new_state, "Cube State after Rotation:")

    action = "U"

    new_state = cube.apply_action(new_state, action)

    # Print the applied action
    print(f"Applied action: {action}")
    print_cube_state(new_state, "Cube State after Rotation:")
