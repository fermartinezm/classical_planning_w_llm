from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    1. The number of tiles that still need to be painted.
    2. The movement distance from the robot's current position to the farthest tile that needs painting.
    3. Whether a color change is needed before painting.

    # Assumptions:
    - The robot can move up, down, left, or right to adjacent tiles.
    - Each movement and painting action counts as one step.
    - If the robot needs to change color, it requires an additional action.

    # Heuristic Initialization
    - Extracts goal conditions and static facts (tile layout and connections) from the task.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify the current position of the robot.
    2. Determine the color the robot is currently holding.
    3. Count the number of tiles that need to be painted and their required colors.
    4. For each required tile, calculate the Manhattan distance from the robot's current position.
    5. Sum the number of paint actions needed and add movement actions based on the farthest tile.
    6. Add an extra action if a color change is needed before painting.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract static information: tile layout and connections
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1:]
                self.up[x] = y
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1:]
                self.down[x] = y
            elif match(fact, "left", "*", "*"):
                y, x = get_parts(fact)[1:]
                self.left[x] = y
            elif match(fact, "right", "*", "*"):
                y, x = get_parts(fact)[1:]
                self.right[x] = y

        # Extract goal information: which tiles need to be painted and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1:]
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """Estimate the minimum number of actions to reach the goal state."""
        state = node.state

        # Extract current state information
        current_pos = None
        current_color = None
        painted = set()
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                current_pos = get_parts(fact)[2]
            elif match(fact, "robot-has", "*", "*"):
                current_color = get_parts(fact)[2]
            elif match(fact, "painted", "*", "*"):
                tile = get_parts(fact)[1]
                painted.add(tile)

        # If no current position, assume the robot is at some default position (should not happen in valid state)
        if current_pos is None:
            return 0

        # Count unpainted goal tiles
        unpainted = 0
        color_changes = 0
        required_color = None
        max_distance = 0
        for tile, color in self.goal_tiles.items():
            if tile not in painted:
                unpainted += 1
                if color != current_color:
                    color_changes += 1
                    required_color = color

                # Calculate Manhattan distance from current position to this tile
                # Extracting coordinates from tile names (e.g., tile_0_1 -> (0,1))
                x1, y1 = current_pos.split('_')
                x2, y2 = tile.split('_')
                distance = abs(int(x1) - int(x2)) + abs(int(y1) - int(y2))
                if distance > max_distance:
                    max_distance = distance

        # Estimate movement actions: moving to the farthest tile and back if needed
        # Since the robot can plan an efficient path, we assume the movement is optimal
        movement_actions = max_distance * 2  # To and from the farthest tile

        # Paint actions: each unpainted tile requires one action
        paint_actions = unpainted

        # Color change actions: needed if the robot doesn't have the required color
        color_change_actions = 1 if color_changes > 0 else 0

        # Total heuristic is the sum of all actions
        total_actions = movement_actions + paint_actions + color_change_actions

        return total_actions
