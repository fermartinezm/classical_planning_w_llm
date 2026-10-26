from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(robot-at robot1 tile_0_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles with the correct colors. It considers both the movement of robots and the need to change colors.

    # Assumptions:
    - The robot can move up, down, left, or right to adjacent tiles.
    - Each tile must be painted with a specific color.
    - The robot can change its color using the `change_color` action.
    - The goal is achieved when all required tiles are painted with their respective colors.

    # Heuristic Initialization
    - Extract the goal conditions to know which tiles need to be painted and with which colors.
    - Build a grid map from static facts to determine tile adjacencies.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. **Extract Goal Conditions**: Identify all tiles that need to be painted and their required colors.
    2. **Identify Current State**: For each robot, determine its current position and the color it is holding.
    3. **Calculate Required Actions**:
       - For each tile that needs painting, determine if it is already painted correctly. If not, add it to the list of required tiles.
       - For each required tile, find the nearest robot and calculate the Manhattan distance from the robot's current position to the tile.
       - If the robot's current color does not match the required color, add a color change action.
       - Sum the movement actions and color change actions for all required tiles.
    4. **Return Total Heuristic Value**: The sum of all required actions gives the heuristic value.

    # Example Calculation
    - If a robot is at tile_0_1 with color black and needs to paint tile_1_1 white:
      - Move down (1 action).
      - Change color to white (1 action).
      - Paint the tile (1 action).
      - Total heuristic value: 3.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Build grid map from static facts
        self.grid = defaultdict(dict)
        for fact in self.static:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[x]['up'] = y
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[x]['down'] = y
            elif match(fact, "left", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[x]['left'] = y
            elif match(fact, "right", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.grid[x]['right'] = y

        # Extract available colors
        self.available_colors = set()
        for fact in self.static:
            if match(fact, "available-color", "*"):
                color = get_parts(fact)[1]
                self.available_colors.add(color)

        # Precompute goal tiles and their required colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_tiles[tile] = color

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract robot information
        robots = {}
        colors = {}
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, tile = get_parts(fact)[1], get_parts(fact)[2]
                robots[robot] = tile
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)[1], get_parts(fact)[2]
                colors[robot] = color

        # If no robots, return 0 (assuming problem is unsolvable)
        if not robots:
            return 0

        # List of required tiles and their required colors
        required = []
        for tile, color in self.goal_tiles.items():
            if f"(painted {tile} {color})" not in state:
                required.append((tile, color))

        if not required:
            return 0

        # Calculate heuristic value
        total_cost = 0

        # For each required tile, find the nearest robot and calculate cost
        for tile, required_color in required:
            min_distance = float('inf')
            selected_robot = None

            # Find the nearest robot
            for robot, pos in robots.items():
                # Calculate Manhattan distance
                x1, y1 = self.parse_tile(pos)
                x2, y2 = self.parse_tile(tile)
                distance = abs(x1 - x2) + abs(y1 - y2)

                if distance < min_distance:
                    min_distance = distance
                    selected_robot = robot

            # Add movement cost
            total_cost += min_distance

            # Check if color change is needed
            current_color = colors[selected_robot]
            if current_color != required_color and required_color in self.available_colors:
                total_cost += 1  # Change color action

            # Add painting action
            total_cost += 1

        return total_cost

    def parse_tile(self, tile_str):
        """
        Parse a tile string (e.g., "tile_0_1") into (x, y) coordinates.
        """
        parts = tile_str.split('_')
        x = int(parts[1])
        y = int(parts[2])
        return x, y
