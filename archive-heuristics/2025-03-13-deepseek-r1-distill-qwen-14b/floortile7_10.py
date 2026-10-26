<code-file-heuristic-3>
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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

class FloortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all required tiles by considering:
    - The distance the robot needs to move to reach each tile
    - Whether the robot needs to change colors before painting
    - The number of tiles that still need to be painted

    # Assumptions:
    - The robot can move up, down, left, or right to adjacent tiles
    - The robot can only paint a tile if it is holding the correct color
    - The robot starts with one color and can change colors as needed

    # Heuristic Initialization
    - Extracts the grid layout and connectivity from static facts
    - Tracks the current position and color of the robot
    - Stores the target tiles and their required colors

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. For each target tile:
       a. If the tile is already painted correctly, no cost is added
       b. If the tile needs to be painted:
          i. Calculate the Manhattan distance from the robot's current position to the tile
          ii. If the robot isn't holding the required color, add a color change cost
          iii. Add the painting cost and movement cost

    2. Sum the costs for all tiles, ensuring that color changes are only counted once when needed
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Grid layout and connectivity from static facts
        - Goal conditions (target tiles and their colors)
        - Current state of the robot (position and held color)
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (grid layout, etc.)

        # Extract grid layout and connectivity
        self.up = {}
        self.down = {}
        self.left = {}
        self.right = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.up[(x, y)] = True
            elif match(fact, "down", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.down[(x, y)] = True
            elif match(fact, "left", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.left[(x, y)] = True
            elif match(fact, "right", "*", "*"):
                y, x = get_parts(fact)[1], get_parts(fact)[2]
                self.right[(x, y)] = True

        # Store target tiles and their required colors
        self.target_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.target_tiles[tile] = color

        # Find robot's initial position and held color
        self.robot_pos = None
        self.held_color = None
        for fact in task.initial_state:
            if match(fact, "robot-at", "*", "*"):
                self.robot_pos = get_parts(fact)[1], get_parts(fact)[2]
            elif match(fact, "robot-has", "*", "*"):
                self.held_color = get_parts(fact)[1]

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Track current state of the world
        current_positions = {}
        current_colors = {}
        painted_tiles = set()
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot, x, y = get_parts(fact)
                current_positions[robot] = (x, y)
            elif match(fact, "robot-has", "*", "*"):
                robot, color = get_parts(fact)
                current_colors[robot] = color
            elif match(fact, "painted", "*", "*"):
                tile, color = get_parts(fact)
                painted_tiles.add((tile, color))

        # If robot position is not tracked, assume it's at the initial position
        if not current_positions:
            current_positions = {self.robot_pos[0]: self.robot_pos[1]}

        # If held color is not tracked, assume it's the initial color
        if not current_colors:
            current_colors = {self.held_color[0]: self.held_color[1]}

        total_cost = 0  # Initialize action cost counter

        # For each target tile, calculate the required actions
        for tile, target_color in self.target_tiles.items():
            # Skip if tile is already painted correctly
            if (tile, target_color) in painted_tiles:
                continue

            # Get the robot's current position
            robot = next(iter(current_positions.keys()))
            current_x, current_y = current_positions[robot]

            # Find the tile's coordinates
            tile_x = tile.split('_')[1]
            tile_y = tile.split('_')[2]
            target_pos = (tile_x, tile_y)

            # Calculate Manhattan distance
            distance = abs(int(tile_x) - int(current_x)) + abs(int(tile_y) - int(current_y))

            # Add movement cost
            total_cost += distance

            # Check if color change is needed
            if current_colors[robot] != target_color:
                # Add cost to change color
                total_cost += 1

            # Add painting cost
            total_cost += 1

        return total_cost
</code-file-heuristic-3>