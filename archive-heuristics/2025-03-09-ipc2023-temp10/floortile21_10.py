from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortile21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors.
    It considers the number of tiles that need to be painted, the number of color changes required,
    and the movements needed to reach the unpainted tiles.

    # Assumptions
    - Each tile needs to be painted only once.
    - The robot can only paint adjacent tiles.
    - The robot needs to have the correct color before painting.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles from static facts.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the tiles that need to be painted according to the goal state.
    2. For each tile, check if it is already painted with the correct color in the current state.
    3. Count the number of tiles that are not painted correctly.
    4. Estimate the number of moves required to reach the unpainted tiles. This is a simplified estimate
       and doesn't consider the actual path.
    5. Estimate the number of color changes required. This is based on the assumption that the robot
       might need to change its current color to match the color required for the next tile.
    6. The heuristic value is the sum of the number of unpainted tiles, a simplified estimate of the number of moves,
       and the number of color changes.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (painted tiles with specific colors).
        - Adjacency information (up, down, left, right) between tiles.
        - Available colors.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal tiles and colors
        self.goal_tiles = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_tiles[tile] = color

        # Extract adjacency information
        self.adjacencies = {}
        for fact in static_facts:
            if match(fact, "up", "*", "*") or match(fact, "down", "*", "*") or \
               match(fact, "left", "*", "*") or match(fact, "right", "*", "*"):
                parts = get_parts(fact)
                tile1 = parts[1]
                tile2 = parts[2]
                if tile1 not in self.adjacencies:
                    self.adjacencies[tile1] = []
                self.adjacencies[tile1].append(tile2)

        # Extract available colors
        self.available_colors = set()
        for fact in static_facts:
            if match(fact, "available-color", "*"):
                parts = get_parts(fact)
                color = parts[1]
                self.available_colors.add(color)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the state is a goal state
        if all(goal in state for goal in self.goals):
            return 0

        unpainted_tiles_count = 0
        color_changes_count = 0
        robot_tile = None
        robot_color = None

        # Find robot's location and color
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                parts = get_parts(fact)
                robot_tile = parts[2]
            if match(fact, "robot-has", "*", "*"):
                parts = get_parts(fact)
                robot_color = parts[2]

        # Count unpainted tiles and estimate color changes
        for tile, color in self.goal_tiles.items():
            painted_correctly = False
            for fact in state:
                if match(fact, "painted", tile, color):
                    painted_correctly = True
                    break

            if not painted_correctly:
                unpainted_tiles_count += 1
                if robot_color != color:
                    color_changes_count += 1

        # Estimate moves required (simplified)
        moves_required = unpainted_tiles_count  # Simplified estimate

        return unpainted_tiles_count + moves_required + color_changes_count
