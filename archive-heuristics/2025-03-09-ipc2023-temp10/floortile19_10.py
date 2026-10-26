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


class floortile19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions needed to paint all tiles to their goal colors,
    considering robot movement and color changes. It prioritizes painting tiles with the color
    the robot currently has and estimates the cost of moving to adjacent tiles and changing colors.

    # Assumptions
    - The robot can only paint adjacent tiles (up, down, left, right).
    - The robot can only hold one color at a time.
    - The heuristic assumes that the robot will always choose the shortest path to the nearest
      tile that needs painting with the color it currently holds.
    - The heuristic does not account for the cost of clearing a tile.

    # Heuristic Initialization
    - Extract the goal conditions (painted tiles with specific colors).
    - Extract the adjacency information (up, down, left, right) between tiles.
    - Identify available colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current robot location and the color the robot is holding.
    2. Identify the tiles that need to be painted according to the goal state.
    3. Calculate the number of tiles that are not painted correctly.
    4. For each unpainted tile, determine if the robot has the correct color.
       - If the robot has the correct color, estimate the cost of moving to that tile and painting it.
       - If the robot does not have the correct color, estimate the cost of changing the color,
         moving to the tile, and painting it.
    5. Estimate the cost of moving to the nearest unpainted tile with the correct color.
       - Use the adjacency information (up, down, left, right) to estimate the number of moves.
    6. Sum the estimated costs for all unpainted tiles to get the heuristic value.
    7. If the goal is already reached, return 0.
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

        self.adj = {}  # Adjacency list: tile -> list of adjacent tiles
        self.available_colors = set()  # Set of available colors

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate in ("up", "down", "left", "right"):
                tile1, tile2 = args
                if tile1 not in self.adj:
                    self.adj[tile1] = []
                if tile2 not in self.adj:
                    self.adj[tile2] = []
                self.adj[tile1].append(tile2)
                self.adj[tile2].append(tile1)
            elif predicate == "available-color":
                self.available_colors.add(args[0])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract robot location and color
        robot_location = None
        robot_color = None
        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_location = get_parts(fact)[2]
            elif match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]

        # Extract goal painted tiles
        goal_painted = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                tile, color = get_parts(goal)[1:]
                goal_painted[tile] = color

        # Check if goal is already reached
        reached = True
        for tile, color in goal_painted.items():
            if f"(painted {tile} {color})" not in state:
                reached = False
                break
        if reached:
            return 0

        # Identify unpainted tiles
        unpainted_tiles = {}
        for tile, color in goal_painted.items():
            if f"(painted {tile} {color})" not in state:
                unpainted_tiles[tile] = color

        # Calculate heuristic value
        heuristic_value = 0
        for tile, color in unpainted_tiles.items():
            if robot_color == color:
                # Estimate cost of moving to the tile and painting it
                heuristic_value += 1  # Painting cost
                if robot_location != tile:
                    heuristic_value += 1 # Move cost
            else:
                # Estimate cost of changing color, moving to the tile, and painting it
                heuristic_value += 1  # Color change cost
                heuristic_value += 1  # Painting cost
                if robot_location != tile:
                    heuristic_value += 1 # Move cost

        return heuristic_value
