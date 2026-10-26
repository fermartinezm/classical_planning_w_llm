from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import math # Used for infinity

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This might happen if the state contains non-fact elements, or is malformed
        # For robustness, handle unexpected input, though state is expected to be frozenset of strings
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Helper function to parse tile names and calculate Manhattan distance
def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_row_col' into (row, col) integer coordinates.
    Assumes tile names follow this format.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected tile name format
            print(f"Warning: Unexpected tile name format: {tile_name}")
            return None # Or raise an error
    except (ValueError, IndexError):
        print(f"Warning: Could not parse tile name coordinates: {tile_name}")
        return None # Or raise an error


def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    Returns infinity if tile names cannot be parsed.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles that are currently clear and unpainted, or painted with the wrong color.
    It calculates the minimum cost for any robot to reach and paint each
    such tile independently and sums these minimum costs.

    # Assumptions:
    - Tiles are arranged in a grid, and tile names follow the format 'tile_row_col'.
    - Movement actions cost 1.
    - Paint actions cost 1.
    - Change color action costs 1.
    - A tile must be 'clear' to be painted.
    - If a goal tile is painted with the wrong color, the state is likely unsolvable
      in this domain version (as there's no explicit unpaint/clear action for painted tiles
      other than potentially moving off them, which doesn't apply to painting adjacent tiles).
      The heuristic returns infinity in this case.
    - The heuristic calculates the cost for each unpainted/wrongly-painted goal tile
      independently, finding the minimum cost across all robots. This ignores
      potential synergies or conflicts between robots.

    # Heuristic Initialization
    - Extracts the goal conditions (which tiles need which color).
    - Identifies all robot names from the initial state.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all goal tiles and their required colors.
    2. For each goal tile:
       a. Check if it is already painted with the correct color in the current state. If yes, cost for this tile is 0.
       b. Check if it is painted with *any* color *other than* the required one. If yes, the state is likely unsolvable; return infinity.
       c. Check if the tile is 'clear'. If it's not clear and not painted correctly, it's likely painted incorrectly (or occupied, which shouldn't happen for painting adjacent tiles); return infinity.
       d. If the tile is 'clear' and needs painting:
          i. Find the current location and color of each robot.
          ii. For each robot, calculate the cost to paint this specific tile:
             - Calculate the Manhattan distance from the robot's current tile to the goal tile.
             - The minimum moves required to get adjacent to the goal tile is `max(0, distance - 1)`.
             - Add 1 for the paint action.
             - If the robot's current color is not the required color, add 1 for the change_color action.
          iii. Find the minimum cost among all robots to paint this tile.
          iv. Add this minimum cost to the total heuristic value.
    3. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and robot names.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state

        # Store goal tiles and their required colors: {tile_name: color_name}
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Goal is (painted ?tile ?color)
                if len(args) == 2:
                    tile, color = args
                    self.goal_tiles[tile] = color
                else:
                    print(f"Warning: Unexpected goal format: {goal}")


        # Identify all robot names from the initial state
        self.robots = set()
        for fact in self.initial_state:
             predicate, *args = get_parts(fact)
             if predicate == "robot-at":
                 # Fact is (robot-at ?r ?x)
                 if len(args) == 2:
                     robot_name = args[0]
                     self.robots.add(robot_name)
                 else:
                     print(f"Warning: Unexpected initial robot-at format: {fact}")

        # print(f"Heuristic initialized. Goal tiles: {self.goal_tiles}")
        # print(f"Heuristic initialized. Robots: {self.robots}")


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state from the current state.
        """
        state = node.state  # Current world state (frozenset of strings)

        # Extract relevant information from the current state
        robot_positions = {} # {robot_name: tile_name}
        robot_colors = {}    # {robot_name: color_name}
        painted_tiles = {}   # {tile_name: color_name}
        clear_tiles = set()  # {tile_name}

        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "robot-at" and len(args) == 2:
                robot, tile = args
                robot_positions[robot] = tile
            elif predicate == "robot-has" and len(args) == 2:
                robot, color = args
                robot_colors[robot] = color
            elif predicate == "painted" and len(args) == 2:
                tile, color = args
                painted_tiles[tile] = color
            elif predicate == "clear" and len(args) == 1:
                tile = args[0]
                clear_tiles.add(tile)
            # Ignore other predicates like up, down, left, right, available-color, free-color

        total_cost = 0

        # Iterate through each goal tile
        for goal_tile, required_color in self.goal_tiles.items():
            current_painted_color = painted_tiles.get(goal_tile)
            is_clear = goal_tile in clear_tiles

            # Case 1: Tile is already painted correctly
            if current_painted_color == required_color:
                continue # This goal is satisfied

            # Case 2: Tile is painted with the wrong color or not clear when it should be painted
            # In this domain, you cannot repaint a tile unless it's clear.
            # If a goal tile is painted with the wrong color, it's likely an unsolvable state.
            # If a goal tile is not clear and not painted correctly, it cannot be painted.
            if current_painted_color is not None and current_painted_color != required_color:
                 # Tile is painted with the wrong color
                 # print(f"Heuristic returning inf: Goal tile {goal_tile} painted with wrong color {current_painted_color}")
                 return float('inf') # State is likely unsolvable

            if not is_clear:
                 # Tile is not clear and not painted correctly (implies painted wrong or occupied)
                 # Given the domain, this means it cannot be painted to the correct color.
                 # print(f"Heuristic returning inf: Goal tile {goal_tile} is not clear and not painted correctly.")
                 return float('inf') # State is likely unsolvable


            # Case 3: Tile is clear and needs painting
            # Calculate the minimum cost for any robot to paint this tile
            min_cost_for_this_tile = float('inf')

            for robot_name in self.robots:
                robot_tile = robot_positions.get(robot_name)
                robot_color = robot_colors.get(robot_name)

                # Ensure robot exists and has a position and color (should always be true in valid states)
                if robot_tile is None or robot_color is None:
                    # print(f"Warning: Robot {robot_name} missing position or color in state.")
                    continue # Cannot use this robot

                # Calculate moves to get adjacent to the goal tile
                dist = manhattan_distance(robot_tile, goal_tile)

                # If distance is infinity (e.g., unparseable tile names), this robot can't reach it
                if dist == float('inf'):
                    continue

                # Minimum moves to get to a tile adjacent to goal_tile
                # If dist is 0 (robot is on the tile), moves_to_adjacent is max(0, -1) = 0. But robot must be adjacent, not on.
                # If dist is 1 (robot is adjacent), moves_to_adjacent is max(0, 0) = 0. Correct.
                # If dist is > 1, moves_to_adjacent is dist - 1. Correct.
                moves_to_adjacent = max(0, dist - 1)

                # Base cost: moves to get adjacent + 1 for the paint action
                current_robot_cost = moves_to_adjacent + 1

                # Add cost for changing color if needed
                if robot_color != required_color:
                    current_robot_cost += 1 # change_color action costs 1

                min_cost_for_this_tile = min(min_cost_for_this_tile, current_robot_cost)

            # If min_cost_for_this_tile is still infinity, it means no robot can reach this tile.
            # This state might be unsolvable or require complex setup not captured by this simple heuristic.
            # For a greedy search, returning infinity is appropriate if a goal is truly unreachable.
            # However, if it's just temporarily unreachable (e.g., path blocked), adding infinity might
            # prune states that could eventually lead to a solution.
            # A simpler non-admissible approach is to just add the calculated min cost (which could be inf).
            # If no robot can reach, min_cost_for_this_tile remains inf, and total_cost becomes inf.
            total_cost += min_cost_for_this_tile


        return total_cost

