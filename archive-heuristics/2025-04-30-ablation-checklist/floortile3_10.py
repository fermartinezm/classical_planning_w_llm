import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Use zip to handle cases where args might be shorter than parts (e.g., with *)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Domain-specific helper functions
def parse_tile_name(tile_name):
    """
    Parses a tile name like 'tile_row_col' into a (row, col) tuple of integers.
    Assumes tile names follow this format.
    """
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers, though unlikely in valid PDDL
            return None
    return None # Return None for unexpected formats

def manhattan_distance(tile1_name, tile2_name):
    """
    Calculates the Manhattan distance between two tiles given their names.
    Returns infinity if parsing fails for either tile.
    """
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)

    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid tile names

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all
    goal tiles with the correct color. It considers the number of tiles
    that need painting, the colors required, and the movement cost for
    robots to reach adjacent tiles.

    # Assumptions
    - Tiles are arranged in a grid and named 'tile_row_col'.
    - Movement is restricted to adjacent clear tiles.
    - Painting requires the robot to be on a tile adjacent to the target tile.
    - Each action (move, change_color, paint) has a cost of 1.

    # Heuristic Initialization
    - Extracts the goal painted states (tile -> color).
    - Builds an adjacency map for tiles based on 'up', 'down', 'left', 'right' facts.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted according to the goal
       but are not currently painted with the correct color.
    2. For each such tile, estimate the minimum cost for *any* robot
       to paint it. This cost includes:
       a. Movement cost: Estimate the Manhattan distance from the robot's
          current location to *any* tile adjacent to the target tile.
       b. Color change cost: 1 if the robot does not currently hold the
          required color, 0 otherwise.
       c. Paint action cost: 1 (for the paint action itself).
    3. The heuristic value is the sum of these minimum costs over all
       tiles that need painting. This assumes tasks are independent,
       which is an overestimate but provides greedy guidance.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal painted states: {tile_name: color_name}
        self.goal_painted_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_painted_tiles[tile] = color

        # Build adjacency map: {tile_name: set(adjacent_tile_names)}
        self.adjacency_map = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Fact is (direction tile1 tile2), meaning tile1 is direction of tile2
                # So tile1 and tile2 are adjacent.
                tile1, tile2 = parts[1], parts[2]
                self.adjacency_map.setdefault(tile1, set()).add(tile2)
                self.adjacency_map.setdefault(tile2, set()).add(tile1)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract current robot locations and colors
        robot_info = {} # {robot_name: {'loc': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['loc'] = tile
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # Extract currently painted tiles: {tile_name: color_name}
        current_painted_tiles = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                current_painted_tiles[tile] = color

        # Identify tiles that need painting (goal color != current color or not painted)
        tiles_to_paint = {} # {tile_name: goal_color}
        for tile, goal_color in self.goal_painted_tiles.items():
            if tile not in current_painted_tiles or current_painted_tiles[tile] != goal_color:
                tiles_to_paint[tile] = goal_color

        # If all goal tiles are painted correctly, heuristic is 0
        if not tiles_to_paint:
            return 0

        total_heuristic_cost = 0

        # Calculate cost for each tile that needs painting
        for tile, goal_color in tiles_to_paint.items():
            min_cost_for_this_tile = float('inf')

            # Find the minimum cost for any robot to paint this tile
            for robot, info in robot_info.items():
                robot_location = info.get('loc')
                robot_color = info.get('color')

                # A robot must have a location and a color to paint
                if robot_location is None or robot_color is None:
                    continue # This robot cannot paint

                # Cost to change color if needed
                color_cost = 1 if robot_color != goal_color else 0

                # Cost to move to an adjacent tile
                min_dist_to_adjacent = float('inf')
                adjacent_tiles = self.adjacency_map.get(tile, set())

                if not adjacent_tiles:
                    # This tile has no adjacent tiles defined, cannot be painted
                    # This indicates a problem with the domain/instance,
                    # but we'll return a high cost to discourage this path.
                    min_cost_for_this_tile = float('inf')
                    break # Cannot paint this tile, consider this path very bad

                for adj_tile in adjacent_tiles:
                    dist = manhattan_distance(robot_location, adj_tile)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

                # If no adjacent tile is reachable (e.g., invalid tile names), this path is bad
                if min_dist_to_adjacent == float('inf'):
                     min_cost_for_this_tile = float('inf')
                     break # Cannot paint this tile, consider this path very bad

                # Total cost for this robot to paint this tile:
                # movement_cost + color_change_cost + paint_action_cost
                cost_for_this_robot = min_dist_to_adjacent + color_cost + 1

                min_cost_for_this_tile = min(min_cost_for_this_tile, cost_for_this_robot)

            # Add the minimum cost found for this tile to the total heuristic
            # If min_cost_for_this_tile is still inf, it means no robot can paint it
            # (e.g., no robots, or tile has no adjacent tiles).
            if min_cost_for_this_tile == float('inf'):
                 # This indicates an unsolvable state or a state from which
                 # a goal tile cannot be reached/painted. Return inf.
                 return float('inf')

            total_heuristic_cost += min_cost_for_this_tile

        return total_heuristic_cost

