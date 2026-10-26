# Assuming the standard structure where heuristic files are in a 'heuristics' directory
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch
import math # For infinity

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(robot-at robot1 tile_0_1)" -> ["robot-at", "robot1", "tile_0_1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip; it stops when the shorter iterable is exhausted.
    # This works correctly for patterns like ("painted", "*", "*") matching ("painted", "tile_1_1", "white")
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


def parse_tile_name(tile_name):
    """Parses 'tile_row_col' into (row, col) integers."""
    parts = tile_name.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        except ValueError:
            # Handle cases where row/col are not integers, though unlikely in valid PDDL
            return None
    return None # Not in 'tile_row_col' format

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if tile names are not in expected format
        return float('inf')
    return abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to paint all goal tiles that are not
    currently painted correctly. It sums the estimated cost for each
    unpainted goal tile independently. The estimated cost for a single tile
    includes the paint action itself, the minimum movement cost for any robot
    to reach an adjacent tile, and the cost for that robot to acquire the
    correct color.

    # Assumptions
    - Goal tiles are either clear or already painted with the correct color
      in the initial state. If a goal tile is painted with the wrong color,
      the heuristic returns infinity, assuming this state is unsolvable.
    - The "clear" precondition for moving into a tile and for painting an
      adjacent tile is ignored for simplicity and efficiency. Movement cost
      is estimated using Manhattan distance, assuming free movement on the grid.
    - A robot can change color in one action if the desired color is available.
      All colors specified in goal conditions are assumed to be available.
    - The cost of actions is 1.

    # Heuristic Initialization
    - Extracts goal conditions to identify which tiles need to be painted
      and with which colors. Stores this in `self.goal_paintings`.
    - Extracts static facts (`up`, `down`, `left`, `right`) to build the
      adjacency information for tiles. Stores this in `self.adjacency`.
      This is used to find tiles adjacent to a goal tile.
    - Stores available colors (extracted from `available-color` facts),
      although the heuristic primarily relies on goal colors being available.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the total heuristic cost to 0.
    2. Extract the current locations and held colors for all robots from the state.
    3. Extract the current painted status and colors, or clear status, for all tiles from the state.
    4. Iterate through each goal tile `T` and its required color `C` as stored during initialization (`self.goal_paintings`).
    5. For the current goal tile `T`:
       a. Check the current state of `T` using the extracted tile states.
       b. If `T` is currently painted with color `C`, the goal for this tile is satisfied. Cost contribution is 0.
       c. If `T` is currently painted with a color `C'` different from `C`, assume this state is unsolvable and return `float('inf')`.
       d. If `T` is not painted with `C` (i.e., it's clear or its state implies it needs painting): This tile needs to be painted with color `C`. Calculate the minimum cost to achieve this for this specific tile.
          - This cost includes 1 action for the `paint` operation itself.
          - It also includes the minimum cost to get a robot into a state where it *can* paint `T` (i.e., adjacent to `T` and holding color `C`).
          - Initialize `min_enable_cost` for this tile to `float('inf')`.
          - Find all tiles adjacent to `T` using the precomputed `self.adjacency`. If `T` has no adjacent tiles, it cannot be painted, so return `float('inf')` for the total heuristic.
          - For each robot `R`:
             - Get `R`'s current location `R_loc` and color `R_color`.
             - Calculate the cost for `R` to acquire color `C`: This is 0 if `R_color` is already `C`, otherwise it's 1 (for a `change_color` action).
             - Calculate the minimum Manhattan distance from `R_loc` to any of the tiles adjacent to `T`.
             - The cost for robot `R` to *enable* painting `T` is (minimum distance to adjacent tile) + (color acquisition cost).
             - Update `min_enable_cost` for tile `T` with the minimum value found across all robots.
          - If after checking all robots, `min_enable_cost` is still `float('inf')`, it means no robot can reach an adjacent tile for this specific goal.
            This could happen if there are no robots or if the grid is disconnected in a way that prevents reaching adjacent tiles.
            Return `float('inf')` in this case.
          - Add `1 (paint action) + min_enable_cost` to the `total_heuristic_cost`.
    6. After iterating through all goal tiles, return the `total_heuristic_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal tiles and their required colors
        self.goal_paintings = {} # {tile_name: color_name}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "painted":
                # Goal is (painted tile color)
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_paintings[tile] = color


        # Build adjacency information {tile_name: [adjacent_tile_name, ...]}
        self.adjacency = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                # Fact is (direction tile1 tile2)
                if len(parts) == 3:
                    _, tile1, tile2 = parts
                    if tile1 not in self.adjacency:
                        self.adjacency[tile1] = []
                    if tile2 not in self.adjacency:
                        self.adjacency[tile2] = []
                    # Adjacency is symmetric
                    if tile2 not in self.adjacency[tile1]:
                         self.adjacency[tile1].append(tile2)
                    if tile1 not in self.adjacency[tile2]:
                         self.adjacency[tile2].append(tile1)

        # Store available colors (extracted from available-color facts)
        self.available_colors = {get_parts(fact)[1] for fact in static_facts if match(fact, "available-color", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract current robot locations and colors
        robot_info = {} # {robot_name: {'location': tile_name, 'color': color_name}}
        # Initialize robots from state facts, as initial state might not list all robots explicitly if they have no predicates
        # However, robot-at and robot-has are the key predicates for robot state.
        # Let's assume robots are introduced via robot-at or robot-has facts in the initial state.
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['location'] = location
            elif parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color

        # Extract current tile states (painted or clear)
        tile_states = {} # {tile_name: {'state': 'painted', 'color': color} or {'state': 'clear'}}
        # Iterate through all facts to find tile states
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                tile_states[tile] = {'state': 'painted', 'color': color}
            elif parts[0] == "clear":
                 tile = parts[1]
                 # Only record clear if the tile hasn't been marked as painted already in this state
                 if tile not in tile_states or tile_states[tile]['state'] != 'painted':
                    tile_states[tile] = {'state': 'clear'}

        total_heuristic_cost = 0

        # Iterate through goal tiles and calculate cost for each unsatisfied goal
        for goal_tile, goal_color in self.goal_paintings.items():
            current_state = tile_states.get(goal_tile)

            # Check if goal is already satisfied for this tile
            if current_state and current_state['state'] == 'painted' and current_state['color'] == goal_color:
                continue # Goal satisfied for this tile

            # Check if tile is painted with the wrong color (assumed unsolvable/very costly)
            if current_state and current_state['state'] == 'painted' and current_state['color'] != goal_color:
                 # If a goal tile is painted the wrong color, it seems impossible to fix in this domain.
                 # Return infinity to prune this branch.
                 return float('inf')

            # Tile needs painting (it's clear or its state is not the correct painted state)
            # Cost includes 1 for the paint action + cost to get a robot with the right color adjacent

            min_enable_cost = float('inf')

            # Find tiles adjacent to the goal tile
            adjacent_tiles = self.adjacency.get(goal_tile, [])

            if not adjacent_tiles or not robot_info:
                 # Cannot paint if no adjacent tiles or no robots exist
                 return float('inf')

            # Calculate min cost for any robot to enable painting this tile
            for robot_name, info in robot_info.items():
                robot_location = info.get('location')
                robot_color = info.get('color')

                if robot_location is None:
                    # Robot location unknown from state facts, cannot use this robot
                    continue

                # Cost to get the correct color
                # Assumes change_color is always possible if the color is available (which goal colors are assumed to be)
                color_cost = 0 if robot_color == goal_color else 1

                # Minimum moves for this robot to reach any adjacent tile
                min_dist_to_adjacent = float('inf')
                for adj_tile in adjacent_tiles:
                    dist = manhattan_distance(robot_location, adj_tile)
                    min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

                # Total cost for this robot to enable painting this tile
                # Only consider if a path exists (distance is not inf)
                if min_dist_to_adjacent != float('inf'):
                    enable_cost = min_dist_to_adjacent + color_cost
                    min_enable_cost = min(min_enable_cost, enable_cost)

            # If after checking all robots, min_enable_cost is still inf for *this tile*,
            # it means no robot can reach an adjacent tile for this specific goal.
            # This could happen if the grid is disconnected in a way that prevents reaching adjacent tiles.
            if min_enable_cost == float('inf'):
                 return float('inf')

            # Add cost for this tile: 1 (paint action) + min_enable_cost
            total_heuristic_cost += (1 + min_enable_cost)

        return total_heuristic_cost
