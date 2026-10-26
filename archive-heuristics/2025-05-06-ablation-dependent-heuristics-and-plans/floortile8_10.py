# Assuming heuristics.heuristic_base.Heuristic is available
# from heuristics.heuristic_base import Heuristic

import math
from fnmatch import fnmatch

# Utility functions used by the heuristic
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty before processing
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Handle potential whitespace issues
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False # Mismatch in number of elements
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Assuming Heuristic base class is imported from heuristics.heuristic_base
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their correct colors. It sums the estimated costs for each unpainted goal tile,
    considering the cost of painting, changing color (if needed for that color),
    and moving a robot close to the tile.

    # Assumptions
    - Tiles are arranged in a grid and named `tile_row_col`.
    - Manhattan distance is a reasonable approximation for movement cost on the grid.
    - Robots always hold a color (based on domain init/actions).
    - If a tile is painted in a reachable state, it is painted with the correct color
      or is not a goal tile that needs a different color.
    - The cost of changing color for a specific color is incurred only once across all
      robots if no robot currently holds that color.
    - The movement cost for a tile is estimated by the minimum distance from any robot
      to that tile, minus one (to reach an adjacent tile).

    # Heuristic Initialization
    - Extract the goal conditions to identify which tiles need to be painted and with which colors.
    - Static facts (like grid structure) are implicitly handled by parsing tile names for coordinates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal tiles and their required colors from the task's goal conditions. Store this mapping.
    2. In the heuristic call for a given state:
       a. Identify the current location of each robot.
       b. Identify the current color held by each robot.
       c. Identify which tiles are already painted with which colors.
       d. Initialize the total heuristic cost to 0.
       e. Keep track of colors for which the 'change_color' cost has already been added.
       f. Identify the set of goal tiles that are not yet painted with their required color.
       g. If there are no unpainted goal tiles, the heuristic is 0 (goal state).
       h. For each unpainted goal tile:
          i. Add 1 to the total cost (representing the 'paint' action for this tile).
          ii. Check if any robot currently holds the required color for this tile.
          iii. If no robot holds the required color AND the 'change_color' cost for this color hasn't been added yet:
              - Add 1 to the total cost (representing one 'change_color' action).
              - Mark this color as having its 'change_color' cost accounted for.
          iv. Calculate the minimum Manhattan distance from any robot's current location to the unpainted goal tile.
          v. Add `max(0, minimum_distance - 1)` to the total cost (representing the minimum moves for the closest robot to get adjacent to the tile).
    4. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        # The base class Heuristic is expected to be initialized with the task
        # super().__init__(task) # Uncomment if the base class requires this

        self.goals = task.goals  # Goal conditions.

        # Store goal locations and colors for each tile.
        self.goal_tiles = {}
        for goal in self.goals:
            # Goal facts are expected to be strings like '(painted tile_1_1 white)'
            parts = get_parts(goal)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                self.goal_tiles[tile] = color
            # Optional: Add error handling or warning for unexpected goal formats
            # else:
            #     print(f"Warning: Unexpected goal format in goals: {goal}")


    def get_tile_coords(self, tile_name):
        """Parses tile name 'tile_row_col' and returns (row, col) integers."""
        try:
            parts = tile_name.split('_')
            if len(parts) == 3 and parts[0] == 'tile':
                row = int(parts[1])
                col = int(parts[2])
                return (row, col)
            else:
                # Return None for unparseable names
                return None
        except (ValueError, IndexError):
             # Return None for unparseable names
             return None


    def get_manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles based on their names."""
        coords1 = self.get_tile_coords(tile1_name)
        coords2 = self.get_tile_coords(tile2_name)

        if coords1 is None or coords2 is None:
            # Cannot calculate distance for unparseable tile names
            return float('inf') # Use infinity to make it non-minimal

        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Identify robot locations and colors
        robot_locations = {}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts and parts[0] == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            # Ignore other fact types or malformed facts


        # Identify painted tiles
        painted_tiles = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles.add((tile, color))
            # Ignore other fact types or malformed facts


        total_cost = 0  # Initialize action cost counter.
        color_change_accounted = set() # Track colors for which change_color cost is added

        # Identify goal tiles that are not yet painted correctly
        unpainted_goal_tiles = {
            tile for tile, required_color in self.goal_tiles.items()
            if (tile, required_color) not in painted_tiles
        }

        # If all goal tiles are painted correctly, the heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # Calculate cost for each unpainted goal tile
        for tile in unpainted_goal_tiles:
            required_color = self.goal_tiles[tile]

            # Cost for painting the tile
            total_cost += 1 # Action: paint_...

            # Cost for color change (if needed for this color)
            # Check if any robot currently has the required color
            has_color = any(color == required_color for color in robot_colors.values())

            # If no robot has the color AND we haven't accounted for changing to this color yet
            if not has_color and required_color not in color_change_accounted:
                total_cost += 1 # Action: change_color
                color_change_accounted.add(required_color)

            # Cost for movement (minimum moves for any robot to get adjacent)
            min_dist_to_tile = float('inf')
            # Iterate through all robots to find the minimum distance from their current tile
            # Only consider robots that exist in the current state
            if robot_locations: # Avoid division by zero or iterating empty dict
                for robot_tile in robot_locations.values():
                     dist = self.get_manhattan_distance(robot_tile, tile)
                     min_dist_to_tile = min(min_dist_to_tile, dist)
            else:
                 # No robots exist? Problem likely unsolvable from this state.
                 # Return infinity to prune this path.
                 return float('inf')


            # Add moves needed to get adjacent (distance - 1, but at least 0)
            # If min_dist_to_tile is inf (e.g., unparseable tile names), this will remain inf
            if min_dist_to_tile != float('inf'):
                 total_cost += max(0, min_dist_to_tile - 1)
            else:
                 # If we can't calculate distance to a goal tile (e.g., tile name issue),
                 # this state might be unsolvable or require actions not modeled by the heuristic.
                 # Returning inf is a safe default for unreachable goals.
                 return float('inf')

        return total_cost
