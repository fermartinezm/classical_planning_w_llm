# Import necessary modules
from fnmatch import fnmatch
from collections import deque
import sys # Needed for float('inf')

# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Helper functions
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the estimated cost for each unpainted goal tile.
    The cost for a single tile includes:
    1. The cost to get the correct color (estimated as 1 if a required color is not
       held by any robot, 0 otherwise). This cost is counted once per distinct missing
       color across all robots and tiles.
    2. The minimum distance (number of moves) for any robot to reach any tile adjacent
       to the target tile.
    3. The cost of the paint action (1).

    It simplifies the problem by ignoring potential conflicts (multiple robots needing
    the same tile or color simultaneously) and doesn't model robot task assignment.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - Tiles that need to be painted are initially 'clear' or unpainted.
    - Once a tile is painted, it cannot be unpainted or repainted in this domain.
    - If a goal tile is found to be painted with the wrong color, the state is unsolvable.
    - All colors required by the goal are 'available-color'.
    - The problem is solvable unless a goal tile is painted with the wrong color or
      a goal tile is unreachable by any robot.

    # Heuristic Initialization
    - Parses goal conditions to identify which tiles need to be painted and with which colors.
    - Builds an adjacency graph of tiles based on 'up', 'down', 'left', 'right' static facts.
    - Identifies all available colors (though this is less critical for the simplified color cost).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal facts of the form `(painted tile color)`. Store these target
       paintings in `self.goal_paintings`.
    2. Build the tile adjacency graph from static 'up', 'down', 'left', 'right' facts.
       Store this in `self.adj_list`.
    3. Identify available colors from static 'available-color' facts (stored in `self.available_colors`).
    4. In the `__call__` method for a given state:
       a. Identify the current location of each robot (`robot_locations`).
       b. Identify the color currently held by each robot (`robot_colors`).
       c. Identify which tiles are currently painted and with what color (`painted_tiles`).
       d. Check for unsolvable states: If any goal tile is currently painted with a
          color different from its target color, return `float('inf')`.
       e. Determine the set of goal tiles that are *not* currently painted with their
          required color (`unpainted_goal_paintings`). If this set is empty,
          the heuristic is 0.
       f. Calculate the 'color cost': Identify all colors required by the tiles in
          `unpainted_goal_paintings`. For each required color, if no robot currently
          holds that color, add 1 to the color cost. This counts the minimum number
          of `change_color` actions needed to make all required colors available
          to at least one robot.
       g. Calculate the 'movement and paint cost': For each tile `T` in
          `unpainted_goal_paintings` that needs color `C`:
          i. Find the set of tiles adjacent to `T` using `self.adj_list`.
          ii. For each robot `R`, calculate the shortest distance from `R`'s current
              location to *any* tile in the set of adjacent tiles of `T`. This requires
              a BFS starting from `R`'s location on the tile graph (`_bfs_distance`).
          iii. Find the minimum of these distances over all robots. Let this be `min_dist`.
          iv. Add `min_dist + 1` (for the paint action) to the total movement and paint cost.
       h. The total heuristic value is the sum of the 'color cost' and the
          'movement and paint cost'.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and building the tile graph.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goal facts to get target paintings {tile: color}
        self.goal_paintings = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                tile, color = args
                self.goal_paintings[tile] = color

        # 2. Build the tile adjacency graph {tile: [adjacent_tiles]}
        self.adj_list = {}
        # Collect all tiles mentioned in adjacency facts
        all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                 tile_a, tile_b = parts[1], parts[2]
                 all_tiles.add(tile_a)
                 all_tiles.add(tile_b)
                 self.adj_list.setdefault(tile_a, []).append(tile_b)
                 self.adj_list.setdefault(tile_b, []).append(tile_a) # Adjacency is symmetric

        # Ensure all tiles mentioned are keys in adj_list
        for tile in all_tiles:
             self.adj_list.setdefault(tile, [])

        # Remove duplicates from adjacency lists
        for tile in self.adj_list:
            self.adj_list[tile] = list(set(self.adj_list[tile]))

        # 3. Identify available colors (less critical for simplified color cost but good practice)
        self.available_colors = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "available-color", "*")
        }

    def _bfs_distance(self, start_tile, target_tiles):
        """
        Performs BFS from start_tile to find the minimum distance to any tile
        in the target_tiles set. Returns infinity if no path exists.
        """
        if not target_tiles:
            return float('inf') # No targets, no path needed

        # Handle case where start_tile is one of the target tiles
        if start_tile in target_tiles:
             return 0

        # Ensure start_tile is a valid tile in our graph
        if start_tile not in self.adj_list:
             return float('inf') # Cannot start BFS from an unknown tile

        queue = deque([(start_tile, 0)])
        visited = {start_tile}

        while queue:
            current_tile, dist = queue.popleft()

            # Check neighbors only if the current tile has neighbors in our graph
            if current_tile in self.adj_list:
                for neighbor in self.adj_list[current_tile]:
                    if neighbor in target_tiles:
                        return dist + 1 # Found a target tile
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # No path found

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 4a. Identify robot locations {robot: tile}
        robot_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile

        # 4b. Identify robot colors {robot: color}
        robot_colors = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        # 4c. Identify painted tiles {tile: color}
        painted_tiles = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_tiles[tile] = color

        # 4d. Check for unsolvable states: goal tile painted with wrong color
        for tile, target_color in self.goal_paintings.items():
             current_painted_color = painted_tiles.get(tile)
             if current_painted_color is not None and current_painted_color != target_color:
                  # Goal tile is painted with the wrong color -> unsolvable
                  return float('inf')

        # 4e. Determine unpainted goal tiles {tile: color}
        unpainted_goal_paintings = {}
        for tile, target_color in self.goal_paintings.items():
            goal_fact_str = f"(painted {tile} {target_color})"
            if goal_fact_str not in state:
                 # Tile needs painting if it's not already painted correctly.
                 # We already checked for wrong color above.
                 # So if it's not painted correctly, it must be clear (or initially unpainted).
                 unpainted_goal_paintings[tile] = target_color


        # If all goal tiles are painted correctly, heuristic is 0.
        if not unpainted_goal_paintings:
            return 0

        # 4f. Calculate color cost
        required_colors = set(unpainted_goal_paintings.values())
        colors_held = set(robot_colors.values())

        color_cost = 0
        # Count how many required colors are NOT held by any robot
        missing_colors = required_colors - colors_held
        color_cost = len(missing_colors)
        # Note: This assumes one change_color action can satisfy the need for one missing color.
        # It doesn't consider if multiple robots need the same missing color, or if a robot
        # needs to change color multiple times. It's a simple lower bound on color changes.


        # 4g. Calculate movement and paint cost
        movement_paint_cost = 0

        # Check if there are any robots to perform actions
        if not robot_locations:
             # If there are unpainted goals but no robots, it's unsolvable
             return float('inf')

        for tile, target_color in unpainted_goal_paintings.items():
            # Find adjacent tiles to the target tile
            adjacent_tiles = set(self.adj_list.get(tile, []))

            if not adjacent_tiles:
                 # This goal tile has no adjacent tiles, likely unsolvable.
                 # A robot cannot paint a tile it is not adjacent to.
                 return float('inf')

            # Find the minimum distance from any robot to any adjacent tile
            min_dist_to_adjacent = float('inf')

            for robot_loc in robot_locations.values():
                 # Calculate distance from robot_loc to any tile in adjacent_tiles
                 dist = self._bfs_distance(robot_loc, adjacent_tiles)
                 min_dist_to_adjacent = min(min_dist_to_adjacent, dist)

            # If min_dist_to_adjacent is still infinity, it means no robot can reach
            # any adjacent tile. Unsolvable.
            if min_dist_to_adjacent == float('inf'):
                 return float('inf')

            # Add the cost: min moves to adjacent + 1 paint action
            movement_paint_cost += min_dist_to_adjacent + 1

        # 4h. Total heuristic
        return color_cost + movement_paint_cost
