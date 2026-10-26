from fnmatch import fnmatch
from collections import deque
# Assume Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle potential empty facts or malformed ones
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args.
    # This is a simplification, assuming '*' matches exactly one part.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# The Heuristic class
# Inherit from Heuristic if available in the planning environment
class floortileHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions (painting tiles with specific colors). It sums up the estimated
    costs for each unsatisfied goal tile, considering the need for the correct
    color, robot movement, and the paint action itself. It also includes a cost
    to clear a goal tile if it is currently occupied by a robot.

    # Assumptions
    - The grid structure defined by up/down/left/right predicates is static and forms a connected graph.
    - Movement cost between adjacent tiles is 1.
    - Changing color costs 1.
    - Painting a tile costs 1.
    - If a goal tile is occupied by a robot, that robot must move off (cost 1).
    - The heuristic calculates costs for each unsatisfied goal tile independently
      and sums them, plus a global cost for acquiring needed colors. This ignores
      synergies (e.g., one robot painting multiple tiles, one color change serving
      multiple tiles).
    - The distance calculation for movement uses precomputed shortest paths on the
      static grid, ignoring the 'clear' precondition for intermediate tiles on the path.
    - Solvable instances do not require unpainting tiles or moving painted tiles.
    - All objects of type 'tile' are part of the grid graph and are mentioned in
      initial state, goals, or static facts.

    # Heuristic Initialization
    - Extracts all tile objects mentioned in the initial state, goals, or static facts.
    - Builds an adjacency list representation of the tile grid from static up/down/left/right facts.
    - Computes all-pairs shortest paths on the static grid using BFS.
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color of each robot.
    2. Identify the set of goal tiles that are not yet painted with the correct color (unsatisfied goals).
    3. Calculate **Color Costs**: Determine which colors are required by the unsatisfied goals. For each required color that is *not* currently held by any robot, add 1 to the total heuristic. This estimates the cost of changing color to make that color available.
    4. Calculate **Tile Costs**: For each tile `T` that is an unsatisfied goal (needs color `C`):
        a. Initialize cost for this tile to 0.
        b. Add 1 for the final `paint_...` action.
        c. Check if tile `T` is currently occupied by a robot. If yes, add 1 to the cost for this tile (estimating the cost to move the robot off).
        d. Estimate the movement cost to get a robot to a tile adjacent to `T`. Find the minimum grid distance (using precomputed distances) from any robot's current location to any tile `AdjT` that is adjacent to `T`.
        e. If no adjacent tile exists for `T` or no robot exists, the minimum distance will be infinity. In this case, the tile is unreachable, and the heuristic should reflect this (e.g., return infinity).
        f. Add this minimum distance to the cost for this tile.
        g. Add the calculated cost for this tile to the total heuristic.
    5. The total heuristic value is the sum of the color costs and the tile costs.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting grid structure and goals."""
        self.goals = task.goals
        static_facts = task.static

        # Extract all tile objects mentioned in initial state, goals, or static facts
        self.all_tiles = set()
        all_facts = task.initial_state | static_facts | self.goals
        for fact in all_facts:
             parts = get_parts(fact)
             # A simple heuristic to identify tiles: starts with "tile_"
             # This relies on naming conventions in the problem instances.
             for part in parts:
                 if part.startswith("tile_"):
                     self.all_tiles.add(part)

        # Build adjacency list from static facts
        self.adj_list = {tile: [] for tile in self.all_tiles}
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Predicate is (dir tile1 tile2) meaning tile1 is dir from tile2
                # This implies tile1 and tile2 are adjacent.
                t1, t2 = parts[1], parts[2]
                # Ensure both are recognized tiles before adding adjacency
                if t1 in self.all_tiles and t2 in self.all_tiles:
                    self.adj_list[t1].append(t2)
                    self.adj_list[t2].append(t1) # Adjacency is symmetric

        # Compute all-pairs shortest paths on the grid
        self.grid_dist = {}
        for start_tile in self.all_tiles:
            self.grid_dist[start_tile] = self._bfs(start_tile)

    def _bfs(self, start_tile):
        """Performs BFS from a start tile to find distances to all other tiles."""
        distances = {tile: float('inf') for tile in self.all_tiles}
        if start_tile not in self.all_tiles:
             # This tile is not in our recognized tile set, cannot compute distance
             return distances

        distances[start_tile] = 0
        queue = deque([start_tile])

        while queue:
            current_tile = queue.popleft()
            current_dist = distances[current_tile]

            # Use .get() with default empty list for tiles that might not be in adj_list keys
            for neighbor in self.adj_list.get(current_tile, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Extract relevant information from the current state
        robot_locs = {}
        robot_colors = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == "robot-at" and len(parts) == 3:
                robot_locs[parts[1]] = parts[2]
            elif parts[0] == "robot-has" and len(parts) == 3:
                robot_colors[parts[1]] = parts[2]

        # Identify unsatisfied goal tiles
        unsatisfied_goals = set()
        for goal in self.goals:
            # Assuming goals are always (painted tile color)
            goal_parts = get_parts(goal)
            if len(goal_parts) == 3 and goal_parts[0] == "painted":
                 goal_tile, goal_color = goal_parts[1], goal_parts[2]
                 # Check if the goal fact is NOT in the current state
                 if goal not in state:
                     unsatisfied_goals.add((goal_tile, goal_color))
            # Note: This heuristic assumes tiles painted with the wrong color
            # are not possible in solvable instances, as there's no unpaint action.
            # If a tile is painted with the wrong color, it will remain in unsatisfied_goals
            # but the heuristic will still calculate a cost, which is fine for non-admissibility.

        # If all goals are satisfied, heuristic is 0
        if not unsatisfied_goals:
            return 0

        total_cost = 0

        # 1. Color Costs: Count how many needed colors are not held by any robot.
        needed_colors = {color for tile, color in unsatisfied_goals}
        held_colors = set(robot_colors.values())
        colors_to_acquire = needed_colors - held_colors
        total_cost += len(colors_to_acquire) # Assume 1 change_color per color type needed

        # 2. Tile Costs (Clearing + Movement + Painting)
        for tile, color in unsatisfied_goals:
            cost_for_tile = 0

            # Cost to make the tile clear if occupied by a robot
            # The paint action requires (clear tile). If a robot is on `tile`, it's not clear.
            is_occupied = any(loc == tile for loc in robot_locs.values())
            if is_occupied:
                cost_for_tile += 1 # Estimate 1 move action to clear the tile

            # Cost to get a robot to an adjacent tile + paint action
            # Find min dist from any robot to any tile adjacent to `tile`.
            min_dist_to_adjacent = float('inf')
            adjacent_to_tile = self.adj_list.get(tile, [])

            # If the goal tile has no adjacent tiles in the grid, it's unreachable for painting
            if not adjacent_to_tile:
                 # This indicates a problem with the grid structure or goal.
                 # Return infinity as the tile cannot be painted by adjacent actions.
                 return float('inf')

            # Find the minimum distance from any robot to any adjacent tile
            if not robot_locs:
                 # No robots exist, cannot paint.
                 return float('inf')

            for robot_loc in robot_locs.values():
                # Ensure robot_loc is a valid tile in our grid distance map
                if robot_loc in self.grid_dist:
                    for adj_tile in adjacent_to_tile:
                         # Use precomputed grid distance (relaxation on path, ignoring clear)
                         dist = self.grid_dist[robot_loc].get(adj_tile, float('inf'))
                         min_dist_to_adjacent = min(min_dist_to_adjacent, dist)
                else:
                    # Robot is at a location not in our tile map? Should not happen in this domain.
                    # Treat as unreachable for this robot.
                    pass # This robot doesn't contribute to min_dist_to_adjacent

            # If min_dist_to_adjacent is still inf, it means no robot can reach any adjacent tile.
            # This shouldn't happen in a connected grid with robots present.
            # If it somehow happens, the problem might be unsolvable from this state.
            # Returning infinity is appropriate.
            if min_dist_to_adjacent == float('inf'):
                return float('inf') # Unreachable goal tile

            # Add movement cost and paint action cost
            cost_for_tile += min_dist_to_adjacent + 1

            total_cost += cost_for_tile

        return total_cost
