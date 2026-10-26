# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# This would be replaced by the actual base class in the planner environment
class Heuristic:
    def __init__(self, task):
        # task object is assumed to have 'goals' (frozenset of goal facts)
        # and 'static' (frozenset of static facts)
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        # node object is assumed to have 'state' (frozenset of facts)
        raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the minimum
    estimated cost for each unpainted goal tile to be painted correctly.
    For each unpainted goal tile, it finds the minimum cost among all robots
    to move to an adjacent tile, acquire the correct color, and paint the tile.
    If a robot is currently on the goal tile, the cost to move it off is included
    in its movement cost calculation for that tile.

    # Assumptions
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - Tile names are in the format 'tile_R_C' where R is row and C is column.
    - Manhattan distance is a reasonable estimate for movement cost between tiles.
    - If a goal tile is painted with the wrong color in a state, the problem is
      considered unsolvable from that state (heuristic returns a large value).
    - If a goal tile is unreachable by any robot, the problem is considered
      unsolvable from that state (heuristic returns a large value).
    - The initial state and static facts provide a complete definition of all tiles
      and their adjacency relations relevant to the goals.

    # Heuristic Initialization
    - Parses goal facts to identify target tiles and their required colors.
    - Parses static facts to build the grid structure:
        - Identifies all tile names involved in adjacency relations or goals.
        - Maps tile names ('tile_R_C') to (row, col) coordinates based on the name format.
        - Builds an adjacency list mapping each tile to its neighbors based on
          'up', 'down', 'left', 'right' static facts.
    - Stores the goal tile requirements and grid information.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic value `h` to 0.
    2. Extract the current location (`robot-at`) and held color (`robot-has`) for each robot from the state.
    3. Extract the current status (`clear` or `painted`) for each tile from the state. Default status is 'unknown' if not explicitly clear or painted.
    4. Iterate through each goal tile and its required color (`goal_tile`, `goal_color`) stored during initialization.
    5. For the current `goal_tile`:
        a. Check if the tile is already painted with `goal_color` in the current state (`(painted goal_tile goal_color)` is in state). If yes, this goal is satisfied; continue to the next goal tile.
        b. Check if the tile is painted with a *different* color in the state. If yes, the problem is likely unsolvable from this state; return a large value (e.g., `float('inf')`).
        c. Calculate the minimum cost for *any* robot to paint this `goal_tile` with `goal_color`. Initialize `min_robot_task_cost` to infinity.
        d. For each robot `r` found in the state (from `robot_locations` keys):
            i. Get the robot's current location `r_loc`. If the robot's location is not found (e.g., robot not mentioned in state facts, though unlikely), skip this robot.
            ii. Calculate the cost for the robot to have the `goal_color`: 0 if `r` already has `goal_color` (fact `(robot-has r goal_color)` is in state), 1 otherwise (for a `change_color` action). Let this be `color_cost`.
            iii. Calculate the movement cost for the robot to be in a position to paint the tile.
                - If the robot `r` is currently *on* the `goal_tile` (`r_loc == goal_tile`): The robot must first move off (cost 1). After moving off, it will be on an adjacent tile, which is adjacent to the `goal_tile`. So, the movement cost to get adjacent is 1 (move off) + 0 (already adjacent) = 1.
                - If the robot `r` is *not* on the `goal_tile` (`r_loc != goal_tile`): The robot must move from `r_loc` to any tile adjacent to `goal_tile`. The minimum cost is the minimum Manhattan distance from `r_loc` to any neighbor of `goal_tile`. Let this be `dist_to_adj`. The movement cost is `dist_to_adj`.
                - If `goal_tile` has no neighbors (unlikely in typical grids but possible), `dist_to_adj` will remain infinity.
            iv. If the calculated movement cost is still infinity (meaning the tile is unreachable by this robot), this robot cannot paint it; continue to the next robot.
            v. The total estimated cost for this robot `r` to paint the `goal_tile` is `movement_cost + color_cost + 1` (the +1 is for the `paint` action itself).
            vi. Update `min_robot_task_cost = min(min_robot_task_cost, robot_task_cost)`.
        e. After checking all robots, if `min_robot_task_cost` is still infinity, it means no robot can reach and paint this goal tile. The problem is likely unsolvable from this state; return a large value (e.g., `float('inf')`).
        f. Add `min_robot_task_cost` to the total heuristic value `h`.
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and grid structure.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        self.static = task.static

        # Store goal locations and colors
        self.goal_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color

        # Extract all tile names and their coordinates
        self.tile_coords = {}
        self.all_tiles = set()

        # Collect all potential tile names from static facts (adjacency) and goals
        potential_tiles = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                 potential_tiles.add(parts[1])
                 potential_tiles.add(parts[2])
        for tile in self.goal_tiles:
             potential_tiles.add(tile)

        # Parse coordinates from tile names (assuming tile_R_C format)
        for tile in potential_tiles:
            try:
                # Extract R and C from 'tile_R_C'
                parts = tile.split('_')
                if len(parts) == 3 and parts[0] == 'tile':
                    row = int(parts[1])
                    col = int(parts[2])
                    self.tile_coords[tile] = (row, col)
                    self.all_tiles.add(tile) # Only add tiles with valid coordinates
            except (ValueError, IndexError):
                # Ignore tiles that don't match the expected format
                pass

        # Build adjacency list using only tiles with valid coordinates
        self.tile_neighbors = {tile: set() for tile in self.all_tiles}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right'] and len(parts) == 3:
                tile1, tile2 = parts[1], parts[2]
                # Ensure both tiles have valid coordinates before adding adjacency
                if tile1 in self.tile_coords and tile2 in self.tile_coords:
                     # PDDL relation (rel tile1 tile2) means tile1 is rel of tile2
                     # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1
                     # So tile_1_1 is a neighbor of tile_0_1, and tile_0_1 is a neighbor of tile_1_1
                     self.tile_neighbors[tile1].add(tile2)
                     self.tile_neighbors[tile2].add(tile1)


    def manhattan_distance(self, tile1, tile2):
        """
        Calculates the Manhattan distance between two tiles.
        Assumes tile names are in 'tile_R_C' format and coordinates are available.
        Returns float('inf') if coordinates are not known for either tile.
        """
        coord1 = self.tile_coords.get(tile1)
        coord2 = self.tile_coords.get(tile2)
        if coord1 is None or coord2 is None:
            # This should ideally not happen if all_tiles and tile_coords are built correctly
            # from the problem definition.
            return float('inf')
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract current robot locations and colors
        robot_locations = {}
        robot_colors = {}
        # Collect all robot names mentioned in the state
        robots_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'robot-at' and len(parts) == 3:
                    robot, tile = parts[1], parts[2]
                    robot_locations[robot] = tile
                    robots_in_state.add(robot)
                elif parts[0] == 'robot-has' and len(parts) == 3:
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
                    robots_in_state.add(robot)

        # Extract current tile status (clear or painted color)
        # Initialize all known tiles to 'unknown' status
        tile_status = {tile: 'unknown' for tile in self.all_tiles}

        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'clear' and len(parts) == 2:
                    tile = parts[1]
                    if tile in tile_status: # Only update status for known tiles
                        tile_status[tile] = 'clear'
                elif parts[0] == 'painted' and len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    if tile in tile_status: # Only update status for known tiles
                         tile_status[tile] = color # Store the color it's painted with

        total_heuristic = 0

        # Iterate through each goal tile
        for goal_tile, goal_color in self.goal_tiles.items():
            # Ensure the goal tile is one we know about
            if goal_tile not in self.all_tiles:
                 # This goal tile wasn't in static facts or other initial parsing.
                 # Problem might be malformed or unsolvable.
                 return float('inf')

            # Check if the goal is already satisfied for this tile
            if tile_status.get(goal_tile) == goal_color:
                continue # This goal is met

            # Check if the tile is painted with the wrong color (unsolvable state)
            current_status = tile_status.get(goal_tile)
            if current_status != 'clear' and current_status != 'unknown' and current_status != goal_color:
                 # Tile is painted, but not with the goal color. Assuming unsolvable.
                 return float('inf') # Return a large value

            # Calculate the minimum cost for any robot to paint this tile
            min_robot_task_cost = float('inf')

            # Iterate through all robots found in the state
            for robot in robots_in_state:
                r_loc = robot_locations.get(robot)
                if r_loc is None:
                    # Robot exists but its location is unknown in this state? Should not happen in valid states.
                    continue # Skip this robot

                r_color = robot_colors.get(robot) # Get robot's current color

                color_cost = 0 if r_color == goal_color else 1

                movement_cost = float('inf')
                if r_loc == goal_tile:
                    # Robot is on the tile. Must move off (cost 1).
                    # After moving off, it will be on an adjacent tile, which is adjacent to goal_tile.
                    # So, movement cost to get adjacent is 1 (move off) + 0 (already adjacent) = 1.
                    movement_cost = 1
                else:
                    # Robot is not on the tile. Find min distance from r_loc to any neighbor of goal_tile.
                    min_dist_to_adj = float('inf')
                    neighbors = self.tile_neighbors.get(goal_tile, set())

                    if not neighbors:
                         # Goal tile has no neighbors, likely unreachable
                         continue # This robot cannot paint it

                    for adj_tile in neighbors:
                        dist = self.manhattan_distance(r_loc, adj_tile)
                        min_dist_to_adj = min(min_dist_to_adj, dist)

                    if min_dist_to_adj != float('inf'):
                         movement_cost = min_dist_to_adj


                # If movement_cost is still infinity, this robot cannot reach an adjacent tile
                if movement_cost == float('inf'):
                    continue

                # Total cost for this robot to paint this tile: move + color change + paint action
                robot_task_cost = movement_cost + color_cost + 1 # +1 for the paint action

                min_robot_task_cost = min(min_robot_task_cost, robot_task_cost)

            # If after checking all robots, no robot can paint this tile, it's unreachable
            if min_robot_task_cost == float('inf'):
                 return float('inf') # Problem likely unsolvable

            total_heuristic += min_robot_task_cost

        return total_heuristic
