from heuristics.heuristic_base import Heuristic
import re

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or return empty list
        # For this problem, assume valid PDDL fact strings
        return []
    return fact[1:-1].split()

def get_coords(tile_name):
    """Parse tile name 'tile_R_C' into (R, C) integer coordinates."""
    match = re.match(r'tile_(\d+)_(\d+)', tile_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Handle cases where tile name doesn't match expected format
    # This might indicate an issue with the problem instance or domain definition
    # For robustness, return None or raise an error
    return None

def manhattan_distance(tile1_name, tile2_name):
    """Calculate Manhattan distance between two tiles using their coordinates."""
    coords1 = get_coords(tile1_name)
    coords2 = get_coords(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if coordinates are invalid
        return float('inf') # Indicate unreachable
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing up
    estimated costs for each individual tile that needs to be painted.
    For each unpainted goal tile, it calculates the minimum cost for any robot
    to move to an adjacent tile, change color if necessary, and paint the tile.
    It also adds a penalty for tiles that are painted with the wrong color,
    as these likely represent states from which the goal is unreachable in this domain.

    # Assumptions
    - The tile names follow the format 'tile_R_C' where R and C are integers representing row and column.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates corresponds to a grid where Manhattan distance is applicable.
    - Robots always hold a color (as seen in initial states).
    - Tiles painted with the wrong color cannot be repainted (based on the domain's 'clear' predicate logic, which requires a tile to be clear to be painted, and painted tiles are not clear). Such states are penalized.
    - Solvable instances only require painting tiles that are currently clear.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which tiles need to be painted and with which color.
    - Builds an adjacency list representation of the tile grid based on the 'up', 'down', 'left', and 'right' static facts.
    - Identifies all robot objects from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location and color held by each robot by iterating through the state facts.
    2. Initialize the total heuristic value to 0 and a counter for wrongly painted goal tiles to 0.
    3. Iterate through each tile and its required color as specified in the goal state (extracted during initialization).
    4. For the current goal tile:
       a. Check if the tile is already painted with the correct color according to the goal. If yes, this goal is satisfied for this tile; continue to the next goal tile.
       b. If the tile is not painted correctly, check if it is painted with *any* color different from the goal color by iterating through state facts. If yes, increment the wrongly painted counter and skip this tile for further cost calculation (it's handled by the penalty).
       c. If the tile is not painted correctly and is not painted wrongly (implying it is clear or unpainted, as per domain assumptions for solvable states):
          i. This tile needs to be painted. Calculate the minimum cost for *any* robot to perform this painting task. Initialize `min_cost_for_tile` to infinity.
          ii. For each robot identified during initialization:
             - Get the robot's current location and color from the state information gathered in step 1.
             - Determine the cost to change the robot's current color to the required goal color (1 if the colors are different, 0 if they are the same).
             - Calculate the minimum Manhattan distance from the robot's current location to *any* tile adjacent to the target tile. Iterate through the adjacent tiles obtained from the precomputed adjacency list.
             - The estimated cost for this robot to paint this specific tile is the minimum movement distance + color change cost + 1 (for the paint action itself).
             - Update `min_cost_for_tile` with the minimum cost found across all robots for this tile.
          iii. After checking all robots, add `min_cost_for_tile` to the total heuristic value. If `min_cost_for_tile` is still infinity (meaning no robot could reach an adjacent tile, e.g., disconnected grid), add a base cost (e.g., 1, representing the paint action) to ensure the heuristic is positive for unpainted clear tiles.
    5. After iterating through all goal tiles, add a penalty for each tile counted in the wrongly painted counter (e.g., 10 per tile). This penalizes states that have wrongly painted goal tiles.
    6. The final heuristic value is the total sum calculated.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        init_facts = task.init # Need init to find robots

        # Extract goal tiles and colors
        self.goal_tiles = {} # {tile_name: color_name}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_tiles[tile] = color

        # Build adjacency list for the grid and identify all tiles
        self.adj = {} # {tile_name: {adjacent_tile_name, ...}}
        self.all_tiles = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                tile1, tile2 = parts[1], parts[2]
                # Add symmetric connections as movement is bidirectional
                self.adj.setdefault(tile1, set()).add(tile2)
                self.adj.setdefault(tile2, set()).add(tile1)
                self.all_tiles.add(tile1)
                self.all_tiles.add(tile2)
        # Convert sets to lists for consistent iteration order if needed, though not strictly necessary here
        self.adj = {tile: list(neighbors) for tile, neighbors in self.adj.items()}


        # Identify robots from initial state
        self.all_robots = set()
        for fact in init_facts:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == "robot-at":
                 self.all_robots.add(parts[1])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot locations and colors
        robot_locs = {} # {robot_name: tile_name}
        robot_colors = {} # {robot_name: color_name}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                robot_locs[robot] = tile
            elif len(parts) == 3 and parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color

        total_heuristic = 0
        wrongly_painted_count = 0

        # Iterate through each tile that needs to be painted according to the goal
        for tile_to_paint, goal_color in self.goal_tiles.items():
            # Check if the tile is already painted correctly
            if f"(painted {tile_to_paint} {goal_color})" in state:
                continue # This goal is already satisfied

            # The tile is not painted correctly. Check its current state.
            is_clear = f"(clear {tile_to_paint})" in state

            # Check if it's painted with the wrong color
            is_wrongly_painted = False
            for fact in state:
                 parts = get_parts(fact)
                 if len(parts) == 3 and parts[0] == "painted" and parts[1] == tile_to_paint and parts[2] != goal_color:
                     is_wrongly_painted = True
                     break

            if is_wrongly_painted:
                # This tile is painted with the wrong color. Assume it's a dead end or requires complex undoing.
                # Add a penalty and move to the next goal tile.
                wrongly_painted_count += 1
            elif is_clear:
                # This tile needs painting and is clear. Estimate the cost to paint it.
                min_cost_for_tile = float('inf')

                # Find the minimum cost for any robot to paint this tile
                for robot in self.all_robots:
                    current_loc = robot_locs.get(robot)
                    current_color = robot_colors.get(robot)

                    # Should always find robot location and color in a valid state
                    if current_loc is None or current_color is None:
                         continue # Skip if robot state is incomplete

                    # Cost to get the right color
                    color_change_cost = 0 if current_color == goal_color else 1

                    # Cost to move to an adjacent tile
                    min_dist_to_adj = float('inf')
                    adjacent_tiles = self.adj.get(tile_to_paint, [])

                    # Calculate distance to each adjacent tile and find the minimum
                    for adj_tile in adjacent_tiles:
                        dist = manhattan_distance(current_loc, adj_tile)
                        if dist != float('inf'): # Ensure coordinates were valid
                            min_dist_to_adj = min(min_dist_to_adj, dist)

                    # If the tile has adjacent tiles and at least one is reachable
                    if min_dist_to_adj != float('inf'):
                        # Estimated cost for this robot to paint this tile:
                        # movement cost + color change cost + paint action cost (1)
                        cost = min_dist_to_adj + color_change_cost + 1
                        min_cost_for_tile = min(min_cost_for_tile, cost)

                # Add the minimum cost found for this tile to the total heuristic
                if min_cost_for_tile != float('inf'):
                    total_heuristic += min_cost_for_tile
                else:
                    # This tile needs painting and is clear, but no robot can reach an adjacent tile.
                    # This shouldn't happen in connected grids, but as a fallback,
                    # add a base cost to ensure heuristic > 0 if there are unpainted clear tiles.
                    # A base cost of 1 represents the paint action itself, assuming reachability eventually.
                    total_heuristic += 1 # Base cost for an unpainted, clear, but seemingly unreachable tile

            # If the tile is neither painted correctly, nor wrongly painted, nor clear,
            # it's in an unexpected state. We ignore it for heuristic calculation,
            # assuming valid problem instances maintain clear/painted state correctly.

        # Add a penalty for each wrongly painted goal tile
        # The penalty value is somewhat arbitrary; 10 is chosen to be larger than typical per-tile painting costs.
        total_heuristic += wrongly_painted_count * 10

        return total_heuristic
