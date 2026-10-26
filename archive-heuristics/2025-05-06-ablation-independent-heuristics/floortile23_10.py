# Need to import Heuristic base class
from heuristics.heuristic_base import Heuristic
# Need Task class definition (although not directly used, good practice)
# from task import Task # Assuming task is available in the environment

import collections # For BFS queue
import math # For infinity

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string into a list of components."""
    # Remove surrounding parentheses and split by spaces
    return fact_str.strip('()').split()

# Helper function to parse tile name into coordinates (assuming tile_row_col format)
def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_row_col' into (row, col) tuple."""
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            return (int(parts[1]), int(parts[2]))
        else:
            # Handle unexpected tile names if necessary
            # For this domain, tile_row_col seems standard
            return None
    except (ValueError, IndexError):
        # Handle parsing errors
        return None

# Helper function to calculate Manhattan distance
def manhattan_distance(coord1, coord2):
    """Calculates Manhattan distance between two (row, col) coordinates."""
    r1, c1 = coord1
    r2, c2 = coord2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up the
    estimated cost for each unpainted goal tile. For each unpainted goal tile
    (T, C), the estimated cost is calculated as:
    (1 if T is not clear) + min_robot_cost_to_paint_T + 1 (for the paint action).
    The min_robot_cost_to_paint_T is the minimum over all robots R of the cost
    for R to be able to paint T with color C. This robot cost is estimated as:
    (1 if R does not have color C) + estimated_move_cost_for_R_to_reach_adjacent_to_T.
    The estimated_move_cost is calculated using Manhattan distance on the grid:
    the minimum Manhattan distance from R's current location R_loc to any tile
    T_adj from which T can be painted.

    Assumptions:
    - The tile names follow the format 'tile_row_col' allowing coordinate extraction.
    - The grid structure defined by 'up', 'down', 'left', 'right' facts is consistent
      with a rectangular grid layout.
    - Solvable instances have a connected tile grid and available colors.
    - The goal only consists of (painted tile color) facts.
    - If a goal tile is not clear, it is because a robot is occupying it (in solvable states).

    Heuristic Initialization:
    1. Parse all tile names from the static connectivity facts ('up', 'down', 'left', 'right').
    2. Build a dictionary mapping target tiles to lists of source tiles from which they can be painted (paintable_from[Y] = [X1, X2, ...] if robot at Xi can paint Y). This is derived from the static facts (direction Y X).
    3. Assign (row, col) coordinates to each tile name by performing a BFS traversal
       starting from an arbitrary tile assigned (0, 0), using the directed movement
       relations (derived from static facts (direction Y X) as X -> Y) to infer relative coordinates. Store this mapping in `self.tile_coords`.
    4. Store the goal facts from the task.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of goal facts `(painted T C)` that are not currently true in the state.
    2. Initialize the total heuristic value to 0.
    3. Parse the current state to find robot locations, robot colors, and clear tiles.
    4. For each unpainted goal fact `(painted T C)`:
        a. Parse the target tile `T` and target color `C`.
        b. Check if tile `T` is currently clear in the state (i.e., `(clear T)` is in the state). If not, add 1 to the cost for this tile (representing the cost for the occupying robot to move away).
        c. Initialize the minimum robot cost for painting tile `T` (`min_robot_cost_for_tile`) to infinity.
        d. Get the coordinates `(tr, tc)` of tile `T` using `self.tile_coords`. If coordinates are not found, return infinity.
        e. Get the list of tiles `paintable_from_tiles` from which `T` can be painted using the precomputed `self.paintable_from` dictionary. If this list is empty, return infinity.
        f. For each robot `R` found in the state:
            i. Get R's current location `R_loc` and color `R_color`.
            ii. Calculate the color cost: 1 if `R_color` is not `C`, otherwise 0.
            iii. Get the coordinates `(rr, rc)` of `R_loc` using `self.tile_coords`. If coordinates are not found, skip this robot.
            iv. Calculate the estimated minimum move cost for robot `R` to reach *any* tile in `paintable_from_tiles`. This is the minimum Manhattan distance from `(rr, rc)` to the coordinates of any tile in `paintable_from_tiles`. Initialize `min_move_cost_for_robot` to infinity and update it by iterating through `paintable_from_tiles`.
            v. If a reachable tile in `paintable_from_tiles` was found (`min_move_cost_for_robot` is not infinity):
                Calculate the total estimated cost for robot `R` to be ready to paint `T`: `color_cost + estimated_move_cost`.
                Update `min_robot_cost_for_tile` with the minimum of its current value and the total estimated cost for robot `R`.
        g. If `min_robot_cost_for_tile` is still infinity after checking all robots, the goal tile is unreachable by any robot; return infinity for the total heuristic.
        h. Add the cost for this unpainted tile (`clear_cost + min_robot_cost_for_tile + 1` for the paint action) to the total heuristic value.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.tile_coords = {}
        # directed_relations: {source_tile: {target_tile: direction}}
        # Derived from static facts (direction Y X), represents movement X -> Y.
        self.directed_relations = collections.defaultdict(dict)
        # paintable_from: {target_tile: [source_tile1, source_tile2, ...]}
        # Derived from static facts (direction Y X), represents Y is paintable from X.
        self.paintable_from = collections.defaultdict(list)
        all_tiles = set()

        # 1, 2, 3: Parse static facts to build directed relations and painting adjacency
        # (direction Y X) means Y is direction from X. Robot moves X -> Y. Robot at X paints Y (in direction).
        # E.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1.
        # Robot moves tile_0_1 -> tile_1_1 (move_up).
        # Robot at tile_0_1 paints tile_1_1 (paint_up).
        # This implies the predicates are (direction tile_target tile_source).
        # Movement edge: source_tile -> target_tile
        # Painting adjacency: target_tile is paintable from source_tile.

        for fact_str in task.static:
            parts = parse_fact(fact_str)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                direction, target_tile, source_tile = parts
                all_tiles.add(source_tile)
                all_tiles.add(target_tile)

                # Store directed relation for coordinate inference
                self.directed_relations[source_tile][target_tile] = direction

                # Store painting adjacency: target_tile is paintable from source_tile
                self.paintable_from[target_tile].append(source_tile)

        # 4. Assign coordinates using BFS on the directed graph
        if not all_tiles:
            # Handle case with no tiles (shouldn't happen in valid problems)
            # Heuristic will return inf later if goals involve tiles
            return

        # Pick an arbitrary starting tile for coordinate assignment
        start_tile = next(iter(all_tiles))
        self.tile_coords[start_tile] = (0, 0)
        queue = collections.deque([(start_tile, (0, 0))])
        visited = {start_tile}

        # Map direction strings to coordinate changes (row_change, col_change)
        # (up Y X): Y is up from X. If X is (r, c), Y is (r-1, c). Change is (-1, 0).
        # (down Y X): Y is down from X. If X is (r, c), Y is (r+1, c). Change is (+1, 0).
        # (left Y X): Y is left from X. If X is (r, c), Y is (r, c-1). Change is (0, -1).
        # (right Y X): Y is right from X. If X is (r, c), Y is (r, c+1). Change is (0, +1).
        direction_changes = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
        }

        while queue:
            current_tile, (r, c) = queue.popleft()

            # Explore neighbors reachable from current_tile
            if current_tile in self.directed_relations:
                for neighbor_tile, direction in self.directed_relations[current_tile].items():
                    if neighbor_tile not in visited:
                        visited.add(neighbor_tile)
                        # Determine neighbor coordinates based on direction
                        dr, dc = direction_changes.get(direction, (0, 0)) # Default to (0,0) if direction unknown
                        neighbor_coords = (r + dr, c + dc)

                        self.tile_coords[neighbor_tile] = neighbor_coords
                        queue.append((neighbor_tile, neighbor_coords))

        # 5. Store goal facts
        self.goals = task.goals


    def __call__(self, node):
        """
        Computes the floortile heuristic for the given state.
        """
        state = node.state

        # 1. Identify unpainted goal tiles
        unpainted_goals = {g for g in self.goals if g not in state}

        # If all goals are met, heuristic is 0
        if not unpainted_goals:
            return 0

        # 2. Initialize total heuristic
        total_heuristic = 0

        # 3. Parse current state for dynamic facts
        robot_locations = {}
        robot_colors = {}
        clear_tiles = set()
        # We don't strictly need occupied_tiles if we rely on (clear T) predicate
        # occupied_tiles = set()

        for fact_str in state:
            parts = parse_fact(fact_str)
            if not parts:
                continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'robot-at' and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
                # occupied_tiles.add(tile) # Not needed if checking (clear T)
            elif predicate == 'robot-has' and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == 'clear' and len(parts) == 2:
                tile = parts[1]
                clear_tiles.add(tile)
            # 'painted' facts in state are implicitly handled by checking against unpainted_goals

        # 4. For each unpainted goal fact (painted T C)
        for goal_fact_str in unpainted_goals:
            # Parse (painted T C)
            parts = parse_fact(goal_fact_str)
            if len(parts) != 3 or parts[0] != 'painted':
                 # Should not happen if goals are well-formed
                 continue
            _, target_tile, target_color = parts

            # a. Check if tile T is clear. Add 1 if not.
            # If (clear T) is not in state, assume it needs 1 action (robot moving away) to become clear.
            # This is a simplification; it could be painted with the wrong color (dead end).
            # Assuming solvable problems.
            clear_cost = 1 if f'(clear {target_tile})' not in clear_tiles else 0

            # b. Initialize min robot cost for this tile
            min_robot_cost_for_tile = math.inf

            # Get target tile coordinates
            if target_tile not in self.tile_coords:
                 # This goal tile is not part of the known grid? Unreachable.
                 return math.inf
            target_coords = self.tile_coords[target_tile]

            # Find tiles adjacent to the target tile for painting (tiles from which robot can paint T)
            paintable_from_tiles = self.paintable_from.get(target_tile, [])
            if not paintable_from_tiles:
                 # No tile adjacent for painting? Unreachable.
                 return math.inf

            # c. For each robot R
            if not robot_locations:
                 # No robots available
                 return math.inf # Cannot paint

            for robot, robot_loc in robot_locations.items():
                # Get robot's current color. Assume robot always has a color if robot-has is present.
                # If robot-has is not present for a robot in the state, maybe it has no color?
                # The domain implies robots always have a color. Use .get() with None default.
                robot_color = robot_colors.get(robot)

                # Calculate color cost
                # If robot_color is None (robot-has fact missing), assume it needs 1 action to get *any* color,
                # then maybe another action to get the target color?
                # Let's assume robot-has is always present for robots in the state.
                color_cost = 1 if robot_color != target_color else 0

                # Get robot location coordinates
                if robot_loc not in self.tile_coords:
                     # Robot is at an unknown tile? Unreachable.
                     continue # Skip this robot

                robot_coords = self.tile_coords[robot_loc]

                # Calculate minimum move cost to reach *any* tile in paintable_from_tiles
                min_move_cost_for_robot = math.inf
                for adj_tile in paintable_from_tiles:
                    if adj_tile in self.tile_coords:
                        adj_coords = self.tile_coords[adj_tile]
                        # Manhattan distance from robot_loc to adj_tile
                        move_dist = manhattan_distance(robot_coords, adj_coords)
                        min_move_cost_for_robot = min(min_move_cost_for_robot, move_dist)

                # If this robot cannot reach any paintable_from tile
                if min_move_cost_for_robot == math.inf:
                     continue # Check other robots

                # Calculate total cost for this robot for this tile
                total_robot_cost = color_cost + min_move_cost_for_robot

                # Update min robot cost for the tile
                min_robot_cost_for_tile = min(min_robot_cost_for_tile, total_robot_cost)

            # If no robot can paint this tile
            if min_robot_cost_for_tile == math.inf:
                 return math.inf # Unreachable goal

            # Add cost for this unpainted tile to total heuristic
            # Cost = clear_cost + min_robot_cost_to_be_ready + paint_action_cost
            total_heuristic += clear_cost + min_robot_cost_for_tile + 1 # Add 1 for the paint action

        return total_heuristic
