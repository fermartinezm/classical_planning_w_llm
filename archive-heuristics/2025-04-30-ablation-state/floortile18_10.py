from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def parse_tile_name(tile_name):
    """Parses a tile name like 'tile_r_c' into (row, col) integers."""
    try:
        parts = tile_name.split('_')
        # Expecting format like 'tile_1_1'
        if len(parts) == 3 and parts[0] == 'tile':
             return (int(parts[1]), int(parts[2]))
        else:
             return None # Indicate parsing failure
    except (ValueError, IndexError):
        return None # Indicate parsing failure

def manhattan_distance(tile1_name, tile2_name):
    """Calculates Manhattan distance between two tiles based on their names."""
    coords1 = parse_tile_name(tile1_name)
    coords2 = parse_tile_name(tile2_name)
    if coords1 is None or coords2 is None:
        # Cannot calculate distance if parsing failed for either tile
        return float('inf') # Use infinity to indicate impossibility

    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated cost for each tile that is not yet painted with its goal color. The estimated cost for a single tile is the cost of painting it (1 action) plus the minimum cost for any robot to get into a position to paint it (adjacent tile) with the correct color.

    # Assumptions
    - Tiles are named in the format 'tile_row_col' allowing coordinate extraction.
    - The grid structure implied by 'up', 'down', etc. predicates is consistent with these coordinates.
    - All tiles that need to be painted are initially 'clear'. Tiles painted with the wrong color are not expected in reachable states of solvable problems.
    - All required colors are available.
    - Robot movement cost is estimated by Manhattan distance to an adjacent tile, ignoring potential path blocking by painted tiles.
    - Changing color costs 1 action (assuming the robot already holds a different color).
    - Painting costs 1 action.

    # Heuristic Initialization
    - Extract the goal conditions, specifically which tiles need to be painted with which colors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all goal conditions of the form `(painted ?tile ?color)`. Store these as the target state for each tile.
    2. Initialize the total heuristic cost to 0.
    3. For each tile `t` and its required goal color `c_goal` from the goal conditions:
        a. Check the current state: Is `(painted t c_goal)` true?
        b. If yes, this goal is met for this tile; continue to the next goal tile.
        c. If no (meaning the tile is either `clear` or painted with a different color - assuming solvable problems, it must be `clear`):
            i. This tile needs to be painted. Add 1 to the cost for the paint action itself.
            ii. This tile needs a robot adjacent to it with the correct color. Estimate the minimum cost for *any* robot to achieve this:
                - Find the current position `pos_r` and current color `color_r` for each robot `r`.
                - Calculate the estimated move cost for robot `r` to get adjacent to tile `t`:
                    - Calculate Manhattan distance `d` from `pos_r` to `t`.
                    - If `d == 0` (robot is at the target tile), the robot must move to an adjacent tile first, costing 1 move.
                    - If `d > 0` (robot is not at the target tile), the minimum moves to reach *any* adjacent tile is `d - 1`.
                    - So, `move_cost = 1` if `d == 0` else `d - 1`.
                - Calculate the estimated color change cost for robot `r`: `1` if `color_r` is not `c_goal`, otherwise `0`. This assumes the robot currently holds *some* color.
                - The estimated cost for robot `r` to prepare to paint tile `t` is `move_cost_r + color_cost_r`.
                - Find the minimum of this estimated preparation cost over all robots `r`.
            iii. Add this minimum robot preparation cost to the total heuristic cost.
    4. After summing costs for all unpainted goal tiles, check if the state is a true goal state (`task.goals <= state`). If it is, the heuristic must be 0. Otherwise, return the calculated total cost. This ensures h=0 only at the goal.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions."""
        self.goals = task.goals
        # Extract goal painted states: {tile_name: color_name}
        self.goal_painted_tiles = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                self.goal_painted_tiles[tile] = color
            # Ignore other potential goal types as per domain analysis and examples.

        # Static facts are not explicitly needed for this heuristic's calculation,
        # as grid structure is inferred from tile names and color availability is assumed.
        # self.static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        task = node.task # Access the task from the node to check full goal

        # Check if it's the goal state first for efficiency and correctness (h=0 only at goal)
        if task.goal_reached(state):
             return 0

        # Find current robot positions and colors
        robot_info = {} # {robot_name: {'pos': tile_name, 'color': color_name}}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot, tile = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['pos'] = tile
            elif len(parts) == 3 and parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                if robot not in robot_info:
                    robot_info[robot] = {}
                robot_info[robot]['color'] = color
            # Ignore other fact types

        # If there are goal tiles but no robots in the state, it's unsolvable.
        if self.goal_painted_tiles and not robot_info:
             return float('inf')

        total_cost = 0

        # Iterate through all tiles that need to be painted according to the goal
        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if the tile is already painted correctly in the current state
            # We check for the exact goal fact string presence in the state set
            is_painted_correctly = f"(painted {goal_tile} {goal_color})" in state

            if not is_painted_correctly:
                # This tile needs to be painted. Add cost for the paint action.
                paint_action_cost = 1

                # Find the minimum cost for any robot to get adjacent and have the right color
                min_robot_prep_cost_for_tile = float('inf')

                for robot_name, info in robot_info.items():
                    robot_pos = info.get('pos')
                    robot_color = info.get('color')

                    # A robot must have both a position and a color to be useful for painting
                    if robot_pos is None or robot_color is None:
                         continue # Skip this robot if info is incomplete

                    # Cost to change color if needed
                    color_change_cost = 0
                    if robot_color != goal_color:
                        color_change_cost = 1 # Assumes robot has *some* color to change from

                    # Cost to move adjacent to the tile
                    dist_to_target = manhattan_distance(robot_pos, goal_tile)

                    if dist_to_target == float('inf'):
                         # Tile name parsing failed for robot pos or goal tile
                         # This robot cannot reach this tile.
                         continue # Skip this robot for this tile

                    # Calculate moves needed to reach an adjacent tile
                    if dist_to_target == 0:
                        move_cost = 1 # Must move off the tile to an adjacent one
                    else:
                        move_cost = dist_to_target - 1 # Must move to an adjacent tile

                    # Total preparation cost for this robot to paint this specific tile
                    robot_prep_cost = move_cost + color_change_cost

                    # Update minimum preparation cost across all robots for this tile
                    min_robot_prep_cost_for_tile = min(min_robot_prep_cost_for_tile, robot_prep_cost)

                # If min_robot_prep_cost_for_tile is still inf, it means no robot could be
                # found or considered that can paint this tile. This implies unsolvable.
                if min_robot_prep_cost_for_tile == float('inf'):
                     return float('inf')

                # Add the cost for this tile (paint action + minimum robot preparation cost)
                total_cost += paint_action_cost + min_robot_prep_cost_for_tile

        # The goal state check at the beginning already handles the case where total_cost is 0.
        # If we reach here, total_cost > 0 unless there were no goal_painted_tiles at all,
        # which would mean the initial state is the goal state (handled).
        return total_cost
