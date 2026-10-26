from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def parse_tile_name(tile_str):
    """Parses 'tile_r_c' string into (row, col) integers."""
    parts = tile_str.split('_')
    if len(parts) == 3 and parts[0] == 'tile':
        try:
            row = int(parts[1])
            col = int(parts[2])
            return row, col
        except ValueError:
            pass # Not a valid tile_r_c format
    return None, None # Indicate parsing failure

def manhattan_distance(tile1_str, tile2_str):
    """Calculates Manhattan distance between two tiles based on their names 'tile_r_c'."""
    r1, c1 = parse_tile_name(tile1_str)
    r2, c2 = parse_tile_name(tile2_str)
    if r1 is not None and r2 is not None:
        return abs(r1 - r2) + abs(c1 - c2)
    return float('inf') # Cannot calculate distance if names are not in expected format


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the total number of actions required to paint all
    goal tiles with their target colors. It sums the estimated cost for each
    unsatisfied goal tile, considering the paint action, the movement cost
    for the closest robot to reach an adjacent tile, and a global cost for
    acquiring necessary colors not currently held by any robot.

    # Assumptions
    - The tiles form a grid structure where movement between adjacent tiles
      (up, down, left, right) costs 1 action. Manhattan distance is used
      to estimate movement cost between any two tiles.
    - A tile named 'tile_r_c' corresponds to grid coordinates (r, c).
    - A tile must be clear before it can be painted.
    - If a goal tile is already painted with a color *different* from the
      goal color, the problem is considered unsolvable in this domain
      (as there is no unpaint action). The heuristic returns infinity in this case.
    - Robots can change color (cost 1) if the desired color is available.
    - The heuristic assumes each unsatisfied goal tile needs one paint action
      and associated movement. Color changes are counted globally for colors
      needed across all unsatisfied goals but not currently held.

    # Heuristic Initialization
    - The heuristic stores the goal conditions.
    - It pre-processes the static facts to build an adjacency map of the tile grid,
      mapping each tile to the set of tiles adjacent to it based on the
      `up`, `down`, `left`, and `right` predicates. This map is used to find
      tiles adjacent to a target goal tile.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Check for Unsolvable States:** Iterate through all goal conditions. For each goal `(painted target_tile target_color)`, check if `target_tile` is currently painted with *any* color `c` where `c != target_color`. If such a state fact exists, the problem is unsolvable, and the heuristic returns `float('inf')`.

    2.  **Identify Unsatisfied Goals:** Determine the set of goal conditions that are not currently true in the state. If this set is empty, the goal is reached, and the heuristic returns 0.

    3.  **Extract State Information:** Identify the current location of each robot (`robot-at`) and the color each robot currently holds (`robot-has`). Store these in dictionaries and a set of held colors.

    4.  **Initialize Cost and Needed Colors:** Initialize the total heuristic cost to 0. Create an empty set `needed_colors` to track all colors required by the unsatisfied goal tiles.

    5.  **Estimate Cost Per Unsatisfied Goal Tile:** Iterate through each unsatisfied goal fact `(painted target_tile target_color)`:
        a.  Add 1 to the total cost. This accounts for the `paint` action required for this tile.
        b.  Add `target_color` to the `needed_colors` set.
        c.  Find the set of tiles adjacent to `target_tile` using the pre-computed adjacency map.
        d.  Calculate the minimum Manhattan distance from *any* robot's current location to *any* tile in the set of adjacent tiles. This estimates the minimum movement cost for a robot to get into a position to paint the tile.
        e.  If no robots exist or no adjacent tiles are found (which would imply an unreachable goal tile in a solvable problem), return `float('inf')`.
        f.  Add this minimum movement distance to the total cost.

    6.  **Estimate Color Acquisition Cost:** After processing all unsatisfied goal tiles, iterate through the `needed_colors` set. For each color in `needed_colors` that is *not* present in the set of colors currently held by robots (`held_colors`), add 1 to the total cost. This estimates the cost of `change_color` actions needed to acquire the required colors.

    7.  **Return Total Cost:** The final accumulated `total_cost` is the heuristic value for the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the tile adjacency map from static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Build adjacency map from static facts (up, down, left, right)
        self.adj_map = {}
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                self.adj_map.setdefault(t1, set()).add(t2)
                self.adj_map.setdefault(t2, set()).add(t1) # Adjacency is symmetric

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state.
        """
        state = node.state

        # 1. Check for Unsolvable States (tile painted with wrong color)
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if len(parts) == 3 and parts[0] == 'painted':
                 target_tile, target_color = parts[1], parts[2]
                 # If the goal is already satisfied, skip this check for this tile
                 if goal_fact in state:
                     continue

                 # Check if the tile is painted with a *different* color
                 for state_fact in state:
                     state_parts = get_parts(state_fact)
                     if len(state_parts) == 3 and state_parts[0] == 'painted':
                         painted_tile, painted_color = state_parts[1], state_parts[2]
                         if painted_tile == target_tile and painted_color != target_color:
                             # Tile is painted with the wrong color - unsolvable
                             return float('inf')

        # 2. Identify Unsatisfied Goals
        unsat_goals = self.goals - state
        if not unsat_goals:
            return 0

        # 3. Extract State Information
        robot_locs = {}
        robot_colors = {}
        held_colors = set()
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3:
                if parts[0] == 'robot-at':
                    robot, tile = parts[1], parts[2]
                    robot_locs[robot] = tile
                elif parts[0] == 'robot-has':
                    robot, color = parts[1], parts[2]
                    robot_colors[robot] = color
                    held_colors.add(color)

        # If no robots exist but there are unsatisfied goals, it's unsolvable
        if not robot_locs and unsat_goals:
             return float('inf')

        # 4. Initialize Cost and Needed Colors
        total_cost = 0
        needed_colors = set()

        # 5. Estimate Cost Per Unsatisfied Goal Tile
        for goal_fact in unsat_goals:
            parts = get_parts(goal_fact)
            if len(parts) == 3 and parts[0] == 'painted':
                target_tile, target_color = parts[1], parts[2]

                # a. Add cost for paint action
                total_cost += 1

                # b. Track needed color
                needed_colors.add(target_color)

                # c. Find adjacent tiles
                adjacent_tiles = self.adj_map.get(target_tile, set())

                # d. Calculate minimum movement cost to an adjacent tile
                min_dist_to_adj = float('inf')

                # e. If no adjacent tiles, goal is unreachable
                if not adjacent_tiles:
                     return float('inf')

                for robot, robot_loc in robot_locs.items():
                    for adj_tile in adjacent_tiles:
                        dist = manhattan_distance(robot_loc, adj_tile)
                        min_dist_to_adj = min(min_dist_to_adj, dist)

                # f. Add movement cost
                if min_dist_to_adj == float('inf'):
                    # No robot can reach any adjacent tile - unsolvable
                    return float('inf')
                total_cost += min_dist_to_adj


        # 6. Estimate Color Acquisition Cost
        color_acquisition_cost = sum(1 for color in needed_colors if color not in held_colors)
        total_cost += color_acquisition_cost

        # 7. Return Total Cost
        return total_cost
