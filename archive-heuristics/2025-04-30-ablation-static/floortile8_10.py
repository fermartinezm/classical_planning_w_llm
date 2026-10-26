from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace after removing parentheses
    return fact[1:-1].strip().split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    that are currently unpainted or painted incorrectly. It sums the estimated cost
    for each unsatisfied goal tile independently. The cost for a single goal tile
    is estimated as 1 (for the paint action) plus the minimum cost to get any robot
    to an adjacent tile with the correct color. The robot approach cost is estimated
    as the grid distance from the robot's current location to the adjacent tile,
    plus 1 if the robot needs to change color.

    # Assumptions
    - The grid structure defined by up/down/left/right predicates is static.
    - Goal tiles are initially either clear or painted with the correct color.
      If a goal tile is found to be painted with the wrong color in a state,
      the heuristic returns infinity, assuming such states are unsolvable
      or highly undesirable.
    - The cost of moving a robot between adjacent tiles is 1, regardless of
      whether the destination tile is clear (this is a relaxation for heuristic
      calculation speed; the actual action requires the destination to be clear).
    - The cost of changing a robot's color is 1.
    - All robots and available colors are present from the initial state.

    # Heuristic Initialization
    The heuristic precomputes the following information from the task definition:
    - The adjacency map of the grid based on up/down/left/right predicates.
    - All-pairs shortest path distances between tiles on the grid using BFS.
    - The target color for each goal tile.
    - The set of all robots and available colors.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Parse the current state to find:
        - The current location of each robot.
        - The current color held by each robot.
        - The current painted status (tile and color) of all tiles.
        - The set of clear tiles.
    2.  Identify the set of unsatisfied goal tiles. A goal tile `tile_Y` needing
        `color_C` is unsatisfied if it is not currently painted with `color_C`.
        If an unsatisfied goal tile is not currently clear, it is assumed to be
        painted with the wrong color, and the heuristic returns infinity.
    3.  If there are no unsatisfied goal tiles, the state is a goal state, and the
        heuristic returns 0.
    4.  Initialize the total heuristic cost to 0.
    5.  For each unsatisfied goal tile `tile_Y` that needs to be painted with
        `color_C`:
        a.  Add 1 to the cost for the paint action itself.
        b.  Identify the set of tiles `tile_X` such that `tile_Y` is adjacent
            to `tile_X` (i.e., a robot at `tile_X` can paint `tile_Y`). This set
            is found using the precomputed reverse adjacency map.
        c.  Initialize the minimum robot approach cost for this specific goal
            tile to infinity.
        d.  Iterate through all robots `R`:
            i.  Get the robot's current location `Loc_R` and color `Color_R`.
            ii. Calculate the cost to change color: 1 if `Color_R` is not the
                target color `color_C`, otherwise 0.
            iii. Find the minimum grid distance from `Loc_R` to any tile `tile_X`
                 in the set of adjacent tiles identified in step 5b. This uses
                 the precomputed all-pairs shortest paths, ignoring the dynamic
                 `clear` status of `tile_X`.
            iv. If a path exists (minimum distance is finite), calculate the
                robot's approach cost for this goal tile as the minimum grid
                distance plus the color change cost.
            v.  Update the minimum robot approach cost for this goal tile with
                the minimum found across all robots so far.
        e.  If, after checking all robots, the minimum robot approach cost for
            this goal tile is still infinity (meaning no robot can reach any
            adjacent tile), the heuristic returns infinity for the total cost,
            as this goal seems unreachable.
        f.  Add the minimum robot approach cost found in step 5d to the total
            heuristic cost.
    6.  Return the total calculated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing grid information and goal details.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Build adjacency map and reverse adjacency map from static facts
        self.adj_map = {}  # tile -> set of neighbors (e.g., tile_0_1 -> {tile_1_1})
        self.reverse_adj_map = {} # neighbor -> set of tiles that have it as neighbor (e.g., tile_1_1 -> {tile_0_1})
        self.all_tiles = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                t1, t2 = parts[1], parts[2]
                self.adj_map.setdefault(t1, set()).add(t2)
                self.reverse_adj_map.setdefault(t2, set()).add(t1)
                self.all_tiles.add(t1)
                self.all_tiles.add(t2)

        # Compute all-pairs shortest paths (grid distances) using BFS
        self.grid_distances = {}
        for start_tile in self.all_tiles:
            self.grid_distances[start_tile] = {}
            queue = deque([(start_tile, 0)])
            visited = {start_tile}
            self.grid_distances[start_tile][start_tile] = 0 # Distance to self is 0

            while queue:
                current_tile, dist = queue.popleft()

                # Explore neighbors using the forward map
                for neighbor in self.adj_map.get(current_tile, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.grid_distances[start_tile][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

        # Store goal conditions (painted tiles)
        self.goal_painted = {}  # Map tile -> target_color
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                self.goal_painted[tile] = color

        # Identify all robots and available colors
        self.all_robots = set()
        self.all_colors = set()

        # Robots are typically defined by their initial location or color
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'robot-at':
                  self.all_robots.add(parts[1])
             elif parts[0] == 'robot-has':
                  self.all_robots.add(parts[1]) # Add robot even if location isn't specified initially (unlikely)

        # Available colors are typically static
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'available-color':
                  self.all_colors.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Extract current state information
        robot_locations = {}
        robot_colors = {}
        current_painted = {}  # Map tile -> color
        current_clear = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'robot-at':
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif parts[0] == 'robot-has':
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif parts[0] == 'painted':
                tile, color = parts[1], parts[2]
                current_painted[tile] = color
            elif parts[0] == 'clear':
                tile = parts[1]
                current_clear.add(tile)

        # Identify unsatisfied goals
        unsatisfied_goals = {}  # Map tile -> target_color
        for goal_tile, target_color in self.goal_painted.items():
            # Goal is unsatisfied if the tile is not painted with the target color
            is_painted_correctly = (goal_tile in current_painted and current_painted[goal_tile] == target_color)

            if not is_painted_correctly:
                # If the goal tile is not painted correctly, it must be clear for a solvable state
                # (assuming no unpaint action). If it's not clear, it's painted wrong.
                if goal_tile not in current_clear:
                    # Goal tile is painted with the wrong color or otherwise blocked.
                    # This state is likely a dead end for solvable problems.
                    # Return a large value to prune this branch.
                    return float('inf') # Use infinity as a large value

                # If it's not painted correctly and is clear, it's an unsatisfied goal
                unsatisfied_goals[goal_tile] = target_color

        # If there are no unsatisfied goals, we are in the goal state.
        if not unsatisfied_goals:
            return 0

        total_cost = 0

        # Sum costs for each unsatisfied goal tile
        for goal_tile, target_color in unsatisfied_goals.items():
            # Cost for this goal tile: 1 (paint) + min_cost_to_get_robot_adjacent_with_color

            min_robot_approach_cost = float('inf')

            # Find tiles tile_X such that goal_tile is adjacent to tile_X.
            # These are the tiles where a robot can be *at* to paint goal_tile.
            # These are the tiles in self.reverse_adj_map[goal_tile].
            adj_tiles_to_be_at = self.reverse_adj_map.get(goal_tile, set())

            if not adj_tiles_to_be_at:
                # This goal tile has no adjacent tiles where a robot can stand to paint it.
                # This indicates a problem with the instance definition or unsolvability.
                return float('inf') # Use infinity as a large value

            # Find the minimum cost for any robot to reach any adjacent tile with the right color
            for robot in self.all_robots:
                loc_R = robot_locations.get(robot)
                # If robot location is not in state, something is wrong, or robot is not active.
                # Assuming valid states have all robots placed.
                if loc_R is None:
                    continue

                color_R = robot_colors.get(robot)
                 # If robot color is not in state, something is wrong.
                 # Assuming valid states have all robots holding a color.
                if color_R is None:
                    continue

                # Cost to change color if needed
                color_change_cost = 1 if color_R != target_color else 0

                min_move_cost_to_adj = float('inf')

                # Find the closest tile in adj_tiles_to_be_at from loc_R using precomputed grid distances
                # We ignore the 'clear' constraint for the destination tile in the heuristic move cost.
                if loc_R in self.grid_distances: # Ensure robot's location is a known tile
                    for tile_X in adj_tiles_to_be_at:
                         if tile_X in self.grid_distances[loc_R]: # Ensure adjacent tile is a known tile
                              min_move_cost_to_adj = min(min_move_cost_to_adj, self.grid_distances[loc_R][tile_X])

                # If it's impossible for this robot to reach any adjacent tile (e.g., disconnected graph)
                if min_move_cost_to_adj == float('inf'):
                     # This robot cannot reach any painting position for this goal tile.
                     pass # Try next robot

                else:
                     # This robot can reach a painting position.
                     robot_approach_cost = min_move_cost_to_adj + color_change_cost
                     min_robot_approach_cost = min(min_robot_approach_cost, robot_approach_cost)


            # If no robot can reach any painting position for this goal tile (even in relaxed sense)
            if min_robot_approach_cost == float('inf'):
                 # This goal is currently unreachable by any robot.
                 # This state is likely very bad or unsolvable.
                 return float('inf') # Use infinity as a large value

            # Add the cost for this goal tile: 1 (paint) + minimum approach cost
            total_cost += 1 + min_robot_approach_cost

        return total_cost

