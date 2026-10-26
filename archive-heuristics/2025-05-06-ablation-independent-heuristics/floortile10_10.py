# Import necessary classes
from heuristics.heuristic_base import Heuristic
from task import Task # Import Task for type hinting the task parameter

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """Parses a PDDL fact string into a (predicate, args) tuple."""
    # Remove leading/trailing parentheses and split by space
    parts = fact_str.strip().strip('()').split()
    if not parts: # Handle empty string or just ()
        return (None, [])
    predicate = parts[0]
    args = parts[1:]
    return (predicate, args)

# Helper function for Manhattan distance
def manhattan_distance(coord1, coord2):
    """Calculates Manhattan distance between two (row, col) coordinates."""
    r1, c1 = coord1
    r2, c2 = coord2
    return abs(r1 - r2) + abs(c1 - c2)

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the floortile domain.

    Summary:
    Estimates the cost to reach the goal by summing up the minimum estimated
    costs for each unpainted goal tile to be painted. The estimated cost for
    a single unpainted goal tile includes the cost of the paint action itself
    (1), plus the minimum cost for any robot to reach an adjacent tile and
    acquire the necessary color. The cost for a robot to reach an adjacent
    tile is estimated using Manhattan distance, and the color acquisition
    cost is estimated as 1 if the robot doesn't currently hold the required
    color. This heuristic is not admissible but aims to guide a greedy search
    efficiently by considering movement and color requirements.

    Assumptions:
    - The tile names follow the format 'tile_X_Y' where X and Y are integers
      representing row and column.
    - The grid structure defined by 'up', 'down', 'left', 'right' predicates
      corresponds to a grid where Manhattan distance is a reasonable
      approximation of movement cost (ignoring obstacles like painted tiles).
    - All goal tiles are initially 'clear' or correctly 'painted'. If a goal
      tile is incorrectly painted, the problem is likely unsolvable, and the
      heuristic might return infinity or a large value. This heuristic assumes
      solvable instances where unpainted goal tiles are clear.
    - Required colors for goal tiles are available in the domain.

    Heuristic Initialization:
    The constructor pre-processes the static information from the task:
    - Parses all tile names ('tile_X_Y') to create a mapping from tile name
      string to (row, col) integer coordinates.
    - Builds an adjacency map based on the 'up', 'down', 'left', 'right'
      predicates, mapping each tile name to a set of adjacent tile names.
    - Stores the goal facts, specifically mapping each goal tile name to its
      required color.
    - Identifies all robot names.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify the set of goal tiles that are not yet painted with the correct
       color in the current state. Store this as a map from tile name to
       required color.
    2. If this set is empty, the goal is reached, and the heuristic is 0.
    3. For each robot, determine its current position (tile name) and the color
       it currently holds, by inspecting the state facts.
    4. Initialize the total heuristic value `h` to 0.
    5. For each unpainted goal tile `T` needing color `C_T`:
        a. Initialize a minimum cost for this tile `min_cost_for_tile` to infinity.
        b. Get the set of tiles adjacent to `T` using the pre-computed adjacency map.
        c. For each robot `R`:
            i. Get the robot's current position `R_pos` and color `R_color`.
            ii. Initialize a minimum movement distance for this robot to reach *any*
                tile adjacent to `T` (`min_dist_to_adj`) to infinity.
            iii. For each tile `A` adjacent to `T`:
                - Calculate the Manhattan distance between `R_pos` and `A` using
                  their coordinates.
                - Update `min_dist_to_adj` with the minimum distance found so far.
            iv. Calculate the estimated cost for robot `R` to be ready to paint `T`:
                `cost_for_robot = min_dist_to_adj + (1 if R_color != C_T else 0)`.
                (This includes movement and potential color change).
            v. Update `min_cost_for_tile` with the minimum of its current value
               and `cost_for_robot`.
        d. If `min_cost_for_tile` is still infinity (e.g., no robot can reach an
           adjacent tile), the problem might be unsolvable from this state;
           return infinity.
        e. Add the estimated cost for painting tile `T` to the total heuristic:
           `h += min_cost_for_tile + 1` (the +1 is for the paint action itself).
    6. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task # Store task for access to goals, static, etc.

        # --- Heuristic Initialization ---
        self.tile_coords = {}
        self.adj_map = {}
        self.goal_tiles_info = {}
        self.robot_names = set()

        # Collect all facts (initial state + static + goals) to find all objects and relations
        all_relevant_facts = set(task.initial_state) | set(task.static) | set(task.goals)

        # Pass 1: Find all tiles and robots, parse tile coordinates
        for fact_str in all_relevant_facts:
            predicate, args = parse_fact(fact_str)
            for arg in args:
                if arg.startswith('tile_'):
                    if arg not in self.tile_coords:
                        try:
                            # Parse tile_X_Y format
                            parts = arg.split('_')
                            # Check if parts has enough elements and the relevant parts are digits
                            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                                row = int(parts[1])
                                col = int(parts[2])
                                self.tile_coords[arg] = (row, col)
                                # Initialize adjacency set for this tile
                                self.adj_map[arg] = set()
                            # else: log warning about unexpected tile name format?
                        except (ValueError, IndexError):
                             # Should not happen with expected tile_X_Y format
                             pass

                elif arg.startswith('robot'):
                    self.robot_names.add(arg)

        # Pass 2: Build adjacency map from static facts
        for fact_str in task.static:
             predicate, args = parse_fact(fact_str)
             if predicate in ['up', 'down', 'left', 'right'] and len(args) == 2:
                 t1, t2 = args
                 # Ensure both tiles were successfully parsed and added to tile_coords/adj_map
                 if t1 in self.adj_map and t2 in self.adj_map:
                     self.adj_map[t1].add(t2)
                     self.adj_map[t2].add(t1) # Adjacency is symmetric

        # Store goal tiles and their required colors
        for goal_fact_str in task.goals:
            predicate, args = parse_fact(goal_fact_str)
            if predicate == 'painted' and len(args) == 2:
                tile_name, color = args
                # Ensure the goal tile was successfully parsed
                if tile_name in self.tile_coords:
                    self.goal_tiles_info[tile_name] = color
                # else: log warning about goal tile with unparseable name?


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic for the floortile domain.
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---

        # 1. Identify unpainted goal tiles and current robot state
        unpainted_goal_tiles = self.goal_tiles_info.copy()
        # Initialize robot state with default values. Robots might not be in initial state facts
        # if they are introduced later, but in this domain, they are usually in init.
        # Let's initialize based on self.robot_names found during init.
        robot_state = {robot_name: {'pos': None, 'color': None} for robot_name in self.robot_names}


        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'painted' and len(args) == 2:
                tile_name, color = args
                # If a goal tile is painted with the correct color, it's done.
                if tile_name in unpainted_goal_tiles and unpainted_goal_tiles[tile_name] == color:
                    del unpainted_goal_tiles[tile_name]
            elif predicate == 'robot-at' and len(args) == 2:
                robot_name, tile_name = args
                if robot_name in robot_state: # Ensure it's a robot we know about
                    robot_state[robot_name]['pos'] = tile_name
            elif predicate == 'robot-has' and len(args) == 2:
                robot_name, color = args
                if robot_name in robot_state: # Ensure it's a robot we know about
                    robot_state[robot_name]['color'] = color

        # 2. Check if goal is reached
        if not unpainted_goal_tiles:
            return 0

        # 3. Calculate heuristic sum
        h = 0
        for tile_name, required_color in unpainted_goal_tiles.items():
            min_cost_for_tile = float('inf')

            # We already ensured tile_name is in self.tile_coords during init
            # if it came from a goal fact with a parseable name.

            adjacent_tile_names = self.adj_map.get(tile_name, set())

            # If a goal tile has no adjacent tiles defined in static facts, it's impossible to paint.
            if not adjacent_tile_names:
                 # This should ideally not happen in a well-formed solvable problem
                 return float('inf') # Unsolvable

            for robot_name, r_state in robot_state.items():
                robot_pos_name = r_state['pos']
                robot_color = r_state['color']

                # Robot must have a position and color in the current state to be considered
                if robot_pos_name is None or robot_color is None:
                    # This robot is not currently active or fully defined in the state.
                    # It cannot contribute to painting this tile right now.
                    continue

                # Ensure robot position tile exists in our coordinate map (should be true if pos is not None)
                if robot_pos_name not in self.tile_coords:
                     # This indicates an inconsistency in the state or init parsing
                     continue # Skip this robot

                min_dist_to_adj = float('inf')

                for adj_tile_name in adjacent_tile_names:
                    # Ensure adjacent tile exists in our coordinate map (should be true if adj_map is built correctly)
                    if adj_tile_name in self.tile_coords:
                        dist = manhattan_distance(self.tile_coords[robot_pos_name], self.tile_coords[adj_tile_name])
                        min_dist_to_adj = min(min_dist_to_adj, dist)

                # If min_dist_to_adj is still inf, this robot cannot reach any adjacent tile
                # (e.g., disconnected graph, or adj_map inconsistency).
                if min_dist_to_adj == float('inf'):
                    cost_for_robot = float('inf')
                else:
                    # Cost includes movement and potential color change
                    cost_for_robot = min_dist_to_adj + (1 if robot_color != required_color else 0)

                min_cost_for_tile = min(min_cost_for_tile, cost_for_robot)

            # If after checking all robots, no robot can reach an adjacent tile
            # and get the right color, the tile cannot be painted.
            if min_cost_for_tile == float('inf'):
                # One required goal tile is unreachable/unpaintable by any robot.
                return float('inf') # Unsolvable from this state

            # Add cost for this tile: min cost to get robot ready + paint action
            h += min_cost_for_tile + 1

        return h
