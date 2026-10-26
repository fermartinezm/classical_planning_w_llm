import collections
from fnmatch import fnmatch
# The base class Heuristic is assumed to be available in the execution environment.
# If it's in a specific module, uncomment the following line:
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not available (e.g., for standalone testing),
# you can use this dummy definition:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError


# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string.
    Example: "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
    Removes parentheses and splits by space.
    """
    # Handles potential extra spaces, removes parentheses
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Checks if a PDDL fact string matches a given pattern.
    Uses fnmatch for wildcard matching ('*').
    Example: match("(at robot1 locA)", "at", "robot*", "*") -> True
    """
    parts = get_parts(fact)
    # The number of parts in the fact must match the number of pattern elements
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'floortile'.

    # Summary
    This heuristic estimates the remaining cost (number of actions) to reach a
    goal state from the current state in the floortile domain. It focuses on
    achieving the `(painted tile color)` goal predicates. The heuristic value
    is the sum of estimated costs for each unsatisfied painting goal. The cost
    for a single goal is the minimum estimated cost for any robot to achieve it,
    considering movement, potential color change, and the painting action itself.
    It is designed for Greedy Best-First Search and does not need to be admissible.

    # Assumptions
    - The tile grid defined by `up/down/left/right` predicates forms a connected
      graph for all relevant tiles, meaning robots can eventually reach any target tile.
    - The cost of moving between tiles is estimated by the shortest path distance
      in the tile graph, ignoring dynamic obstacles like other robots or
      already painted tiles (this is a standard relaxation).
    - Each unsatisfied `(painted tile color)` goal requires at least three components
      of cost: moving the robot to the target tile's vicinity (>=0 actions),
      potentially changing the robot's color (0 or 1 action), and performing
      the paint action (1 action).
    - The heuristic sums the minimum estimated costs for each goal independently.
      This might overestimate the total cost if one robot can efficiently paint
      multiple tiles sequentially, but serves as an informative estimate for guiding
      the search. It avoids solving the complex optimal assignment problem (which
      robot paints which tile).

    # Heuristic Initialization (`__init__`)
    - Stores the goal conditions from the task, specifically focusing on
      `(painted ?t ?c)` predicates.
    - Parses the static facts (`up`, `down`, `left`, `right`) to:
        a. Identify all unique tile objects.
        b. Build an adjacency list representation of the tile grid graph.
    - Precomputes all-pairs shortest path distances between all identified tiles
      using Breadth-First Search (BFS) on the tile graph. This ignores dynamic
      predicates like `clear` or `robot-at`. The distances are stored in
      `self.distances[tile1][tile2]`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Identify Unsatisfied Goals**: Determine the set of `(painted target_tile target_color)`
        predicates specified in the task's goal that are *not* present in the
        current state `S`. If this set is empty, the state `S` is a goal state,
        and the heuristic value is 0.
    2.  **Initialize Total Cost**: Set the heuristic estimate `h = 0`.
    3.  **Iterate Through Unsatisfied Goals**: For each unsatisfied goal predicate
        `G = (painted target_tile target_color)`:
        a.  Initialize Minimum Cost for this Goal: `min_goal_cost = float('inf')`.
        b.  **Iterate Through Robots**: For each robot `R` identified in the current state `S`:
            i.  Get Robot's Current State: Find `R`'s current location `robot_tile`
                (from `robot-at` fact) and the color it currently holds `robot_color`
                (from `robot-has` fact).
            ii. Calculate Color Change Cost (`cost_color`): If `robot_color` is
                different from `target_color`, `cost_color = 1` (representing one
                `change_color` action). Otherwise, `cost_color = 0`.
            iii.Calculate Movement Cost (`cost_move`): Retrieve the precomputed
                shortest path distance from `robot_tile` to `target_tile` using
                `self.distances[robot_tile].get(target_tile, float('inf'))`. This estimates
                the minimum number of `move_*` actions required. If the distance
                is infinity (no path found), this robot cannot reach the target.
            iv. Calculate Painting Cost (`cost_paint`): This is always 1, representing
                the single `paint_up` or `paint_down` action needed once the robot
                is adjacent and has the right color.
            v.  Calculate Total Cost for this Robot (`cost_robot`): Sum the components:
                `cost_robot = cost_color + cost_move + cost_paint`. If `cost_move`
                was infinity, `cost_robot` becomes infinity.
            vi. Update Minimum Goal Cost: Keep track of the minimum cost found so far
                among all robots for achieving goal `G`:
                `min_goal_cost = min(min_goal_cost, cost_robot)`.
        c.  **Add Minimum Cost to Total**: Add the calculated `min_goal_cost` for the
            current goal `G` to the total heuristic estimate `h`. If `min_goal_cost`
            remained infinity after checking all robots (meaning no robot can
            achieve this goal), then the current state `S` is considered a dead end,
            and the function returns `float('inf')`.
    4.  **Return Total Cost**: After iterating through all unsatisfied goals, return
        the final computed heuristic value `h`. This value represents the summed
        minimum estimated costs for achieving all remaining painting goals.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task:
        - Extracts all unique tile names.
        - Builds the adjacency graph representing tile connectivity.
        - Computes and stores all-pairs shortest path distances between tiles.
        - Stores the set of painting goal predicates.
        """
        # Assuming Heuristic base class is properly handled by the environment
        # super().__init__(task) # Call base class init if necessary
        self.task = task # Store task for reference if needed
        self.goals = task.goals
        static_facts = task.static

        # --- Extract tiles and build adjacency map ---
        self.tiles = set()
        self.adj = collections.defaultdict(list)

        # Parse connectivity predicates (up, down, left, right) from static facts
        for fact in static_facts:
             parts = get_parts(fact)
             # Ensure fact is correctly formatted
             if not parts: continue
             pred = parts[0]
             # Check for connectivity predicates with exactly 3 parts (pred, tile1, tile2)
             if pred in ["up", "down", "left", "right"] and len(parts) == 3:
                 t1, t2 = parts[1], parts[2]
                 self.tiles.add(t1)
                 self.tiles.add(t2)
                 # Add bidirectional edge for adjacency graph used in BFS distance calculation
                 # Assumes if t1 is 'up' from t2, they are adjacent for movement/painting.
                 self.adj[t1].append(t2)
                 self.adj[t2].append(t1)

        # Ensure all tiles mentioned in initial state or goals are included in the set of tiles
        # This handles cases where a tile might be a goal but not connected initially,
        # or mentioned in predicates like 'clear' or 'painted'.
        for fact_set in [task.initial_state, task.goals]:
            for fact in fact_set:
                parts = get_parts(fact)
                if not parts: continue
                pred = parts[0]
                # Add tiles from relevant predicates (robot-at, clear, painted)
                if pred in ["robot-at", "clear", "painted"] and len(parts) >= 2:
                    self.tiles.add(parts[1])
                # Also consider tiles from connectivity predicates if they appear in init/goal
                elif pred in ["up", "down", "left", "right"] and len(parts) >= 3:
                     self.tiles.add(parts[1])
                     self.tiles.add(parts[2])

        # Remove potential duplicate entries in adjacency lists
        for tile in list(self.adj.keys()): # Iterate over a copy of keys
            self.adj[tile] = list(set(self.adj[tile]))

        # --- Precompute all-pairs shortest paths using BFS ---
        # self.distances[t1][t2] will store the shortest distance from t1 to t2
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))

        for start_node in self.tiles:
            # Ensure start_node is a valid tile name
            if not start_node: continue

            # Distance from a node to itself is 0
            self.distances[start_node][start_node] = 0
            # Initialize queue for BFS starting from start_node
            queue = collections.deque([(start_node, 0)])
            # Keep track of visited nodes for this specific BFS run
            visited_bfs = {start_node}

            while queue:
                current_tile, dist = queue.popleft()

                # Explore neighbors of the current tile
                for neighbor in self.adj.get(current_tile, []):
                    # If neighbor hasn't been visited in this BFS run
                    if neighbor not in visited_bfs:
                        visited_bfs.add(neighbor)
                        # Record the distance (dist + 1)
                        self.distances[start_node][neighbor] = dist + 1
                        # Add neighbor to the queue for further exploration
                        queue.append((neighbor, dist + 1))

        # Store only the painting goal predicates for efficient lookup during heuristic calculation
        self.goal_predicates = {goal for goal in self.goals if match(goal, "painted", "*", "*")}


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the minimum number of actions required to reach the goal.
        Returns float('inf') if the goal seems unreachable from the current state.
        """
        state = node.state # The current state is a frozenset of facts

        # --- Extract dynamic information from the current state ---
        robot_locs = {} # Maps robot name -> current tile location
        robot_cols = {} # Maps robot name -> current color held
        current_painted = {} # Maps tile name -> color it is currently painted

        for fact in state:
            parts = get_parts(fact)
            # Skip empty or malformed facts
            if not parts: continue
            pred = parts[0]

            # Parse robot location facts
            if pred == "robot-at" and len(parts) == 3:
                robot_locs[parts[1]] = parts[2] # parts[1]=robot, parts[2]=tile
            # Parse robot color facts
            elif pred == "robot-has" and len(parts) == 3:
                robot_cols[parts[1]] = parts[2] # parts[1]=robot, parts[2]=color
            # Parse currently painted tile facts
            elif pred == "painted" and len(parts) == 3:
                current_painted[parts[1]] = parts[2] # parts[1]=tile, parts[2]=color

        # --- Calculate heuristic value based on unsatisfied goals ---
        total_heuristic_cost = 0
        unsatisfied_goal_found = False # Flag to check if any goal needs work

        # Iterate through all painting goals defined in the task
        for goal_fact in self.goal_predicates:
             # Extract target tile and color from the goal fact
             _, target_tile, target_color = get_parts(goal_fact)

             # Check if this specific goal is already satisfied in the current state
             if current_painted.get(target_tile) == target_color:
                 continue # This goal is met, move to the next one

             # If we reach here, the goal is not satisfied
             unsatisfied_goal_found = True
             min_cost_for_this_goal = float('inf')

             # If there are no robots in the current state, goals cannot be achieved
             if not robot_locs:
                 return float('inf') # Return infinity indicating unreachability

             # Calculate the minimum cost for any available robot to achieve this goal
             for r, robot_tile in robot_locs.items():
                 robot_color = robot_cols.get(r)
                 # A robot must have a color to paint (based on domain actions)
                 if robot_color is None:
                     # This indicates an unexpected state, skip this robot
                     continue

                 # Estimate Cost Component 1: Color Change
                 # Cost is 1 if the robot needs to change color, 0 otherwise.
                 cost_color = 0 if robot_color == target_color else 1

                 # Estimate Cost Component 2: Movement
                 # Retrieve precomputed shortest path distance. Defaults to infinity if no path exists.
                 cost_move = self.distances[robot_tile].get(target_tile, float('inf'))

                 # Estimate Cost Component 3: Painting Action
                 # Cost is always 1 for the final paint action.
                 cost_paint = 1

                 # Calculate the total estimated cost for this robot to achieve this goal
                 # If movement cost is infinity, the total cost is infinity for this robot.
                 if cost_move == float('inf'):
                     current_robot_cost = float('inf')
                 else:
                     current_robot_cost = cost_color + cost_move + cost_paint

                 # Update the minimum cost found so far for achieving this goal
                 min_cost_for_this_goal = min(min_cost_for_this_goal, current_robot_cost)

             # After checking all robots:
             # If min_cost_for_this_goal is still infinity, it means no robot can reach
             # or perform the action for this goal from the current state.
             if min_cost_for_this_goal == float('inf'):
                 # This state is considered a dead end towards achieving this goal.
                 return float('inf')

             # Add the minimum cost found for satisfying this goal to the total heuristic cost
             total_heuristic_cost += min_cost_for_this_goal

        # After checking all goals:
        # If no unsatisfied goals were found, the state is a goal state.
        if not unsatisfied_goal_found:
            return 0
        else:
            # Return the sum of minimum costs for all unsatisfied goals.
            # Ensure the heuristic value is non-negative.
            return max(0, total_heuristic_cost)

