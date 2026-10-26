import math
from collections import deque
# The problem description implies the existence of a base class Heuristic.
# We assume it's available in the specified path.
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL fact strings like '(predicate obj1 obj2)'
def get_parts(fact_str):
    """Removes parentheses and splits the fact string into predicate and arguments."""
    return fact_str[1:-1].split()

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Floortile domain using a greedy assignment approach.

    # Summary
    This heuristic estimates the remaining cost to reach the goal state by summing
    the estimated costs for painting each required tile that is not yet painted
    with the correct color. It employs a greedy strategy: in each step of the
    heuristic calculation, it identifies the minimum cost action sequence
    (considering movement, potential color change, and painting) for any available
    robot to paint any single remaining goal tile. This minimum cost is added to
    the total heuristic value. To account for sequential dependencies partially,
    the state of the chosen robot (its position and held color) is hypothetically
    updated for subsequent cost estimations within the same heuristic evaluation.

    # Assumptions
    - All actions defined in the domain (move_*, change_color, paint_*) have a uniform cost of 1.
    - The heuristic simplifies complex interactions:
        - It ignores potential dead ends that might arise from robot interference or
          unresolvable blocking situations.
        - It adds a small fixed penalty (+1) if a target tile for painting is not
          'clear' in the input state, approximating the cost to potentially clear it.
    - Movement cost is estimated using Breadth-First Search (BFS) on the tile grid,
      assuming robots follow optimal paths.
    - The BFS pathfinding considers only tiles marked as 'clear' in the input state
      as traversable. It does not simulate changes to the 'clear' status during
      pathfinding or between greedy assignment steps within one heuristic call.
    - Robot interference during movement (beyond occupying tiles, which makes them
      not 'clear') is ignored.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task's static facts (`up`, `down`,
      `left`, `right`, `available-color`) to build an adjacency map of the tile
      grid and identify all available paint colors.
    - It stores the set of goal conditions, specifically the `(painted tile color)` facts.
    - It identifies all robot and tile objects defined in the task by examining
      static facts, the initial state, and goal conditions.
    - Internal data structures (like the adjacency map) are initialized for
      efficient lookups during heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse Current State: Extract the current positions of all robots (`robot-at`),
        the color each robot holds (`robot-has`), the set of currently painted tiles
        and their colors (`painted`), and the set of clear tiles (`clear`) from the
        input state (`node.state`).
    2.  Identify Unsatisfied Goals: Determine the set of `(painted tile color)` goals
        (defined in `task.goals`) that are not currently satisfied in the state.
        If this set is empty, the state is a goal state, and the heuristic value is 0.
    3.  Initialize Simulation State: Create mutable copies of the current robot
        positions and held colors. These copies will be updated during the greedy
        assignment process within this heuristic calculation. Also, create a list
        of the unsatisfied goals identified in step 2.
    4.  Greedy Assignment Loop: Repeat the following steps until all unsatisfied
        goals have been assigned an estimated cost:
        a.  Find Best Robot-Goal Pair: Iterate through every available robot (`r`)
            and every remaining unsatisfied goal (`g = (goal_tile, goal_color)`).
            For each pair, calculate the estimated cost for that robot to achieve
            that goal:
            i.   Precondition Cost: Check if `goal_tile` was present in the set of
                 `clear` tiles derived from the original input state. If not, add a
                 penalty cost of 1.
            ii.  Movement Cost: Calculate the shortest path distance for robot `r`
                 from its current *simulated* position to any tile that is adjacent
                 to `goal_tile`. This is done using BFS. The BFS explores paths only
                 through tiles that were `clear` in the *original* input state. If
                 no path exists to any adjacent tile, the movement cost is infinity.
            iii. Color Cost: If robot `r`'s current *simulated* color is different
                 from the required `goal_color`, add a cost of 1 (for the `change_color`
                 action). Otherwise, the cost is 0.
            iv.  Paint Cost: Add a cost of 1 for the `paint_up` or `paint_down` action.
            v.   Total Cost: Sum the costs from steps i, ii, iii, and iv.
        b.  Select Minimum Cost Action: Identify the (robot, goal) pair that resulted
            in the minimum total cost calculated in step 4a across all pairs.
        c.  Handle Impossibility: If no robot-goal pair has a finite cost (i.e.,
            all calculated minimum costs are infinity, perhaps due to blocked paths),
            assign a large penalty based on the number of remaining unassigned goals
            and terminate the loop.
        d.  Update Heuristic Value: Add the minimum total cost found in step 4b to
            the overall heuristic value being computed.
        e.  Update Simulated State: Remove the selected goal from the list of
            remaining goals. Update the chosen robot's simulated position to be the
            adjacent tile that was determined as the endpoint for the BFS path
            calculation in step 4a.ii. Update the robot's simulated color to be the
            `goal_color` (reflecting the state after the potential color change).
    5.  Return Heuristic Value: Once the loop terminates (either all goals assigned
        or impossibility detected), return the total accumulated heuristic value,
        rounded to the nearest integer.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # --- Internal Data Structures ---
        self.adj = {} # Adjacency map: tile -> {neighbor1, neighbor2, ...}
        self.tiles = set()
        self.robots = set()
        self.colors = set()
        self.goal_conditions = set() # Stores required (tile, color) tuples

        # --- Parse Static Facts (Grid structure, available colors) ---
        for fact_str in self.static_facts:
            parts = get_parts(fact_str)
            predicate = parts[0]
            args = parts[1:]

            if predicate in ["up", "down", "left", "right"]:
                # Adjacency is defined relative to the second argument, e.g.,
                # (up target source) means target is up from source.
                t_target, t_source = args[0], args[1]
                self.tiles.add(t_source)
                self.tiles.add(t_target)
                # Store symmetric adjacency since domain typically includes inverse relations
                self.adj.setdefault(t_source, set()).add(t_target)
                self.adj.setdefault(t_target, set()).add(t_source)
            elif predicate == "available-color":
                self.colors.add(args[0])

        # --- Extract All Objects (Robots, Tiles, Colors) ---
        # Ensure all objects mentioned anywhere are recorded
        all_facts = self.initial_state.union(self.goals).union(self.static_facts)
        for fact_str in all_facts:
             parts = get_parts(fact_str)
             predicate = parts[0]
             args = parts[1:]
             # Infer object types and add them to the respective sets
             if predicate == "robot-at":
                 self.robots.add(args[0])
                 self.tiles.add(args[1])
             elif predicate == "painted":
                 self.tiles.add(args[0])
                 self.colors.add(args[1])
             elif predicate == "robot-has":
                 self.robots.add(args[0])
                 self.colors.add(args[1])
             elif predicate == "clear":
                 self.tiles.add(args[0])
             elif predicate in ["up", "down", "left", "right"]:
                 self.tiles.add(args[0])
                 self.tiles.add(args[1])
             elif predicate == "available-color":
                 self.colors.add(args[0])

        # --- Parse Goal Conditions ---
        for goal_str in self.goals:
            parts = get_parts(goal_str)
            if parts[0] == "painted":
                # Store the target state for each tile required by the goal
                self.goal_conditions.add((parts[1], parts[2])) # (tile, color)

        # Convert robots set to list for consistent iteration order during calculation
        self.robots = list(self.robots)


    def _bfs_distance(self, start_tile, end_tile, clear_tiles_set):
        """
        Calculates the shortest path distance (number of moves) between two tiles
        using Breadth-First Search. Movement is restricted to tiles present in
        the provided `clear_tiles_set`.

        Args:
            start_tile (str): The starting tile name.
            end_tile (str): The destination tile name.
            clear_tiles_set (set): A set of tile names that are currently clear
                                   and thus traversable.

        Returns:
            int: The shortest distance in number of moves.
            math.inf: If the end_tile is unreachable from the start_tile through
                      clear tiles.
        """
        if start_tile == end_tile:
            return 0

        # Initialize queue for BFS: stores (tile_name, distance_from_start)
        queue = deque([(start_tile, 0)])
        # Keep track of visited tiles to avoid cycles and redundant exploration
        visited = {start_tile}

        while queue:
            current_tile, dist = queue.popleft()

            # Explore neighbors of the current tile
            for neighbor in self.adj.get(current_tile, set()):
                # Check if the neighbor is the target destination tile
                if neighbor == end_tile:
                    return dist + 1 # Path found

                # Check if the neighbor is traversable (clear) and not already visited
                if neighbor in clear_tiles_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1)) # Add to queue for further exploration

        # If the queue becomes empty and the end_tile was not reached
        return math.inf


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: A state node object containing the current state (`node.state`).

        Returns:
            int: The estimated cost (number of actions) to reach a goal state.
        """
        state = node.state

        # --- 1. Parse Current State ---
        robot_positions = {} # robot -> tile
        robot_colors = {}    # robot -> color
        current_painted_tuples = set() # {(tile, color)}
        clear_tiles = set()  # {tile}

        for fact_str in state:
            parts = get_parts(fact_str)
            predicate = parts[0]
            args = parts[1:]

            if predicate == "robot-at":
                robot_positions[args[0]] = args[1]
            elif predicate == "robot-has":
                robot_colors[args[0]] = args[1]
            elif predicate == "painted":
                current_painted_tuples.add((args[0], args[1]))
            elif predicate == "clear":
                clear_tiles.add(args[0])

        # --- 2. Identify Unsatisfied Goals ---
        unsatisfied_goals = set()
        for goal_tile, goal_color in self.goal_conditions:
            if (goal_tile, goal_color) not in current_painted_tuples:
                 unsatisfied_goals.add((goal_tile, goal_color))

        # If no unsatisfied goals, the heuristic value is 0 (goal reached)
        if not unsatisfied_goals:
            return 0

        # --- 3. Initialize Simulation State ---
        h_value = 0
        # Convert set to list to allow indexed removal during iteration
        remaining_goals = list(unsatisfied_goals)

        # Create copies of dynamic state parts for simulation within this calculation
        sim_robot_positions = robot_positions.copy()
        sim_robot_colors = robot_colors.copy()
        # Use the set of tiles clear at the *start* of this state evaluation for all BFS calls
        # This simplifies the calculation by not simulating clear state changes.
        initial_clear_tiles = clear_tiles.copy()

        # --- 4. Greedy Assignment Loop ---
        while remaining_goals:
            min_total_cost = math.inf
            best_robot = None
            best_goal_index = -1 # Index in the remaining_goals list
            best_adj_tile_for_move = None # Store the adjacent tile the robot moves to

            # --- 4a. Find Best Action (Robot-Goal Pair) ---
            for robot in self.robots:
                # Ensure robot exists in the current simulation state
                if robot not in sim_robot_positions:
                    continue # Should not happen in standard PDDL tasks

                current_sim_pos = sim_robot_positions[robot]
                current_sim_color = sim_robot_colors.get(robot)

                # Robot needs a color to perform paint or change_color actions
                if current_sim_color is None:
                    continue # Skip robots without a color (if possible in domain)

                # Iterate through remaining goals
                for g_idx, goal in enumerate(remaining_goals):
                    goal_tile, goal_color = goal

                    # --- 4a.i. Precondition Cost ---
                    # Check if the target tile for painting was clear in the original state
                    precondition_cost = 0
                    if goal_tile not in initial_clear_tiles:
                        # Add a penalty if target tile wasn't clear (needs clearing action)
                        precondition_cost = 1

                    # --- 4a.ii. Movement Cost ---
                    # Find the minimum distance from the robot's current simulated position
                    # to any tile adjacent to the goal tile.
                    min_dist_to_adj = math.inf
                    target_adj_tile = None # The specific adjacent tile to move to

                    # Iterate through tiles adjacent to the goal tile
                    for adj_tile in self.adj.get(goal_tile, set()):
                        # Calculate distance using BFS, restricted to initially clear tiles
                        dist = self._bfs_distance(current_sim_pos, adj_tile, initial_clear_tiles)

                        # Update if this path is shorter
                        if dist < min_dist_to_adj:
                            min_dist_to_adj = dist
                            target_adj_tile = adj_tile # Store the best adjacent tile found

                    # If no adjacent tile is reachable via a clear path for this robot
                    if target_adj_tile is None or min_dist_to_adj == math.inf:
                        continue # This robot cannot reach this goal's vicinity currently

                    move_cost = min_dist_to_adj

                    # --- 4a.iii. Color Cost ---
                    color_cost = 0 if current_sim_color == goal_color else 1

                    # --- 4a.iv. Paint Cost ---
                    paint_cost = 1

                    # --- 4a.v. Total Cost ---
                    total_cost = precondition_cost + move_cost + color_cost + paint_cost

                    # --- 4b. Select Minimum Cost Action (Update Best Choice) ---
                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        best_robot = robot
                        best_goal_index = g_idx
                        best_adj_tile_for_move = target_adj_tile # Record where the robot moves

            # --- 4c. Handle Impossibility ---
            # If no robot could be assigned to any remaining goal in this iteration
            if best_robot is None or min_total_cost == math.inf:
                 # Assign a large penalty for remaining goals and stop calculation
                 h_value += len(remaining_goals) * 100 # Arbitrary large penalty
                 break # Exit the while loop

            # --- 4d. Update Heuristic Value ---
            h_value += min_total_cost

            # --- 4e. Update Simulated State ---
            # Remove the assigned goal from the list
            chosen_goal = remaining_goals.pop(best_goal_index)
            _goal_tile, goal_color = chosen_goal # Extract the required color

            # Update the chosen robot's simulated position and color
            sim_robot_positions[best_robot] = best_adj_tile_for_move
            sim_robot_colors[best_robot] = goal_color

        # --- 5. Return Total Heuristic Value ---
        # Ensure the heuristic returns an integer value
        return int(round(h_value))
