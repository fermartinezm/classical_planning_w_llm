import collections
from heuristics.heuristic_base import Heuristic
# Using collections.deque for efficient BFS queue

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space. Handles potential whitespace.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    return fact.strip()[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It primarily considers the sum of shortest path distances each box needs to travel
    to reach its target location, plus the shortest path distance the robot needs to
    travel to reach the location of the nearest misplaced box. Distances are
    calculated using Breadth-First Search (BFS) on the static map layout,
    representing the minimum number of moves/pushes required assuming no obstacles
    other than walls. This heuristic is designed for Greedy Best-First Search and
    is not necessarily admissible.

    # Assumptions
    - The cost of moving the robot ('move' action) is 1.
    - The cost of pushing a box ('push' action) is 1.
    - The heuristic ignores dynamic obstacles (other boxes, robot position relative
      to push direction) when calculating distances. It uses the empty grid distances.
    - It assumes the shortest path for a box on an empty grid is a reasonable estimate
      of the number of pushes needed for that box.
    - Location connectivity is fully defined by the static 'adjacent' predicates.
    - Goal is defined by a set of (at box location) predicates.

    # Heuristic Initialization
    - Extracts all unique location objects from the task definition (static facts,
      initial state, goals).
    - Builds an adjacency graph representing the possible movements between locations
      based on the static 'adjacent' predicates. Assumes adjacency is symmetric,
      meaning if loc1 is adjacent to loc2, loc2 is also adjacent to loc1.
    - Computes All-Pairs Shortest Paths (APSP) using BFS starting from each location.
      This precomputes the `distance(loc1, loc2)` for all reachable pairs. Stores
      distances in `self.distances`. Unreachable pairs implicitly have infinite distance.
    - Parses the goal conditions to create a mapping `self.box_goals` from each
      box involved in the goal to its target location.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Identify the current location of the robot (`robot_loc`)
        and the location of each box (`box_locs`).
    2.  **Check Goal Completion:** Determine if all goal conditions `(at box goal_loc)`
        specified in `self.box_goals` are currently satisfied in the state. If yes,
        the heuristic value is 0.
    3.  **Calculate Total Box Distance:**
        - Initialize `total_box_distance = 0`.
        - For each box `b` that has a defined goal location `goal_loc` in `self.box_goals`:
            - Get the box's current location `current_loc` from `box_locs`. If the box
              is not found in the current state (e.g., PDDL error or unusual state),
              return infinity as the goal is unreachable.
            - If `current_loc != goal_loc`:
                - Retrieve the precomputed shortest path distance `dist = self.get_dist(current_loc, goal_loc)`.
                - If `dist` is infinity (goal location is unreachable for this box from
                  its current position based on static map connectivity), return infinity
                  for the heuristic value, as the state is unsolvable.
                - Add `dist` to `total_box_distance`.
    4.  **Calculate Minimum Robot-to-Misplaced-Box Distance:**
        - Initialize `min_robot_to_box_dist = infinity`.
        - Keep track if any boxes are misplaced (`misplaced_boxes_exist = False`).
        - For each box `b` that is misplaced (identified in step 3):
            - Set `misplaced_boxes_exist = True`.
            - Get its current location `current_loc`.
            - Retrieve the precomputed shortest path distance from the robot's current
              location to the box's location: `dist = self.get_dist(robot_loc, current_loc)`.
            - Update `min_robot_to_box_dist = min(min_robot_to_box_dist, dist)`. This finds
              the distance to the *nearest* misplaced box.
        - If `misplaced_boxes_exist` is true but `min_robot_to_box_dist` remains infinity
          (meaning the robot cannot reach *any* of the misplaced boxes based on static
          map connectivity), return infinity for the heuristic value (unsolvable state).
    5.  **Combine Costs:** The final heuristic value is the sum of the total box distance
        and the minimum robot-to-box distance.
        `h(state) = total_box_distance + min_robot_to_box_dist`
        (If `misplaced_boxes_exist` is false, the goal check in step 2 should have
         returned 0, so this calculation step is effectively skipped for goal states).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing distances and goal locations.
        """
        self.goals = task.goals
        static_facts = task.static
        # Combine initial state and static facts to find all potential locations
        all_init_and_static = task.initial_state.union(static_facts)

        # Extract all unique location objects mentioned anywhere
        potential_locations = set()
        for fact in all_init_and_static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            # Check arguments based on known predicates that use locations
            if predicate == 'adjacent':
                if len(parts) >= 3: # (adjacent loc1 loc2 dir)
                    potential_locations.add(parts[1])
                    potential_locations.add(parts[2])
            elif predicate == 'at-robot':
                 if len(parts) == 2: # (at-robot loc)
                     potential_locations.add(parts[1])
            elif predicate == 'at':
                 if len(parts) == 3: # (at box loc) or (at truck loc) etc.
                     # Assume 3rd part is location if it's 'at' predicate
                     potential_locations.add(parts[2])
            elif predicate == 'clear':
                 if len(parts) == 2: # (clear loc)
                     potential_locations.add(parts[1])

        # Add locations mentioned in goals
        for fact in task.goals:
            parts = get_parts(fact)
            if not parts: continue
            # Assuming goal is (at box loc)
            if parts[0] == 'at' and len(parts) == 3:
                potential_locations.add(parts[2])

        self.locations = potential_locations # Use all found locations

        # Build adjacency list graph from static 'adjacent' facts
        self.adj = collections.defaultdict(list)
        adj_pairs = set() # To handle symmetric definition in PDDL if present
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            # PDDL format: (adjacent ?l1 - location ?l2 - location ?d - direction)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                # Ensure locations are known before adding edge
                if loc1 in self.locations and loc2 in self.locations:
                    pair = tuple(sorted((loc1, loc2)))
                    if pair not in adj_pairs:
                        self.adj[loc1].append(loc2)
                        self.adj[loc2].append(loc1) # Assume symmetry for physical adjacency
                        adj_pairs.add(pair)

        # Compute All-Pairs Shortest Paths (APSP) using BFS
        self.distances = {}
        for start_node in self.locations:
            # Run BFS from each location to find distances to all reachable locations
            q = collections.deque([(start_node, 0)])
            # Store distances from start_node to all reachable nodes
            dist_from_start = {start_node: 0}
            self.distances[(start_node, start_node)] = 0

            while q:
                curr_loc, dist = q.popleft()
                # Iterate through neighbors based on the adjacency graph
                for neighbor in self.adj.get(curr_loc, []): # Use .get for safety
                    if neighbor not in dist_from_start:
                        dist_from_start[neighbor] = dist + 1
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Store goal locations for each box specified in the goal
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            # Goal format: (at ?o - box ?l - location)
            if parts[0] == 'at' and len(parts) == 3:
                box, goal_loc = parts[1], parts[2]
                # Ensure goal location is valid and known
                if goal_loc in self.locations:
                    self.box_goals[box] = goal_loc
                else:
                    # This indicates an issue with the problem PDDL file.
                    print(f"Warning: Goal location '{goal_loc}' for box '{box}' "
                          f"is not found among known locations. This goal may be unachievable.")
                    # If a goal location doesn't exist, the problem is likely unsolvable.
                    # The get_dist function will return infinity later if needed.


    def get_dist(self, loc1, loc2):
        """
        Returns the precomputed shortest path distance between two locations.
        Returns float('inf') if locations are invalid or unreachable based on static map.
        """
        # Check if locations are valid (part of the recognized locations)
        if loc1 not in self.locations or loc2 not in self.locations:
            return float('inf')
        # Retrieve distance, default to infinity if no path exists in self.distances map
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state

        # --- 1. Parse Current State ---
        robot_loc = None
        box_locs = {} # Stores {box_name: location}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == 'at' and len(parts) == 3:
                # Assuming it's (at box loc) based on domain structure
                box, loc = parts[1], parts[2]
                # Only track boxes relevant to the goal to avoid issues with other objects
                # if box in self.box_goals: # Optimization: only track goal boxes? No, need all for checks.
                box_locs[box] = loc

        # Validate robot location - essential for calculating robot distances
        if robot_loc is None or robot_loc not in self.locations:
             # Robot location missing or invalid, cannot compute heuristic reliably
             return float('inf')

        # --- 2. Check Goal Completion ---
        num_goals = len(self.box_goals)
        if num_goals == 0: # Handle case with no box goals defined
             return 0

        met_goals = 0
        all_goal_boxes_present_and_correct = True
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locs.get(box)
            if current_loc == goal_loc:
                met_goals += 1
            else:
                # If any goal box is not at its goal location (or missing)
                all_goal_boxes_present_and_correct = False
                # No need to break, continue checking all goals for heuristic calculation later

        if all_goal_boxes_present_and_correct and met_goals == num_goals:
             return 0 # Goal state reached

        # --- 3. Calculate Total Box Distance & 4. Min Robot Distance ---
        total_box_distance = 0
        min_robot_to_box_dist = float('inf')
        misplaced_boxes_exist = False

        for box, goal_loc in self.box_goals.items():
            current_loc = box_locs.get(box)

            # If a goal box is not present in the state, the goal is impossible from here
            if current_loc is None:
                 # print(f"Error: Goal box {box} not found in state. Returning inf.")
                 return float('inf') # Goal cannot be achieved if a required box is missing

            if current_loc != goal_loc:
                misplaced_boxes_exist = True # Mark that at least one box needs moving

                # --- Calculate distance from box to its goal ---
                box_dist = self.get_dist(current_loc, goal_loc)
                if box_dist == float('inf'):
                    # If a box cannot reach its goal location based on the static map,
                    # this state is unsolvable.
                    return float('inf')
                total_box_distance += box_dist

                # --- Calculate distance from robot to this misplaced box ---
                robot_dist = self.get_dist(robot_loc, current_loc)
                # Update minimum distance if robot can reach this box
                # (robot_dist will be inf if robot cannot reach current_loc)
                min_robot_to_box_dist = min(min_robot_to_box_dist, robot_dist)

        # --- Handle edge cases ---
        # If there are misplaced boxes, but the robot cannot reach *any* of them
        if misplaced_boxes_exist and min_robot_to_box_dist == float('inf'):
            # This implies the robot is trapped or all misplaced boxes are in areas
            # unreachable by the robot according to the static map. Unsolvable state.
            return float('inf')

        # If no boxes were misplaced (implies goal was met, handled by check 2)
        # This path should ideally not be reached if check 2 is correct.
        if not misplaced_boxes_exist:
             # This implies all boxes listed in box_goals are at their goal locations.
             # Check 2 should have returned 0. Return 0 here for robustness.
             return 0

        # --- 5. Combine Costs ---
        # Heuristic is the sum of distances boxes need to move plus the distance
        # for the robot to get to the nearest box that needs moving.
        heuristic_value = total_box_distance + min_robot_to_box_dist

        # Ensure non-negativity (should be guaranteed by BFS distances >= 0)
        return max(0, heuristic_value)

