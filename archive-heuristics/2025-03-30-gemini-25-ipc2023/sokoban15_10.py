import heapq
from collections import deque
from fnmatch import fnmatch
# Assuming the planner infrastructure provides this base class
# If not available, define a simple placeholder:
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Utility functions
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Checks if a PDDL fact matches a given pattern using fnmatch for wildcard support.
    The number of parts in the fact must match the number of arguments in the pattern.
    Example: match("(at box1 loc_1_1)", "at", "*", "loc_1_1") -> True
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    # Optimization: Check predicate name first if it's not a wildcard
    if args[0] != '*' and parts[0] != args[0]:
        return False
    # Check if all parts match the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban for use
    with Greedy Best-First Search. It calculates the sum of shortest path distances
    for each box from its current location to its goal location, plus the shortest
    distance from the robot to the nearest misplaced box. Distances are precomputed
    on the empty grid graph derived from static 'adjacent' facts.

    # Assumptions
    - The grid structure (walls and open spaces defined by 'adjacent' facts) is static.
    - Connectivity between locations is bidirectional as is typical in Sokoban grids.
    - The cost of moving the robot ('move' action) is 1 per step.
    - The cost of pushing a box ('push' action) is 1 per step (this action moves both the box and the robot).
    - The heuristic does not need to be admissible.
    - Goal states involve specific boxes being at specific locations defined by '(at boxX locY)' facts in the goal description.

    # Heuristic Initialization
    - Parses static 'adjacent' facts from `task.static` to build a map representation as an adjacency list (`self.adj`). It assumes bidirectional connectivity.
    - Identifies all unique location names found in the `adjacent` facts (`self.locations`).
    - Computes all-pairs shortest path distances between all identified locations using Breadth-First Search (BFS). The distances are stored in `self.distances[loc1][loc2]`. This computation ignores dynamic obstacles (robot, other boxes) and only considers the static grid layout.
    - Parses the goal conditions (`task.goals`) to identify the target location for each required box, stored in `self.goal_locations`. It also stores the set of boxes involved in the goal (`self.boxes`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the current location of the robot (`at-robot` fact) and the current location of each box relevant to the goal (`at box loc` facts) from the input `node.state`.
    2.  **Check State Validity:** Ensure the robot location is found and all boxes required for the goal are present in the state. If not, return infinity (invalid/unsolvable state).
    3.  **Check Goal Completion:** Iterate through all goal conditions (`self.goal_locations`). If all relevant boxes are currently at their respective goal locations, the state is a goal state, return 0.
    4.  **Calculate Total Box Distance (`box_heuristic`):**
        a. Initialize `box_heuristic = 0`.
        b. Create a list `misplaced_boxes_locs` to store the locations of boxes not yet at their goal.
        c. For each box `b` that is required for a goal and is not currently at its goal location `g_loc = self.goal_locations[b]`:
            i. Find its current location `c_loc`.
            ii. Retrieve the precomputed shortest path distance `d = self.distances[c_loc][g_loc]` from the `self.distances` map.
            iii. If `d` is infinity, it signifies that the goal location `g_loc` is unreachable from the box's current location `c_loc` based on the static grid structure. This state cannot lead to a solution, so return `float('inf')`.
            iv. Add `d` to `box_heuristic`. This value estimates the minimum number of pushes required for this box, ignoring interactions.
            v. Add the box's current location `c_loc` to the `misplaced_boxes_locs` list.
    5.  **Calculate Robot Distance (`robot_heuristic`):**
        a. Initialize `robot_heuristic = 0`.
        b. If `misplaced_boxes_locs` is not empty (i.e., the goal is not achieved):
            i. Get the robot's current location `r_loc`.
            ii. Initialize `min_dist_to_box = float('inf')`.
            iii. Retrieve the precomputed distances from the robot's location: `robot_distances = self.distances[r_loc]`.
            iv. For each location `box_loc` in `misplaced_boxes_locs`:
                - Get the distance `d = robot_distances[box_loc]` from the robot to the box's location.
                - Update `min_dist_to_box = min(min_dist_to_box, d)`.
            v. If `min_dist_to_box` remains infinity after checking all misplaced boxes, it means the robot cannot reach any of the boxes that need to be moved. Return `float('inf')`.
            vi. Set `robot_heuristic = min_dist_to_box`. This estimates the number of robot moves required to reach the *location* of the nearest box that needs pushing.
    6.  **Combine:** The final heuristic value is the sum `h = box_heuristic + robot_heuristic`. This combines the estimated total push cost for all boxes with the estimated cost for the robot to initiate interaction with the nearest misplaced box.
    7.  **Return Value:** Return the calculated heuristic value `h`. It is 0 for goal states, a positive integer for reachable non-goal states, and infinity for states detected as unsolvable due to grid connectivity issues or missing objects.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing distances based on static grid layout
        and identifying goal locations for boxes.
        """
        super().__init__(task) # Initialize base class if necessary
        self.goals = task.goals
        static_facts = task.static

        # 1. Build adjacency list and identify all locations from static 'adjacent' facts
        self.adj = {}
        self.locations = set()
        adj_relations = set() # Use a set to store pairs of connected locations to handle duplicates
        for fact in static_facts:
            # Example PDDL fact: (adjacent loc_1_1 loc_1_2 right)
            if match(fact, "adjacent", "?l1", "?l2", "?dir"):
                _, l1, l2, _ = get_parts(fact)
                # Store locations
                self.locations.add(l1)
                self.locations.add(l2)
                # Store connection canonically (sorted tuple) to avoid duplicates if facts are symmetric
                pair = tuple(sorted((l1, l2)))
                if pair not in adj_relations:
                    # Add edges in both directions, assuming grid connectivity is bidirectional
                    self.adj.setdefault(l1, []).append(l2)
                    self.adj.setdefault(l2, []).append(l1)
                    adj_relations.add(pair)

        # 2. Compute all-pairs shortest paths using BFS on the empty grid
        self.distances = {}
        for start_node in self.locations:
            # Compute distances from start_node to all other locations in the set
            self.distances[start_node] = self._bfs(start_node)

        # 3. Parse goal locations and identify the set of boxes involved in the goal
        self.goal_locations = {}
        self.boxes = set()
        for goal in self.goals:
            # Example PDDL goal fact: (at box1 loc_2_4)
            if match(goal, "at", "?box", "?loc"):
                _, box, loc = get_parts(goal)
                if loc not in self.locations:
                     # If a goal location wasn't mentioned in 'adjacent' facts, it might be isolated.
                     # Add it to locations; distances involving it will likely be infinity unless
                     # it's reachable via paths not explicitly listed (unlikely in standard PDDL).
                     print(f"Warning: Goal location {loc} for box {box} not found among 'adjacent' facts. Assuming it exists but may be unreachable.")
                     self.locations.add(loc)
                     # Ensure distance entries exist for this potentially isolated location
                     if loc not in self.distances:
                         self.distances[loc] = {l: float('inf') for l in self.locations}
                         self.distances[loc][loc] = 0
                         for other_loc, dist_map in self.distances.items():
                             if other_loc != loc:
                                 dist_map[loc] = float('inf')

                self.goal_locations[box] = loc
                self.boxes.add(box)


    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from start_node to find shortest path
        distances to all reachable locations based on the static adjacency graph `self.adj`.
        Returns a dictionary mapping reachable locations to their distances.
        """
        # Initialize distances for all known locations to infinity
        distances = {node: float('inf') for node in self.locations}

        if start_node not in self.locations:
             # This case should ideally not happen if start_node is from self.locations
             print(f"Error: BFS start node {start_node} is not in the set of known locations.")
             return distances # Return all infinities

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            # Explore neighbors based on the precomputed adjacency list
            for neighbor in self.adj.get(current_node, []):
                # If neighbor is a known location and we haven't found a path yet (dist is inf)
                if neighbor in distances and distances[neighbor] == float('inf'):
                    # Update distance and add neighbor to the queue
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the cost to reach the goal.
        Returns 0 for goal states, infinity for detected unsolvable states.
        """
        state = node.state

        # 1. Parse current state: find robot location and locations of goal-relevant boxes
        robot_loc = None
        box_locations = {}
        for fact in state:
            if match(fact, "at-robot", "?loc"):
                robot_loc = get_parts(fact)[1]
            elif match(fact, "at", "?box", "?loc"):
                _, box, loc = get_parts(fact)
                # Only track locations of boxes that are part of the goal definition
                if box in self.boxes:
                    box_locations[box] = loc

        # 2. Basic state validity checks
        if robot_loc is None:
            print("Warning: Robot location ('at-robot') not found in the current state.")
            return float('inf') # Cannot proceed without robot location

        # Check if all boxes required for the goal are present in the current state
        missing_boxes = self.boxes - set(box_locations.keys())
        if missing_boxes:
             print(f"Warning: Goal-relevant boxes are missing from the state description: {missing_boxes}")
             return float('inf') # Cannot achieve goal if required boxes are missing


        # 3. Check goal completion & calculate total box distance heuristic
        box_heuristic = 0
        misplaced_boxes_locs = [] # Keep track of locations of boxes not in their goal spots
        goal_achieved = True

        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box) # We know the box exists from the check above

            if current_loc != goal_loc:
                goal_achieved = False # At least one box is not at its goal

                # Retrieve precomputed distance from the box's current location to its goal
                current_loc_distances = self.distances.get(current_loc)
                if current_loc_distances is None:
                    # This implies the box is at a location not found during initialization (e.g., inside a wall?)
                    print(f"Warning: Box {box} is at an unknown location {current_loc} with no precomputed distances.")
                    return float('inf') # Invalid state

                dist = current_loc_distances.get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # If distance is infinity, the goal is unreachable for this box from its current spot
                    # based on the empty grid layout. This state is unsolvable.
                    return float('inf')

                # Add the distance (estimated pushes) to the heuristic total
                box_heuristic += dist
                # Record the location of this misplaced box
                misplaced_boxes_locs.append(current_loc)

        # If all boxes are at their goal locations, the heuristic value is 0
        if goal_achieved:
            return 0

        # 4. Calculate robot distance heuristic (cost to reach nearest misplaced box)
        robot_heuristic = 0
        # This part only runs if goal_achieved is False, meaning misplaced_boxes_locs is not empty
        if misplaced_boxes_locs:
            min_dist_to_box = float('inf')
            # Get the dictionary of distances from the robot's current location
            robot_distances = self.distances.get(robot_loc)
            if robot_distances is None:
                 # Robot is at an unknown location?
                 print(f"Warning: Robot is at an unknown location {robot_loc} with no precomputed distances.")
                 return float('inf') # Invalid state

            # Find the minimum distance from the robot to any of the misplaced box locations
            for box_loc in misplaced_boxes_locs:
                 dist = robot_distances.get(box_loc, float('inf'))
                 min_dist_to_box = min(min_dist_to_box, dist)

            if min_dist_to_box == float('inf'):
                 # If the minimum distance is still infinity, the robot cannot reach any box that needs moving.
                 return float('inf') # Unsolvable state

            # The heuristic contribution is the distance to the nearest misplaced box
            robot_heuristic = min_dist_to_box
        # If misplaced_boxes_locs was empty, robot_heuristic remains 0 (already handled by goal_achieved check)


        # 5. Combine the heuristics: total estimated pushes + estimated moves to nearest box
        total_heuristic = box_heuristic + robot_heuristic

        # Return the final heuristic value, ensuring it's non-negative
        return max(0, total_heuristic)

