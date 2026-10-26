import sys
from collections import deque
from fnmatch import fnmatch
# Assuming the base class Heuristic is defined in the following path
# If the path is different, you may need to adjust the import statement.
from heuristics.heuristic_base import Heuristic 

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    
    Args:
        fact (str): A PDDL fact string.

    Returns:
        list: A list containing the predicate name and its arguments.
    """
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban. It calculates
    the sum of the shortest path distances for each box from its current location to 
    its designated goal location. Additionally, it adds the shortest path distance 
    for the robot to reach the location of the nearest misplaced box. The distances 
    are pre-calculated based on the empty grid layout (ignoring dynamic obstacles 
    like other boxes or the robot position during pathfinding). This heuristic is
    designed to be informative for a greedy best-first search but is not admissible,
    as it simplifies the complex interactions and path blockages in Sokoban.

    # Assumptions
    - The main effort and cost in Sokoban involves pushing boxes to their goals. Pushes are estimated by the grid distance.
    - Robot movement cost is secondary but necessary to position for pushes. This is estimated by the distance to the nearest misplaced box.
    - The heuristic simplifies the problem by ignoring potential blockages caused by other boxes or the robot's position when calculating distances between locations.
    - Each box mentioned in the goal has a unique, fixed target location specified in the problem's goal description (e.g., `(at box1 goal_loc1)`).
    - Distances correspond to the minimum number of steps (adjacent moves or pushes) on the grid graph defined by `adjacent` predicates.
    - The level layout (walls and passable squares) is static.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the planning task's static information and goal conditions once upon instantiation.
    - It parses the goal statements (`task.goals`) to create a mapping (`self.goal_map`) from each relevant box to its target location.
    - It builds an adjacency list representation (`self.adj`) of the level grid using the static `adjacent` facts provided in `task.static`.
    - It precomputes all-pairs shortest path distances (`self.distances`) between all locations using Breadth-First Search (BFS) on the grid graph. This distance represents the minimum steps needed to travel between two points assuming the path is clear of dynamic obstacles.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Get Current State Information:** In the `__call__` method, extract the robot's current location (`robot_loc`) and the current location of each box (`box_locs`) from the input state (`node.state`). Only consider boxes that are part of the goal definition.
    2.  **Initialize Heuristic Value:** Set `total_heuristic = 0`. Initialize `min_dist_robot_to_misplaced_box = float('inf')` to track the shortest distance from the robot to any box not yet at its goal. Set a flag `misplaced_boxes_exist = False`.
    3.  **Iterate Over Goal Boxes:** Loop through each box (`box`) listed in `self.boxes` (boxes that have a defined goal location).
        a.  Find the box's current location `current_loc` from the parsed state `box_locs`. Handle cases where a goal box might be missing from the state (return `inf`).
        b.  Get the box's target location `goal_loc` from the precomputed `self.goal_map`.
        c.  **Check if Box is Misplaced:** If `current_loc` is different from `goal_loc`:
            i.  Set `misplaced_boxes_exist = True`.
            ii. **Calculate Box Distance:** Retrieve the precomputed shortest distance `box_dist` between `current_loc` and `goal_loc` from `self.distances`. If this distance is infinite (meaning the goal is unreachable on the empty grid), return `float('inf')` immediately, as the state is unsolvable. Add `box_dist` to `total_heuristic`. This component estimates the minimum number of pushes required for this specific box.
            iii. **Calculate Robot-to-Box Distance:** Retrieve the precomputed shortest distance `robot_dist` between the `robot_loc` and the `current_loc` of the box. This estimates the number of moves the robot might need to reach this box. Handle cases where the distance is infinite.
            iv. **Track Nearest Misplaced Box:** Update `min_dist_robot_to_misplaced_box` by taking the minimum of its current value and `robot_dist`.
    4.  **Add Robot Movement Component:** After checking all boxes, if `misplaced_boxes_exist` is `True`:
        a.  Check if `min_dist_robot_to_misplaced_box` is still `float('inf')`. If so, it means the robot cannot reach *any* of the boxes that need to be moved; return `float('inf')` as the state seems unsolvable from the robot's position.
        b.  Otherwise, add `min_dist_robot_to_misplaced_box` to `total_heuristic`. This adds an estimate for the robot's initial travel cost to get to the closest box that requires pushing.
    5.  **Return Heuristic Value:** Return the final `total_heuristic`. If no boxes were misplaced, the value will be 0, correctly identifying states where all box-related goal conditions are met.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information and goals.
        
        Args:
            task: The planning task object containing goals, static facts, 
                  initial state, operators, etc.
        """
        self.goals = task.goals
        static_facts = task.static
        
        # Build the adjacency graph representing the level layout
        self.adj = self._build_adjacency_graph(static_facts)
        
        # Precompute shortest path distances between all locations on the empty grid
        self.distances = self._compute_all_pairs_shortest_paths(self.adj)
        
        # Parse goal conditions to find target locations for boxes
        self.goal_map = self._parse_goal_locations(task.goals)
        
        # Store the names of boxes that are part of the goal definition
        self.boxes = list(self.goal_map.keys())

    def _build_adjacency_graph(self, static_facts):
        """
        Builds an adjacency list graph from 'adjacent' predicates found in static facts.
        Assumes the grid connectivity is defined by these facts.
        """
        adj = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            # Example fact: "(adjacent loc_1_1 loc_1_2 right)"
            if parts[0] == 'adjacent' and len(parts) >= 3:
                loc1, loc2 = parts[1], parts[2]
                locations.add(loc1)
                locations.add(loc2)
                # Add edge loc1 -> loc2. Assumes grid allows movement between adjacent locations.
                if loc1 not in adj: adj[loc1] = []
                # Avoid adding duplicate neighbors if multiple directions connect the same pair
                if loc2 not in adj.get(loc1, []):
                    adj[loc1].append(loc2)
        
        # Ensure all identified locations exist as keys in the adjacency list, even if isolated.
        for loc in locations:
             if loc not in adj:
                 adj[loc] = [] 
        return adj

    def _compute_all_pairs_shortest_paths(self, adj):
        """
        Computes shortest path distances between all pairs of locations using BFS 
        starting from each location. Returns a dictionary of dictionaries where
        distances[start][end] gives the shortest distance.
        """
        locations = list(adj.keys())
        if not locations:
            # Return an empty dictionary if there are no locations
            return {} 
            
        distances = {loc: {} for loc in locations}
        
        for start_node in locations:
            # Initialize distances from start_node to all other locations as infinity
            for loc in locations:
                distances[start_node][loc] = float('inf')
            
            # Distance from a node to itself is 0
            distances[start_node][start_node] = 0
            # Initialize queue for BFS starting from start_node
            queue = deque([(start_node, 0)])
            
            # Perform BFS
            while queue:
                current_node, dist = queue.popleft()
                
                # Explore neighbors of the current node
                for neighbor in adj.get(current_node, []):
                    # If we found a shorter path (or the first path) to the neighbor
                    if distances[start_node][neighbor] == float('inf'): 
                        distances[start_node][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
                        
        return distances

    def _parse_goal_locations(self, goals):
        """
        Parses goal facts like '(at box loc)' into a dictionary mapping 
        each box to its required goal location.
        """
        goal_map = {}
        for goal in goals:
            parts = get_parts(goal)
            # Example goal: "(at box1 loc_2_4)"
            if parts[0] == 'at' and len(parts) == 3:
                # Assuming the predicate structure is (at <object> <location>)
                # And that the object is a box relevant to the problem goals.
                box, loc = parts[1], parts[2]
                # We might want to check if 'box' is actually of type 'box' if types are available
                goal_map[box] = loc
        return goal_map

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.

        Args:
            node: The search node object, which contains the current state (`node.state`). 
                  The state is expected to be a frozenset of PDDL fact strings.

        Returns:
            float: An estimate of the cost to reach the goal. Returns 0 for goal states 
                   (where all goal boxes are in place). Returns float('inf') if the state 
                   is detected as unsolvable by this heuristic's checks (e.g., box goal 
                   unreachable, robot cannot reach any misplaced box).
        """
        state = node.state
        robot_loc = None
        box_locs = {} # Dictionary to store current location of goal-relevant boxes

        # Parse the current state to find the robot's location and locations of goal boxes
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == 'at' and len(parts) == 3:
                # Check if the object 'at' this location is one of the boxes we care about for the goal
                obj, loc = parts[1], parts[2]
                if obj in self.goal_map: # Only track boxes that are part of the goal definition
                     box_locs[obj] = loc

        # --- Heuristic Calculation ---

        # Handle trivial case: if there are no boxes defined in the goal, heuristic is 0.
        if not self.boxes: 
             return 0
             
        # Check if robot location was found (essential for the heuristic)
        if robot_loc is None:
             # This indicates an invalid state or problem setup for Sokoban.
             # print("Warning: Robot location ('at-robot') not found in state.", file=sys.stderr)
             return float('inf') # Cannot calculate heuristic without robot position

        total_heuristic = 0.0 # Use float for potential infinity values
        min_dist_robot_to_misplaced_box = float('inf')
        misplaced_boxes_exist = False

        # Iterate through all boxes that have a specified goal location
        for box in self.boxes:
            current_loc = box_locs.get(box)
            
            # Check if the box's location was found in the current state
            if current_loc is None:
                 # This implies a goal box is missing from the state, which is problematic.
                 # print(f"Warning: Location for goal box '{box}' not found in state.", file=sys.stderr)
                 return float('inf') # State seems invalid

            goal_loc = self.goal_map[box]

            # If the box is not in its goal location
            if current_loc != goal_loc:
                misplaced_boxes_exist = True
                
                # --- Calculate Box Contribution ---
                # Check if distances were precomputed for these locations
                if current_loc not in self.distances or goal_loc not in self.distances.get(current_loc, {}):
                    # Should not happen if BFS covered all reachable locations from valid starts
                    # print(f"Warning: Distance lookup failed between '{current_loc}' and '{goal_loc}'.", file=sys.stderr)
                    return float('inf') # Indicate error or inconsistency
                
                box_dist = self.distances[current_loc][goal_loc]
                
                # If the distance is infinite, the goal location is unreachable for this box
                if box_dist == float('inf'):
                    # This state is unsolvable regarding this box placement
                    return float('inf') 

                total_heuristic += box_dist # Add estimated pushes cost

                # --- Calculate Robot Distance to this Box ---
                if robot_loc not in self.distances or current_loc not in self.distances.get(robot_loc, {}):
                     # print(f"Warning: Distance lookup failed between robot at '{robot_loc}' and box at '{current_loc}'.", file=sys.stderr)
                     # If robot cannot reach the box location (e.g., disconnected graph component)
                     # Assign infinity to robot_dist; the min calculation will handle this.
                     robot_dist = float('inf')
                else:
                     robot_dist = self.distances[robot_loc][current_loc]
                
                # Update the minimum distance from the robot to any misplaced box
                min_dist_robot_to_misplaced_box = min(min_dist_robot_to_misplaced_box, robot_dist)

        # Add the estimated cost for the robot to move to the nearest misplaced box
        if misplaced_boxes_exist:
            # If min_dist remained infinity, it means the robot cannot reach *any* box that needs moving.
            if min_dist_robot_to_misplaced_box == float('inf'):
                return float('inf') # State likely unsolvable from robot's current position
            
            total_heuristic += min_dist_robot_to_misplaced_box

        # The final heuristic value is the sum of estimated pushes and initial robot movement.
        # It will be 0 if misplaced_boxes_exist is False.
        return int(total_heuristic) # Return as integer if finite
