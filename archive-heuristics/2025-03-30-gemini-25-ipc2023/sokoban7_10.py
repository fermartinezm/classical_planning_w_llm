import collections
import math
from fnmatch import fnmatch
# Assuming the base class is correctly located based on the project structure.
# If the heuristic base class is in a different location, adjust the import path.
from heuristics.heuristic_base import Heuristic 

def get_parts(fact):
    """
    Helper function to extract predicate and arguments from a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    # Remove the first '(' and last ')' characters, then split by space
    return fact[1:-1].split()

def bfs(start_node, adj):
    """
    Performs Breadth-First Search to find shortest path distances from a single 
    source node in an unweighted graph represented by an adjacency list. This 
    calculates distances on the empty grid, ignoring dynamic obstacles.

    Args:
        start_node: The name (string) of the starting location.
        adj: A dictionary representing the graph's adjacency list.
             Keys are location names (string).
             Values are lists of neighbor location names (string).
             Example: {'loc_1_1': ['loc_1_2', 'loc_2_1'], 'loc_1_2': ['loc_1_1']}

    Returns:
        A dictionary mapping each reachable location name (string) to its shortest
        distance (integer) from the start_node. Unreachable locations will have
        a distance of math.inf.
    """
    # Initialize distances: infinity for all nodes known in the graph, 0 for the start node
    distances = {node: math.inf for node in adj}
    
    # Check if the start node exists in the graph representation
    if start_node not in distances:
        # This could happen if the start node is isolated or not part of the locations
        # derived from 'adjacent' predicates. Return distances assuming it's unreachable.
        # Or, if it's a valid location but has no outgoing edges, adj might not have it as a key.
        # The __init__ method ensures all locations are keys, so this check is mainly defensive.
        return distances # Or potentially raise an error if start_node should always be valid

    distances[start_node] = 0
    # Queue for BFS, initialized with the starting node
    queue = collections.deque([start_node])
    
    while queue:
        current_node = queue.popleft()
        
        # Iterate through neighbors of the current node
        # Use adj.get(current_node, []) to safely handle nodes that might have no neighbors listed
        for neighbor in adj.get(current_node, []):
            # If the neighbor is a valid location in our graph and hasn't been reached yet
            if neighbor in distances and distances[neighbor] == math.inf:
                # Update distance and add the neighbor to the queue for exploration
                distances[neighbor] = distances[current_node] + 1
                queue.append(neighbor)
                
    return distances

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It primarily calculates the sum of the shortest path distances for each box from its current
    location to its designated goal location (ignoring obstacles for distance calculation).
    To account for the robot's actions needed to enable pushes, it adds the shortest path 
    distance from the robot's current location to the location of the nearest box that is 
    not yet at its goal. The heuristic incorporates a basic dead-end detection mechanism: 
    if any box is located in a pre-calculated 'dead location' (a simple trap like a corner 
    or edge from which it cannot be pushed out in opposite directions) and that location 
    is not its goal, the state is assigned an infinite heuristic value, effectively pruning 
    it from the search in algorithms like Greedy Best-First Search.

    # Assumptions
    - The Sokoban grid layout (locations and their adjacency) is static and defined by
      the 'adjacent' predicates.
    - The goal is defined by a set of '(at boxX locY)' predicates, specifying a unique
      target location for each relevant box.
    - All boxes mentioned in the goal conditions must reach their target locations for the
      problem to be solved.
    - The shortest path distances used are based on the empty grid (ignoring the dynamic
      positions of the robot and other boxes) for efficient precomputation. This distance
      represents the minimum number of cells to traverse between two points.

    # Heuristic Initialization (`__init__`)
    - Parses the task's goal conditions (`task.goals`) to build a mapping from each
      box name to its target location name (`self.goal_locations`).
    - Parses the static facts (`task.static`), specifically the 'adjacent' predicates, to:
        a) Collect all unique location names (`self.all_locations`).
        b) Build two representations of the grid graph:
           - `self.adj_dirs`: Maps each location to its neighbors based on the direction
             of adjacency (e.g., `{'loc_1_1': {'down': 'loc_2_1', 'right': 'loc_1_2'}}`). 
             This is used for the dead-end checks.
           - `self.simple_adj`: A standard adjacency list (`{'loc_1_1': ['loc_2_1', 'loc_1_2']}`). 
             This is used as input for the BFS distance calculation.
    - Precomputes the shortest path distances between all pairs of locations on the
      empty grid using the BFS algorithm. Results are stored in `self.distances` 
      (`{loc1: {loc2: distance}}`).
    - Precomputes a set of 'dead locations' (`self.dead_locations`). A location is
      identified as dead if it is not a goal location for any box AND it represents a 
      simple static trap (e.g., a corner where pushes are blocked in two perpendicular 
      directions, or along a wall segment where pushes are blocked in opposite directions 
      like North/South or East/West).

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Get Current State:** Obtain the set of facts (`state`) for the current node in the search.
    2.  **Goal Check:** First, check if the current state satisfies all goal conditions defined
        in `self.goals`. If `self.goals <= state`, the goal is reached, return 0.
    3.  **Parse State:** Identify the robot's current location (`robot_loc`) by finding the
        `(at-robot ?loc)` fact, and the current location of each box (`box_locations`) by
        finding the `(at ?box ?loc)` facts.
    4.  **Dead End Check:** Iterate through each box and its current location `l_curr`.
        - Retrieve the goal location `l_goal` for this box.
        - If `l_curr` is present in the precomputed `self.dead_locations` set AND `l_curr` is not 
          equal to `l_goal`, then this state contains a box in a detected dead-end situation. 
          Return `math.inf` to indicate this state is considered unsolvable.
    5.  **Calculate Box Costs (h_boxes):**
        - Initialize `h_boxes = 0`.
        - Create an empty list `misplaced_boxes` to track boxes not at their goal.
        - For each box `b` and its current location `l_curr`:
            - Get its target location `l_goal` from `self.goal_locations`.
            - If a `l_goal` exists for this box and `l_curr != l_goal`:
                - Retrieve the precomputed shortest path distance `d = self.distances[l_curr][l_goal]`.
                - Check for reachability: If `d` is `math.inf`, the goal is statically unreachable. Return `math.inf`.
                - Add `d` to `h_boxes`. This distance estimates the minimum number of single-step pushes required for this box.
                - Add the box name `b` to the `misplaced_boxes` list.
    6.  **Calculate Robot Cost (h_robot):**
        - Initialize `h_robot = 0`.
        - If the `misplaced_boxes` list is not empty:
            - Ensure `robot_loc` was found and is a valid location in the precomputed distances map. If not, return `math.inf`.
            - Find the minimum shortest path distance from `robot_loc` to the current location (`l_box`)
              of any box `b` listed in `misplaced_boxes`. Let this be `min_dist`.
            - Check for reachability: If `min_dist` is `math.inf`, the robot cannot statically reach any 
              of the boxes that need to be moved. Return `math.inf`.
            - Set `h_robot = min_dist`. This estimates the number of robot 'move' actions needed to 
              get adjacent to the nearest box that requires pushing.
    7.  **Combine Costs:** The final heuristic value is the sum of the estimated push costs and the estimated robot movement cost: `h = h_boxes + h_robot`.
    8.  **Return Value:** Return the calculated heuristic value `h`. It should be 0 only for goal states,
        `math.inf` for detected dead-ends or statically unreachable goals/boxes, and a positive integer otherwise,
        representing an estimate of the remaining actions.
    """

    def __init__(self, task):
        """Initializes the heuristic by precomputing distances and dead ends."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goals to find target locations for boxes
        self.goal_locations = {}  # {box_name: goal_location}
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Ensure it's an 'at' predicate with 3 parts, and the object is a box
            if parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                self.goal_locations[parts[1]] = parts[2]
        # Create a set of locations that are goals for efficient lookup
        goal_locs_set = set(self.goal_locations.values())

        # 2. Parse static facts ('adjacent') to build grid graph representations
        all_locations = set()
        # Adjacency list based on direction {loc: {direction: neighbor}}
        self.adj_dirs = {}
        # Simple adjacency list for BFS {loc: [neighbor1, neighbor2, ...]}
        self.simple_adj = {}

        for fact in static_facts:
            parts = get_parts(fact)
            # Process only 'adjacent' predicates with the correct number of arguments
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                # Add both locations to the set of all known locations
                all_locations.add(l1)
                all_locations.add(l2)

                # Populate adj_dirs (maps location -> direction -> neighbor)
                if l1 not in self.adj_dirs: self.adj_dirs[l1] = {}
                self.adj_dirs[l1][direction] = l2

                # Populate simple_adj (maps location -> list of neighbors)
                if l1 not in self.simple_adj: self.simple_adj[l1] = []
                # Avoid adding duplicate neighbors
                if l2 not in self.simple_adj[l1]:
                    self.simple_adj[l1].append(l2)
                
                # Sokoban domains typically define adjacency symmetrically (e.g., right implies left exists too).
                # We rely on the PDDL file providing both directions explicitly.

        # Ensure all identified locations exist as keys in the adjacency lists,
        # even if they are isolated (have no neighbors).
        self.all_locations = all_locations
        for loc in self.all_locations:
            if loc not in self.adj_dirs: self.adj_dirs[loc] = {}
            if loc not in self.simple_adj: self.simple_adj[loc] = []

        # 3. Precompute all-pairs shortest paths using BFS
        self.distances = {} # Stores {loc1: {loc2: distance}}
        for loc in self.all_locations:
            # Calculate distances from 'loc' to all other locations
            self.distances[loc] = bfs(loc, self.simple_adj)

        # 4. Precompute dead locations (simple static trap check)
        self.dead_locations = set()
        for loc in self.all_locations:
            # A location cannot be dead if it serves as a goal for any box
            if loc in goal_locs_set:
                continue

            # Get the neighbors of 'loc' organized by direction
            loc_neighbors_by_dir = self.adj_dirs.get(loc, {})
            has_neighbor_up = 'up' in loc_neighbors_by_dir
            has_neighbor_down = 'down' in loc_neighbors_by_dir
            has_neighbor_left = 'left' in loc_neighbors_by_dir
            has_neighbor_right = 'right' in loc_neighbors_by_dir

            # Check if pushing is blocked in opposite vertical directions
            stuck_vert = not has_neighbor_up and not has_neighbor_down
            # Check if pushing is blocked in opposite horizontal directions
            stuck_horiz = not has_neighbor_left and not has_neighbor_right

            # A location is considered dead if it's a non-goal corner trap.
            # Condition 1: Stuck vertically AND also blocked on at least one horizontal side.
            if stuck_vert and (not has_neighbor_left or not has_neighbor_right):
                self.dead_locations.add(loc)
            # Condition 2: Stuck horizontally AND also blocked on at least one vertical side.
            elif stuck_horiz and (not has_neighbor_up or not has_neighbor_down):
                 self.dead_locations.add(loc)
            # Note: This doesn't capture all deadlocks (like 2x2 patterns), only simple traps.


    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state

        # 1. Goal Check: The most reliable way to check for goal achievement.
        if self.goals <= state:
             return 0 # State is a goal state, cost-to-go is 0.

        # 2. Parse State: Find current locations of robot and boxes.
        robot_loc = None
        box_locations = {} # {box_name: current_location}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('box'):
                box_locations[parts[1]] = parts[2]

        # 3. Dead End Check: See if any box is in a precomputed dead location.
        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box)
            # If the box is in a dead location AND this location is not its goal
            if current_loc in self.dead_locations and current_loc != goal_loc:
                 return math.inf # This state leads to a dead end.

        # 4. Calculate Box Costs (h_boxes): Sum of distances for misplaced boxes.
        h_boxes = 0
        misplaced_boxes = [] # Keep track of boxes not at their goal
        for box, current_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box)
            # Process only if the box has a goal and is not currently there
            if goal_loc and current_loc != goal_loc:
                # Ensure locations are valid keys in the precomputed distances
                if current_loc not in self.distances or goal_loc not in self.distances[current_loc]:
                     # Indicates an issue like disconnected graph or invalid location
                     return math.inf 

                dist = self.distances[current_loc][goal_loc]
                
                # If the goal is statically unreachable from the current location
                if dist == math.inf:
                    return math.inf
                
                h_boxes += dist # Add distance (estimated pushes) to total
                misplaced_boxes.append(box) # Mark this box as needing attention

        # 5. Calculate Robot Cost (h_robot): Distance to nearest misplaced box.
        h_robot = 0
        if misplaced_boxes: # Only calculate if there are boxes to move
            # Check if robot location is valid
            if not robot_loc or robot_loc not in self.distances:
                 return math.inf # Robot location invalid or not found
                 
            min_dist_to_box = math.inf
            # Find the minimum distance from robot to any misplaced box
            for box in misplaced_boxes:
                box_loc = box_locations[box]
                # Check if the box location is reachable by the robot
                if box_loc not in self.distances[robot_loc]:
                     # Box is in a location statically unreachable by the robot
                     dist = math.inf 
                else:
                     dist = self.distances[robot_loc][box_loc]
                
                min_dist_to_box = min(min_dist_to_box, dist)

            # If robot cannot reach any misplaced box statically
            if min_dist_to_box == math.inf:
                return math.inf
            else:
                # Add the distance to the nearest misplaced box as the robot cost estimate
                h_robot = min_dist_to_box

        # 6. Combine Costs: Total heuristic is sum of box distances and robot distance.
        total_heuristic = h_boxes + h_robot
        
        # Final sanity check: heuristic should only be 0 for goal states.
        # The initial goal check handles this. If total_heuristic is 0 here,
        # it implies h_boxes=0 (all boxes at goals) and h_robot=0 (no misplaced boxes),
        # confirming a goal state, which should have already returned 0.
        
        return total_heuristic
