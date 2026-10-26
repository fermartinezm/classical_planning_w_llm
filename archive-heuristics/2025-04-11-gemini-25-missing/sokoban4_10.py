import sys
import collections

# Ensure the base class can be imported. If the environment setup assumes
# it's in a specific directory structure (like 'heuristics/heuristic_base.py'),
# this import should work directly. Otherwise, adjust sys.path if needed.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails (e.g., for standalone testing)
    # This allows the code to be syntactically correct even if the base class isn't found.
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper function to parse PDDL facts like '(predicate arg1 arg2)'
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: '(at box1 loc_1_1)' -> ['at', 'box1', 'loc_1_1']
    Returns an empty list if the fact is malformed or empty after stripping.
    """
    stripped_fact = fact.strip()
    if len(stripped_fact) < 2 or stripped_fact[0] != '(' or stripped_fact[-1] != ')':
        return [] # Malformed fact
    return stripped_fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban by summing
    two components:
    1. The total shortest path distance for each box from its current location to its
       goal location, calculated on the empty grid graph (ignoring other boxes/robot).
       This estimates the minimum number of pushes required.
    2. The shortest path distance for the robot to reach a "useful" position. A useful
       position is one from which it can immediately push *any* misplaced box one step
       closer towards its goal. If no such ideal "push spot" is reachable, it uses
       the distance to the nearest clear location adjacent to any misplaced box as a fallback.

    The heuristic aims to provide good guidance for a greedy best-first search by
    prioritizing states where boxes are closer to their goals and the robot is
    positioned to make progress. It is not guaranteed to be admissible.

    # Assumptions
    - The primary cost driver is moving boxes (pushes). The heuristic approximates
      the number of pushes using empty-grid distances.
    - Robot movement cost is estimated based on reaching the nearest position
      that enables a productive push.
    - The map is defined by static 'adjacent' predicates.
    - Goals are specified as `(at boxN locM)` for specific boxes.
    - Complex dead-end situations (e.g., boxes blocking each other irrevocably)
      are not explicitly detected beyond basic graph reachability.

    # Heuristic Initialization
    - Parses the task's goal conditions `(at box loc)` to create a mapping
      `self.goal_locations` from each goal box to its target location.
    - Parses the task's static 'adjacent' facts to build:
        - `self.graph`: An undirected adjacency list representing the map layout
          (ignoring push/move mechanics, just connectivity).
        - `self.adj`: A directed adjacency dictionary storing the direction between
          adjacent locations `{loc1: {loc2: direction}}`.
    - Precomputes all-pairs shortest paths (APSP) on the empty location graph
      using Breadth-First Search (BFS) starting from each location. Results are
      stored in `self.distances[start_loc][end_loc]`. This gives the minimum
      number of steps (moves or pushes on an empty grid) between any two locations.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:**
        - Extract the robot's current location (`robot_loc`).
        - Extract the current location of each box relevant to the goal (`box_locations`).
        - Store all `(clear loc)` facts for quick lookup (`clear_location_facts`).
    2.  **Calculate Total Box Distance (`box_heuristic`):**
        - Initialize `box_heuristic = 0`.
        - Create a list `misplaced_boxes` to track boxes not at their goal.
        - For each box `b` tracked in `box_locations`:
            - Get its current location `box_loc` and target `goal_loc`.
            - If `box_loc != goal_loc`:
                - Retrieve the precomputed shortest path distance `d = self.distances[box_loc].get(goal_loc, float('inf'))`.
                - If `d` is infinity, the goal is fundamentally unreachable on the grid -> return infinity (state is likely unsolvable).
                - Add `d` to `box_heuristic`.
                - Add box `b` to `misplaced_boxes`.
    3.  **Check for Goal State:**
        - If `misplaced_boxes` is empty, all goal boxes are in their target locations. Return 0.
    4.  **Calculate Minimum Robot Distance (`min_robot_dist`):**
        - Initialize `min_robot_dist = float('inf')`.
        - Get the set of locations currently occupied by any box (`current_obstacle_locations`).
        - **Find Best Push Spot:** Iterate through each `box` in `misplaced_boxes`:
            - Find its `box_loc` and `goal_loc`.
            - Calculate `current_box_dist_to_goal`.
            - Iterate through locations `p` that are adjacent to `box_loc` (these are potential push spots).
            - For each `p`, determine the direction `dir` required to push the box *from* `box_loc` *towards* a `target_loc`. This `dir` is the same as the direction from `p` to `box_loc`.
            - Find the `target_loc` that is adjacent to `box_loc` in this direction `dir`.
            - If `target_loc` exists and is closer to the goal (`distances[target_loc][goal_loc] < current_box_dist_to_goal`):
                - Then `p` is a valid "push spot" for making progress on this box.
                - Check if `p` is currently occupied by another box. If yes, this spot is unusable right now, so skip it.
                - Calculate the shortest path distance for the robot from `robot_loc` to `p` using BFS (`_robot_bfs`), treating other boxes (`current_obstacle_locations - {box_loc}`) as obstacles. Let this be `robot_dist`.
                - Update `min_robot_dist = min(min_robot_dist, robot_dist)`. This finds the minimum distance to *any* productive push spot for *any* misplaced box.
    5.  **Handle Robot Unreachability (Fallback):**
        - If after checking all boxes, `min_robot_dist` is still infinity (meaning no ideal push spot is reachable, perhaps because they are all blocked or the robot is trapped):
            - Calculate the minimum robot distance to any *clear* location that is adjacent to *any* misplaced box. This represents the cost to get next to a box, even if not perfectly positioned for the best push.
            - Iterate through `misplaced_boxes` and their neighbors `n`.
            - If `n` is clear (check against `clear_location_facts`), calculate the robot BFS distance to `n` (avoiding all box locations).
            - Update `min_robot_dist` with the minimum distance found in this fallback step.
            - If `min_robot_dist` *still* remains infinity, the robot appears completely unable to reach any useful position near a misplaced box -> return infinity.
    6.  **Combine Heuristics:**
        - The final heuristic value is `box_heuristic + min_robot_dist`.
        - Ensure the returned value is non-negative (it should be, barring calculation errors). Return infinity if calculated as such.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        self.adj = {} # Stores {loc1: {loc2: direction}}
        self.graph = collections.defaultdict(set) # Undirected graph {loc: {neighbor1, ...}}

        # 1. Parse Goals to find target locations for boxes
        self.goal_locations = {}
        for fact in self.goals:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Basic check: Assume the first argument of a goal 'at' is a box
                self.goal_locations[box] = loc

        # 2. Build Graph from Static Adjacency Facts
        locations = set()
        adj_tuples = [] # Store (l1, l2, dir) to process after finding all locations
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                locations.add(l1)
                locations.add(l2)
                adj_tuples.append((l1, l2, direction))

        # Populate graph (undirected) and directed adjacency map
        for l1, l2, direction in adj_tuples:
            self.graph[l1].add(l2)
            # Assuming adjacency might only be listed once, add reverse for undirected graph
            self.graph[l2].add(l1)
            # Store directed adjacency for push direction logic
            if l1 not in self.adj: self.adj[l1] = {}
            self.adj[l1][l2] = direction

        # 3. Precompute All-Pairs Shortest Paths (APSP) using BFS
        self.distances = collections.defaultdict(lambda: collections.defaultdict(lambda: float('inf')))
        all_locs = list(locations) # Use a fixed list of all known locations

        for start_node in all_locs:
            if start_node not in locations: continue # Skip if somehow not in the set
            self.distances[start_node][start_node] = 0

            queue = collections.deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                curr_loc, dist = queue.popleft()

                # Use the undirected graph for pathfinding
                for neighbor in self.graph.get(curr_loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        # Ensure neighbor is a known location before assigning distance
                        if neighbor in locations:
                            self.distances[start_node][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))


    def _robot_bfs(self, start_loc, end_loc, obstacles):
        """
        Helper BFS: Finds shortest path distance for robot from start_loc to end_loc,
        avoiding locations in the obstacles set. Uses the undirected self.graph.
        Returns float('inf') if end_loc is unreachable or invalid.
        """
        if start_loc == end_loc:
            return 0
        # Check if start/end locations are known and valid
        if start_loc not in self.graph or end_loc not in self.graph:
             return float('inf')
        if start_loc in obstacles:
            return float('inf') # Cannot start in an obstacle

        queue = collections.deque([(start_loc, 0)])
        visited = set(obstacles) | {start_loc} # Avoid obstacles and already visited

        while queue:
            curr_loc, dist = queue.popleft()

            for neighbor in self.graph.get(curr_loc, set()):
                if neighbor == end_loc:
                    return dist + 1
                # Check if neighbor is valid and not visited/obstacle
                if neighbor in self.graph and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf') # End location not reachable

    def __call__(self, node):
        state = node.state
        robot_loc = None
        box_locations = {} # {box_name: location}
        clear_location_facts = set() # Store raw '(clear loc)' facts

        # Parse current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Track location only if obj is a box relevant to the goal
                if obj in self.goal_locations:
                     box_locations[obj] = loc
            elif parts[0] == 'clear' and len(parts) == 2:
                 clear_location_facts.add(fact)

        if robot_loc is None or robot_loc not in self.graph:
             # Invalid state if robot location unknown or not on the map
             return float('inf')

        box_heuristic = 0
        misplaced_boxes = []
        current_obstacle_locations = set(box_locations.values()) # Locations occupied by boxes

        # Calculate sum of box-to-goal distances
        for box, box_loc in box_locations.items():
            goal_loc = self.goal_locations.get(box) # Should exist

            # Check validity of locations before accessing distances
            if box_loc not in self.distances or not goal_loc or goal_loc not in self.distances:
                 return float('inf') # Cannot compute heuristic if locations are unknown/invalid

            if box_loc != goal_loc:
                # Use precomputed distance, default to infinity if somehow missing
                dist = self.distances[box_loc].get(goal_loc, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal is fundamentally unreachable
                box_heuristic += dist
                misplaced_boxes.append(box)

        # If all boxes are in place, heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate minimum robot distance to a valid push spot
        min_robot_dist = float('inf')

        for box in misplaced_boxes:
            box_loc = box_locations[box]
            goal_loc = self.goal_locations[box]
            current_box_dist_to_goal = self.distances[box_loc].get(goal_loc, float('inf'))

            if current_box_dist_to_goal == float('inf'): continue # Should be caught

            # Find potential push spots 'p' adjacent to 'box_loc'
            # Iterate through neighbors 'p' of box_loc in the undirected graph
            for push_spot in self.graph.get(box_loc, set()):
                # Check if push_spot is a valid location on the map
                if push_spot not in self.graph: continue

                # Find direction from push_spot to box_loc using the directed adj map
                direction = self.adj.get(push_spot, {}).get(box_loc)
                if not direction: continue # Need direction info for push logic

                # Find the target location 'target_loc' when pushing box from box_loc in 'direction'
                target_loc = None
                if box_loc in self.adj:
                    for potential_target, dir_from_box in self.adj[box_loc].items():
                        if dir_from_box == direction:
                            target_loc = potential_target
                            break

                # Check if target location is valid and makes progress
                if target_loc and target_loc in self.distances:
                    target_dist_to_goal = self.distances[target_loc].get(goal_loc, float('inf'))

                    if target_dist_to_goal < current_box_dist_to_goal:
                        # 'push_spot' is valid for making progress.
                        # Check if push_spot is blocked by another box.
                        if push_spot in current_obstacle_locations:
                             continue # Cannot use push_spot if occupied

                        # Calculate robot path cost to this push_spot
                        obstacles_for_robot = current_obstacle_locations - {box_loc}
                        robot_dist = self._robot_bfs(robot_loc, push_spot, obstacles_for_robot)
                        min_robot_dist = min(min_robot_dist, robot_dist)


        # Fallback if no ideal push spot is reachable
        if min_robot_dist == float('inf'):
             min_dist_to_clear_adj = float('inf')
             for box in misplaced_boxes:
                 box_loc = box_locations[box]
                 if box_loc not in self.graph: continue # Box location invalid

                 for neighbor in self.graph.get(box_loc, set()):
                     # Check if neighbor is clear using the '(clear loc)' fact
                     if f'(clear {neighbor})' in clear_location_facts:
                         # Calculate distance to this clear adjacent spot
                         # Obstacles are all current box locations
                         obstacles_for_robot = current_obstacle_locations
                         dist = self._robot_bfs(robot_loc, neighbor, obstacles_for_robot)
                         min_dist_to_clear_adj = min(min_dist_to_clear_adj, dist)

             if min_dist_to_clear_adj == float('inf'):
                 # Robot seems truly stuck or cannot reach any useful adjacent spot
                 return float('inf')
             else:
                 min_robot_dist = min_dist_to_clear_adj

        # Final heuristic: sum of box distances + min robot distance to enable a push
        final_h = box_heuristic + min_robot_dist
        # Ensure heuristic is non-negative, return infinity if calculated as such
        return max(0.0, final_h) if final_h != float('inf') else float('inf')

