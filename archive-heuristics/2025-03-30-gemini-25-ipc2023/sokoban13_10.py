import math
from collections import deque
# Assuming the heuristic base class is available at this path
# If the environment uses a different path, this import needs adjustment.
from heuristics.heuristic_base import Heuristic 

# Helper function outside the class for parsing PDDL fact strings
def get_parts(fact):
    """
    Parses a PDDL fact string like '(predicate obj1 obj2 ...)' into a list of strings.
    Removes parentheses and splits by whitespace. Handles potential extra whitespace.
    """
    return fact.strip()[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban by summing two components:
    1. The sum of shortest path distances for each box from its current location to its goal location.
       This estimates the minimum number of push actions required. Distances are calculated on the grid
       ignoring other dynamic objects (boxes, robot).
    2. The shortest path distance for the robot to reach the required 'push position' for the *nearest*
       misplaced box. The push position is the location the robot must occupy to push a box one step
       along its shortest path towards the goal. This estimates the robot's movement cost to initiate
       the pushing process.

    The heuristic is designed for Greedy Best-First Search and is not necessarily admissible. It aims
    to provide an informative estimate to guide the search effectively.

    # Assumptions
    - The grid layout (locations and their connectivity) is static, defined by `adjacent` predicates
      in the PDDL file. These are preprocessed in the constructor.
    - Goals are specified as `(at boxName goalLocation)` predicates. Only these goals are considered.
    - Box names follow the convention `box...` (e.g., `box1`, `box_a`). This assumption is used
      to identify boxes in the goal predicates and state facts. If box names differ, this logic
      might need adjustment.
    - The shortest path distance on an empty grid (ignoring other boxes/robot) is a reasonable proxy
      for the minimum number of pushes required for a box.
    - The cost of robot movement needed to start pushing is approximated by the distance to the
      push position associated with the misplaced box that is 'easiest' for the robot to reach.
    - The heuristic does not perform dead-end detection. States where boxes are trapped in
      non-goal locations (due to static walls or corners) might receive finite heuristic values,
      potentially misleading the search in complex maps. Detecting dead ends would require more
      complex static map analysis.

    # Heuristic Initialization (`__init__`)
    - Parses all static `adjacent` facts from the `task.static` information.
    - Builds an undirected graph representing location connectivity (`undirected_adj`) used for pathfinding.
    - Stores directional adjacency information (`self.adj_directional`, `self.adj_reverse`)
      to determine push directions and the required robot positions ('push positions') relative to boxes.
    - Computes all-pairs shortest path distances (`self.distances`) and predecessors
      (`self.predecessors`) between all connected locations using Breadth-First Search (BFS)
      on the undirected graph. This allows for efficient O(1) lookups during heuristic evaluation.
    - Parses the `task.goals` to identify target locations for each box (`self.goal_locations`)
      and compiles a set of all goal-relevant boxes (`self.boxes`).
    - Stores the set of all unique locations identified from the `adjacent` facts (`self.locations`).

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse State:** Extract the current location of the robot (`robot_loc`) and the
        location of each goal-relevant box (`box_locations`) from the input `node.state` (a frozenset of facts).
    2.  **Validity Checks:** Ensure the robot exists (`robot_loc` is not None) and is on a known map
        location (`robot_loc` in `self.locations`). If checks fail, return infinity (state is invalid or unsolvable).
    3.  **Goal Check:** Verify if all boxes listed in `self.goal_locations` are currently
        at their respective goal locations by comparing `box_locations` with `self.goal_locations`.
        If the state satisfies all goals, return 0.0.
    4.  **Initialize Components:** Set `h_box_goals = 0.0` (to accumulate box-to-goal distances) and
        `min_dist_robot_to_push_pos = float('inf')` (to track the minimum robot travel needed).
        Set a flag `misplaced_boxes_exist = False`.
    5.  **Iterate Over Boxes:** For each box `b` with a goal location `gloc` defined in `self.goal_locations`:
        a.  Get the box's current location `bloc` from `box_locations`. Perform validity checks: ensure the
            box exists in the state and its location `bloc` is a known map location. If not, return infinity.
        b.  If `bloc != gloc` (the box is misplaced):
            i.   Mark `misplaced_boxes_exist = True`.
            ii.  **Box Distance Component:** Retrieve the precomputed shortest distance
                 `dist_box_goal = self.distances.get((bloc, gloc), float('inf'))`.
                 If this distance is infinity, the goal is physically unreachable for this box; return infinity.
                 Add `dist_box_goal` to `h_box_goals`.
            iii. **Robot Distance Component:**
                 - Determine the required robot position (`push_pos`) to push box `b` from `bloc`
                   one step towards `gloc`. This involves:
                     - Getting the predecessor map for paths starting at `bloc`: `pred_map = self.predecessors.get(bloc)`.
                     - Finding the next location (`next_box_loc`) on the shortest path from `bloc` to `gloc`
                       using the helper method `self._get_first_step(bloc, gloc, pred_map)`.
                     - If `next_box_loc` is found, determine the `push_direction` from `bloc` to `next_box_loc`
                       by checking `self.adj_directional`.
                     - If `push_direction` is found, find the `push_pos` using the `self.adj_reverse` map.
                       `push_pos` is the location `l` such that `adjacent(l, bloc, push_direction)` holds.
                 - Determine the `target_loc_for_robot`: Use `push_pos` if it was successfully found and is a valid
                   location on the map. Otherwise, fallback to using the box's current location `bloc` as the target.
                 - Calculate the robot's shortest distance to this target:
                   `dist_robot = self.distances.get((robot_loc, target_loc_for_robot), float('inf'))`.
                 - Update the minimum robot distance found so far:
                   `min_dist_robot_to_push_pos = min(min_dist_robot_to_push_pos, dist_robot)`.
    6.  **Combine Components and Return:**
        a.  If no boxes were found to be misplaced (`misplaced_boxes_exist` is False), the state must be a goal
            (this case should ideally be caught by the initial goal check); return 0.0.
        b.  If `min_dist_robot_to_push_pos` remains infinity, it means the robot cannot reach any valid
            position to start pushing any of the misplaced boxes; return infinity (unsolvable state).
        c.  The final heuristic value is the sum of the estimated pushes and the estimated robot positioning cost:
            `h = h_box_goals + min_dist_robot_to_push_pos`.
    7.  **Return Value:** Return the calculated heuristic value `h` as a float, ensuring it's non-negative.
    """

    def __init__(self, task):
        """Initialize the heuristic by precomputing distances and goal info."""
        self.goals = task.goals
        static_facts = task.static

        self.locations = set()
        undirected_adj = {}
        self.adj_directional = {} # Map: (location, direction) -> adjacent_location
        self.adj_reverse = {}     # Map: (location, direction) -> source_location such that adjacent(source_location, location, direction)

        # Process static adjacency facts to build graph representations
        for fact in static_facts:
            parts = get_parts(fact)
            # Ensure the fact is an 'adjacent' predicate with correct arity
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.locations.add(l1)
                self.locations.add(l2)
                # Build undirected graph for BFS pathfinding
                undirected_adj.setdefault(l1, set()).add(l2)
                undirected_adj.setdefault(l2, set()).add(l1)
                # Store directional adjacency for push logic
                self.adj_directional[(l1, direction)] = l2
                # Store reverse adjacency: given l2 and direction, find l1
                self.adj_reverse[(l2, direction)] = l1

        # Compute all-pairs shortest paths and predecessors using BFS
        self.distances = {} # Stores shortest path distances: (loc1, loc2) -> distance
        self.predecessors = {} # Stores predecessor maps: start_loc -> {loc -> predecessor_on_path_from_start}
        
        if not self.locations:
             print("Warning: No locations found from 'adjacent' predicates. Heuristic may not work.")
             # Handle case with no locations if necessary, e.g., by always returning 0 or infinity.

        for start_loc in self.locations:
            # Initialize distances and predecessors for BFS starting from start_loc
            dist = {loc: float('inf') for loc in self.locations}
            pred = {loc: None for loc in self.locations}
            dist[start_loc] = 0
            self.distances[(start_loc, start_loc)] = 0 # Distance to self is 0
            
            q = deque([start_loc])
            visited = {start_loc} # Keep track of visited nodes during this BFS run

            while q:
                curr_loc = q.popleft()
                current_distance = dist[curr_loc]

                # Explore neighbors in the undirected graph
                for neighbor in undirected_adj.get(curr_loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = current_distance + 1
                        dist[neighbor] = new_dist
                        pred[neighbor] = curr_loc # Record predecessor for path reconstruction
                        # Store distance from start_loc to neighbor
                        self.distances[(start_loc, neighbor)] = new_dist
                        q.append(neighbor)
            
            # Store the computed predecessor map for paths originating from start_loc
            self.predecessors[start_loc] = pred

        # Store goal locations for boxes by parsing goal facts
        self.goal_locations = {} # Map: box_name -> goal_location
        self.boxes = set()       # Set of box names involved in goals
        for goal in self.goals:
            parts = get_parts(goal)
            # Check for goal predicates like (at boxName locationName)
            if parts[0] == 'at' and len(parts) == 3:
                 # Use name convention to identify boxes (e.g., starts with 'box')
                 # This might need adjustment based on actual PDDL naming.
                 if parts[1].startswith('box'):
                    box, loc = parts[1], parts[2]
                    # Check if the goal location is valid according to the map
                    if loc not in self.locations:
                         # Log a warning if a goal location isn't part of the connected map
                         print(f"Warning: Goal location '{loc}' for box '{box}' is not in the set of known locations derived from 'adjacent' facts. This goal might be unreachable.")
                    self.goal_locations[box] = loc
                    self.boxes.add(box)

    def _get_first_step(self, start, end, pred_map):
        """
        Helper method to find the first node after 'start' on a shortest path to 'end',
        using the precomputed predecessor map 'pred_map' (generated by BFS starting at 'start').
        Returns the next node's name, or None if start == end, path doesn't exist, or map is invalid.
        """
        # Basic cases where no step is needed or possible
        if start == end or pred_map is None or end not in pred_map:
            return None
        
        # Check reachability: if 'end' has no predecessor in this map (and isn't 'start'), it's unreachable
        if pred_map.get(end) is None and start != end:
             # This check might be redundant if distances are handled correctly, but adds safety
             return None 

        # Trace the path backwards from 'end' using the predecessor map
        curr = end
        # Keep moving backwards until the node whose predecessor is 'start' is found
        while pred_map.get(curr) is not None and pred_map[curr] != start:
            curr = pred_map[curr]
            # Safety check: detect potential cycles or errors if we loop back to 'end'
            if curr == end: return None 

        # After the loop:
        # If pred_map[curr] == start, then 'curr' is the first node visited after 'start' on the path to 'end'.
        if pred_map.get(curr) == start:
            return curr 
        else:
            # This case implies start == end, or 'end' was unreachable from 'start' via this predecessor map.
            return None

    def __call__(self, node):
        """Calculate the heuristic value for the given state represented by the search node."""
        state = node.state # The state is a frozenset of PDDL fact strings

        # 1. Parse current state to find robot and box locations
        robot_loc = None
        box_locations = {} # Stores current location for boxes relevant to goals
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                 # Check if this fact involves a box we care about (i.e., one with a goal)
                 if parts[1] in self.boxes:
                    box_locations[parts[1]] = parts[2]

        # 2. Perform basic validity checks on the state
        if not self.goal_locations: return 0.0 # No goals defined, trivially solved
        if robot_loc is None: return float('inf') # Robot is essential, state invalid/unsolvable if missing
        if robot_loc not in self.locations: return float('inf') # Robot is off the map, unsolvable

        # 3. Check if the current state is a goal state
        goal_reached = True
        for box, gloc in self.goal_locations.items():
            # Check if the box exists in the state and is at its goal location
            if box_locations.get(box) != gloc:
                goal_reached = False
                break
        if goal_reached:
            return 0.0 # State is a goal, cost-to-go is 0

        # 4. Calculate heuristic components: box distances and robot distance
        h_box_goals = 0.0 # Accumulator for sum of box-to-goal distances
        min_dist_robot_to_push_pos = float('inf') # Tracks min robot distance to initiate a push
        misplaced_boxes_exist = False # Flag to check if any work needs to be done

        for box, gloc in self.goal_locations.items():
            bloc = box_locations.get(box)
            # Check if the box's location is known and valid
            if bloc is None: return float('inf') # Box mentioned in goal is missing from state
            if bloc not in self.locations: return float('inf') # Box is somehow off the valid map locations

            # Process only if the box is not already at its goal
            if bloc != gloc:
                misplaced_boxes_exist = True

                # --- Calculate Box Distance Component ---
                dist_box_goal = self.distances.get((bloc, gloc), float('inf'))
                # If distance is infinite, the goal location is unreachable for this box
                if dist_box_goal == float('inf'):
                    return float('inf') # State is unsolvable if any box cannot reach its goal
                h_box_goals += dist_box_goal

                # --- Calculate Robot Distance Component ---
                push_pos = None # The location the robot needs to be to push this box
                pred_map = self.predecessors.get(bloc) # Get predecessors for paths starting from bloc

                # Find the next step for the box towards its goal using the shortest path
                next_box_loc = self._get_first_step(bloc, gloc, pred_map)

                if next_box_loc:
                    # Determine the direction of the push from bloc to next_box_loc
                    push_direction = None
                    # Check standard directions (assuming they are named 'up', 'down', 'left', 'right')
                    for direction_candidate in ['up', 'down', 'left', 'right']:
                        if self.adj_directional.get((bloc, direction_candidate)) == next_box_loc:
                            push_direction = direction_candidate
                            break
                    
                    if push_direction:
                        # Find the required robot position (push_pos) using the reverse map
                        # push_pos is the location l such that adjacent(l, bloc, push_direction) holds
                        push_pos = self.adj_reverse.get((bloc, push_direction))

                # Determine the target location for the robot for this specific box push
                # Use push_pos if it was successfully found and is a valid location, otherwise fallback to the box's location
                target_loc_for_robot = push_pos if push_pos and push_pos in self.locations else bloc
                
                # Calculate the robot's shortest distance to this target location
                dist_robot = self.distances.get((robot_loc, target_loc_for_robot), float('inf'))
                
                # Update the minimum robot distance needed across all misplaced boxes
                min_dist_robot_to_push_pos = min(min_dist_robot_to_push_pos, dist_robot)

        # 5. Combine heuristic components and return the final value
        # If no boxes were misplaced, goal should have been detected earlier. Safety check.
        if not misplaced_boxes_exist: 
             return 0.0 

        # If the robot cannot reach any position to push any of the misplaced boxes, the state is unsolvable
        if min_dist_robot_to_push_pos == float('inf'):
             return float('inf')

        # Total heuristic value is the sum of estimated pushes + estimated robot positioning cost
        heuristic_value = h_box_goals + min_dist_robot_to_push_pos
        
        # Ensure the heuristic returns a non-negative float value
        return max(0.0, heuristic_value)

