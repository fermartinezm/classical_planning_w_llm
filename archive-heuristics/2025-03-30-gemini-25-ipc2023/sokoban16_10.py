import math
from collections import deque
from fnmatch import fnmatch
# Assuming the heuristic base class is available in this path
# If the environment uses a different structure, adjust the import accordingly.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Simple fallback if the base class structure isn't strictly defined
    # or if running this script standalone for testing.
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban
    for use with greedy best-first search. It is designed to be informative
    but not necessarily admissible. It primarily calculates the sum of shortest
    path distances for each box from its current location to its designated
    goal location. It adds an estimate for the robot's movement cost, namely
    the shortest path distance from the robot to the location of the nearest
    misplaced box. The heuristic relies on precomputed shortest path distances
    between all locations based on the static map layout (ignoring dynamic
    obstacles like other boxes and the robot during path calculation). It also
    includes a basic static dead-end detection mechanism to identify states
    from which the goal is likely unreachable.

    # Assumptions
    - The main effort in Sokoban involves pushing boxes. Each push action has a cost of 1.
    - Robot movement is also costly (1 per move action) and necessary to position for pushes.
    - The heuristic estimates the total number of pushes by summing the shortest path distances (number of steps) for each box to its goal on an empty grid.
    - It estimates the initial robot movement cost by adding the shortest path distance from the robot's current location to the location of the *nearest* box that is not yet in its goal position.
    - Shortest paths between locations are precomputed using BFS on the static map connectivity defined by `adjacent` facts. Dynamic obstacles (robot, other boxes) are ignored for this path calculation.
    - The goal specifies target locations (`at box loc`) for a set of boxes. Only these boxes are considered by the heuristic.
    - Static Dead-End Detection: Locations are pre-identified as 'dead squares' if they are not goal locations and if, based on the static map structure, a box placed there cannot be pushed out along two perpendicular axes (e.g., it's in a corner or a narrow corridor with no exit perpendicular to the corridor). If a box is found in such a dead square (and it's not its goal), the state is assigned an infinite heuristic value.

    # Heuristic Initialization
    - Parses static facts (`adjacent`) to build graph representations (adjacency lists for forward and reverse connections) of the locations.
    - Identifies all unique locations from the adjacency facts.
    - Parses goal facts (`at box loc`) to create a map (`goal_map`) storing the target location for each relevant box. It also identifies the set of all goal locations.
    - Computes all-pairs shortest path distances between all identified locations using Breadth-First Search (BFS). Distances are stored in the `distances` dictionary. Unreachable pairs implicitly have infinite distance.
    - Performs static dead-end analysis: Identifies 'dead squares' based on map connectivity and whether they are goal locations. Stores these in the `dead_squares` set.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse State:** Extract the current location of the robot (`robot_loc`) and the current locations of all boxes defined in the `goal_map` from the input state's facts.
    2.  **Dead-End Check:** Iterate through the current box locations. If any box `b` is at a location `loc` such that `loc` is in the precomputed `dead_squares` set AND `loc` is not the `goal_map[b]`, return `math.inf` immediately, as the goal is deemed unreachable from this state.
    3.  **Calculate Box Costs:**
        - Initialize total heuristic value `h = 0`.
        - Initialize `min_robot_dist_to_misplaced_box = math.inf`.
        - Initialize `misplaced_boxes_count = 0`.
        - For each box `b` tracked in `goal_map`:
            - Get its current location `current_loc` and its target goal location `goal_loc = self.goal_map[b]`.
            - If `current_loc != goal_loc`:
                - Increment `misplaced_boxes_count`.
                - Retrieve the precomputed shortest path distance `box_dist = self.distances.get((current_loc, goal_loc), math.inf)`.
                - If `box_dist == math.inf` (goal is statically unreachable for this box), return `math.inf`.
                - Add `box_dist` to `h`. This estimates the minimum number of pushes for this box.
                - Retrieve the precomputed shortest path distance from the robot's location to the box's current location: `robot_dist = self.distances.get((robot_loc, current_loc), math.inf)`.
                - If `robot_dist != math.inf`, update `min_robot_dist_to_misplaced_box = min(min_robot_dist_to_misplaced_box, robot_dist)`.
    4.  **Calculate Robot Cost:**
        - If `misplaced_boxes_count > 0` (meaning at least one box needs to be moved):
            - If `min_robot_dist_to_misplaced_box == math.inf` (the robot cannot statically reach any of the misplaced boxes), return `math.inf`.
            - Otherwise, add `min_robot_dist_to_misplaced_box` to `h`. This estimates the cost for the robot to move to the nearest box requiring a push.
    5.  **Return Value:** Return the total calculated heuristic value `h`. A value of 0 indicates that all boxes considered in the goal are already at their target locations. Infinite values indicate detected dead ends or unreachable goals/boxes.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # --- Data Structures ---
        self.locations = set()
        self.boxes = set() # Boxes mentioned in the goal
        self.adj = {} # Adjacency list: loc -> {neighbor: direction, ...}
        self.rev_adj = {} # Reverse adjacency: loc -> {neighbor: direction, ...} (locations leading *to* loc)
        self.goal_map = {} # box -> goal_location
        self.distances = {} # (loc1, loc2) -> distance (int)
        self.dead_squares = set() # Set of non-goal locations identified as static dead ends

        # --- Parse Static Facts to build graph and identify locations ---
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                self.locations.add(l1)
                self.locations.add(l2)
                # Forward edge: l1 -> l2
                if l1 not in self.adj: self.adj[l1] = {}
                self.adj[l1][l2] = direction
                # Backward edge representation: l2 <- l1
                if l2 not in self.rev_adj: self.rev_adj[l2] = {}
                # Store the neighbor (l1) and the direction *of the forward edge* (l1->l2)
                self.rev_adj[l2][l1] = direction

        # --- Parse Goals to identify target boxes and locations ---
        goal_locations_set = set()
        for goal in self.goals:
            parts = get_parts(goal)
            # Only consider 'at' predicates for box goals
            if parts[0] == 'at' and len(parts) == 3:
                 # Assume format is (at <box_name> <location_name>)
                 box, loc = parts[1], parts[2]
                 # Check if the location is known from static facts
                 if loc in self.locations:
                     self.boxes.add(box)
                     self.goal_map[box] = loc
                     goal_locations_set.add(loc)
                 else:
                      # This might happen if a goal location is isolated or not part of the grid
                      # Or if location names in goal don't match objects/static facts.
                      print(f"Warning: Goal location '{loc}' for box '{box}' not found among locations defined by 'adjacent' facts. Ignoring this goal for heuristic calculation.")


        # --- Compute All-Pairs Shortest Paths using BFS ---
        # Initialize all distances implicitly to infinity
        for start_node in self.locations:
            self.distances[(start_node, start_node)] = 0
            q = deque([(start_node, 0)])
            # visited_in_run prevents cycles and redundant work within this BFS
            visited_in_run = {start_node}

            while q:
                curr_loc, dist = q.popleft()

                # Combine neighbors from both forward and backward connections for BFS exploration
                neighbors_to_explore = set()
                if curr_loc in self.adj:
                    neighbors_to_explore.update(self.adj[curr_loc].keys())
                if curr_loc in self.rev_adj:
                    neighbors_to_explore.update(self.rev_adj[curr_loc].keys())

                for neighbor in neighbors_to_explore:
                    if neighbor not in visited_in_run:
                        visited_in_run.add(neighbor)
                        new_dist = dist + 1
                        # Store the shortest distance found so far
                        if self.distances.get((start_node, neighbor), math.inf) > new_dist:
                             self.distances[(start_node, neighbor)] = new_dist
                        q.append((neighbor, new_dist))


        # --- Basic Static Dead Square Detection ---
        for loc in self.locations:
            if loc in goal_locations_set:
                continue # Goal squares are never dead ends by this definition

            # Check if a box at 'loc' can be pushed out vertically AND horizontally
            # based on the static map structure. Requires checking for paths of length 2.
            push_options = {'up': False, 'down': False, 'left': False, 'right': False}

            # Check structural possibility of pushing UP from loc
            # Need: loc_A (above loc), loc_AA (above loc_A)
            for loc_A, dir_A in self.rev_adj.get(loc, {}).items():
                # Check if loc_A is indeed above loc (connected by 'down' edge from loc_A)
                if self.adj.get(loc_A, {}).get(loc) == 'down':
                    # Check if there is a loc_AA above loc_A
                    for loc_AA, dir_AA in self.rev_adj.get(loc_A, {}).items():
                        if self.adj.get(loc_AA, {}).get(loc_A) == 'down':
                            push_options['up'] = True
                            break
                    if push_options['up']: break

            # Check structural possibility of pushing DOWN from loc
            # Need: loc_B (below loc), loc_BB (below loc_B)
            for loc_B, dir_B in self.adj.get(loc, {}).items():
                if dir_B == 'down':
                    # Check if there is a loc_BB below loc_B
                    for loc_BB, dir_BB in self.adj.get(loc_B, {}).items():
                        if dir_BB == 'down':
                            push_options['down'] = True
                            break
                    if push_options['down']: break

            # Check structural possibility of pushing LEFT from loc
            # Need: loc_L (left of loc), loc_LL (left of loc_L)
            for loc_L, dir_L in self.rev_adj.get(loc, {}).items():
                # Check if loc_L is indeed left of loc (connected by 'right' edge from loc_L)
                if self.adj.get(loc_L, {}).get(loc) == 'right':
                    # Check if there is a loc_LL left of loc_L
                    for loc_LL, dir_LL in self.rev_adj.get(loc_L, {}).items():
                        if self.adj.get(loc_LL, {}).get(loc_L) == 'right':
                            push_options['left'] = True
                            break
                    if push_options['left']: break

            # Check structural possibility of pushing RIGHT from loc
            # Need: loc_R (right of loc), loc_RR (right of loc_R)
            for loc_R, dir_R in self.adj.get(loc, {}).items():
                if dir_R == 'right':
                    # Check if there is a loc_RR right of loc_R
                    for loc_RR, dir_RR in self.adj.get(loc_R, {}).items():
                        if dir_RR == 'right':
                            push_options['right'] = True
                            break
                    if push_options['right']: break

            can_push_vert = push_options['up'] or push_options['down']
            can_push_horiz = push_options['left'] or push_options['right']

            # If a box at this non-goal location cannot be pushed both vertically AND horizontally
            # based on the static map structure, mark it as a dead square.
            if not (can_push_vert and can_push_horiz):
                 self.dead_squares.add(loc)


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # --- Parse current state ---
        robot_loc = None
        box_locs = {} # Track locations of boxes relevant to the goal

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at-robot':
                robot_loc = parts[1]
            elif predicate == 'at':
                # Only track boxes that are part of our goal_map
                box = parts[1]
                if box in self.goal_map:
                    loc = parts[2]
                    box_locs[box] = loc
            # 'clear' facts are not directly used in this heuristic calculation

        # --- Basic check if state is valid for heuristic calculation ---
        if robot_loc is None:
             # This indicates an invalid state or problem setup
             # print("Error: Robot location not found in state.")
             return math.inf # Cannot evaluate heuristic

        # Check if all goal boxes have a location in the current state
        # This might not be strictly necessary if planner handles partial states,
        # but good for robustness.
        if len(box_locs) != len(self.goal_map):
             # print(f"Warning: Mismatch between goal boxes ({len(self.goal_map)}) and boxes found in state ({len(box_locs)}).")
             # Decide how to handle: return inf, or calculate based on found boxes?
             # Let's calculate based on found boxes, assuming missing ones might appear later.
             pass


        h = 0
        min_robot_dist_to_misplaced_box = math.inf
        misplaced_boxes_count = 0

        # --- Calculate heuristic value ---
        for box, current_loc in box_locs.items():
            goal_loc = self.goal_map.get(box) # Should exist if box is in box_locs

            # 1. Dead-End Check: Is the box in a static dead square?
            if current_loc in self.dead_squares and current_loc != goal_loc:
                # print(f"DEBUG: Box {box} at dead square {current_loc} (goal: {goal_loc})")
                return math.inf # Goal likely unreachable

            # 2. Calculate distance if box is misplaced
            if current_loc != goal_loc:
                misplaced_boxes_count += 1

                # Get precomputed distance for the box to its goal
                box_dist = self.distances.get((current_loc, goal_loc), math.inf)

                if box_dist == math.inf: # Goal is statically unreachable for this box
                    # print(f"DEBUG: Goal {goal_loc} statically unreachable for box {box} from {current_loc}")
                    return math.inf

                # Add estimated pushes for this box
                h += box_dist

                # Find distance from robot to this misplaced box's location
                robot_dist = self.distances.get((robot_loc, current_loc), math.inf)

                # Update the minimum distance to a misplaced box if the robot can reach it
                if robot_dist != math.inf:
                    min_robot_dist_to_misplaced_box = min(min_robot_dist_to_misplaced_box, robot_dist)


        # 3. Add robot travel cost estimate if needed
        if misplaced_boxes_count > 0:
            # If min_robot_dist is still infinity, the robot cannot reach *any* misplaced box
            if min_robot_dist_to_misplaced_box == math.inf:
                 # print(f"DEBUG: Robot at {robot_loc} cannot statically reach any misplaced box.")
                 return math.inf # Problem seems unsolvable from this state
            else:
                 # Add the cost for the robot to get to the nearest misplaced box
                 h += min_robot_dist_to_misplaced_box

        # Return the final heuristic value
        # h=0 implies all boxes are at their goals (goal state reached)
        return h
