import sys
from collections import deque
# Assuming the Heuristic base class is available in the environment.
# If not, a simple placeholder can be used for standalone testing:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): raise NotImplementedError
from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL fact strings like "(predicate obj1 obj2)"
def get_parts(fact):
    """Removes parentheses and splits the fact string into parts."""
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    Estimates the cost to reach the goal state in Sokoban. The heuristic value is
    calculated as the sum of two components:
    1. The sum of shortest path distances for each box from its current location
       to its designated goal location (estimated number of pushes).
    2. The shortest path distance for the robot to reach the location of the
       nearest misplaced box (estimated number of robot moves to initiate the first push).

    This heuristic aims to guide a greedy best-first search by prioritizing states
    where boxes are closer to their goals and the robot is closer to a box that
    needs moving. It is not necessarily admissible.

    # Assumptions
    - The primary cost contributor is moving boxes to their goal locations (pushes).
    - Robot movement cost is approximated by the distance to the nearest misplaced box's location.
    - Box paths are calculated using Breadth-First Search (BFS) on the static location
      graph (defined by 'adjacent' predicates), ignoring dynamic obstacles like the
      robot and other boxes. This simplification might underestimate the true path
      length if dynamic obstacles block the way, but keeps the calculation efficient.
    - Robot paths consider locations currently occupied by other boxes as obstacles.
    - The location connectivity is defined solely by the static 'adjacent' predicates.
    - Goals consist primarily of '(at boxX locY)' predicates, specifying a unique
      target location for each relevant box. Other types of goal predicates are
      currently ignored by this heuristic's calculation.
    - Box object names contain the substring 'box' (e.g., 'box1', 'mybox'). This is used
      to distinguish boxes from other objects like locations.

    # Heuristic Initialization
    - Parses static 'adjacent' facts from `task.static` to build an adjacency graph
      (`self.adj`) representing the traversable grid/map. This graph stores directed
      edges based on the `adjacent` predicates.
    - Identifies all unique locations mentioned in the 'adjacent' facts (`self.locations`).
    - Parses goal facts '(at boxX locY)' from `task.goals` to create a map
      (`self.goal_map`) storing the target location for each box involved in the goal.
    - Identifies the set of all box objects (`self.boxes`) based on goal conditions
      and initial state predicates, using the naming convention assumption ('box' in name).

    # Step-By-Step Thinking for Computing Heuristic
    1.  Parse the current `state` (a set of PDDL fact strings) to find:
        - The robot's current location (`robot_loc`).
        - The current location of each box (`box_locs` map: box -> location).
        - The set of all locations currently occupied by any box (`current_box_positions`).
    2.  If the robot's location cannot be determined, return infinity (invalid state).
    3.  Initialize `total_heuristic_value = 0`.
    4.  Initialize `min_robot_dist_to_box = float('inf')` (tracks distance to nearest misplaced box).
    5.  Set `misplaced_boxes_exist = False` (flag to check if any box needs moving).
    6.  Iterate through each box `b` identified during initialization (`self.boxes`):
        a.  Retrieve its current location `bloc` from `box_locs`. If a box is unexpectedly
            missing from the state, return infinity.
        b.  Retrieve its designated goal location `gloc` from `self.goal_map`.
        c.  If `gloc` exists (i.e., this box has a target location) and `bloc != gloc`:
            i.   Mark `misplaced_boxes_exist = True`.
            ii.  Calculate `box_dist`: the shortest path distance for the box from `bloc`
                 to `gloc` using BFS (`self.bfs`) on the static graph (`self.adj`).
                 Obstacles are ignored for this box path calculation (pass `obstacles=set()`).
            iii. If `box_dist` is infinity, the goal is topologically unreachable for this box
                 from its current position (indicates a dead end on the static map). Return
                 `float('inf')` immediately.
            iv.  Add `box_dist` to `total_heuristic_value`. This estimates the pushes needed for this box.
            v.   Calculate `robot_dist`: the shortest path distance for the robot from
                 `robot_loc` to the box's current location `bloc`. The BFS for the robot
                 considers the locations of *other* boxes (`current_box_positions - {bloc}`)
                 as obstacles.
            vi.  Update `min_robot_dist_to_box = min(min_robot_dist_to_box, robot_dist)`.
                 This tracks the minimum distance the robot needs to travel to get to *any*
                 misplaced box's location.
    7.  After checking all boxes:
        a.  If `misplaced_boxes_exist` is `False`:
            - Check if the current state satisfies all goal conditions (`self.goals <= state`).
            - If yes, return 0 (goal state reached).
            - If no, return 1 (not the goal state, but all boxes are placed or have no goals;
              return a small positive value to indicate it's not the goal).
        b.  If `misplaced_boxes_exist` is `True`:
            i.   If `min_robot_dist_to_box` is infinity, it means the robot cannot reach
                 any of the misplaced boxes (e.g., trapped or blocked by static obstacles
                 or other boxes). Return `float('inf')` (unsolvable state from here).
            ii.  Otherwise, add `min_robot_dist_to_box` to `total_heuristic_value`. This
                 incorporates the estimated robot movement cost.
            iii. Return `max(1, total_heuristic_value)`. This ensures the heuristic is
                 always positive if the state is not a goal state (since `misplaced_boxes_exist`
                 is True).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information (adjacency)
        and goal conditions (box target locations).
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify all locations and build adjacency graph from static facts
        self.locations = set()
        # First pass to find all locations
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Initialize adjacency list for all known locations
        self.adj = {loc: [] for loc in self.locations}
        # Second pass to populate adjacency list based on directed 'adjacent' facts
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == 'adjacent':
                 loc1, loc2 = parts[1], parts[2]
                 # Ensure loc1 is a known location before adding edge
                 if loc1 in self.adj:
                     # Avoid duplicates if domain file lists adjacency redundantly
                     if loc2 not in self.adj[loc1]:
                         self.adj[loc1].append(loc2)
                 # else: # Log warning or handle error if loc1 wasn't found? Assumes consistency.
                 #    print(f"Warning: Location {loc1} from adjacent fact not found in location set.")


        # Store goal locations for boxes, assuming 'box' in name identifies boxes
        self.goal_map = {}
        self.boxes = set() # Keep track of all unique box objects
        for goal in self.goals:
            parts = get_parts(goal)
            # Check if the goal is an 'at' predicate for a box
            if parts[0] == 'at' and 'box' in parts[1]:
                box, loc = parts[1], parts[2]
                self.goal_map[box] = loc
                self.boxes.add(box)

        # Also identify any other boxes mentioned in the initial state
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 # Add to our set of boxes if name suggests it's a box
                 if 'box' in obj:
                     self.boxes.add(obj)


    def bfs(self, start, goal, obstacles):
        """
        Performs Breadth-First Search on the location graph `self.adj`.

        Args:
            start (str): The starting location name.
            goal (str): The target location name.
            obstacles (set): A set of location names that cannot be traversed.

        Returns:
            int: The shortest distance (number of steps) from start to goal.
            float('inf'): If the goal is unreachable from the start.
        """
        if start == goal:
            return 0
        # Check if start location is valid and part of the graph
        if start not in self.locations:
             return float('inf') # Start location itself is invalid

        queue = deque([(start, 0)])
        # Initialize visited set: add start node and ensure obstacles are valid locations
        valid_obstacles = {obs for obs in obstacles if obs in self.locations}
        visited = {start} | valid_obstacles

        while queue:
            current_loc, dist = queue.popleft()

            # Explore neighbors using the precomputed adjacency list
            # Check if current_loc has outgoing edges defined in self.adj
            if current_loc in self.adj:
                for neighbor in self.adj[current_loc]:
                    if neighbor == goal:
                        return dist + 1
                    # Check if neighbor is traversable (not an obstacle and not visited)
                    # Also ensure neighbor is a valid location (should be if graph is consistent)
                    if neighbor not in visited and neighbor in self.locations:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            # If current_loc not in self.adj, it has no outgoing edges -> dead end for this path

        return float('inf') # Goal not reachable


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        """
        state = node.state
        robot_loc = None
        box_locs = {} # Map: box -> current location
        current_box_positions = set() # Set of locations occupied by boxes

        # Parse current state to find robot and box locations
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at-robot':
                robot_loc = parts[1]
            elif predicate == 'at':
                obj, loc = parts[1], parts[2]
                # Check if the object is one of the boxes we track
                if obj in self.boxes:
                    box_locs[obj] = loc
                    current_box_positions.add(loc)

        # If robot location is not found, the state is invalid or incomplete
        if robot_loc is None:
             return float('inf')

        total_heuristic_value = 0
        min_robot_dist_to_box = float('inf')
        misplaced_boxes_exist = False

        # Iterate through all boxes identified during initialization
        for box in self.boxes:
            # Get the box's current location from the parsed state
            bloc = box_locs.get(box)
            if bloc is None:
                # A box identified initially is missing from the current state.
                # This is unexpected in standard Sokoban. Return infinity.
                return float('inf')

            # Get the goal location for this box, if one is defined
            gloc = self.goal_map.get(box)

            # Check if this box has a goal and is not currently at it
            if gloc is not None and bloc != gloc:
                misplaced_boxes_exist = True

                # 1. Calculate box distance to goal (ignoring dynamic obstacles)
                #    This estimates the minimum number of pushes required for this box.
                box_dist = self.bfs(bloc, gloc, obstacles=set())

                # If the box cannot reach its goal topologically, this state leads to a dead end.
                if box_dist == float('inf'):
                    return float('inf')
                total_heuristic_value += box_dist

                # 2. Calculate robot distance to this box's current location
                #    This estimates the robot moves needed to get to this box.
                #    Obstacles for the robot path are the locations of OTHER boxes.
                robot_obstacles = current_box_positions - {bloc}
                robot_dist = self.bfs(robot_loc, bloc, robot_obstacles)

                # Update the minimum distance for the robot to reach *any* misplaced box.
                # If robot_dist is inf, min() correctly keeps the lower finite value or inf.
                min_robot_dist_to_box = min(min_robot_dist_to_box, robot_dist)

        # After checking all boxes, determine the final heuristic value:
        if not misplaced_boxes_exist:
            # If no boxes were misplaced, check if the state meets all goal conditions.
            # The heuristic must be 0 if and only if the state is a goal state.
            if self.goals <= state:
                 return 0 # State is a goal state
            else:
                 # State is not goal, but all boxes are placed (or no box goals defined).
                 # Return 1 to indicate it's not the goal but potentially close.
                 return 1
        else:
            # Misplaced boxes exist. Now, factor in the robot's accessibility.
            if min_robot_dist_to_box == float('inf'):
                # The robot cannot reach any of the misplaced boxes from its current position.
                # This implies the state is unsolvable.
                return float('inf')
            else:
                # Add the minimum robot distance component to the total heuristic value.
                total_heuristic_value += min_robot_dist_to_box
                # Ensure the heuristic is positive (>0) because we know it's not a goal state
                # (since misplaced_boxes_exist is True). max(1, ...) handles the edge case
                # where calculated value might accidentally be 0.
                return max(1, total_heuristic_value)

