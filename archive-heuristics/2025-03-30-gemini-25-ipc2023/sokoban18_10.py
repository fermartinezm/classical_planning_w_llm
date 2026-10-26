import sys
from collections import deque
# Assuming the heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extracts the components of a PDDL fact string.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in Sokoban by summing
    the shortest path distances for each box from its current location to its goal
    location, and adding the shortest distance from the robot to the nearest misplaced box.
    The distances are calculated on the static grid graph, ignoring dynamic obstacles.
    It also includes a basic check for static deadlocks (boxes in non-goal corners).

    # Assumptions
    - The primary cost driver is the number of box pushes. Each push moves a box one step.
    - The cost of robot movement is approximated by the distance required to reach the
      nearest misplaced box initially.
    - Shortest path distances for boxes are calculated based on the static grid layout
      (using 'adjacent' predicates) and ignore dynamic elements like other boxes or
      the robot's current position. This provides a lower bound on pushes if paths were clear.
    - Each box mentioned in the goal has a unique, fixed target location.
    - The heuristic is designed for Greedy Best-First Search and does not need to be admissible.

    # Heuristic Initialization
    - Extracts the target location for each box from the `task.goals`.
    - Identifies all unique locations and boxes involved in the problem.
    - Builds an adjacency list representation of the Sokoban grid using the static
      `adjacent` predicates from `task.static`.
    - Precomputes all-pairs shortest path distances between reachable locations on this
      grid using Breadth-First Search (BFS). The distance represents the minimum number
      of steps (moves or pushes) required on the empty grid.
    - Identifies simple static deadlock locations (non-goal locations that are corners
      from which a box cannot be pushed out).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Identify the robot's current location (`robot_loc`) and
        the current location of each box (`box_locations`) from the input `state`.
    2.  **Initialize:** Set `heuristic_value = 0` and `min_robot_to_box_dist = float('inf')`.
        Keep track of the number of misplaced boxes (`misplaced_boxes_count = 0`).
    3.  **Iterate Through Boxes:** For each box `b` tracked by the heuristic:
        a. Get its current location `bloc` from `box_locations`.
        b. Get its target location `gloc` from the precomputed `self.goal_locations`.
        c. **Check for Deadlock:** If `bloc` is in the precomputed `self.deadlock_locations`,
           return `float('inf')` as this state is considered a dead end.
        d. **Check if Misplaced:** If `bloc != gloc`:
            i.  Increment `misplaced_boxes_count`.
            ii. Retrieve the precomputed shortest path distance `dist(bloc, gloc)` using
                `self.distances[bloc][gloc]`. If this distance is `inf`, the goal is
                unreachable from this state; return `float('inf')`.
            iii. Add `dist(bloc, gloc)` to `heuristic_value`. This estimates the pushes needed for this box.
            iv. Retrieve the precomputed shortest path distance `dist(robot_loc, bloc)`
                using `self.distances[robot_loc][bloc]`.
            v.  Update `min_robot_to_box_dist = min(min_robot_to_box_dist, dist(robot_loc, bloc))`.
    4.  **Check Goal Reached:** If `misplaced_boxes_count == 0`, all boxes are in their goal
        locations. Return 0.
    5.  **Check Robot Reachability:** If `min_robot_to_box_dist` is still `inf` after checking
        all misplaced boxes, it means the robot cannot reach any of the boxes that need
        to be moved. Return `float('inf')`.
    6.  **Combine Costs:** Add the `min_robot_to_box_dist` to `heuristic_value`. This
        approximates the initial cost for the robot to move towards the nearest task.
    7.  **Return Value:** Return the calculated `heuristic_value`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the task, building the grid graph,
        precomputing distances, and identifying deadlocks.
        """
        self.task = task
        self.goal_locations = {} # box -> goal_location
        self.boxes = set()       # set of boxes relevant to goals
        self.locations = set()   # set of all locations
        self.adj = {}            # adjacency list: loc -> [neighbor_locs]
        self.adj_dir = {}        # adjacency with direction: loc -> {neighbor: direction}
        self.distances = {}      # precomputed distances: loc1 -> {loc2: dist}

        # Parse goals to find target locations for boxes
        for goal in task.goals:
            parts = get_parts(goal)
            # Ensure the goal is an 'at' predicate for a box
            if parts[0] == 'at' and task.types.get(parts[1]) == 'box':
                box, loc = parts[1], parts[2]
                self.goal_locations[box] = loc
                self.boxes.add(box)
                self.locations.add(loc) # Goal locations are also locations

        # Build graph, find all locations, and precompute distances
        self._build_graph_and_distances()

        # Identify potential static deadlock locations
        self.deadlock_locations = self._identify_deadlocks()

    def _build_graph_and_distances(self):
        """
        Builds the location graph from static 'adjacent' facts and computes
        all-pairs shortest paths using BFS.
        """
        # Parse static facts to build adjacency graph and identify all locations
        for fact in self.task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                l1, l2, direction = parts[1], parts[2], parts[3]
                # Add locations to the set of known locations
                self.locations.add(l1)
                self.locations.add(l2)
                # Build adjacency list
                if l1 not in self.adj: self.adj[l1] = []
                self.adj[l1].append(l2)
                # Store direction information
                if l1 not in self.adj_dir: self.adj_dir[l1] = {}
                self.adj_dir[l1][l2] = direction

        # Precompute all-pairs shortest paths using BFS from each location
        for start_node in self.locations:
            # Initialize distances from start_node
            self.distances[start_node] = {loc: float('inf') for loc in self.locations}
            self.distances[start_node][start_node] = 0

            queue = deque([(start_node, 0)])
            visited = {start_node} # Keep track of visited nodes for this BFS run

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors
                if current_node in self.adj:
                    for neighbor in self.adj[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[start_node][neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))

    def _identify_deadlocks(self):
        """
        Identifies simple static deadlock locations based on grid topology.
        Currently checks for non-goal locations that are corners (only two exits
        in orthogonal directions).
        """
        deadlocks = set()
        goal_locs_set = set(self.goal_locations.values())

        for loc in self.locations:
            # Skip if the location is a goal location for any box
            if loc in goal_locs_set:
                continue

            neighbors = self.adj.get(loc, [])
            num_neighbors = len(neighbors)

            # A simple check: locations with only one exit (dead ends) are deadlocks if not goals
            if num_neighbors <= 1:
                 deadlocks.add(loc)
                 continue

            # Check for corners: Requires exactly two neighbors, and they must be in
            # orthogonal directions relative to 'loc'.
            if num_neighbors == 2:
                n1, n2 = neighbors
                dir1 = self.adj_dir.get(loc, {}).get(n1)
                dir2 = self.adj_dir.get(loc, {}).get(n2)

                if dir1 and dir2:
                    # Check if directions are orthogonal (one horizontal, one vertical)
                    horizontal1 = dir1 in ['left', 'right']
                    vertical1 = dir1 in ['up', 'down']
                    horizontal2 = dir2 in ['left', 'right']
                    vertical2 = dir2 in ['up', 'down']

                    # If directions are orthogonal (e.g., right and down), it's a corner.
                    if horizontal1 != horizontal2 and vertical1 != vertical2:
                        deadlocks.add(loc)

        # print(f"DEBUG: Identified {len(deadlocks)} potential deadlock locations: {deadlocks}")
        return deadlocks

    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state
        robot_loc = None
        box_locations = {} # box -> current_location

        # Parse current state to find robot and box locations
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at-robot':
                robot_loc = parts[1]
            elif predicate == 'at':
                # Ensure it's a box we care about (one with a goal location)
                box = parts[1]
                if box in self.boxes:
                    loc = parts[2]
                    box_locations[box] = loc

        # Basic check: If robot location is unknown, state is invalid
        if robot_loc is None:
             # This might happen if the initial state doesn't define the robot's position
             # Or if the state representation is unexpected.
             # Returning infinity signals an issue or an invalid state.
             print("Warning: Robot location not found in state.")
             return float('inf')

        heuristic_value = 0
        min_robot_to_box_dist = float('inf')
        misplaced_boxes_count = 0
        can_robot_reach_any_misplaced_box = False

        # Calculate cost for each box
        for box in self.boxes:
            bloc = box_locations.get(box)
            gloc = self.goal_locations.get(box) # Should always exist for boxes in self.boxes

            # If box location is missing in state, something is wrong
            if bloc is None:
                print(f"Warning: Location for box {box} not found in state.")
                return float('inf') # Cannot evaluate heuristic

            # Check if the box is in a deadlock location
            if bloc in self.deadlock_locations:
                 # print(f"DEBUG: Deadlock detected - Box {box} at {bloc}")
                 return float('inf') # This state leads to a known deadlock

            # If the box is not at its goal location
            if bloc != gloc:
                misplaced_boxes_count += 1

                # Get precomputed distance for the box to its goal
                box_dist = self.distances.get(bloc, {}).get(gloc, float('inf'))

                # If the goal is unreachable from the box's current location
                if box_dist == float('inf'):
                    # print(f"DEBUG: Goal {gloc} unreachable for box {box} from {bloc}")
                    return float('inf') # Problem might be unsolvable from this state

                heuristic_value += box_dist

                # Get precomputed distance from the robot to this box
                robot_dist = self.distances.get(robot_loc, {}).get(bloc, float('inf'))

                # If robot can reach this box, update min distance and flag reachability
                if robot_dist != float('inf'):
                     min_robot_to_box_dist = min(min_robot_to_box_dist, robot_dist)
                     can_robot_reach_any_misplaced_box = True
                # else: robot cannot reach this specific box location

        # If all boxes are in their goal locations, heuristic is 0
        if misplaced_boxes_count == 0:
            return 0

        # If there are misplaced boxes, but the robot cannot reach any of them
        if not can_robot_reach_any_misplaced_box:
            # print(f"DEBUG: Robot at {robot_loc} cannot reach any misplaced boxes.")
            return float('inf') # State seems unsolvable if robot intervention is needed

        # Add the cost for the robot to move to the nearest misplaced box
        # If min_robot_to_box_dist remained inf here, it implies an issue,
        # but the 'can_robot_reach_any_misplaced_box' check should handle it.
        if min_robot_to_box_dist == float('inf'):
             # This case should ideally not be reached if the above check works.
             print("Warning: min_robot_to_box_dist is infinity despite reachable boxes.")
             return float('inf')

        heuristic_value += min_robot_to_box_dist

        return heuristic_value

