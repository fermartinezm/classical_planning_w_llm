import math
from collections import deque
# No need for fnmatch or heapq based on the final heuristic logic
from heuristics.heuristic_base import Heuristic # Assuming this is the correct import path

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    """
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It calculates the sum of shortest path distances for each box from its current
    location to its goal location, based on the empty grid layout. It adds the
    shortest path distance for the robot to reach a position adjacent to the *nearest*
    misplaced box (the box requiring the fewest robot moves to reach an adjacent square).
    The heuristic aims to guide a greedy best-first search towards states where boxes
    are closer to their targets and the robot is positioned to push them efficiently.
    It is designed to be informative but not necessarily admissible.

    # Assumptions
    - The primary cost driver is moving boxes (pushes). Robot movement cost is
      approximated by the estimated moves needed to reach the first box to push.
    - Shortest paths for boxes and the robot are calculated on the static grid
      structure defined by 'adjacent' facts, ignoring the dynamic presence of
      the robot and other boxes as obstacles during the heuristic calculation.
      This simplification makes the heuristic non-admissible but computationally cheaper.
    - Each box has a unique, fixed goal location specified in the problem's goal
      definition (e.g., "(at box1 goal_loc)").
    - Locations are treated as nodes in a graph, and their names (e.g., 'loc_row_col')
      are not parsed for geometric information; connectivity is solely based on
      'adjacent' facts.
    - The 'adjacent' predicates define a symmetric relationship (if A is adjacent to B,
      B is adjacent to A), as is typical in grid-based domains like Sokoban.

    # Heuristic Initialization
    - Parses static 'adjacent' facts from the task's static information to build an
      undirected graph representation of the level layout (adjacency list).
    - Identifies all unique locations present in the map.
    - Computes all-pairs shortest path distances between all locations using Breadth-First
      Search (BFS) starting from each location. These distances represent the minimum
      number of steps (moves or pushes) required to travel between locations on an
      empty grid. Distances are stored for efficient lookup during heuristic evaluation.
      Unreachable locations have a distance of infinity.
    - Extracts the specific goal location for each box from the task's goal specification
      (e.g., `(at box1 loc_goal)`).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Goal Check:** Check if the current state satisfies all goal conditions. If yes,
        the heuristic value is 0, as no more actions are needed.
    2.  **State Parsing:** Extract the current location of the robot (`rloc`) and the
        current location of each box (`box_locs`) from the set of facts in the state.
    3.  **Initialization:** Initialize the total heuristic value `h` to 0. Initialize
        `min_robot_dist_to_adjacent` to positive infinity. This variable will track the
        shortest distance from the robot to a square adjacent to *any* box that is not
        yet at its goal location. Set a flag `any_box_misplaced` to `False`.
    4.  **Box Evaluation Loop:** Iterate through each box `b` and its designated goal
        location `gloc` (stored during initialization).
        a.  Retrieve the box's current location `bloc` from `box_locs`.
        b.  **Check if Box is Placed:** If `bloc == gloc`, this box is already at its
            goal. Continue to the next box.
        c.  **Mark Misplaced:** If the box is not at its goal, set `any_box_misplaced = True`.
        d.  **Box-Goal Distance:** Look up the precomputed shortest path distance
            `dist(bloc, gloc)` between the box's current location and its goal location.
            If the distance is infinity (goal is unreachable from current location on
            the empty grid), return infinity for the heuristic value, as the goal is
            unreachable from this state (potential deadlock or disconnected map).
            Otherwise, add this distance to `h`. This component estimates the minimum
            number of pushes required for this box, ignoring interference.
        e.  **Robot-to-Adjacent Distance:** Find all locations `adj_loc` that are adjacent
            to the box's current location `bloc` (using the precomputed adjacency list).
            For each `adj_loc`, look up the precomputed shortest path distance
            `dist(rloc, adj_loc)` from the robot's current location `rloc` to `adj_loc`.
            Update `min_robot_dist_to_adjacent = min(min_robot_dist_to_adjacent, dist(rloc, adj_loc))`.
            This step finds the minimum number of moves the robot needs to make to get
            next to *any* of the misplaced boxes.
    5.  **Combine Costs:** After checking all boxes:
        a.  If `any_box_misplaced` is `True`:
            i.  If `min_robot_dist_to_adjacent` is still infinity, it means the robot
                cannot reach any position adjacent to any misplaced box. Return infinity
                for the heuristic value (unsolvable state).
            ii. Otherwise, add `min_robot_dist_to_adjacent` to the total heuristic value `h`.
                This adds the estimated cost for the robot's initial movement to position
                itself for the first push.
        b.  If `any_box_misplaced` is `False` (all boxes are at their goals), the heuristic
            value should be 0 (this case is handled by the initial goal check, but ensures
            correctness if the goal involves more than just box positions).
    6.  **Return Value:** Return the calculated heuristic value `h`. Ensure it's non-negative
        (although the calculation should naturally produce non-negative values or infinity).
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information:
        - Builds the adjacency graph from 'adjacent' facts.
        - Computes all-pairs shortest paths using BFS.
        - Stores goal locations for each box.
        """
        self.goals = task.goals
        static_facts = task.static

        # 1. Build adjacency graph and identify all locations
        self.adj = {}
        self.locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                # adjacent ?l1 - location ?l2 - location ?d - direction
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Add edges in both directions assuming symmetry
                self.adj.setdefault(l1, []).append(l2)
                self.adj.setdefault(l2, []).append(l1)

        # Ensure all locations are keys in adj, even if they have no neighbors
        for loc in self.locations:
            if loc not in self.adj:
                self.adj[loc] = []

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            # Initialize distances from start_node to all others as infinity
            self.distances[start_node] = {loc: float('inf') for loc in self.locations}
            self.distances[start_node][start_node] = 0

            queue = deque([(start_node, 0)]) # (node, distance)
            visited = {start_node}

            while queue:
                current_node, dist = queue.popleft()

                for neighbor in self.adj.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = dist + 1
                        self.distances[start_node][neighbor] = new_dist
                        queue.append((neighbor, new_dist))

        # 3. Extract goal locations for boxes
        self.box_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal format: (at ?b - box ?l - location)
            if parts[0] == 'at' and len(parts) == 3:
                 # Simple check: assume second argument is a box if it's not a location?
                 # A better approach might involve checking types if available,
                 # but for Sokoban, 'at' usually involves a box.
                 # Let's assume the structure is always (at box loc) in goals.
                 box, loc = parts[1], parts[2]
                 if loc in self.locations: # Check if the location is valid
                    self.box_goals[box] = loc
                 # else: print(f"Warning: Goal location {loc} not found in map.")


    def __call__(self, node):
        """
        Computes the heuristic value for a given state node.
        """
        state = node.state

        # 1. Check if goal is already reached
        # The goal_reached method is provided by the Task object usually,
        # but comparing sets is equivalent for conjunctive goals.
        if self.goals <= state:
            return 0

        # 2. Parse current state for robot and box locations
        robot_loc = None
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                # Assume it's (at box loc)
                box, loc = parts[1], parts[2]
                if box in self.box_goals: # Only track boxes that have goals
                     box_locs[box] = loc

        # Basic check: If robot location is unknown, state is invalid/unreachable
        if robot_loc is None or robot_loc not in self.locations:
             # print(f"Warning: Robot location '{robot_loc}' invalid or not found.")
             return float('inf')

        # 3. Initialize heuristic value and tracking variables
        h = 0
        min_robot_dist_to_adjacent = float('inf')
        any_box_misplaced = False

        # 4. Box Evaluation Loop
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locs.get(box)

            # If a box specified in the goal is not in the current state,
            # something is wrong.
            if current_loc is None:
                # print(f"Warning: Box '{box}' from goal not found in state.")
                return float('inf')

            # 4b. Check if Box is Placed
            if current_loc != goal_loc:
                # 4c. Mark Misplaced
                any_box_misplaced = True

                # 4d. Box-Goal Distance
                # Use .get for safer dictionary access
                dist_box_goal = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if dist_box_goal == float('inf'):
                    # Goal is unreachable for this box from its current position
                    return float('inf')
                h += dist_box_goal

                # 4e. Robot-to-Adjacent Distance
                # Find locations adjacent to the current box location
                for adj_loc in self.adj.get(current_loc, []):
                    # Find distance from robot to this adjacent location
                    dist_robot_adj = self.distances.get(robot_loc, {}).get(adj_loc, float('inf'))
                    min_robot_dist_to_adjacent = min(min_robot_dist_to_adjacent, dist_robot_adj)

        # 5. Combine Costs
        if any_box_misplaced:
            # If no path exists for the robot to get next to any misplaced box
            if min_robot_dist_to_adjacent == float('inf'):
                 # This implies the robot is trapped or the map is disconnected in a way
                 # that prevents reaching necessary push positions.
                 return float('inf')
            h += min_robot_dist_to_adjacent
        # else: If no box was misplaced, h remains 0 (or whatever sum was accumulated,
        # which should be 0 if all boxes started at goal or were moved there).
        # The initial goal check handles the h=0 case correctly.

        # 6. Return Value (ensure non-negative)
        # The calculation naturally yields non-negative or inf.
        return h

