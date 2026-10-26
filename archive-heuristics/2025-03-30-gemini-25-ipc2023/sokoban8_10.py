import sys
from collections import deque
# Assuming the planner infrastructure provides the Heuristic base class
# and Task/Operator classes as described.
# If running standalone, you might need to define a placeholder for Heuristic.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a placeholder if the base class is not available in the environment
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper function to parse PDDL fact strings like '(predicate arg1 arg2)'
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    return fact.strip()[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It primarily considers the sum of shortest path distances each box needs to travel
    on the grid to reach its target location (ignoring other boxes/robot). This
    estimates the minimum number of pushes required. It also adds an estimate of the
    robot's travel distance to get adjacent to the nearest misplaced box. A simple
    deadlock detection mechanism is included to assign infinite cost to states where
    boxes are stuck in non-goal corner locations.

    # Assumptions
    - The cost of both 'move' and 'push' actions is 1.
    - The heuristic estimates cost based on pushes (box distances) and initial robot
      positioning towards the nearest task. It doesn't model complex interactions,
      path blocking by other objects, or the full cost of robot movement between pushes.
    - Only simple "corner" deadlocks are detected (non-goal locations where a box
      cannot be pushed out along either the vertical or horizontal axis).

    # Heuristic Initialization (`__init__`)
    - Stores the task object.
    - Parses goal conditions `(at box loc)` to map each box to its target location (`self.goal_locations`).
    - Identifies all unique boxes mentioned in the goals (`self.boxes`).
    - Parses static `adjacent` facts to build a detailed adjacency map `self.adj_details`,
      storing `{loc: {direction: neighbor}}`, and identifies all unique locations (`self.locations`).
    - Computes all-pairs shortest path distances between reachable locations using BFS,
      treating the grid as empty. Stores distances in `self.distances`. Unreachable
      pairs get infinite distance.
    - Identifies simple "corner" dead-end locations using `_identify_dead_ends()`
      and stores them in `self.dead_ends`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Get the current state `node.state`.
    2. Check if the goal is already reached using `task.goal_reached(state)`. If yes, return 0.
    3. Parse the current state to find the robot's location (`robot_loc`) and the
       location of each box (`box_locs`). Return infinity if parsing fails.
    4. **Deadlock Check:**
       - Iterate through each box `b` and its current location `l_b`.
       - If `l_b` is in `self.dead_ends` AND `l_b` is not the goal location for `b`,
         return `float('inf')` immediately, as this state is considered a dead end.
    5. **Calculate Box Contributions:**
       - Initialize `total_box_distance = 0`.
       - Initialize `min_robot_dist_to_box = float('inf')`.
       - Set `misplaced_boxes_exist = False`.
       - For each box `b` tracked:
         - Get its current location `l_b` and goal location `g_b`.
         - If `l_b != g_b`:
           - Set `misplaced_boxes_exist = True`.
           - Retrieve the precomputed shortest path distance `dist = self.distances[l_b][g_b]`.
           - If `dist` is infinity (goal is unreachable for this box), return `inf`.
           - Add `dist` to `total_box_distance` (counts estimated pushes).
           - Calculate the distance from the robot to this box: `robot_dist = self.distances[robot_loc][l_b]`.
           - Update `min_robot_dist_to_box = min(min_robot_dist_to_box, robot_dist)`.
    6. **Handle Goal State / Unsolvable:**
       - If `not misplaced_boxes_exist`, return 0 (this confirms goal state).
       - If `min_robot_dist_to_box` is infinity, it means the robot cannot reach any
         misplaced box (or robot/box location is invalid). Return `float('inf')`.
    7. **Calculate Robot Positioning Cost:**
       - Estimate the moves needed for the robot to get adjacent to the *nearest*
         misplaced box.
       - `robot_positioning_cost = max(0, min_robot_dist_to_box - 1)`. If the robot
         is already adjacent (`dist=1`) or at the box location (`dist=0`), this cost is 0.
    8. **Combine Costs:**
       - The final heuristic value is `h = total_box_distance + robot_positioning_cost`.
    9. Return `h`.
    """

    def __init__(self, task):
        self.task = task
        self.goal_locations = {} # {box_name: goal_loc}
        self.adj_details = {} # {loc: {direction: neighbor}}
        self.locations = set()
        self.boxes = set()
        self.opposites = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        # Parse goals to find target location for each box
        for goal in task.goals:
            parts = get_parts(goal)
            # Ensure it's a valid (at box loc) goal predicate
            if parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Basic check if box/loc names seem valid (optional)
                if box.startswith('box') and loc.startswith('loc'):
                    self.goal_locations[box] = loc
                    self.boxes.add(box)

        # Parse static facts for adjacency and identify all locations
        for fact in task.static:
            parts = get_parts(fact)
            # Ensure it's a valid (adjacent l1 l2 dir) predicate
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                # Basic check if location/direction names seem valid (optional)
                if l1.startswith('loc') and l2.startswith('loc') and direction in self.opposites:
                    self.locations.add(l1)
                    self.locations.add(l2)
                    # Store forward adjacency
                    if l1 not in self.adj_details: self.adj_details[l1] = {}
                    self.adj_details[l1][direction] = l2
                    # Store backward adjacency using opposite direction
                    opposite_dir = self.opposites[direction]
                    if l2 not in self.adj_details: self.adj_details[l2] = {}
                    self.adj_details[l2][opposite_dir] = l1

        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths()

        # Identify dead-end locations (simple corners)
        self.dead_ends = self._identify_dead_ends()
        # print(f"Identified {len(self.dead_ends)} dead ends: {self.dead_ends}") # Optional debug print

    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        # Initialize all distances to infinity
        distances = {loc: {other_loc: float('inf') for other_loc in self.locations} for loc in self.locations}

        for start_node in self.locations:
            # Distance from a node to itself is 0
            distances[start_node][start_node] = 0
            # Initialize queue for BFS starting from start_node
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()
                current_dist = distances[start_node][current_node]

                # Explore neighbors based on adj_details (all directions)
                for direction, neighbor in self.adj_details.get(current_node, {}).items():
                    # If this neighbor hasn't been reached yet from start_node
                    if distances[start_node][neighbor] == float('inf'):
                        distances[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)
        return distances

    def _identify_dead_ends(self):
        """
        Identifies simple corner dead-end locations.
        A location is considered a dead end if it's not a goal location,
        and a box pushed there cannot be pushed out along at least one axis
        (i.e., it's blocked vertically AND horizontally).
        """
        dead_ends = set()
        goal_locs_set = set(self.goal_locations.values())

        for loc in self.locations:
            # Skip if it's a goal location for any box
            if loc in goal_locs_set:
                continue

            # Check available neighbors in all four cardinal directions
            neighbors = self.adj_details.get(loc, {})
            has_neighbor = {
                'up': 'up' in neighbors,
                'down': 'down' in neighbors,
                'left': 'left' in neighbors,
                'right': 'right' in neighbors
            }

            # Check if movement is blocked along the vertical axis (cannot push both up AND down)
            stuck_vertical = not (has_neighbor['up'] and has_neighbor['down'])
            # Check if movement is blocked along the horizontal axis (cannot push both left AND right)
            stuck_horizontal = not (has_neighbor['left'] and has_neighbor['right'])

            # If it's blocked along both axes, it's a simple dead end (corner)
            if stuck_vertical and stuck_horizontal:
                dead_ends.add(loc)
        return dead_ends


    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state
        # Check goal condition first - most efficient check
        if self.task.goal_reached(state):
            return 0

        robot_loc = None
        box_locs = {} # {box_name: location}

        # Parse current state to find robot and box locations
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif pred == 'at' and len(parts) == 3:
                # Check if it's one of the boxes we are tracking from the goal
                box_name = parts[1]
                if box_name in self.boxes:
                    box_locs[box_name] = parts[2]

        # --- State Validity and Parsing Checks ---
        # Check if robot location was found
        if robot_loc is None:
            # print("Warning: Robot location not found in state.", file=sys.stderr)
            return float('inf')
        # Check if all goal boxes were found in the state
        if len(box_locs) != len(self.boxes):
            # print(f"Warning: Mismatch in box count. Found {len(box_locs)}, expected {len(self.boxes)}.", file=sys.stderr)
            return float('inf') # State might be inconsistent
        # Check if robot and box locations are valid known locations
        if robot_loc not in self.locations:
             # print(f"Warning: Robot location '{robot_loc}' unknown.", file=sys.stderr)
             return float('inf')
        for box, loc in box_locs.items():
             if loc not in self.locations:
                  # print(f"Warning: Box '{box}' location '{loc}' unknown.", file=sys.stderr)
                  return float('inf')


        total_box_distance = 0
        min_robot_dist_to_box = float('inf')
        misplaced_boxes_exist = False

        # --- Deadlock Check and Box Distance Calculation ---
        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)
            # This check should ideally not fail if init was correct
            if goal_loc is None: return float('inf')

            # 1. Deadlock Check: Is the box in a non-goal dead-end location?
            if current_loc in self.dead_ends and current_loc != goal_loc:
                return float('inf')

            # 2. Calculate distance if the box is misplaced
            if current_loc != goal_loc:
                misplaced_boxes_exist = True
                # Retrieve precomputed distance: current_loc -> goal_loc
                # Ensure keys exist in the precomputed distances dictionary
                if current_loc not in self.distances or goal_loc not in self.distances[current_loc]:
                     # This indicates an issue with BFS or location tracking
                     # print(f"Warning: Distance not found for {current_loc} -> {goal_loc}", file=sys.stderr)
                     return float('inf')
                dist = self.distances[current_loc][goal_loc]

                # If distance is infinite, the goal is statically unreachable for this box
                if dist == float('inf'):
                    return float('inf')
                total_box_distance += dist

                # 3. Find distance from robot to this misplaced box: robot_loc -> current_loc
                if robot_loc not in self.distances or current_loc not in self.distances[robot_loc]:
                     # print(f"Warning: Distance not found for {robot_loc} -> {current_loc}", file=sys.stderr)
                     return float('inf') # Should not happen if locations are valid
                robot_dist = self.distances[robot_loc][current_loc]

                # If robot cannot reach this box, the state is unsolvable
                if robot_dist == float('inf'):
                    return float('inf')
                # Update the minimum distance from robot to *any* misplaced box
                min_robot_dist_to_box = min(min_robot_dist_to_box, robot_dist)


        # --- Final Heuristic Value Calculation ---
        # If no boxes were misplaced, it's a goal state.
        if not misplaced_boxes_exist:
            # This case should have been caught by task.goal_reached at the start.
            # If reached here, it might indicate an issue, but returning 0 is consistent.
            return 0

        # If min_robot_dist_to_box is still infinity here, it means misplaced boxes exist,
        # but the robot couldn't reach any of them (already handled by checks inside the loop).
        # This check is redundant but safe:
        if min_robot_dist_to_box == float('inf'):
             return float('inf')

        # Estimate robot positioning cost: moves needed to get adjacent to the nearest misplaced box.
        # If robot is at the box (dist=0) or adjacent (dist=1), cost is 0.
        # Otherwise, cost is dist-1.
        robot_positioning_cost = max(0, min_robot_dist_to_box - 1)

        # Combine: Sum of box-to-goal distances (pushes) + robot positioning cost (moves)
        heuristic_value = total_box_distance + robot_positioning_cost

        return heuristic_value
