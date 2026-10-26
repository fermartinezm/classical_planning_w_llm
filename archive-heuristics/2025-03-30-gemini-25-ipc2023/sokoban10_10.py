import math
from collections import deque

# Try to import the base class - provide a dummy if not found
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the import fails
    class Heuristic:
        def __init__(self, task):
            """Dummy initializer."""
            pass
        def __call__(self, node):
            """Dummy call method."""
            raise NotImplementedError("Heuristic base class not found.")

def get_parts(fact: str):
    """
    Helper function to extract predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at box1 loc1)" -> ["at", "box1", "loc1"]
    """
    # Ensure fact is a string and has content
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return [] # Return empty list for invalid input


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain.

    # Summary
    Estimates the cost to reach the goal state in Sokoban. The heuristic combines
    the sum of shortest path distances for each box to its target location
    with the shortest path distance for the robot to reach the nearest misplaced box.
    The distances are precomputed based on the static grid layout, representing
    the minimum number of robot moves between locations (ignoring dynamic obstacles like boxes).
    This heuristic is designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - The goal is specified by `(at boxX locY)` predicates, providing a unique target for each relevant box.
    - The grid structure (adjacency defined by `adjacent` facts) is static.
    - Robot movement cost is 1 per step. The heuristic uses this distance as a proxy for box push counts.
    - The heuristic estimates the sum of (approximated) pushes plus the robot moves needed to reach the first box to push.
    - Adjacency is symmetric (if A is adjacent to B via some direction, B is adjacent to A via some direction). The specific directions are ignored for pathfinding.

    # Heuristic Initialization (`__init__`)
    - Parses goal conditions (`task.goals`) to create a map from boxes to their target locations (`self.goal_map`).
    - Identifies all boxes involved in the goal (`self.boxes`).
    - Builds an adjacency list representation of the grid (`self.adj`) from static `adjacent` facts found in `task.static`.
    - Collects all unique location names found in static facts, initial state, and goals (`self.locations`).
    - Precomputes all-pairs shortest path distances for robot movement (`self.dist`) using Breadth-First Search (BFS) starting from each location. Unreachable locations will have infinite distance (`math.inf`). This represents the minimum number of 'move' actions between locations.
    # - Optional: Identification of dead-end squares is not implemented in this version for simplicity. Adding dead-end detection could improve heuristic guidance but increases complexity.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse Current State:** Iterate through the facts in the current state (`node.state`, a frozenset of strings) to find the robot's location (`robot_loc`) and the current location of each goal-relevant box (`box_locs`). Handle cases where the robot or a goal box might be missing from the state representation (return infinity, indicating an invalid or unreachable goal state).
    2.  **Dead-End Check:** (Not Implemented) If dead-end detection were implemented, this step would check if any box is in a pre-identified dead-end square (unless that square is its goal). If so, return infinity.
    3.  **Calculate Total Box Distance (`box_dist_sum`):**
        - Initialize `box_dist_sum = 0`.
        - Keep track if any box is misplaced (`misplaced_boxes_exist = False`).
        - Iterate through each box (`box`) and its target location (`goal_loc`) defined in `self.goal_map`.
        - Get the box's current location (`current_loc`) from `box_locs`. If a goal box is not found in the state, return infinity.
        - If `current_loc` is not equal to `goal_loc`:
            - Mark `misplaced_boxes_exist = True`.
            - Retrieve the precomputed shortest path distance `d = self.dist.get(current_loc, {}).get(goal_loc, math.inf)`. This uses the robot's path distance as an estimate for the minimum number of pushes required for this box, ignoring obstacles and robot positioning requirements for the push itself.
            - If `d` is infinity, the goal location is statically unreachable for this box from its current position based on the grid connectivity. Return `math.inf`.
            - Add `d` to `box_dist_sum`.
    4.  **Calculate Minimum Robot Distance (`min_robot_dist_to_misplaced_box`):**
        - Initialize `min_robot_dist_to_misplaced_box = math.inf`.
        - During the loop in step 3, if a box (`box`) is found to be misplaced at `current_loc`:
            - Retrieve the precomputed shortest path distance from the robot's current location (`robot_loc`) to the box's current location (`current_loc`): `d_robot = self.dist.get(robot_loc, {}).get(current_loc, math.inf)`. This estimates the cost for the robot to reach the location of the misplaced box.
            - Update `min_robot_dist_to_misplaced_box = min(min_robot_dist_to_misplaced_box, d_robot)`.
    5.  **Combine and Return:**
        - If no boxes were misplaced (`misplaced_boxes_exist` is False), the goal state is reached regarding box positions. Return 0.
        - If `min_robot_dist_to_misplaced_box` is still infinity (and boxes *are* misplaced), it means the robot cannot reach any of the boxes that need moving from its current position based on the static grid connectivity. Return `math.inf`.
        - Otherwise, return the sum `box_dist_sum + min_robot_dist_to_misplaced_box`. This combines the estimated total pushes with the estimated cost for the robot to reach the nearest box needing a push. The result is clamped to be non-negative.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class
        self.task = task
        self.adj = {}           # Adjacency list for locations: loc -> {neighbor1, neighbor2,...}
        self.locations = set()  # Set of all known location names
        self.goal_map = {}      # Map from box name to its goal location: box -> loc
        self.boxes = set()      # Set of box names mentioned in the goal

        # 1. Parse static facts (adjacency) and goals (at box loc)
        static_facts = task.static
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts
            if parts[0] == 'adjacent':
                # adjacent ?l1 - location ?l2 - location ?d - direction
                if len(parts) == 4:
                    loc1, loc2 = parts[1], parts[2]
                    # Add locations and build adjacency list (assume symmetric)
                    self.locations.add(loc1)
                    self.locations.add(loc2)
                    self.adj.setdefault(loc1, set()).add(loc2)
                    self.adj.setdefault(loc2, set()).add(loc1)

        for goal in task.goals:
            parts = get_parts(goal)
            if not parts: continue
            # at ?o - box ?l - location
            if parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.goal_map[box] = loc
                self.boxes.add(box)
                self.locations.add(loc) # Ensure goal locations are known

        # Ensure locations from initial state are included
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'at-robot' and len(parts) == 2:
                 self.locations.add(parts[1])
             elif parts[0] == 'at' and len(parts) == 3:
                 # Add box location
                 self.locations.add(parts[2])
                 # We only care about boxes that are part of the goal state definition
                 # box = parts[1]
                 # if box in self.goal_map:
                 #     pass # Already tracked if it's a goal box

        # Convert locations set to list for consistent iteration order during BFS
        self.locations = list(self.locations)

        # 2. Precompute all-pairs shortest paths (robot movement distance) using BFS
        self.dist = {loc: {} for loc in self.locations} # {start_loc: {end_loc: distance}}
        for start_node in self.locations:
            # Initialize distance from start_node to itself as 0
            self.dist[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node} # Keep track of visited nodes for this specific BFS run

            while queue:
                curr = queue.popleft()
                current_dist = self.dist[start_node][curr]

                # Use self.adj.get(curr, set()) to safely handle locations that might not be keys
                # (e.g., isolated locations mentioned in init/goal but not adjacent facts)
                for neighbor in self.adj.get(curr, set()):
                    if neighbor not in visited:
                        # Ensure neighbor is in our known locations list before adding distance
                        if neighbor in self.dist:
                            visited.add(neighbor)
                            self.dist[start_node][neighbor] = current_dist + 1
                            queue.append(neighbor)
                        # else:
                            # This case should ideally not happen if all locations are collected properly
                            # print(f"Warning: Neighbor '{neighbor}' of '{curr}' not in known locations.")


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the cost to reach the goal.
        Returns 0 for a goal state.
        Returns math.inf for states detected as unsolvable or invalid.
        """
        state = node.state
        robot_loc = None
        box_locs = {} # Stores current location {box: loc} for goal boxes

        # 1. Parse current state for robot and box locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                if box in self.boxes: # Only track boxes relevant to the goal
                    box_locs[box] = loc

        # --- Sanity Checks ---
        if robot_loc is None:
             # print("Warning: Robot location not found in state.")
             return math.inf # Cannot evaluate heuristic without robot position
        if robot_loc not in self.dist:
             # print(f"Warning: Robot location '{robot_loc}' is unknown or isolated.")
             return math.inf # Robot is in an unexpected place

        # Check if all goal boxes were found in the state
        if len(box_locs) != len(self.boxes):
             # print(f"Warning: Not all goal boxes found in state. Found {len(box_locs)}/{len(self.boxes)}.")
             return math.inf # Cannot satisfy goal if a required box is missing

        # --- Heuristic Calculation ---
        box_dist_sum = 0
        min_robot_dist_to_misplaced_box = math.inf
        misplaced_boxes_exist = False

        # 3. & 4. Calculate box distances and find min robot distance to misplaced boxes
        for box, goal_loc in self.goal_map.items():
            # We already checked that all goal boxes are in box_locs above
            current_loc = box_locs[box]

            # Check if the box location is known/valid (should be, as locations are collected)
            if current_loc not in self.dist:
                # print(f"Warning: Box {box} is at unknown location '{current_loc}'.")
                return math.inf

            if current_loc != goal_loc:
                misplaced_boxes_exist = True

                # Get distance from box's current location to its goal
                # Use precomputed robot distances as approximation for pushes
                dist_box_to_goal = self.dist.get(current_loc, {}).get(goal_loc, math.inf)

                if dist_box_to_goal == math.inf:
                    # Goal location is statically unreachable for this box
                    return math.inf

                box_dist_sum += dist_box_to_goal

                # Get distance from robot to this misplaced box's current location
                dist_robot_to_box = self.dist.get(robot_loc, {}).get(current_loc, math.inf)

                # Update the minimum distance to any misplaced box
                min_robot_dist_to_misplaced_box = min(min_robot_dist_to_misplaced_box, dist_robot_to_box)

        # 5. Combine and return heuristic value
        if not misplaced_boxes_exist:
            # Goal state reached (all boxes are in their target locations)
            return 0
        elif min_robot_dist_to_misplaced_box == math.inf:
             # There are misplaced boxes, but the robot cannot reach any of them
             # This implies the state is unsolvable if robot needs to move boxes
             return math.inf
        else:
            # Heuristic = (Sum of distances for boxes to goals) + (Distance for robot to nearest misplaced box)
            heuristic_value = box_dist_sum + min_robot_dist_to_misplaced_box
            # Ensure non-negative value
            return max(0, heuristic_value)

