import sys
from collections import deque
from fnmatch import fnmatch
import itertools # Only needed if complex logic requires it, currently not used.

# Ensure the base class is importable. Adjust path if necessary.
try:
    # Attempt standard import relative to a 'heuristics' package
    from heuristics.heuristic_base import Heuristic
except ModuleNotFoundError:
    # Fallback for running script directly or different project structure
    try:
        # Assumes heuristic_base.py is in the parent directory relative to where this file is
        # Adjust '..' based on your actual project structure if needed.
        sys.path.append('..')
        from heuristics.heuristic_base import Heuristic
    except (ImportError, ModuleNotFoundError):
        # If still not found, provide a dummy class for basic functionality
        # This helps in environments where the base class path is unconventional.
        print("Warning: Heuristic base class not found at expected paths. Using dummy base class.", file=sys.stderr)
        class Heuristic:
            def __init__(self, task): pass
            def __call__(self, node): raise NotImplementedError("Dummy Heuristic Base Class")


def get_parts(fact):
    """
    Safely extracts predicate and arguments from a PDDL fact string.

    Handles potential empty strings or malformed facts (e.g., missing parentheses,
    empty content like '( )'). Returns an empty list if the fact is malformed
    or cannot be parsed.

    Args:
        fact (str): The PDDL fact string, e.g., "(at box1 loc_1_1)".

    Returns:
        list: A list of strings representing the predicate and its arguments,
              e.g., ['at', 'box1', 'loc_1_1']. Returns an empty list if
              parsing fails.
    """
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Fact is not a string, too short, or missing parentheses
        return []
    content = fact[1:-1].strip() # Remove parentheses and leading/trailing whitespace
    if not content:
        # Handle empty facts like "()" or "( )"
        return []
    return content.split()


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It calculates the sum of shortest path distances for each box from its current location
    to its target goal location. Additionally, it adds the shortest path distance from the
    robot's current location to the nearest box that is not yet in its goal location.
    The heuristic relies on precomputed all-pairs shortest paths on the location grid,
    ignoring dynamic obstacles like other boxes or the robot's position during path calculation.
    It is designed to be informative for greedy best-first search but is not necessarily admissible.

    # Assumptions
    - Action costs are uniform: 'move' costs 1, 'push' costs 1. A push action moves both the box and the robot one step.
    - The map is represented by 'location' objects and 'adjacent' predicates defining connectivity between locations.
    - Walls or static obstacles are implicitly defined by the absence of 'adjacent' predicates connecting locations.
    - The heuristic uses grid distances (minimum number of steps/actions) assuming intermediate locations can be cleared when needed.
    - It does not perform complex dead-end detection (e.g., recognizing configurations where boxes block each other permanently) beyond basic reachability checks based on the static map layout.
    - Goal conditions explicitly assign each relevant box to a specific target location (e.g., `(at box1 goal_loc1)`). Problems where any box can go to any goal location are not directly supported by the goal parsing logic.

    # Heuristic Initialization
    - Identifies all unique 'location' objects defined in the task by scanning static facts, initial state, and goals.
    - Parses static 'adjacent' facts to build an adjacency list representing the map connectivity. Only connectivity is stored; directionality is ignored for distance calculation.
    - Computes all-pairs shortest path distances between all identified locations using Breadth-First Search (BFS). Unreachable pairs are assigned a distance of infinity. Stores these distances in `self.distances`.
    - Parses the goal conditions (`task.goals`) to identify the target location for each relevant box and stores this mapping in `self.goal_locations`. Also maintains a set `self.boxes` of all boxes mentioned in the goals.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse Current State:** Extract the robot's current location (`robot_loc`) and the current location of each box (`box_locs`) from the set of facts in the input state (`node.state`). Perform validation checks to ensure the robot and relevant boxes are found and their locations are valid (exist in the precomputed map). If critical information is missing or invalid, return infinity.
    2.  **Identify Misplaced Boxes:** Iterate through all boxes relevant to the goal (`self.boxes`). For each box, compare its current location (`box_locs.get(box)`) with its target location (`self.goal_locations[box]`). Collect all boxes that are not currently at their target location into a list called `misplaced_boxes`.
    3.  **Goal Check:** If the `misplaced_boxes` list is empty, it means all goal conditions related to box positions are satisfied. The heuristic value is 0.
    4.  **Calculate Box Component (`H_box`):**
        - Initialize `H_box = 0`.
        - For each `box` in `misplaced_boxes`:
            - Get its current location `l_b = box_locs[box]`.
            - Get its goal location `g_b = self.goal_locations[box]`.
            - Look up the precomputed shortest path distance `d = self.distances[l_b][g_b]`. This estimates the minimum number of pushes required for this box.
            - If `d` is infinity, it signifies that the box cannot reach its goal location due to the static map layout (e.g., separated by walls). The state is considered unsolvable from this point, so return `float('inf')`.
            - Add `d` to `H_box`. The sum `H_box` represents the total estimated pushing effort required.
    5.  **Calculate Robot Component (`H_robot`):**
        - Initialize `min_robot_dist_to_box = float('inf')`.
        - This component estimates the cost for the robot to move to the first box it needs to push.
        - If `misplaced_boxes` is not empty:
            - For each `box` in `misplaced_boxes`:
                - Get the box's current location `l_b = box_locs[box]`.
                - Look up the precomputed shortest path distance `d = self.distances[robot_loc][l_b]` from the robot to the box.
                - Update `min_robot_dist_to_box = min(min_robot_dist_to_box, d)`.
            - If `min_robot_dist_to_box` remains infinity after checking all misplaced boxes, it means the robot cannot reach any of the boxes that need to be moved. The state is considered unsolvable, return `float('inf')`.
        - If `misplaced_boxes` is empty (goal state), `H_robot = 0`.
    6.  **Combine Components:** The final heuristic value is calculated as `H = H_box + H_robot`. This sums the estimated cost of all required pushes (`H_box`) and the estimated cost for the robot to initiate the pushing process (`H_robot`).
    7.  **Return Value:** Return the calculated heuristic value `H`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic: parses static info, builds graph,
        computes all-pairs shortest paths via BFS, and stores goal locations.
        """
        super().__init__(task) # Initialize base class if it has an __init__
        self.task = task
        self.static_facts = task.static
        self.goals = task.goals

        # --- Data Structures ---
        self.locations = set()
        # Adjacency list stores neighbors for BFS (direction ignored here)
        self.adj = {}
        # Stores all-pairs shortest path distances
        self.distances = {} # Format: {loc1: {loc2: distance, ...}, ...}
        # Stores the target location for each box mentioned in the goal
        self.goal_locations = {} # Format: {box_name: goal_location_name}
        # Stores the names of boxes that are part of the goal conditions
        self.boxes = set()

        # --- Identify All Locations ---
        # Scan static facts, initial state, and goals to find all location names
        potential_locations = set()
        for fact_set in [self.static_facts, task.initial_state, self.goals]:
            for fact in fact_set:
                parts = get_parts(fact)
                if not parts: continue
                # Check predicates known to involve locations
                if parts[0] == 'adjacent' and len(parts) >= 3:
                    potential_locations.add(parts[1])
                    potential_locations.add(parts[2])
                elif parts[0] in ['at', 'at-robot', 'clear'] and len(parts) >= 2:
                    potential_locations.add(parts[-1]) # Location is typically last arg

        self.locations = potential_locations
        if not self.locations:
             # This would indicate a problem with the PDDL instance definition
             print("Warning: No locations were identified from the PDDL instance.", file=sys.stderr)

        # --- Build Adjacency List (for BFS distance calculation) ---
        # Initialize adjacency list for all found locations
        for loc in self.locations:
            self.adj[loc] = []
        # Populate adjacency list from static 'adjacent' facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            # Ensure fact is correctly parsed 'adjacent loc1 loc2 direction'
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                # Add edge only if both locations are known
                if loc1 in self.locations and loc2 in self.locations:
                    # Add edges for BFS (undirected reachability)
                    # Check if neighbor already added to avoid duplicates if PDDL is symmetric
                    if loc2 not in self.adj.get(loc1, []):
                         self.adj.setdefault(loc1, []).append(loc2)
                    # If adjacency is not guaranteed to be symmetric in PDDL, add reverse edge:
                    # if loc1 not in self.adj.get(loc2, []):
                    #     self.adj.setdefault(loc2, []).append(loc1)
                    # Assuming the provided PDDL implies symmetry or provides both directions

        # --- Compute All-Pairs Shortest Paths using BFS ---
        for start_loc in self.locations:
            # Initialize distances from start_loc to all other locations as infinity
            self.distances[start_loc] = {loc: float('inf') for loc in self.locations}
            # Distance from a location to itself is 0
            if start_loc in self.locations: # Check if start_loc is valid
                 self.distances[start_loc][start_loc] = 0
            else:
                 continue # Skip if start_loc is somehow invalid

            # Initialize BFS queue
            q = deque([start_loc])

            while q:
                curr_loc = q.popleft()
                current_dist = self.distances[start_loc][curr_loc]

                # Explore neighbors
                for neighbor in self.adj.get(curr_loc, []):
                    # Process neighbor only if it's a known location and not yet reached
                    if neighbor in self.distances[start_loc] and self.distances[start_loc][neighbor] == float('inf'):
                        self.distances[start_loc][neighbor] = current_dist + 1
                        q.append(neighbor)

        # --- Parse Goal Locations and Identify Relevant Boxes ---
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Expecting goal facts like "(at boxName locationName)"
            if parts and parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Store goal location only if the location is valid
                if loc in self.locations:
                    self.goal_locations[box] = loc
                    self.boxes.add(box) # Keep track of boxes involved in goals
                else:
                    # This indicates an issue with the goal specification in the PDDL file
                    print(f"Warning: Goal location '{loc}' for box '{box}' is not a known location.", file=sys.stderr)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Returns an estimate of the cost to reach the goal from the current state.
        """
        state = node.state

        # --- Get Current State Information (Robot and Box Locations) ---
        robot_loc = None
        box_locs = {} # Stores current location for boxes relevant to the goal

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            # Find robot location
            if parts[0] == 'at-robot' and len(parts) == 2:
                # Basic check for multiple robot locations (shouldn't happen in standard Sokoban)
                if robot_loc is not None and robot_loc != parts[1]:
                     print(f"Warning: Multiple different robot locations detected ('{robot_loc}', '{parts[1]}'). Using first found.", file=sys.stderr)
                if robot_loc is None:
                     robot_loc = parts[1]
            # Find locations of boxes relevant to the goal
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.boxes: # Check if this object is a box we care about
                     # Basic check for multiple locations for the same box
                     if obj in box_locs and box_locs[obj] != loc:
                          print(f"Warning: Box '{obj}' found at multiple locations ('{box_locs[obj]}', '{loc}'). Using first found.", file=sys.stderr)
                     if obj not in box_locs:
                          box_locs[obj] = loc

        # --- Validate State Information ---
        # Check if robot location was found
        if robot_loc is None:
            # print("Error: Robot location ('at-robot') not found in the current state.", file=sys.stderr)
            return float('inf') # Cannot compute heuristic without robot position
        # Check if robot location is valid
        if robot_loc not in self.locations:
            # print(f"Error: Robot location '{robot_loc}' is not a known location.", file=sys.stderr)
            return float('inf')

        # --- Identify Misplaced Boxes ---
        misplaced_boxes = []
        for box in self.boxes: # Iterate over boxes that have a goal location
            goal_loc = self.goal_locations.get(box) # Should always exist if box is in self.boxes
            current_loc = box_locs.get(box)

            # Check if the box's current location was found in the state
            if current_loc is None:
                 # print(f"Error: Box '{box}' (required for goal) not found in the current state ('at' fact missing).", file=sys.stderr)
                 return float('inf') # State seems inconsistent or box is lost

            # Check if the box's current location is a valid, known location
            if current_loc not in self.locations:
                 # print(f"Error: Current location '{current_loc}' for box '{box}' is not a known location.", file=sys.stderr)
                 return float('inf')

            # If the box is not at its goal location, add it to the list
            if current_loc != goal_loc:
                misplaced_boxes.append(box)

        # --- Goal Check ---
        # If there are no misplaced boxes, the goal state (regarding boxes) is reached
        if not misplaced_boxes:
            return 0

        # --- Calculate Heuristic Components: H_box and H_robot ---
        h_box = 0 # Sum of distances for boxes to reach their goals
        min_robot_dist_to_box = float('inf') # Min distance from robot to any misplaced box

        # Calculate sum of box-to-goal distances (H_box)
        for box in misplaced_boxes:
            current_loc = box_locs[box]
            goal_loc = self.goal_locations[box]

            # Retrieve precomputed distance; use .get for safety, though keys should exist
            dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

            # If distance is infinity, the goal is unreachable for this box
            if dist == float('inf'):
                # print(f"Info: Box '{box}' at '{current_loc}' cannot reach goal '{goal_loc}'. State potentially unsolvable.", file=sys.stderr)
                return float('inf')
            h_box += dist

        # Calculate minimum robot-to-misplaced-box distance (H_robot component)
        for box in misplaced_boxes:
            current_box_loc = box_locs[box]
            # Retrieve precomputed distance from robot to the box's current location
            dist = self.distances.get(robot_loc, {}).get(current_box_loc, float('inf'))
            min_robot_dist_to_box = min(min_robot_dist_to_box, dist)

        # If robot cannot reach any misplaced box, the state is likely unsolvable
        if min_robot_dist_to_box == float('inf'):
             # print(f"Info: Robot at '{robot_loc}' cannot reach any of the misplaced boxes. State potentially unsolvable.", file=sys.stderr)
             return float('inf')

        # --- Combine Components ---
        # Heuristic is the sum of total box push distance and distance to nearest misplaced box
        heuristic_value = h_box + min_robot_dist_to_box

        # The heuristic value should naturally be non-negative if distances are non-negative
        return heuristic_value
