import sys
from collections import deque
# Import the base class for heuristics.
# The exact path might depend on the planner's structure.
from heuristics.heuristic_base import Heuristic

# Define a large number to represent infinity or unsolvable states.
# Using a large integer avoids potential issues with float('inf').
LARGE_NUMBER = 1_000_000_000

def get_parts(fact_string):
    """
    Safely extracts predicate and arguments from a PDDL fact string like '(predicate arg1 arg2)'.
    It removes the surrounding parentheses and splits the string by spaces.
    Handles potential empty strings or malformed facts gracefully by returning an empty list.

    Args:
        fact_string (str): The PDDL fact string.

    Returns:
        list: A list containing the predicate name followed by its arguments,
              or an empty list if the input is malformed.
    """
    if not fact_string or len(fact_string) < 2 or fact_string[0] != '(' or fact_string[-1] != ')':
        # Return an empty list for malformed or empty strings.
        return []
    # Remove parentheses and split by space.
    return fact_string[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Sokoban domain, designed for use
    with Greedy Best-First Search.

    # Summary
    This heuristic estimates the cost (number of actions) to reach the goal state
    from a given state in a Sokoban problem. It combines two main components:
    1. The sum of shortest path distances for each box from its current location
       to its designated goal location. This estimates the total number of 'push' actions.
    2. The shortest path distance for the robot to reach the location of the
       *nearest* box that is currently not in its goal location. This estimates
       the initial 'move' actions required for the robot to start pushing.
    The heuristic aims to provide informative guidance for greedy search by
    prioritizing states where boxes are closer to their goals and the robot is
    well-positioned to move them. It is not guaranteed to be admissible.

    # Assumptions
    - Both 'move' and 'push' actions have a uniform cost of 1.
    - The goal state is defined by a set of `(at boxN locX)` predicates,
      specifying a unique target location for each relevant box.
    - The map layout and connectivity are static, defined by `adjacent` predicates
      in the problem's initial state or domain file. Walls are implicitly defined
      by the absence of adjacency.
    - The state provided to the `__call__` method is a `frozenset` of PDDL fact strings.
    - The `task` object passed during initialization contains `goals` (goal facts)
      and `static` facts (like `adjacent` predicates).

    # Heuristic Initialization (`__init__`)
    - **Parse Connectivity:** Reads static `adjacent` facts to build an undirected graph
      representing the traversable locations and their connections (`self.adj`).
      This graph is used for calculating shortest path distances.
    - **Identify Locations:** Collects all unique location names mentioned in `adjacent` facts
      (`self.locations`).
    - **Parse Goals:** Extracts the target location for each box specified in the
      `task.goals` (`self.goal_locations`) and identifies the set of boxes that
      are part of the goal condition (`self.goal_boxes`).
    - **Precompute Distances:** Calculates all-pairs shortest path distances between
      all locations using Breadth-First Search (BFS) starting from each location.
      The results are stored in `self.distances[loc1][loc2]`. This distance represents
      the minimum number of steps (moves or pushes) required between two locations,
      ignoring dynamic obstacles like the robot or other boxes. Unreachable locations
      have a distance of `float('inf')`.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. **Goal Check:** Verify if the current state (`node.state`) already satisfies all
       goal conditions defined in `self.goals`. If yes, the heuristic value is 0,
       as no more actions are needed.
    2. **State Parsing:** Extract the current location of the robot (`robot_loc`) and
       the current location of each box relevant to the goal (`box_locs`) from the state facts.
    3. **Validation:** If the robot's location cannot be determined from the state,
       return a large value (`LARGE_NUMBER`) to indicate an error or invalid state.
    4. **Box Distance Sum Calculation (`box_distance_sum`):**
       - Iterate through each box listed in `self.goal_locations`.
       - If a box is not currently at its designated goal location:
         - Retrieve the precomputed shortest path distance from its current location
           to its goal location using `self.distances`.
         - If the distance is infinite (`float('inf')`), it means the box cannot reach
           its goal. The state is considered unsolvable, so return `LARGE_NUMBER`.
         - Otherwise, add this distance to `box_distance_sum`.
       - Keep track of all boxes that are currently misplaced.
    5. **Robot Distance Calculation (`min_robot_to_box_dist`):**
       - If there are any misplaced boxes:
         - Find the minimum precomputed shortest path distance from the robot's
           current location (`robot_loc`) to the current location of *any* of the
           misplaced boxes.
         - If the robot cannot reach *any* of the misplaced boxes (all distances are
           infinite), the state is considered unsolvable, so return `LARGE_NUMBER`.
         - Set `min_robot_to_box_dist` to this minimum distance.
       - If all boxes are already in their goal locations (no misplaced boxes),
         set `min_robot_to_box_dist` to 0.
    6. **Combine Components:** The final heuristic value is calculated as:
       `h = box_distance_sum + min_robot_to_box_dist`.
       This sum estimates the total pushes needed plus the initial robot moves required.
    7. **Return Value:** Return the calculated heuristic value `h` as an integer.
       If the state was determined to be unsolvable at any point, `LARGE_NUMBER` is returned.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class if needed by the framework
        self.goals = task.goals
        static_facts = task.static

        # --- 1. Build adjacency graph and identify locations ---
        self.adj = {} # Adjacency list: {location: [neighbor1, neighbor2, ...]}
        self.locations = set() # Set of all unique location names
        adj_relations = set() # Store unique undirected edges as tuples (loc1, loc2) with loc1 <= loc2

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            # Process (adjacent loc1 loc2 dir) facts
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                # Add locations to the set of known locations
                self.locations.add(loc1)
                self.locations.add(loc2)
                # Create a representation for the undirected edge
                # Sort locations alphabetically to ensure uniqueness (e.g., ('loc_1_1', 'loc_1_2'))
                edge = tuple(sorted((loc1, loc2)))
                # Add the edge if it connects two different locations
                if loc1 != loc2:
                    adj_relations.add(edge)

        # Build the adjacency list from the collected unique edges
        for loc1, loc2 in adj_relations:
            # Ensure both locations have entries in the adjacency dictionary
            if loc1 not in self.adj: self.adj[loc1] = []
            if loc2 not in self.adj: self.adj[loc2] = []
            # Add the connection in both directions for the undirected graph
            self.adj[loc1].append(loc2)
            self.adj[loc2].append(loc1)

        # Ensure all locations identified exist as keys in self.adj, even if isolated
        for loc in self.locations:
            if loc not in self.adj:
                self.adj[loc] = []

        # --- 2. Parse goal locations for boxes ---
        self.goal_locations = {} # Stores {box_name: goal_location}
        self.goal_boxes = set() # Stores names of boxes that are part of the goal

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            # Expect goal facts like (at box_name location_name)
            if parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Check if the specified location is valid (exists in our set of locations)
                if loc in self.locations:
                    self.goal_locations[box] = loc
                    self.goal_boxes.add(box)
                else:
                    # Log a warning if a goal refers to an unknown location
                    print(f"Warning: Goal location '{loc}' for box '{box}' "
                          f"in goal '{goal}' is not among known locations. Ignoring this goal condition.")

        # --- 3. Precompute all-pairs shortest paths using BFS ---
        self.distances = {} # Stores {start_loc: {end_loc: distance}}

        # Run BFS starting from each location to find shortest paths to all others
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)


    def _bfs(self, start_node):
        """
        Performs Breadth-First Search (BFS) starting from `start_node` on the graph
        defined by `self.adj` to find shortest path distances.

        Args:
            start_node (str): The location name to start the BFS from.

        Returns:
            dict: A dictionary mapping each location name (str) to its shortest distance (int)
                  from `start_node`. Unreachable locations will have a distance of `float('inf')`.
        """
        # Initialize distances: infinity for all locations, 0 for the start node
        distances = {loc: float('inf') for loc in self.locations}

        # Validate start_node before proceeding
        if start_node not in self.locations:
            print(f"Error: BFS start node '{start_node}' is not a known location.")
            return distances # Return distances as all infinity

        distances[start_node] = 0
        # Use deque for efficient queue operations
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Explore neighbors of the current location
            if current_loc in self.adj:
                for neighbor in self.adj[current_loc]:
                    # If neighbor is a valid location and we haven't found a path yet (dist is inf)
                    if neighbor in distances and distances[neighbor] == float('inf'):
                        # Update distance and add neighbor to the queue
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.

        Args:
            node: A state node object, expected to have an attribute `state` which is a
                  frozenset of PDDL fact strings representing the current state.

        Returns:
            int: The estimated cost (number of actions) to reach the goal state.
                 Returns 0 if the state is already a goal state.
                 Returns `LARGE_NUMBER` if the state is determined to be unsolvable.
        """
        state = node.state

        # --- 1. Goal Check ---
        # Check if all goal facts are present in the current state
        is_goal = all(goal_fact in state for goal_fact in self.goals)
        if is_goal:
            return 0 # State is already a goal state

        # --- 2. State Parsing ---
        robot_loc = None
        box_locs = {} # Stores {box_name: current_location}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            # Find robot location: (at-robot location)
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            # Find box locations: (at box_name location)
            elif predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Store location only if this object is a box relevant to the goal
                if obj in self.goal_boxes:
                    box_locs[obj] = loc

        # --- 3. Validation ---
        # If robot location wasn't found, the state is invalid or incomplete
        if robot_loc is None:
            print("Error: Robot location ('at-robot') not found in the current state.")
            return LARGE_NUMBER

        # --- 4. Box Distance Sum Calculation ---
        box_distance_sum = 0
        misplaced_boxes = [] # List of box names not currently at their goal location

        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locs.get(box)

            # If a goal box's location is missing from the state, it's an error
            if current_box_loc is None:
                print(f"Error: Location for goal box '{box}' not found in the current state.")
                return LARGE_NUMBER

            # If the box is not at its goal location
            if current_box_loc != goal_loc:
                try:
                    # Retrieve the precomputed shortest path distance
                    dist = self.distances[current_box_loc][goal_loc]
                except KeyError:
                    # This might happen if locations parsed from state aren't in self.locations
                    print(f"Error: Distance lookup failed for box '{box}' from "
                          f"'{current_box_loc}' to '{goal_loc}'. Assuming unreachable.")
                    dist = float('inf') # Treat as unreachable

                # Check if the goal is reachable for this box
                if dist == float('inf'):
                    # If any box cannot reach its goal, the state is unsolvable
                    return LARGE_NUMBER

                # Add the distance to the sum and mark the box as misplaced
                box_distance_sum += dist
                misplaced_boxes.append(box)

        # --- 5. Robot Distance Calculation ---
        min_robot_to_box_dist = 0
        # Only calculate if there are boxes that need moving
        if misplaced_boxes:
            min_dist = float('inf')
            # Track if the robot can reach at least one misplaced box
            possible_robot_reach = False

            for box in misplaced_boxes:
                box_loc = box_locs[box]
                try:
                    # Get distance from robot's current location to the box's location
                    dist = self.distances[robot_loc][box_loc]

                    # If the box is reachable by the robot
                    if dist != float('inf'):
                        possible_robot_reach = True
                        # Update the minimum distance found so far
                        min_dist = min(min_dist, dist)
                except KeyError:
                    # Handle potential lookup errors (e.g., robot_loc is invalid)
                    print(f"Error: Distance lookup failed for robot at '{robot_loc}' "
                          f"to box '{box}' at '{box_loc}'.")
                    # Continue checking other boxes

            # If the robot cannot reach any of the boxes that need moving, state is unsolvable
            if not possible_robot_reach:
                 return LARGE_NUMBER

            # Store the minimum distance found
            min_robot_to_box_dist = min_dist

        # --- 6. Combine Components ---
        # Heuristic is the sum of box distances plus the robot's distance to the nearest misplaced box
        heuristic_value = box_distance_sum + min_robot_to_box_dist

        # --- 7. Return Value ---
        # Ensure the value is non-negative (should be, but max(0,...) is safe)
        # Convert to integer (distances are integers from BFS)
        return int(round(max(0, heuristic_value)))

