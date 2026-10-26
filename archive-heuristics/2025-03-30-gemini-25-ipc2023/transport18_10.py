import math
from collections import deque
from heuristics.heuristic_base import Heuristic # Ensure this import path is correct for the environment

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string."""
    # Handles potential extra spaces, removes parentheses
    try:
        # Remove leading/trailing whitespace, then remove '(' and ')' before splitting
        return fact.strip()[1:-1].split()
    except IndexError:
        # Handle empty or malformed facts if necessary
        return []

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Transport domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    by focusing on the state of the packages relative to their goal locations.
    It sums the estimated costs for each package individually. The cost for a
    package includes the necessary pickup action (if on the ground and not at the
    goal), the necessary drop action (if not already at the goal), and the
    driving distance required to move the package from its current location (or
    the location of the vehicle carrying it) to its goal location.

    # Assumptions
    - The primary goal conditions are of the form `(at package location)`. Other
      goal conditions are ignored by this heuristic calculation but are checked
      by the `task.goal_reached` method.
    - Road connections define the possible movements for vehicles. The heuristic
      uses the `(road l1 l2)` facts provided in the static information. It assumes
      the graph might be directed based on these facts unless pairs like
      `(road l1 l2)` and `(road l2 l1)` are both present.
    - The cost of driving between two locations is the number of `drive` actions
      in the shortest path, computed via Breadth-First Search (BFS).
    - The heuristic simplifies the problem by ignoring vehicle capacity constraints.
      This means it might underestimate the cost if multiple trips are needed due
      to capacity limits.
    - It simplifies by not explicitly modeling which vehicle serves which package
      or the cost for vehicles to reach packages for pickup. It focuses on the
      actions directly related to the package's journey (pickup, drive-while-carrying, drop).

    # Heuristic Initialization
    - Identifies package, vehicle, and location objects by parsing the initial
      state and static facts. It relies on the presence of objects in relevant
      predicates (`at`, `in`, `capacity`, `road`). A more robust implementation
      might use typed object lists if provided by the PDDL parser/task object.
    - Extracts the goal location for each identified package from the task's goals.
    - Parses static `road` facts to build a graph representation of the locations.
    - Computes all-pairs shortest path distances between locations using
      Breadth-First Search (BFS) and stores these distances. Infinite distance
      indicates unreachability.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the total heuristic cost `h` to 0.
    2. Parse the current state (`node.state`) to determine:
        - The location `p_loc` for each package `p` currently on the ground (`at p p_loc`).
        - Which vehicle `v` carries which package `p` (`in p v`).
        - The location `v_loc` for each vehicle `v` (`at v v_loc`).
        - Store this information efficiently, e.g., in dictionaries mapping packages
          to their status (`('at', location)` or `('in', vehicle)`).
    3. For each package `p` that has a defined goal location `g_loc`:
        a. Retrieve the current status of `p` from the parsed state.
        b. **If `p` is on the ground at `p_loc`**:
           - If `p_loc == g_loc`, cost for `p` is 0.
           - If `p_loc != g_loc`, calculate `dist = shortest_path(p_loc, g_loc)`.
             If `dist` is infinity, return infinity (unreachable goal).
             Otherwise, cost for `p` is `1 (pickup) + dist + 1 (drop)`.
        c. **If `p` is in vehicle `v`**:
           - Find the vehicle's location `v_loc`. If unknown, return infinity (inconsistent state).
           - If `v_loc == g_loc`, the cost for `p` is `1 (drop)`.
           - If `v_loc != g_loc`, calculate `dist = shortest_path(v_loc, g_loc)`.
             If `dist` is infinity, return infinity.
             Otherwise, cost for `p` is `dist + 1 (drop)`.
        d. **If `p`'s status is unknown**:
           - Check if the fact `(at p g_loc)` exists in the state. If yes, cost is 0.
           - If no, the state is inconsistent or the package is missing; return infinity.
        e. Add the calculated cost for package `p` to the total heuristic cost `h`.
    4. Check if the current state is a goal state using `self.task.goal_reached(node.state)`. If it is, return 0.
    5. Return the total calculated cost `h`. If `h` is 0 but the state is not a goal state (due to non-package goals), it still returns 0, letting the search algorithm handle the final goal test.
    """

    def __init__(self, task):
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # --- Object Identification ---
        self.packages = set()
        self.vehicles = set()
        self.locations = set()

        # Combine initial state and static facts for parsing objects
        all_facts = task.initial_state.union(static_facts)
        roads_tuples = set()

        # Infer objects and locations from all known facts
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'at':
                obj, loc = args[0], args[1]
                self.locations.add(loc)
                # Basic type inference (improve if task object has types)
                # Check if obj is already known, otherwise infer
                if obj.startswith('p'): self.packages.add(obj)
                elif obj.startswith('v'): self.vehicles.add(obj)
            elif pred == 'capacity':
                 vehicle = args[0]
                 self.vehicles.add(vehicle)
            elif pred == 'in':
                 package, vehicle = args[0], args[1]
                 self.packages.add(package)
                 self.vehicles.add(vehicle)
            elif pred == 'road':
                l1, l2 = args[0], args[1]
                self.locations.add(l1)
                self.locations.add(l2)
                roads_tuples.add((l1, l2))
            # Ignore capacity-predecessor for object identification

        # Extract goal locations specifically for identified packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 # Only consider goals for objects identified as packages
                 if obj in self.packages:
                     self.goal_locations[obj] = loc
                     self.locations.add(loc) # Ensure goal locations are known

        # --- Compute Distances ---
        self.distances = self._compute_all_pairs_shortest_paths(self.locations, roads_tuples)


    def _compute_all_pairs_shortest_paths(self, locations, roads):
        """Computes shortest path distances using BFS."""
        # Initialize distances: infinity for all pairs, 0 for self-loops
        dist = {loc: {other_loc: float('inf') for other_loc in locations} for loc in locations}
        for loc in locations:
            if loc in dist: # Check if location is actually in the keys
                 dist[loc][loc] = 0

        # Build adjacency list for graph traversal
        adj = {loc: [] for loc in locations}
        for u, v in roads:
            # Add edge only if both locations are known and exist in adj keys
            if u in adj and v in locations:
                 adj[u].append(v)
            # Note: If roads are bidirectional, PDDL should define both (road u v) and (road v u)

        # Run BFS from each location that exists in the adjacency list keys
        for start_node in locations:
            if start_node not in adj: continue # Skip if location has no entry (e.g., isolated)

            # Check if start_node has outgoing edges or self-loop distance is 0
            # If dist[start_node][start_node] is inf, it means the location wasn't properly added.
            if dist.get(start_node, {}).get(start_node, float('inf')) != 0:
                 continue # Skip invalid start nodes

            queue = deque([(start_node, 0)])
            # Keep track of visited nodes in this specific BFS run to avoid cycles and redundant work
            visited_in_bfs = {start_node}

            while queue:
                current_node, d = queue.popleft()

                # Update the final distance matrix for the start_node
                # We already set dist[start_node][start_node] = 0
                # For other nodes, the distance is d
                if current_node != start_node:
                    dist[start_node][current_node] = d

                # Explore neighbors
                for neighbor in adj.get(current_node, []):
                    if neighbor not in visited_in_bfs:
                        visited_in_bfs.add(neighbor)
                        # Check if neighbor is a valid location before adding
                        if neighbor in locations:
                             queue.append((neighbor, d + 1))
                        # else: neighbor mentioned in road but not in locations set? Ignore.

        return dist


    def __call__(self, node):
        """Calculate the heuristic value for the given state."""
        # Optimization: If the state is the goal, heuristic is 0.
        if self.task.goal_reached(node.state):
            return 0

        state = node.state
        h_value = 0

        # --- Parse current state ---
        package_locs = {} # package -> location
        package_in = {}   # package -> vehicle
        vehicle_locs = {} # vehicle -> location

        # Store current status for faster lookup
        current_package_state = {} # package -> ('at', location) or ('in', vehicle)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at':
                # Ensure args has at least 2 elements
                if len(args) >= 2:
                    obj, loc = args[0], args[1]
                    if obj in self.packages:
                        package_locs[obj] = loc
                        current_package_state[obj] = ('at', loc)
                    elif obj in self.vehicles:
                        vehicle_locs[obj] = loc
            elif predicate == 'in':
                 # Ensure args has at least 2 elements
                 if len(args) >= 2:
                    package, vehicle = args[0], args[1]
                    if package in self.packages:
                        package_in[package] = vehicle
                        current_package_state[package] = ('in', vehicle)

        # --- Calculate cost for each package ---
        for package, goal_loc in self.goal_locations.items():
            cost_p = 0
            state_info = current_package_state.get(package)

            if state_info:
                status, value = state_info

                if status == 'at':
                    current_loc = value
                    if current_loc != goal_loc:
                        # Package on ground, not at goal
                        # Use .get for safer dictionary access in distances
                        distance = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                        if distance == float('inf'):
                            return float('inf') # Unreachable goal state
                        cost_p = 1 + distance + 1 # pickup + drive + drop

                elif status == 'in':
                    vehicle = value
                    # Package in vehicle, find vehicle location
                    current_loc = vehicle_locs.get(vehicle)
                    if current_loc is None:
                         # Vehicle location unknown - inconsistent state?
                         # This might happen if a vehicle is mentioned 'in' but not 'at'
                         return float('inf')

                    if current_loc != goal_loc:
                         distance = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                         if distance == float('inf'):
                             return float('inf') # Unreachable goal state
                         cost_p = distance + 1 # drive + drop
                    else:
                         # Vehicle is at goal loc, just need to drop
                         cost_p = 1 # drop
            else:
                # Package state not found in current facts ('at' or 'in').
                # This implies the package might already be at the goal, or is missing.
                # We construct the goal fact string to check against the raw state set.
                goal_fact_str = f"(at {package} {goal_loc})"
                if goal_fact_str not in state:
                     # Package not found via parsed state AND not explicitly at goal in state facts.
                     # This indicates an issue (e.g., package doesn't exist in current state).
                     # print(f"Warning: Package {package} state unknown and not at goal {goal_loc}")
                     return float('inf')
                # else: Package is implicitly at goal (cost_p remains 0).

            h_value += cost_p

        # Heuristic value calculated based on package goals.
        # Return the sum. If h_value is 0, it means all package goals are met.
        # The initial check ensures 0 is returned if the overall task goal is met.
        return h_value
