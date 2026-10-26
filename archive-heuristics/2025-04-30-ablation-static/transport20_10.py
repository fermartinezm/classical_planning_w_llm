from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions from examples
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS implementation
def bfs(graph, start_node, all_nodes):
    """
    Performs BFS from start_node to find distances to all other nodes within the set all_nodes.
    Returns a dictionary {node: distance}.
    """
    distances = {node: float('inf') for node in all_nodes}
    if start_node not in all_nodes:
        # Start node is not in the set of relevant nodes, cannot reach anything relevant
        return distances

    distances[start_node] = 0
    queue = [start_node]
    visited = {start_node}

    while queue:
        current_node = queue.pop(0)

        # Check if current_node exists as a key in graph and has neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                # Only process neighbors that are relevant locations and haven't been visited
                if neighbor in all_nodes and neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

    return distances


class transportHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It simplifies the problem by ignoring vehicle capacity
    and assuming vehicles are always available when needed for a package. The cost
    is calculated independently for each package not at its goal.

    # Assumptions
    - Vehicle capacity constraints are ignored.
    - Vehicles are assumed to be available to pick up packages when needed.
    - Roads are bidirectional (inferred from example instances).
    - All locations relevant to the problem (initial package/vehicle locations,
      goal package locations) are connected by roads in solvable problems.

    # Heuristic Initialization
    - Identify all package and vehicle objects by scanning initial state and goals.
    - Extract goal locations for each package.
    - Build a graph of locations based on `road` predicates.
    - Identify all locations relevant to the problem (mentioned in initial 'at', goal 'at', or 'road' facts).
    - Compute shortest path distances between all pairs of relevant locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is the sum of estimated costs for each
    package that is not yet at its goal location.

    For each package `p` with goal location `l_goal`:
    1. Check if the package is already at `l_goal`. If `(at p l_goal)` is in the state,
       the cost for this package is 0.
    2. If the package is at a different location `l_curr` (i.e., `(at p l_curr)` is in the state
       where `l_curr != l_goal`):
       - This package needs to be picked up (1 action).
       - It needs to be transported from `l_curr` to `l_goal`. The minimum number of
         drive actions required is the shortest path distance between `l_curr` and `l_goal`.
       - It needs to be dropped at `l_goal` (1 action).
       - The estimated cost for this package is 1 (pick-up) + distance(`l_curr`, `l_goal`) (drive) + 1 (drop).
    3. If the package is inside a vehicle `v` (i.e., `(in p v)` is in the state):
       - Find the current location `l_v_curr` of vehicle `v` (i.e., `(at v l_v_curr)` is in the state).
       - The package needs to be transported from `l_v_curr` to `l_goal` while inside the vehicle.
         The minimum number of drive actions is the shortest path distance between `l_v_curr` and `l_goal`.
       - It needs to be dropped at `l_goal` (1 action).
       - The estimated cost for this package is distance(`l_v_curr`, `l_goal`) (drive) + 1 (drop).

    The total heuristic value is the sum of these estimated costs for all packages.
    If a location is unreachable, the distance is considered infinite, resulting in a
    very high heuristic value, discouraging paths leading to such states (though
    unreachable goals usually mean the problem is unsolvable).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts,
           identifying objects, and precomputing distances."""
        # Assuming Heuristic base class exists and takes task in init
        # super().__init__(task)
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Identify packages and vehicles and collect all relevant locations
        self.packages = set()
        self.vehicles = set()
        all_locations = set()

        # First pass: Identify packages from 'in' facts in initial state and goals
        for fact in initial_state | self.goals:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == 'in' and len(parts) == 3:
                # (in ?package ?vehicle)
                self.packages.add(parts[1])

        # Second pass: Identify vehicles and collect locations from 'at' facts in initial state and goals
        for fact in initial_state | self.goals:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'at' and len(parts) == 3:
                 # (at ?locatable ?location)
                 obj = parts[1]
                 loc = parts[2]
                 all_locations.add(loc)
                 if obj not in self.packages:
                     self.vehicles.add(obj)

        # Collect locations from road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                all_locations.add(l1)
                all_locations.add(l2)

        # 2. Extract goal locations for packages
        self.package_goals = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure the goal fact is an 'at' predicate for a package
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1] in self.packages:
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
                # Goal locations are already added to all_locations in the previous step

        # 3. Build location graph from road facts
        graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                graph.setdefault(l1, []).append(l2)
                graph.setdefault(l2, []).append(l1) # Assuming roads are bidirectional

        # Ensure all relevant locations are in the graph keys, even if they have no roads
        # This is important for BFS to iterate over all potential start nodes
        for loc in all_locations:
             graph.setdefault(loc, [])

        # 4. Compute all-pairs shortest paths for relevant locations
        self.distances = {}
        relevant_locations = list(all_locations) # Use a list for consistent iteration order
        for start_loc in relevant_locations:
            distances_from_start = bfs(graph, start_loc, relevant_locations)
            for end_loc in relevant_locations:
                 self.distances[(start_loc, end_loc)] = distances_from_start[end_loc]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current locations/status of packages and vehicles
        pkg_locs = {} # package -> location (if at a location)
        pkg_in_veh = {} # package -> vehicle (if in a vehicle)
        veh_locs = {} # vehicle -> location (where the vehicle is)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    pkg_locs[obj] = loc
                elif obj in self.vehicles:
                    veh_locs[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                 pkg, veh = parts[1], parts[2]
                 # Ensure they are identified objects before adding
                 if pkg in self.packages and veh in self.vehicles:
                     pkg_in_veh[pkg] = veh


        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package not at its goal
        for package, goal_location in self.package_goals.items():
            # Check if package is already at goal
            if package in pkg_locs and pkg_locs[package] == goal_location:
                continue # Package is at goal, cost is 0 for this package

            # Package is not at goal, calculate its contribution to the heuristic
            package_cost = 0

            if package in pkg_locs:
                # Package is on the ground at pkg_locs[package]
                current_location = pkg_locs[package]
                # Cost: pick-up (1) + drive (distance) + drop (1)
                # Look up distance, default to infinity if locations are not in our precomputed set
                drive_distance = self.distances.get((current_location, goal_location), float('inf'))

                if drive_distance == float('inf'):
                    # Goal is unreachable from current location, very high cost
                    return float('inf') # Problem likely unsolvable from this state
                package_cost = 1 + drive_distance + 1

            elif package in pkg_in_veh:
                # Package is inside a vehicle pkg_in_veh[package]
                vehicle = pkg_in_veh[package]
                # Find vehicle's location
                if vehicle not in veh_locs:
                     # Vehicle location unknown (shouldn't happen in valid state), high cost
                     # This could indicate an inconsistent state representation
                     return float('inf')
                vehicle_location = veh_locs[vehicle]
                # Cost: drive (distance) + drop (1)
                # Look up distance, default to infinity
                drive_distance = self.distances.get((vehicle_location, goal_location), float('inf'))

                if drive_distance == float('inf'):
                     # Goal is unreachable from vehicle's current location, very high cost
                     return float('inf') # Problem likely unsolvable from this state
                package_cost = drive_distance + 1
            else:
                # Package status unknown (e.g., package exists but is neither at a location nor in a vehicle)
                # This shouldn't happen in a valid state based on domain effects.
                # Assign a high cost to penalize such states.
                return float('inf') # Or a large constant like 1000

            total_cost += package_cost

        # If the loop finishes, total_cost is the sum of costs for all packages
        # not at their goal. If all were at goal, total_cost is 0.
        return total_cost
