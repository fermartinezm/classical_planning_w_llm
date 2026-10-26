import collections
import math
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It considers the cost of picking up, dropping,
    and driving packages based on shortest path distances in the road network.

    # Assumptions
    - The primary cost is associated with moving packages between locations.
    - Picking up a package costs 1 action.
    - Dropping a package costs 1 action.
    - Driving a vehicle between locations costs the shortest path distance
      in the road network (each road segment traversal is 1 action).
    - Vehicle capacity constraints are ignored. The heuristic assumes a vehicle
      is available and has sufficient capacity if it's at the required location.
    - The heuristic sums the estimated costs for each package independently.

    # Heuristic Initialization
    - The goal location for each package is extracted from the task goals.
    - The road network is built as an undirected graph from the static facts
      defining road connectivity `(road l1 l2)`.
    - Shortest path distances between all pairs of locations connected by roads
      are precomputed using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1. Initialize the total heuristic cost to 0.
    2. Determine the current physical location of every locatable object (packages and vehicles).
       - Iterate through the state facts to find objects directly `(at ?x ?l)` a location.
       - Iterate again to find packages `(in ?p ?v)` a vehicle, and record which vehicle
         each package is in.
       - After finding vehicle locations, update the physical location of packages
         that are inside vehicles to be the physical location of the vehicle they are in.
    3. For each package `p` that has a goal location `goal_l` (extracted during initialization):
       - Check if the package is already at its goal location on the ground, i.e., if `(at p goal_l)` is a fact in the current state. If yes, the cost for this package is 0, and we move to the next package.
       - If the package is not at its goal location:
         - Find the package's current physical location, `current_l`. If the location is unknown (e.g., package in vehicle but vehicle location not in state), return infinity.
         - Get the shortest path distance `dist` from `current_l` to `goal_l` from the precomputed distances. If `goal_l` is unreachable from `current_l`, return infinity.
         - Determine if the package is currently inside a vehicle by checking the record created in step 2.
         - If the package is inside a vehicle:
           - The estimated cost for this package is the distance the vehicle needs to drive (`dist`) plus the cost to drop the package (1 action). Total: `dist + 1`.
         - If the package is on the ground:
           - The estimated cost for this package is the cost to pick it up (1 action) at `current_l`, plus the distance to drive (`dist`) to `goal_l`, plus the cost to drop it (1 action) at `goal_l`. Total: `1 + dist + 1`.
       - Add the estimated cost for this package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        road network graph, and precomputing shortest path distances.
        """
        # Assuming task object has 'goals' (frozenset of goal facts) and 'static' (frozenset of static facts)
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                # Fact is (road l1 l2)
                l1, l2 = parts[1], parts[2]
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1) # Roads are typically bidirectional
                locations.add(l1)
                locations.add(l2)

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in locations:
            queue = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while queue:
                current_loc, dist = queue.popleft()

                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # Track where locatable objects (packages and vehicles) are currently located.
        # This map stores the *physical* location for all objects.
        current_physical_locations = {}
        package_is_in_vehicle = {} # Keep track if a package is inside a vehicle

        # First pass: find locations of objects directly 'at' a location.
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                current_physical_locations[obj] = loc

        # Second pass: find packages 'in' vehicles and record the vehicle.
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                package_is_in_vehicle[package] = vehicle # Record that package is in vehicle

        # Now, update the physical location for packages that are inside vehicles
        # to be the vehicle's location.
        for package, vehicle in package_is_in_vehicle.items():
             vehicle_location = current_physical_locations.get(vehicle)
             if vehicle_location is not None:
                 current_physical_locations[package] = vehicle_location
             # else: vehicle location unknown, handled as unreachable later if package is a goal

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that needs to reach a goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at the goal location (on the ground)
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at its goal, cost is 0 for this package

            # Find the package's current physical location
            current_location = current_physical_locations.get(package)

            if current_location is None:
                 # Package is a goal package but its location is unknown in the state.
                 # This might indicate an invalid state or an unreachable goal.
                 # Assign a very high cost, effectively pruning this path in greedy search.
                 return float('inf')

            # Get the shortest path distance from the package's current physical location to its goal location
            dist = self.distances.get((current_location, goal_location), float('inf'))

            if dist == float('inf'):
                # Goal location is unreachable from the package's current location via roads
                return float('inf') # Problem is unsolvable from this state

            # Determine if the package is currently inside a vehicle
            is_in_vehicle = package in package_is_in_vehicle

            # Calculate cost based on package's current state (on ground or in vehicle)
            if is_in_vehicle:
                # Package is in a vehicle at current_location.
                # Needs drive (dist) to goal_location and then drop (1).
                # If current_location is already the goal (dist=0), cost is 1 (drop).
                total_cost += dist + 1
            else:
                # Package is on the ground at current_location.
                # Needs pick (1) at current_location, drive (dist) to goal_location, and drop (1) at goal_location.
                total_cost += 1 + dist + 1

        return total_cost
