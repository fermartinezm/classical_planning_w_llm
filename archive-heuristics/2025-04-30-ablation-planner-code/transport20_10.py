# from heuristics.heuristic_base import Heuristic # Assuming this is provided
import sys
from collections import deque

# Mock Heuristic base class for demonstration/testing purposes if not provided by the environment
# In a real planning framework, you would import it.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Could not import Heuristic base class. Using mock class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static
            # Assuming task.objects is a dict like {'type': [obj1, obj2, ...]}
            self.objects = getattr(task, 'objects', {}) # Use getattr for safety
            # Assuming task has an initial_state attribute
            self.initial_state = getattr(task, 'initial_state', frozenset())


        def __call__(self, node):
            raise NotImplementedError("Heuristic __call__ method not implemented.")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove parentheses and split by whitespace
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the cost to transport all packages to their goal locations.
    It sums the minimum estimated cost for each package independently.
    The cost for a single package is estimated as:
    - 1 (pick-up) + shortest_path(current_location, goal_location) + 1 (drop)
      if the package is currently on the ground.
    - shortest_path(vehicle_location, goal_location) + 1 (drop)
      if the package is currently inside a vehicle.
    It ignores vehicle capacity and availability constraints for simplicity and efficiency.

    # Assumptions
    - The goal is to move specific packages to specific locations (`(at ?p ?l)`).
    - Roads are bidirectional (or defined explicitly in static facts). The current implementation assumes bidirectional based on typical transport domains and examples.
    - Vehicle capacity and availability are not bottlenecks (simplified assumption for heuristic).
    - All packages mentioned in goal conditions are present in the initial state.
    - All vehicles and packages are initially located somewhere (`at` or `in`).
    - The state representation is valid (objects mentioned in 'in' facts are also mentioned in 'at' facts for vehicles).

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build the road network graph from static `road` facts and initial `at` facts for locatables.
    - Compute all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. For each package `p` that has a goal location `loc_goal`:
    3. Determine the package's current state: Is it `(at p loc_p)` or `(in p v)`?
    4. If `(at p loc_p)`:
       - If `loc_p` is the same as `loc_goal`, the package is already at its goal; add 0 cost for this package.
       - If `loc_p` is different from `loc_goal`:
         - Find the shortest path distance `d` between `loc_p` and `loc_goal` in the road network.
         - If `loc_goal` is unreachable from `loc_p`, the problem is unsolvable; return infinity.
         - Add `1` (for pick-up) + `d` (for driving) + `1` (for drop) to the total cost.
    5. If `(in p v)`:
       - Find the current location `loc_v` of vehicle `v` (`(at v loc_v)`).
       - If `loc_v` is the same as `loc_goal`:
         - The package is in a vehicle at the goal location; add `1` (for drop) to the total cost.
       - If `loc_v` is different from `loc_goal`:
         - Find the shortest path distance `d` between `loc_v` and `loc_goal`.
         - If `loc_goal` is unreachable from `loc_v`, the problem is unsolvable; return infinity.
         - Add `d` (for driving) + `1` (for drop) to the total cost.
    6. If the package's state (`at` or `in`) is not found in the current state facts, this indicates an invalid state representation; return infinity.
    7. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network, and computing shortest paths.
        """
        super().__init__(task)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Goal is (at package location)
                if len(parts) == 3:
                    package, location = parts[1:]
                    self.goal_locations[package] = location
                else:
                    # Handle unexpected goal format
                    print(f"Warning: Unexpected goal format in goals: {goal}", file=sys.stderr)


        # Build the road network graph
        self.road_network = {} # Adjacency list: {location: [neighbor1, neighbor2, ...]}
        locations = set()

        # Collect locations and road connections from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                if len(parts) == 3:
                    l1, l2 = parts[1:]
                    locations.add(l1)
                    locations.add(l2)
                    self.road_network.setdefault(l1, []).append(l2)
                    # Assuming roads are bidirectional based on examples
                    self.road_network.setdefault(l2, []).append(l1)
                else:
                     print(f"Warning: Unexpected static road fact format: {fact}", file=sys.stderr)

        # Collect locations from initial 'at' facts for locatable objects (vehicles, packages)
        # This ensures we include locations where objects start, even if not connected by roads
        # in the static facts (though unreachable locations will have infinite path cost).
        # We use self.initial_state which is assumed to be provided by the base class/task object.
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 if len(parts) == 3:
                     obj, loc = parts[1:]
                     # Add the location if it's not already known.
                     if loc not in locations:
                         locations.add(loc)
                         self.road_network.setdefault(loc, []) # Add isolated location node

                 # else: Warning about malformed fact could be added here if needed


        self.locations = list(locations) # Store list of all known locations

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {} # {(loc_a, loc_b): distance}

        for start_loc in self.locations:
            distances = {loc: float('inf') for loc in self.locations}
            distances[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                u = queue.popleft()

                # Get neighbors from the road network graph
                neighbors = self.road_network.get(u, [])

                for v in neighbors:
                    if distances[v] == float('inf'):
                        distances[v] = distances[u] + 1
                        queue.append(v)

            # Store distances from start_loc to all other locations
            for end_loc in self.locations:
                self.shortest_paths[(start_loc, end_loc)] = distances[end_loc]

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # Map locatable objects (vehicles, packages) to their physical location
        current_physical_locations = {}
        # Map packages to the vehicle they are inside
        package_in_vehicle = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "at":
                # (at ?x - locatable ?v - location)
                if len(parts) == 3:
                    obj, loc = parts[1:]
                    current_physical_locations[obj] = loc
                # else: Warning about malformed fact could be added here if needed

            elif predicate == "in":
                # (in ?x - package ?v - vehicle)
                if len(parts) == 3:
                    package, vehicle = parts[1:]
                    package_in_vehicle[package] = vehicle
                # else: Warning about malformed fact could be added here if needed

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that need to reach a specific goal location
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at its goal location and not inside a vehicle
            # The goal is (at package goal_location), not (in package vehicle) at goal_location
            if package in current_physical_locations and current_physical_locations[package] == goal_location and package not in package_in_vehicle:
                continue # Package is already at goal, cost is 0 for this package

            # Determine the effective current location for path calculation
            current_effective_location = None
            package_is_in_vehicle = package in package_in_vehicle

            if package_is_in_vehicle:
                # Package is in a vehicle, its effective location is the vehicle's location
                vehicle = package_in_vehicle[package]
                current_effective_location = current_physical_locations.get(vehicle)
                if current_effective_location is None:
                    # Vehicle location unknown - indicates an invalid state representation
                    # or a vehicle exists but is not 'at' any location.
                    # This state is likely unreachable or indicates an unsolvable problem.
                    # Return infinity to prune this path.
                    return float('inf')

                # Cost: Drive from vehicle_location to goal + Drop
                drive_cost = self.shortest_paths.get((current_effective_location, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # Goal location is unreachable from the vehicle's current location
                    # Return infinity to prune this path.
                    return float('inf')
                total_cost += drive_cost + 1 # Drive actions + Drop action

            elif package in current_physical_locations:
                # Package is on the ground at a location
                current_effective_location = current_physical_locations[package]

                # Cost: Pick + Drive from package_location to goal + Drop
                drive_cost = self.shortest_paths.get((current_effective_location, goal_location), float('inf'))
                if drive_cost == float('inf'):
                    # Goal location is unreachable from the package's current location
                    # Return infinity to prune this path.
                    return float('inf')
                total_cost += 1 + drive_cost + 1 # Pick action + Drive actions + Drop action
            else:
                # Package is not 'at' a location and not 'in' a vehicle.
                # This indicates an invalid state representation.
                # Return infinity to prune this path.
                return float('inf')

        # The heuristic value is the sum of costs for all misplaced packages.
        # If all packages with 'at' goals are at their goals, total_cost will be 0.
        # This correctly reflects the goal state for typical transport problems.
        return total_cost
