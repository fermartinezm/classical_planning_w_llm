# from heuristics.heuristic_base import Heuristic # Assuming this base class exists

from fnmatch import fnmatch
from collections import deque # Used for BFS

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Return an empty list for invalid fact strings
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define a dummy Heuristic base class if not provided in the environment
# In a real environment, you would uncomment the import from heuristics.heuristic_base
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy base class for standalone execution."""
        def __init__(self, task):
            self.task = task
            pass

        def __call__(self, node):
            pass


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently. The cost for a package is estimated as the number of
    actions needed to pick it up (if on the ground), drive it to its goal
    location, and drop it. Driving cost is the shortest path distance in the
    road network. Capacity constraints and vehicle availability/location
    (beyond carrying the package) are ignored.

    # Assumptions
    - The goal state is defined solely by the locations of packages specified
      in the task's goal conditions using the `(at ?p ?l)` predicate.
    - Any vehicle can pick up any package (capacity is ignored).
    - A vehicle is assumed to be available to pick up a package if it's on the ground, or
      is already carrying the package. The cost of the vehicle reaching the package's
      initial location (if needed) is not explicitly calculated.
    - All actions (drive, pick-up, drop) have a cost of 1.
    - The road network is connected, or unreachable goal locations imply a very high cost.
    - Every package listed in the goal has its location or carrier specified in any valid state.

    # Heuristic Initialization
    - Extracts all locations and road connections from static facts to build
      a graph of the road network.
    - Computes the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS). These distances represent the minimum
      number of `drive` actions needed between locations.
    - Extracts the goal location for each package from the task's goal conditions
      that use the `(at ?p ?l)` predicate.
    - Identifies all vehicles defined in the problem by looking for the
      `(capacity ?v ?s)` predicate in the initial state or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state (represented as a frozenset of facts):
    1. Identify the current location or carrier for every package that has a goal location.
       - Iterate through the state facts. If a fact is `(at package location)`, record the package's location. If a fact is `(in package vehicle)`, record the package's carrier vehicle.
       - Also, record the current location `(at vehicle location)` for all identified vehicles.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that is listed in the `self.goal_locations` dictionary (meaning it has a target location `l_goal`):
       a. Determine the package's current status (on the ground at `l_current` or inside vehicle `v` at `l_v`). This information is retrieved from the state facts processed in step 1.
       b. Check if the package is already at its goal location `l_goal`. This happens if:
          - The package is on the ground at `l_goal` (`(at p l_goal)` is in state).
          - OR the package is inside a vehicle `v` and that vehicle is at `l_goal` (`(in p v)` and `(at v l_goal)` are in state).
       c. If the package is already at its goal, add 0 to the total cost for this package and move to the next package.
       d. If the package is not at its goal:
          - If the package is on the ground at `l_current` (`l_current != l_goal`):
            - The estimated cost for this package is 1 (pick-up action) + the shortest path distance from `l_current` to `l_goal` (minimum drive actions) + 1 (drop action).
            - Add `1 + self.distances.get((current_loc_or_carrier, goal_location), self.unreachable_cost) + 1` to the total cost.
          - If the package is inside a vehicle `v` which is currently at `l_v` (`l_v != l_goal`):
            - The estimated cost for this package is the shortest path distance from `l_v` to `l_goal` (minimum drive actions) + 1 (drop action).
            - Add `self.distances.get((current_vehicle_location, goal_location), self.unreachable_cost) + 1` to the total cost.
          - If a required distance lookup fails (e.g., locations are disconnected), a large finite number is used instead of infinity.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the road graph and computing distances."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all vehicles

        # 1. Build the road graph
        self.locations = set()
        graph = {} # Adjacency list: location -> list of connected locations

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                graph.setdefault(l1, []).append(l2)
                graph.setdefault(l2, []).append(l1) # Roads are bidirectional

        # Ensure all locations mentioned in the problem are in the graph keys
        for loc in self.locations:
             graph.setdefault(loc, [])

        self.graph = graph

        # 2. Compute all-pairs shortest paths using BFS
        self.distances = {} # (start_loc, end_loc) -> distance

        for start_loc in self.locations:
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                self.distances[(start_loc, current_loc)] = dist

                for neighbor in self.graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # Define a large cost for unreachable locations
        # This value is returned if BFS couldn't find a path.
        # It should be larger than any possible path length in a connected graph.
        # Max possible path length is |locations| - 1.
        # A large multiplier ensures it dominates other costs.
        self.unreachable_cost = len(self.locations) * 1000 + 1


        # 3. Extract goal locations for packages
        self.goal_locations = {} # package_name -> goal_location_name
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Assuming goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other types of goal predicates if any

        # 4. Identify all vehicles
        self.vehicles = set()
        # Look for capacity facts in initial state and static facts
        # Vehicles are objects that have a capacity predicate applied to them.
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "capacity":
                 vehicle_name = parts[1]
                 self.vehicles.add(vehicle_name)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located or contained.
        # Map object name -> its current location name (if at) or carrier name (if in)
        current_status = {}
        # Map vehicle name -> its current location name
        vehicle_locations = {}

        # Populate current_status and vehicle_locations from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            if predicate == "at":
                obj, location = parts[1], parts[2]
                current_status[obj] = location # Could be package or vehicle
                if obj in self.vehicles:
                    vehicle_locations[obj] = location

            elif predicate == "in":
                 package, vehicle = parts[1], parts[2]
                 # Only track packages that are in our goal list
                 if package in self.goal_locations:
                     current_status[package] = vehicle # Package is inside a vehicle


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Find the package's current status (location or carrier)
            # If a package exists in goal_locations, it must have a status in any valid state.
            # If for some reason it's not in the state facts, treat as unreachable.
            current_loc_or_carrier = current_status.get(package)

            if current_loc_or_carrier is None:
                 # Package exists in goal but not in state? Problematic state representation.
                 # Assume it's infinitely far or unreachable.
                 total_cost += self.unreachable_cost
                 continue

            # Determine if the package is currently at its goal location
            is_at_goal = False
            if current_loc_or_carrier == goal_location:
                 # Package is on the ground at the goal location
                 is_at_goal = True
            elif current_loc_or_carrier in self.vehicles: # It's inside a vehicle
                 carrier_vehicle = current_loc_or_carrier
                 carrier_location = vehicle_locations.get(carrier_vehicle)
                 if carrier_location == goal_location:
                     # Package is inside a vehicle that is at the goal location
                     is_at_goal = True
                 # If carrier_location is None, the vehicle is not located, treat as unreachable below.


            if is_at_goal:
                continue # Package is already at its goal, cost is 0 for this package

            # Package is not at its goal. Calculate cost.
            if current_loc_or_carrier in self.vehicles: # Package is inside a vehicle
                carrier_vehicle = current_loc_or_carrier
                current_vehicle_location = vehicle_locations.get(carrier_vehicle)

                # If the vehicle's location is unknown, this is an unexpected state.
                # Assign a very high cost.
                if current_vehicle_location is None:
                     total_cost += self.unreachable_cost
                     continue # Cannot calculate further for this package

                # Cost = drive from current vehicle location to goal + drop
                # Need distance from current_vehicle_location to goal_location
                dist = self.distances.get((current_vehicle_location, goal_location), self.unreachable_cost)

                total_cost += dist + 1 # drive + drop

            else: # Package is on the ground at current_loc_or_carrier
                current_package_location = current_loc_or_carrier

                # current_package_location is already guaranteed not to be None by the check above

                # Cost = pick-up + drive from current package location to goal + drop
                # Need distance from current_package_location to goal_location
                dist = self.distances.get((current_package_location, goal_location), self.unreachable_cost)

                total_cost += 1 + dist + 1 # pick-up + drive + drop


        return total_cost
