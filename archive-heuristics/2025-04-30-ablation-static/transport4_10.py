from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque # For BFS

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at p1 l1)" -> ["at", "p1", "l1"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the minimum number of actions (pick-up, drop, drive)
    required to move each package from its current location to its goal location,
    summing the costs for all packages that are not yet at their goal.
    The cost includes picking up the package (if on the ground), driving the
    distance to the goal location, and dropping the package.

    # Assumptions
    - Vehicle capacity constraints are ignored. Any vehicle can carry any package.
    - Vehicle availability is ignored. A vehicle is assumed to be available
      at the required location when needed.
    - Packages are moved independently. The cost for each package is calculated
      as if it were the only package being moved, and these costs are summed.
    - The cost of driving between locations is the shortest path distance
      in the road network, where each road segment has a cost of 1.

    # Heuristic Initialization
    - The road network is extracted from the static facts to build a graph
      of locations.
    - All-pairs shortest path distances between locations are precomputed
      using Breadth-First Search (BFS).
    - The goal location for each package is extracted from the task goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic cost to 0.
    2. For each package that has a specified goal location:
       a. Check if the package is already on the ground at its goal location. If yes, cost is 0 for this package, continue to the next package.
       b. Determine the package's current status: Is it on the ground at some location `l` (fact `(at package l)`), or inside a vehicle `v` (fact `(in package v)`)?
       c. Find the package's current physical location `current_l`. If on the ground, `current_l` is `l`. If inside vehicle `v`, `current_l` is the location of vehicle `v` (fact `(at v current_l)`). If the vehicle's location is unknown, this package's cost cannot be estimated (and it's likely an inconsistent state or unreachable goal).
       d. If the package's current physical location `current_l` is the same as its goal location `goal_l`:
          - If the package is inside a vehicle, it needs to be dropped (1 action). Add 1 to total cost.
          - If the package is on the ground, this case is already handled by 2a (cost is 0).
       e. If the package's current physical location `current_l` is different from its goal location `goal_l`:
          - Calculate the shortest path distance `dist` from `current_l` to `goal_l` in the road network.
          - If the package is currently on the ground, it needs to be picked up (1 action). Add 1 to total cost.
          - Add the distance `dist` as the estimated drive cost to total cost.
          - Add 1 for the drop action to total cost.
          - If `dist` was infinite, the total cost will become infinite.
    3. If the total calculated cost is infinite (meaning at least one package
       needed an unreachable drive in the simplified model), return a large finite number (e.g., 1000000).
    4. Otherwise, return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes shortest path distances between all locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract road network and locations
        self.locations = set()
        self.road_graph = {} # Adjacency list: location -> [neighbor1, neighbor2, ...]
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                if l2 not in self.road_graph:
                    self.road_graph[l2] = []
                # Add road in both directions as per example instances
                self.road_graph[l1].append(l2)
                self.road_graph[l2].append(l1)

        # Collect all locations mentioned in initial state and goals
        all_mentioned_locations = set()
        all_locatables = set()
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 all_mentioned_locations.add(loc)
                 all_locatables.add(obj)
             elif match(fact, "in", "*", "*"):
                 _, obj, veh = get_parts(fact)
                 all_locatables.add(obj)
                 all_locatables.add(veh) # vehicles are also locatables

        for goal in self.goals:
             if match(goal, "at", "*", "*"):
                 _, obj, loc = get_parts(goal)
                 all_mentioned_locations.add(loc)
                 all_locatables.add(obj)

        # Add any mentioned locations that weren't in road facts to our set of locations
        self.locations.update(all_mentioned_locations)

        # Ensure all locations are keys in the road_graph dictionary, even if they have no roads
        for loc in self.locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = []

        # Compute all-pairs shortest paths using BFS
        self.distances = {} # (l1, l2) -> distance
        for start_node in self.locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Ensure current_loc is in road_graph keys before accessing neighbors
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_node, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify packages: locatables that are not vehicles.
        vehicles = set()
        for fact in task.initial_state:
             if match(fact, "capacity", "*", "*"):
                 _, veh, _ = get_parts(fact)
                 vehicles.add(veh)

        packages = all_locatables - vehicles

        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                obj, location = args
                if obj in packages: # Only consider packages as per domain
                    self.goal_locations[obj] = location
                # Ignore goals about vehicles or other object types if any

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        # This maps object name (package or vehicle) to its location string.
        # If a package is 'in' a vehicle, its entry will be the vehicle name.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                obj, location = args
                current_locations[obj] = location
            elif predicate == "in":
                package, vehicle = args
                current_locations[package] = vehicle # Store the vehicle name

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if (f"(at {package} {goal_location})") in state:
                continue # Package is already at goal, cost is 0 for this package

            # Find the package's current status (location string or vehicle name)
            current_status = current_locations.get(package)

            if current_status is None:
                 # Package status not found in state. This indicates an issue
                 # with the state representation or problem definition if a package
                 # with a goal is not located anywhere. Skip this package.
                 continue

            # Determine the package's current physical location
            is_on_ground = current_status in self.locations
            if is_on_ground:
                current_location = current_status
            else: # Package is inside a vehicle
                vehicle = current_status
                vehicle_location = current_locations.get(vehicle)
                if vehicle_location is None:
                    # Vehicle location not found. Cannot determine package location.
                    # Skip this package or add a large penalty. Skipping is simpler
                    # for a non-admissible heuristic, assuming valid states.
                    continue
                current_location = vehicle_location

            # If the package is already at the goal location (but inside a vehicle)
            if current_location == goal_location and not is_on_ground:
                 # Package is in a vehicle, and the vehicle is at the goal location.
                 # Needs 1 action: drop.
                 total_cost += 1
                 continue # Done with this package

            # If the package is not at the goal location (either on ground or in vehicle)
            # Cost includes drive + drop, and potentially pick-up if on ground.
            cost_for_package = 0

            # Add cost for pick-up if the package is on the ground
            if is_on_ground:
                cost_for_package += 1 # pick-up action

            # Add cost for driving
            drive_cost = self.distances.get((current_location, goal_location), float('inf'))

            if drive_cost == float('inf'):
                 # Goal location is unreachable from current location in the road network.
                 # This package cannot reach its goal via simple driving in the relaxed model.
                 # Mark this package's contribution as infinite.
                 cost_for_package = float('inf')
            else:
                 cost_for_package += drive_cost
                 # Add cost for drop
                 cost_for_package += 1 # drop action

            total_cost += cost_for_package


        # If the total calculated cost is infinite (meaning at least one package
        # needed an unreachable drive), return a large finite number.
        # Otherwise, return the calculated total cost.
        if total_cost == float('inf'):
             # Use a large number that is unlikely to be reached by summing finite costs.
             # This acts as a penalty for states where a package goal is unreachable
             # in the simplified distance model.
             return 1000000
        else:
             return total_cost
