import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, independently. It considers
    whether the package is on the ground or inside a vehicle and uses shortest
    path distances on the road network for drive actions. Capacity constraints
    and vehicle availability are ignored for simplicity and efficiency.

    # Assumptions
    - The goal is always to have packages at specific locations: (at package location).
    - Drive actions have a cost of 1 and correspond to edges in the road network.
    - Pick-up and drop-off actions have a cost of 1.
    - Capacity constraints are ignored.
    - Vehicle availability is ignored (assumes a vehicle is available when needed).
    - Roads are bidirectional if the PDDL defines road(l1, l2) and road(l2, l1).
      The precomputation assumes bidirectionality if either direction is present.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Builds the road network graph from static facts.
    - Computes all-pairs shortest paths between locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For each package that is not yet at its goal location:
    1. Determine the package's current status: Is it on the ground at a location `L_p`, or is it inside a vehicle `V`?
    2. If the package is inside vehicle `V`, find the current location `L_v` of vehicle `V`. The effective current location for transport is `L_v`.
    3. If the package is on the ground at `L_p`, the effective current location for transport is `L_p`.
    4. Calculate the shortest path distance `D` from the effective current location to the package's goal location `L_goal` using the precomputed distances. If unreachable, the state is likely a dead end, return infinity.
    5. Estimate the actions needed for this package:
       - If the package was on the ground at `L_p` (`L_p != L_goal`): Needs 1 (pick-up) + `D` (drive actions) + 1 (drop). Total: `D + 2`.
       - If the package was inside vehicle `V` at `L_v`: Needs `D` (drive actions) + 1 (drop). Total: `D + 1`.
    6. The total heuristic value is the sum of these estimated costs for all packages not at their goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify packages and vehicles based on how they appear in predicates
        self.packages = set()
        self.vehicles = set()

        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package, location = get_parts(goal)[1:]
                self.goal_locations[package] = location
                self.packages.add(package)

        # Collect all objects that are packages or vehicles from initial state and static facts
        # This helps distinguish objects in (at ?x ?l)
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == "capacity" and len(parts) == 3:
                 self.vehicles.add(parts[1])
             elif predicate == "in" and len(parts) == 3:
                 self.packages.add(parts[1])
                 self.vehicles.add(parts[2])
             elif predicate == "at" and len(parts) == 3:
                 # We can't definitively say if it's a package or vehicle just from 'at'
                 # unless we have type information here, which we don't directly.
                 # Relying on 'capacity', 'in', and goal 'at' is more robust.
                 pass


        # Build the road network graph and compute shortest paths.
        self.locations = set()
        graph = collections.defaultdict(list)

        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                # Assuming roads are bidirectional
                graph[l1].append(l2)
                graph[l2].append(l1)

        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_node in self.locations:
            distances = {loc: float('inf') for loc in self.locations}
            distances[start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                u = queue.popleft()
                if u in graph: # Check if node has any roads connected
                    for v in graph[u]:
                        if distances[v] == float('inf'):
                            distances[v] = distances[u] + 1
                            queue.append(v)
            self.shortest_paths[start_node] = distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track current status of packages (location or vehicle) and vehicle locations.
        package_status = {}  # package -> location or vehicle
        vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1:]
                if obj in self.packages:
                    package_status[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
                # Note: An object could be in both self.packages and self.vehicles
                # if the problem file is malformed, but we assume valid PDDL.
                # If an object is in 'at', it's either a package on the ground
                # or a vehicle at a location.
            elif predicate == "in" and len(parts) == 3:
                p, v = parts[1:]
                # This fact tells us package p is inside vehicle v.
                # We don't need to check if p/v are in self.packages/vehicles here,
                # as the predicate structure implies their types.
                package_status[p] = v


        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if (f"(at {package} {goal_location})") in state:
                continue # Package is already at its goal, cost is 0 for this package

            # Find the package's current status (on ground or in vehicle)
            current_status_p = package_status.get(package)

            if current_status_p is None:
                 # Package is not mentioned in 'at' or 'in' facts. This indicates
                 # an invalid state or a package not in the initial state.
                 # For heuristic purposes, we can treat this as an unreachable goal
                 # or simply skip if it wasn't in the initial goal set.
                 # Assuming valid states where all goal packages are initially present.
                 # If a package somehow disappeared, returning inf might be appropriate.
                 # Let's return inf as it's likely a dead end.
                 return float('inf')


            effective_current_loc = None
            cost_to_start_moving = 0 # Cost to get the package ready for driving

            if current_status_p in self.vehicles:
                # Package is inside a vehicle. The effective location is the vehicle's location.
                vehicle = current_status_p
                effective_current_loc = vehicle_locations.get(vehicle)
                if effective_current_loc is None:
                    # Vehicle carrying package is not at any location? Invalid state.
                    return float('inf')
                # Cost starts from driving the vehicle. No pick-up needed.
                cost_to_start_moving = 0
            else:
                # Package is on the ground at current_status_p
                current_loc_p = current_status_p
                effective_current_loc = current_loc_p
                # Needs a pick-up action before driving.
                cost_to_start_moving = 1 # pick-up cost

            # Calculate distance from effective current location to goal location
            if effective_current_loc not in self.shortest_paths or goal_location not in self.shortest_paths[effective_current_loc]:
                 # Goal location is unreachable from the package's current effective location
                 return float('inf') # Dead end

            drive_distance = self.shortest_paths[effective_current_loc][goal_location]

            # Cost for this package: cost_to_start_moving + drive_distance + drop_cost
            # Drop cost is always 1 if the package is not already at the goal location on the ground
            drop_cost = 1

            total_cost += cost_to_start_moving + drive_distance + drop_cost

        return total_cost

