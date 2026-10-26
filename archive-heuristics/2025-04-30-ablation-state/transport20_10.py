# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If running standalone for testing, uncomment the mock class:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         pass

from fnmatch import fnmatch
import collections # Using collections.deque for BFS queue

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the individual costs. It considers whether a
    package is on the ground or in a vehicle and uses shortest path distances
    between locations.

    # Assumptions
    - The road network is connected (or relevant parts are connected).
    - Any vehicle can transport any package (ignores capacity constraints).
    - Any necessary vehicle is available when needed (ignores vehicle conflicts).
    - Action costs are uniform (implicitly 1 per action).
    - Goal conditions primarily involve packages being at specific locations.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task goals.
    - Builds a graph of locations connected by road facts from static information.
    - Identifies all relevant locations from initial state, goals, and road facts.
    - Computes all-pairs shortest path distances between all identified locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For each package `p` that has a goal location `goal_l`:
    1. Check if the package is already at its goal location (`(at p goal_l)` is in the state). If this specific goal fact is true, the cost for this package is 0.
    2. If the package is not at its goal location fact, determine its current status:
       - Find the current location of the package. This could be a location name (if on the ground) or a vehicle name (if inside a vehicle). This information is extracted from `(at ?p ?l)` or `(in ?p ?v)` facts in the current state.
       - If the package is inside a vehicle `v` (i.e., its current status is a vehicle name `v`):
         - Find the current location `current_l` of vehicle `v` from the state (`(at ?v ?current_l)`).
         - If `current_l` is the same as `goal_l`, the package only needs to be dropped. Estimated cost for this package: 1 (drop).
         - If `current_l` is different from `goal_l`, the vehicle needs to drive from `current_l` to `goal_l`, and then the package needs to be dropped. Estimated cost: `distance(current_l, goal_l)` (drives) + 1 (drop).
       - If the package is on the ground at `current_l` (i.e., its current status is a location name `current_l`):
         - Since we already checked if `current_l == goal_l` in step 1, we know `current_l` is different from `goal_l`.
         - The package needs to be picked up at `current_l`, transported to `goal_l`, and dropped at `goal_l`.
         - This sequence requires a pick-up action, one or more drive actions (corresponding to the shortest path distance), and a drop action.
         - Estimated cost: 1 (pick-up) + `distance(current_l, goal_l)` (drives) + 1 (drop). Total: `distance(current_l, goal_l) + 2`.
    3. The total heuristic value for the state is the sum of the estimated costs calculated for each package that is not yet at its goal location fact.
    4. If the state is the goal state (checked explicitly using `task.goal_reached`), the heuristic value is 0. This is the definitive check for the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        road network graph, and computing shortest path distances.
        """
        self.task = task

        self.goal_locations = {}
        locations = set()
        graph = collections.defaultdict(set) # Use set for neighbors to avoid duplicates

        # Extract goal locations and identify locations from goals
        for goal in self.task.goals:
            parts = get_parts(goal)
            # Assuming goal facts are primarily of the form (at package location)
            if len(parts) == 3 and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                locations.add(location)
            # Add any locations mentioned in other goal predicates if necessary,
            # though the domain description and examples suggest 'at' is the main goal type.


        # Identify locations from initial state (packages and vehicles)
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if len(parts) == 3 and parts[0] == "at":
                 obj, location = parts[1], parts[2]
                 locations.add(location)

        # Build graph and identify locations from static facts (roads)
        for fact in self.task.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                graph[l1].add(l2)
                graph[l2].add(l1) # Roads are bidirectional
                locations.add(l1)
                locations.add(l2)

        self.locations = list(locations) # Store locations list
        self.distances = {} # Store shortest path distances

        # Compute all-pairs shortest paths using BFS
        for start_node in self.locations:
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Store distance from start_node to current_loc
                self.distances[(start_node, current_loc)] = dist

                # Explore neighbors
                for neighbor in graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

            # After BFS from start_node, any location in self.locations not in visited
            # is unreachable from start_node. We don't explicitly store infinity here,
            # but the .get() call in __call__ will return None, which we handle.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the state is the goal state first.
        if self.task.goal_reached(state):
             return 0

        # Track where packages and vehicles are currently located.
        # Maps object name (package or vehicle) to its location (location name)
        # or package name to vehicle name if inside a vehicle.
        current_locations = {}
        # Also track vehicles to distinguish them from locations
        vehicles = set()

        # First pass to identify vehicles and initial locations
        for fact in state:
             parts = get_parts(fact)
             if len(parts) == 3:
                 predicate, arg1, arg2 = parts[0], parts[1], parts[2]
                 if predicate == "capacity":
                     vehicles.add(arg1)
                 elif predicate == "in":
                     vehicles.add(arg2) # arg2 is the vehicle

        # Second pass to populate current_locations
        for fact in state:
             parts = get_parts(fact)
             if len(parts) == 3:
                 predicate, obj, loc_or_veh = parts[0], parts[1], parts[2]
                 if predicate == "at":
                     current_locations[obj] = loc_or_veh
                 elif predicate == "in":
                     # If a package is 'in' a vehicle, its location is the vehicle object itself
                     package, vehicle = obj, loc_or_veh
                     current_locations[package] = vehicle


        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the package is currently at its goal location fact
            if f"(at {package} {goal_location})" in state:
                 continue # Package is at goal, cost is 0 for this package

            # Package is not at its goal location fact. Find its current state.
            current_loc_or_vehicle = current_locations.get(package)

            if current_loc_or_vehicle is None:
                 # This case indicates the package's location is not represented.
                 # This is unexpected in a typical STRIPS state. Treat as unreachable.
                 return float('inf')

            # Case 1: Package is inside a vehicle
            if current_loc_or_vehicle in vehicles: # Check if the value is a known vehicle
                 vehicle = current_loc_or_vehicle
                 vehicle_location = current_locations.get(vehicle) # Get vehicle's location

                 if vehicle_location is None:
                     # Vehicle location is not known. Cannot estimate cost. Unreachable.
                     return float('inf')

                 current_location = vehicle_location

                 # Cost to move package from current_location (where vehicle is) to goal_location
                 dist = self.distances.get((current_location, goal_location))
                 if dist is None:
                     # Goal location unreachable from current location. Infinite cost.
                     return float('inf')

                 # Needs drive + drop
                 total_cost += dist + 1

            # Case 2: Package is on the ground
            elif current_loc_or_vehicle in self.locations: # Check if the value is a known location
                 current_location = current_loc_or_vehicle

                 # Cost to move package from current_location to goal_location
                 dist = self.distances.get((current_location, goal_location))
                 if dist is None:
                     # Goal location unreachable from current location. Infinite cost.
                     return float('inf')

                 # Needs pick-up + drive + drop
                 total_cost += dist + 2
            else:
                 # current_loc_or_vehicle is neither a known vehicle nor a known location.
                 # This indicates a problem with the state representation. Treat as unreachable.
                 return float('inf')

        # If we reach here, it's not the goal state (checked at the beginning),
        # and we've summed costs for all packages not at their goal.
        # If total_cost is 0 here, it means all packages in goal_locations were
        # found to be at their goal location fact, but the overall goal state
        # was not reached (perhaps other goal conditions exist, or the state
        # representation is minimal and only lists facts that are true).
        # The initial check `if self.task.goal_reached(state): return 0` handles
        # the h=0 case correctly. The sum `total_cost` is the estimate for non-goal states.
        # If total_cost is 0 here, it implies all package location goals are met,
        # but the overall goal is not. This shouldn't happen if package locations
        # are the *only* goal conditions, as assumed. If other goals exist, this
        # heuristic might return 0 prematurely. However, based on the examples,
        # package locations are the only goals.

        return total_cost
