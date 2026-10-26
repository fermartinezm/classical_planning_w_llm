from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all
    misplaced packages to their goal locations. It calculates the shortest
    path distance for each package from its current physical location to its goal
    location in the road network and adds the cost of necessary loading and unloading
    actions for that package. It sums these costs for all packages that are not
    yet at their goals on the ground.

    # Assumptions
    - The cost of each action (drive, load, unload) is 1.
    - Vehicle capacity constraints are ignored for simplicity and efficiency.
    - Vehicle availability is ignored; it is assumed a suitable vehicle is
      available when needed at the package's current location or vehicle's
      current location.
    - Packages can be moved independently (additive assumption).
    - The shortest path in the road network represents the minimum number
      of drive actions required for a vehicle to traverse that path.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Identify all packages that have a goal location.
    - Build a graph representing the road network from static facts.
    - Identify all relevant locations from road facts, initial state, and goals.
    - Compute all-pairs shortest paths between relevant locations using BFS.
    - Store the computed distances.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is the goal state. If yes, the heuristic is 0.
    2. If not the goal state, initialize total estimated cost to 0.
    3. For each package `p` that has a goal location:
       - Determine its goal location `goal_loc`.
       - Check if the package is already at its goal location on the ground
         (`(at p goal_loc)` is in the state). If yes, this package contributes 0
         to the heuristic, continue to the next package.
       - If the package is not at its goal on the ground:
         - Find the package's current physical location `current_loc`.
           - If `(at p loc)` is in the state, `current_loc` is `loc`, and the package is `on_ground`.
           - If `(in p v)` is in the state, find the vehicle's location `(at v loc_v)` in the state;
             `current_loc` is `loc_v`, and the package is not `on_ground`.
           - If the package's physical location cannot be determined (e.g., vehicle location unknown),
             the problem is likely unsolvable from this state; return infinity.
         - If `current_loc` is found:
           - If the package is in a vehicle (`not on_ground`) and the vehicle is already
             at the goal location (`current_loc == goal_loc`), the package only needs
             to be unloaded. Add 1 to the total cost for this package.
           - Otherwise (package is on the ground, or in a vehicle not at the goal):
             - Get the shortest path distance `dist` from `current_loc` to `goal_loc`
               from the pre-computed distances.
             - If `dist` is infinity, the goal is unreachable for this package; return infinity.
             - The estimated cost for this package is `dist` (for vehicle drive actions)
               + 1 (for the final unload action) + 1 (for a load action, if the package
               is currently on the ground). Add this cost to the total cost.
    4. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and pre-computing
        shortest path distances between all relevant locations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package and identify all packages with goals.
        self.package_goals = {}
        self.all_packages = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location
                self.all_packages.add(package)

        # Build road graph and identify all relevant locations.
        locations = set()
        roads = []
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                roads.append((loc1, loc2))
                locations.add(loc1)
                locations.add(loc2)

        # Add locations mentioned in initial state and goals to ensure all relevant
        # locations are included in the distance calculation.
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 _, loc = get_parts(fact)
                 locations.add(loc)
        for goal in task.goals:
             if match(goal, "at", "*", "*"):
                 _, loc = get_parts(goal)
                 locations.add(loc)

        # Build adjacency list for the road network.
        adj = {loc: [] for loc in locations}
        for l1, l2 in roads:
             adj[l1].append(l2)

        # Compute all-pairs shortest paths using BFS from each location.
        self.location_dist = {}
        for start_loc in locations:
            self.location_dist[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            self.location_dist[start_loc][start_loc] = 0 # Distance from a location to itself is 0

            while queue:
                current_loc, d = queue.popleft()

                # Ensure current_loc is in adj (it should be if added to locations)
                if current_loc in adj:
                    for neighbor in adj[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.location_dist[start_loc][neighbor] = d + 1
                            queue.append((neighbor, d + 1))

        # Fill in unreachable locations with infinity.
        for l1 in locations:
            for l2 in locations:
                if l2 not in self.location_dist[l1]:
                    self.location_dist[l1][l2] = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if the state is the goal state.
        if self.goals <= state:
            return 0

        # Map current location for all objects (packages and vehicles) that are 'at' a location.
        current_obj_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                current_obj_locations[obj] = loc

        # Map packages that are 'in' a vehicle.
        package_in_vehicle = {}
        for fact in state:
            if match(fact, "in", "*", "*"):
                package, vehicle = get_parts(fact)
                package_in_vehicle[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through all packages that have a goal location.
        for package in self.all_packages:
            goal_loc = self.package_goals.get(package)

            # This package must have a goal location if it's in self.all_packages,
            # but check defensively.
            if goal_loc is None:
                continue

            # Check if the package is already at its goal location on the ground.
            if (f"(at {package} {goal_loc})") in state:
                continue # Package is already at goal, cost is 0 for this package.

            # Package is not at its goal on the ground. Calculate cost to move it.
            current_loc = None
            on_ground = False # Assume package is not on the ground initially

            if package in current_obj_locations: # Package is on the ground somewhere.
                current_loc = current_obj_locations[package]
                on_ground = True
            elif package in package_in_vehicle: # Package is in a vehicle.
                vehicle = package_in_vehicle[package]
                if vehicle in current_obj_locations: # Vehicle is at a location.
                     current_loc = current_obj_locations[vehicle]
                     on_ground = False # Package is inside vehicle, not on the ground.
                else:
                     # Vehicle location is unknown. This state might be invalid or
                     # the package is effectively stuck. Assume unsolvable.
                     return float('inf')
            else:
                # Package location is completely unknown (not at a location, not in a vehicle).
                # Assume unsolvable.
                return float('inf')

            # If we are here, the package is not at its goal on the ground,
            # and its current physical location is known (current_loc).

            # If the package is in a vehicle (`not on_ground`) and the vehicle is already
            # at the goal location (`current_loc == goal_loc`), it just needs unloading.
            if not on_ground and current_loc == goal_loc:
                 total_cost += 1 # Unload action cost
                 continue # Done calculating cost for this package.

            # Otherwise (package is on the ground, or in a vehicle not at the goal),
            # it needs driving and unloading, and potentially loading.

            # Get drive distance from current physical location to goal location.
            # Check if locations exist in the pre-computed distances (should always if logic is correct).
            if current_loc not in self.location_dist or goal_loc not in self.location_dist[current_loc]:
                 # This indicates an issue with location extraction or distance computation.
                 # Treat as unreachable goal.
                 return float('inf')

            dist = self.location_dist[current_loc][goal_loc]

            if dist == float('inf'):
                # Goal is unreachable for this package via the road network.
                return float('inf')

            # Estimated cost for this package:
            # dist: Minimum drive actions for a vehicle carrying the package.
            # + 1: Unload action at the goal location.
            # + 1: Load action if the package is currently on the ground.
            package_cost = dist + 1 # Drive distance + Unload cost
            if on_ground:
                package_cost += 1 # Add Load cost if package is on the ground

            total_cost += package_cost

        # Return the total estimated cost, which is the sum of costs for all
        # packages not yet at their goal on the ground.
        return total_cost
