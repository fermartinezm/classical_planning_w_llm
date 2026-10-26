from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this is available

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
    # Ensure the number of parts is at least the number of args for a valid match
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions needed to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, assuming each package needs a load, a series of drives,
    and an unload action. The drive cost is estimated by the shortest path
    distance in the road network.

    # Assumptions
    - Any package can be loaded into any vehicle with capacity > c0.
    - Vehicles are always available when needed at the package's current location
      or the vehicle's current location (this is where non-admissibility comes from).
    - The cost of driving a vehicle between two connected locations is 1.
    - The cost of loading a package is 1.
    - The cost of unloading a package is 1.

    # Heuristic Initialization
    - Extract the goal location for each package from the task goals.
    - Build the road network graph from the static facts.
    - Compute all-pairs shortest paths on the road network using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize total heuristic cost to 0.
    2. Create mappings for current package status (on ground location or in vehicle)
       and vehicle locations by iterating through the state facts.
       - For each `(at obj loc)` fact: if `obj` is a goal package, record its ground location; otherwise (assume it's a vehicle), record its location.
       - For each `(in package vehicle)` fact: if `package` is a goal package, record that it's in the vehicle.
    3. Iterate through each package `p` that has a goal location `goal_loc_p`:
        a. Retrieve the package's current status (on ground at `current_loc_p` or in vehicle `v`) from the mappings.
        b. If the package is in vehicle `v`, retrieve the vehicle's current location `current_loc_v` from the vehicle locations mapping.
        c. Determine the effective current location for travel: `current_loc_p` if on ground, `current_loc_v` if in vehicle. Handle cases where vehicle location is unknown (shouldn't happen in valid states).
        d. If the effective current location is the same as the goal location:
           - If the package is on the ground, its contribution is 0.
           - If the package is in a vehicle, it still needs to be unloaded, so its contribution is 1.
        e. If the effective current location is different from the goal location:
           - Calculate the shortest path distance `dist` between the effective current location and the goal location using the precomputed distances. If unreachable, the problem is unsolvable from this state (return infinity).
           - If the package is on the ground: Add 1 (load) + `dist` (drive) + 1 (unload) to the total cost.
           - If the package is in a vehicle: Add `dist` (drive) + 1 (unload) to the total cost.
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the road network."""
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # 2. Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Roads are bidirectional

        self.locations = list(locations) # Store list of all locations

        # 3. Compute all-pairs shortest paths.
        self.shortest_paths = {}
        for start_loc in self.locations:
            self._bfs(start_loc)

    def _bfs(self, start_node):
        """Perform BFS from start_node to find shortest paths to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.road_graph:
                for neighbor in self.road_graph[current_loc]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        # Store distances from start_node
        for end_loc in self.locations:
             if distances[end_loc] != float('inf'):
                 self.shortest_paths[(start_node, end_loc)] = distances[end_loc]
             # If not reachable, distance remains inf. Heuristic will be inf if a goal is unreachable.

    def get_distance(self, loc1, loc2):
        """Get the shortest path distance between two locations."""
        if loc1 == loc2:
            return 0
        # Look up precomputed distance. Return a large value if no path exists.
        return self.shortest_paths.get((loc1, loc2), float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the state is a goal state, the heuristic is 0.
        if self.task.goal_reached(state):
             return 0

        # Map packages to their current status: (location_on_ground, None) or (None, vehicle_name)
        package_status = {}
        # Map vehicles to their current location
        vehicle_locations = {}

        # Populate status and location maps by iterating through the state facts
        # First pass to find package status and all objects with 'at'
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.goal_locations: # It's a package we care about
                    package_status[obj] = (loc, None)
            elif parts[0] == "in":
                 package, vehicle = parts[1], parts[2]
                 if package in self.goal_locations: # It's a package we care about
                    package_status[package] = (None, vehicle)

        # Second pass to identify vehicles and their locations from 'at' facts
        # Objects with 'at' that are not goal packages are assumed to be vehicles
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == "at":
                  obj, loc = parts[1], parts[2]
                  if obj not in self.goal_locations: # It's not a goal package, assume it's a vehicle
                       vehicle_locations[obj] = loc


        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is not found in state facts, it's an issue.
            # Assuming valid states where all goal packages are accounted for.
            if package not in package_status:
                 # This should not happen in valid states derived from initial state
                 # where all goal packages are present.
                 # Return infinity as a safeguard.
                 return float('inf')

            current_ground_loc, current_vehicle = package_status[package]

            if current_ground_loc is not None: # Package is on the ground
                package_current_loc = current_ground_loc
                if package_current_loc != goal_location:
                    # Needs load (1) + drive (dist) + unload (1)
                    dist = self.get_distance(package_current_loc, goal_location)
                    if dist == float('inf'): return float('inf') # Goal unreachable
                    total_cost += 1 + dist + 1
                # Else: package is on ground at goal, cost contribution is 0.

            elif current_vehicle is not None: # Package is in a vehicle
                 vehicle_name = current_vehicle
                 if vehicle_name not in vehicle_locations:
                      # Vehicle location unknown - should not happen in valid states
                      return float('inf')

                 vehicle_current_loc = vehicle_locations[vehicle_name]

                 # Needs drive (dist) + unload (1)
                 # If vehicle is at goal, dist is 0, cost is 1 (unload)
                 dist = self.get_distance(vehicle_current_loc, goal_location)
                 if dist == float('inf'): return float('inf') # Goal unreachable
                 total_cost += dist + 1

            # If package is not on ground and not in vehicle, something is wrong.
            # This case is implicitly handled by the initial check `if package not in package_status:`.
            # If it were in package_status but both elements of the tuple were None,
            # that would also indicate an issue, but the logic above covers the two valid states.


        return total_cost
