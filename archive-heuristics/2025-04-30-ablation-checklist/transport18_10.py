import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    # Ensure the number of parts matches the number of args, unless args has a wildcard at the end
    if len(parts) != len(args) and args[-1] != '*':
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - The cost for a package involves picking it up (if on the ground),
      driving it to the destination, and dropping it off.
    - Vehicle capacity and availability are not strictly modeled, assuming
      a suitable vehicle is available when needed.
    - Road network is static and provides connections between locations.

    # Heuristic Initialization
    - Extract goal locations for each package from the task's goal conditions.
    - Build a graph of locations based on `road` facts.
    - Precompute all-pairs shortest paths between locations using BFS.

    # Step-by-Step Thinking for Computing the Heuristic Value
    For a given state:
    1. Identify the current location or status (in a vehicle) of every package.
    2. Identify the current location of every vehicle.
    3. For each package that is not yet at its goal location:
       a. Determine the package's current physical location (either directly
          on the ground or the location of the vehicle it's in).
       b. Calculate the shortest path distance from the package's current
          physical location to its goal location using the precomputed distances.
       c. Estimate the actions needed for this package:
          - If the package is on the ground at `l_current` (and `l_current != l_goal`):
            Cost = 1 (pick-up) + distance(l_current, l_goal) (drive) + 1 (drop).
          - If the package is inside a vehicle `v` which is at `l_v_current`
            (and `l_v_current != l_goal`):
            Cost = distance(l_v_current, l_goal) (drive) + 1 (drop).
          - If the package is inside a vehicle `v` which is already at `l_goal`:
            Cost = 1 (drop).
       d. Add this estimated cost for the package to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and precomputing
        shortest paths in the road network.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the graph of locations based on road facts.
        self.location_graph = collections.defaultdict(list)
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1) # Assuming roads are bidirectional
                locations.add(loc1)
                locations.add(loc2)
        self.locations = list(locations) # Store locations for BFS

        # Precompute all-pairs shortest paths.
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances from every location to every other
        location using BFS.
        """
        all_distances = {}
        for start_node in self.locations:
            all_distances[start_node] = self._bfs(start_node)
        return all_distances

    def _bfs(self, start_node):
        """
        Performs a Breadth-First Search starting from start_node to find
        distances to all reachable locations.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = collections.deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node is in the graph keys before iterating neighbors
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state.

        # Map packages and vehicles to their current status/location.
        package_status = {} # Maps package -> ('at', loc) or ('in', vehicle)
        vehicle_locations = {} # Maps vehicle -> loc

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                # Determine if the object is a package or a vehicle
                # We need a way to distinguish types. PDDL state facts don't
                # explicitly include types like (package p1). We can infer
                # from predicate usage or rely on object names if consistent.
                # A robust way is to check if the object appears in 'in' facts.
                # Let's assume objects starting with 'p' are packages and 'v' are vehicles
                # based on the example instance files. A more general approach
                # would require parsing object types from the PDDL problem file.
                # Given the constraints, inferring from names or common predicates is reasonable.
                # Let's check if the object is a package by seeing if it's a key in goal_locations
                if obj in self.goal_locations:
                     package_status[obj] = ('at', loc)
                # Check if the object is a vehicle by seeing if it appears in capacity facts
                # Or simply assume 'v' prefix based on examples
                elif obj.startswith('v'): # Assuming 'v' prefix for vehicles
                     vehicle_locations[obj] = loc
                # If we can't determine type, ignore for this heuristic's purpose
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                package_status[package] = ('in', vehicle)

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that has a goal location.
        for package, goal_location in self.goal_locations.items():
            # If package is not in the current state, it might be an error or
            # the problem is malformed for this heuristic. Assume it exists.
            if package not in package_status:
                 # This package is not mentioned in 'at' or 'in' facts.
                 # This shouldn't happen in a valid state representation
                 # where all locatables have a location or are in a vehicle.
                 # For robustness, we could add a large penalty or skip.
                 # Let's assume valid states for now.
                 continue

            status_type, current_loc_or_vehicle = package_status[package]

            # If the package is already at its goal location on the ground, it's done.
            if status_type == 'at' and current_loc_or_vehicle == goal_location:
                continue # Package is already at goal, cost is 0 for this package.

            # Calculate cost for packages not at their goal.
            if status_type == 'at':
                # Package is on the ground at current_loc_or_vehicle (l_current)
                l_current = current_loc_or_vehicle
                # Cost = pick-up + drive + drop
                # Need to pick up (1 action)
                total_cost += 1
                # Need to drive from l_current to goal_location
                # Distance lookup: handle cases where l_current or goal_location might not be in graph (malformed problem?)
                if l_current in self.distances and goal_location in self.distances[l_current]:
                    dist = self.distances[l_current][goal_location]
                    if dist == float('inf'):
                        # Goal is unreachable from current location
                        return float('inf') # Return infinity heuristic
                    total_cost += dist
                else:
                    # Location not found in precomputed distances - problem with graph or locations?
                    # Treat as unreachable
                    return float('inf')
                # Need to drop off (1 action)
                total_cost += 1

            elif status_type == 'in':
                # Package is inside a vehicle (current_loc_or_vehicle is the vehicle name)
                vehicle = current_loc_or_vehicle
                # Find the vehicle's current location
                if vehicle not in vehicle_locations:
                    # Vehicle location not found - malformed state?
                    # Treat as unreachable
                    return float('inf')

                l_v_current = vehicle_locations[vehicle]

                # If vehicle is already at the goal location, just need to drop.
                if l_v_current == goal_location:
                    total_cost += 1 # Drop action
                else:
                    # Vehicle needs to drive from l_v_current to goal_location, then drop.
                    # Cost = drive + drop
                    # Need to drive from l_v_current to goal_location
                    if l_v_current in self.distances and goal_location in self.distances[l_v_current]:
                        dist = self.distances[l_v_current][goal_location]
                        if dist == float('inf'):
                            # Goal is unreachable from vehicle's current location
                            return float('inf') # Return infinity heuristic
                        total_cost += dist
                    else:
                         # Location not found in precomputed distances
                         return float('inf')
                    # Need to drop off (1 action)
                    total_cost += 1

        return total_cost

