from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args):
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, considering its current state (on ground or in vehicle)
    and the shortest path distance on the road network. It sums the estimated
    costs for each package independently, ignoring vehicle capacity and shared trips.

    # Assumptions
    - The goal is always a set of `(at ?p ?l)` facts for packages.
    - The road network is static (extracted from static facts).
    - Shortest path distance on the road network is a reasonable estimate for drive actions.
    - A suitable vehicle is assumed to be available when a package needs loading.
    - Vehicle capacity is ignored.
    - Shared vehicle trips are ignored (each package's cost is calculated independently).
    - Objects starting with 'p' are packages, and objects starting with 'v' are vehicles.

    # Heuristic Initialization
    - Extract goal locations for each package from `task.goals`.
    - Build the road network graph from `task.static` facts `(road ?l1 ?l2)`.
    - Precompute all-pairs shortest paths on the road network using BFS from each location.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and vehicle. Also, determine which packages are inside which vehicles.
    2. Initialize the total heuristic cost to 0.
    3. For each package `p` that has a goal location `l_goal`:
        a. Find the package's current position (`l_current_pos`). This is the location name if on the ground, or the vehicle name if inside a vehicle.
        b. Determine the package's physical location (`l_current_phys`). This is `l_current_pos` if on the ground, or the location of the vehicle (`at ?v ?l`) if inside a vehicle.
        c. Check if the package is currently inside a vehicle.
        d. If `l_current_phys` is the same as `l_goal`:
            - If the package is inside a vehicle, it needs 1 `unload` action. Add 1 to the total cost.
            - If the package is on the ground, it is already at its goal location and requires 0 further actions.
        e. If `l_current_phys` is different from `l_goal`:
            - Calculate the shortest path distance `d` between `l_current_phys` and `l_goal` using the precomputed distances.
            - If the package is on the ground at `l_current_phys`: It needs to be loaded (1 action), transported by a vehicle (`d` drive actions), and unloaded (1 action). Add `1 + d + 1` to the total cost.
            - If the package is inside a vehicle at `l_current_phys`: It needs to be transported by the vehicle (`d` drive actions) and unloaded (1 action). Add `d + 1` to the total cost.
    4. The total heuristic value is the sum of costs calculated for each package.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph.
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Roads are bidirectional

        self.locations = list(locations) # Store locations list

        # Precompute all-pairs shortest paths using BFS from each location.
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """Perform BFS to find shortest distances from start_node to all other nodes."""
        distances = {node: float('inf') for node in self.locations}

        # If start_node is not in the graph (e.g., an isolated location not in any road fact),
        # it can only reach itself (distance 0). Other locations remain unreachable (inf).
        if start_node not in self.locations:
             # Add the isolated node to distances if it's not already there
             distances[start_node] = 0
             # No neighbors to add to queue, so BFS finishes here.
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node has neighbors in the graph
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        package_positions = {} # Maps package -> location (if on ground) or vehicle (if in vehicle)
        vehicle_locations = {} # Maps vehicle -> location
        package_is_in_vehicle = set() # Set of packages currently in a vehicle

        for fact in state:
            parts = get_parts(fact)
            # Skip facts that don't indicate location or containment
            if not parts or parts[0] not in ["at", "in"]:
                continue

            predicate = parts[0]
            if predicate == "at":
                # Ensure fact has expected number of parts for 'at'
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Assume objects starting with 'p' are packages and 'v' are vehicles
                    if obj.startswith('p'):
                        package_positions[obj] = loc
                    elif obj.startswith('v'):
                        vehicle_locations[obj] = loc
            elif predicate == "in":
                 # Ensure fact has expected number of parts for 'in'
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    # Assume package starts with 'p' and vehicle starts with 'v'
                    if package.startswith('p') and vehicle.startswith('v'):
                        package_positions[package] = vehicle # Package is *in* the vehicle
                        package_is_in_vehicle.add(package)


        total_cost = 0  # Initialize action cost counter.

        # Consider only packages that have a goal location defined and are present in the state
        packages_to_move = [p for p in self.goal_locations if p in package_positions]

        for package in packages_to_move:
            goal_location = self.goal_locations[package]
            current_pos = package_positions[package] # This could be a location name or a vehicle name

            is_in_vehicle = package in package_is_in_vehicle
            current_physical_location = None # The actual location on the map

            if is_in_vehicle:
                current_vehicle = current_pos # current_pos is the vehicle name
                # The physical location is where the vehicle is
                current_physical_location = vehicle_locations.get(current_vehicle)
                # If vehicle location is unknown, this state is likely invalid or unreachable.
                # Return infinity in such cases.
                if current_physical_location is None:
                     # print(f"Error: Vehicle {current_vehicle} carrying {package} has no location in state.")
                     return float('inf') # Should not happen in valid states

            else: # Package is on the ground
                 current_physical_location = current_pos # current_pos is the location name

            # Check if the package is already at its goal location on the ground
            if current_physical_location == goal_location and not is_in_vehicle:
                # Package is at goal and on the ground - goal achieved for this package
                pass # Cost is 0 for this package

            # Check if the package is at the goal location but still in a vehicle
            elif current_physical_location == goal_location and is_in_vehicle:
                 # Needs 1 unload action
                 total_cost += 1

            # Package is not at the goal location
            elif current_physical_location != goal_location:
                # Get shortest path distance from current physical location to goal location
                # Ensure both locations are in our precomputed distances.
                # If goal_location is not in self.dist[current_physical_location], it's unreachable.
                if current_physical_location not in self.dist or goal_location not in self.dist[current_physical_location]:
                     # This implies the goal location is unreachable from the package's current location.
                     # This state is likely not on a path to the goal in a solvable problem.
                     # Return infinity.
                     # print(f"Error: Goal location {goal_location} unreachable from {current_physical_location}.")
                     return float('inf')

                dist_to_goal = self.dist[current_physical_location][goal_location]

                if is_in_vehicle:
                    # Package is in a vehicle, needs drive + unload
                    # Cost = drive actions + unload action
                    total_cost += dist_to_goal + 1
                else:
                    # Package is on the ground, needs load + drive + unload
                    # Cost = load action + drive actions + unload action
                    total_cost += 1 + dist_to_goal + 1 # load + drive + unload

        # The heuristic is 0 if and only if all goal conditions are met.
        # Our calculation sums costs for *misplaced* packages. If all packages
        # are at their goal locations on the ground, the loop won't add cost,
        # resulting in 0. This satisfies the requirement.
        # If any package is at the goal location but in a vehicle, cost is > 0.
        # If any package is not at the goal location, cost is > 0 (assuming reachable).

        return total_cost
