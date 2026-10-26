from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to match PDDL facts (similar to Logistics example)
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
    This heuristic estimates the cost to move packages to their goal locations.
    It sums the estimated costs for each package that is not yet at its goal
    location on the ground. The estimated cost for a package depends on whether
    it's on the ground or inside a vehicle, and the shortest path distance
    between its current location (or its vehicle's location) and its goal location.
    It uses a relaxed plan where capacity constraints and vehicle availability
    are simplified, and assumes shortest path drives.

    # Assumptions
    - The road network is static and defines possible vehicle movements.
    - Capacity constraints are ignored in the heuristic calculation for simplicity and speed.
    - Each pick-up and drop action costs 1.
    - Each drive action costs 1, and the cost of moving between two locations
      is the shortest path distance in the road network.
    - Goals are always of the form (at package location).
    - Objects starting with 'v' are vehicles, and objects starting with 'p' are packages relevant to goals.

    # Heuristic Initialization
    - Extract goal locations for each package from the task goals.
    - Build the road network graph from static `road` facts.
    - Compute all-pairs shortest paths on the road network graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state by verifying if all goal facts are present. If yes, return 0.
    2. Initialize total heuristic cost to 0.
    3. Identify the current location of all locatable objects (packages and vehicles)
       and which packages are inside which vehicles by iterating through the state facts.
       Store vehicle locations and packages currently inside vehicles.
    4. For each package `p` that has a goal location `goal_loc` defined in the task goals:
        a. Check if the goal fact `(at p goal_loc)` is present in the state. If yes, this package is already at its final goal location on the ground; continue to the next package as it requires no further actions.
        b. If the goal fact is not present, the package needs further actions. Determine its current status and location:
            i. If `p` is inside a vehicle `v` (i.e., `(in p v)` is true in the state):
                - Find the current location `current_loc` of vehicle `v` (i.e., `(at v current_loc)` is true).
                - If the vehicle's location is unknown, the state is invalid; return infinity.
                - If the vehicle is already at the goal location (`current_loc == goal_loc`), the package only needs to be dropped. Estimated cost = 1 (drop).
                - If the vehicle is not at the goal location (`current_loc != goal_loc`), the package needs to be transported and dropped. Estimated cost = shortest_path_distance(current_loc, goal_loc) (drives) + 1 (drop). If the goal is unreachable, return infinity.
                - Add this estimated cost to the total.
            ii. If `p` is on the ground at `current_loc` (i.e., `(at p current_loc)` is true in the state):
                - Ensure `current_loc` is a valid location in the road network. If not, the state is invalid; return infinity.
                - This package needs to be picked up, transported, and dropped.
                - Estimated cost = 1 (pick-up) + shortest_path_distance(current_loc, goal_location) (drives) + 1 (drop). If the goal is unreachable, return infinity.
                - Add this estimated cost to the total.
            iii. If the package's status cannot be determined (e.g., not 'at' any location and not 'in' any vehicle), this indicates an unexpected or invalid state; return infinity.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        road network graph, and computing shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package involved in a goal.
        self.goal_locations = {}
        # Identify all packages involved in goals
        self.packages_in_goals = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
                self.packages_in_goals.add(package)
            # Assuming goals are always (at package location)

        # Build the road network graph
        self.road_graph = {}
        self.locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Assuming roads are bidirectional
                self.locations.add(loc1)
                self.locations.add(loc2)

        # Compute all-pairs shortest paths
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Perform BFS from a start node to find shortest distances to all other nodes.
        Returns a dictionary {location: distance}.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # If current_node is not in graph (shouldn't happen if locations are from road facts)
            if current_node not in self.road_graph:
                 continue

            for neighbor in self.road_graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If the state contains all goal facts, heuristic is 0
        if self.goals <= state:
             return 0

        total_cost = 0

        # Find current locations of all locatable objects (packages and vehicles)
        # and which packages are inside which vehicles.
        current_at_locations = {} # {object: location} for 'at' facts
        packages_in_vehicles = {} # {package: vehicle} for 'in' facts
        vehicle_locations = {} # {vehicle: location} derived from 'at' facts

        # Populate location dictionaries
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_at_locations[obj] = loc
                # Assume objects starting with 'v' are vehicles based on domain examples
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif parts[0] == "in":
                 package, vehicle = parts[1], parts[2]
                 # Only track packages that are relevant to goals
                 if package in self.packages_in_goals:
                      packages_in_vehicles[package] = vehicle


        # Compute cost for each package that is part of a goal
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at its final goal location on the ground
            if f"(at {package} {goal_location})" in state:
                 continue # This package is done, cost is 0 for this package

            # Package is not yet at its final goal location on the ground.
            # Find its current status and location.

            package_current_loc = None
            is_in_vehicle = False

            if package in packages_in_vehicles:
                 is_in_vehicle = True
                 vehicle_carrying_package = packages_in_vehicles[package]
                 # The package's "location" is the vehicle's location
                 if vehicle_carrying_package in vehicle_locations:
                      package_current_loc = vehicle_locations[vehicle_carrying_package]
                 else:
                      # Vehicle location unknown - indicates invalid state or missing facts
                      # Assign a very high cost
                      return float('inf')
            elif package in current_at_locations and current_at_locations[package] in self.locations:
                 # Package is on the ground at a known location
                 package_current_loc = current_at_locations[package]
            else:
                 # Package is not 'at' a known location and not 'in' a vehicle.
                 # Indicates invalid state or missing facts.
                 return float('inf')


            # Now calculate the cost based on the package's current status and location
            if is_in_vehicle:
                # Package is in a vehicle at package_current_loc
                # Needs transport (if not already at goal vehicle loc) and drop
                # Cost = dist(package_current_loc, goal_location) + 1 (drop)

                # If the vehicle is already at the goal location, it just needs a drop
                if package_current_loc == goal_location:
                    total_cost += 1 # drop action
                else:
                    # Needs drive + drop
                    # Ensure locations are in our graph and goal is reachable
                    if package_current_loc in self.shortest_paths and goal_location in self.shortest_paths[package_current_loc]:
                         drive_cost = self.shortest_paths[package_current_loc][goal_location]
                         if drive_cost == float('inf'):
                              # Goal location unreachable from current vehicle location
                              return float('inf')
                         total_cost += drive_cost + 1 # drive + drop
                    else:
                         # Locations not found in graph - indicates parsing error or invalid state
                         return float('inf')

            else: # Package is on the ground at package_current_loc
                # Needs pickup + transport + drop
                # Cost = 1 (pickup) + dist(package_current_loc, goal_location) + 1 (drop)

                # If the package is already on the ground at the goal location, it's done (handled by the initial check)
                # So, package_current_loc must be different from goal_location here.

                # Ensure locations are in our graph and goal is reachable
                if package_current_loc in self.shortest_paths and goal_location in self.shortest_paths[package_current_loc]:
                     drive_cost = self.shortest_paths[package_current_loc][goal_location]
                     if drive_cost == float('inf'):
                          # Goal location unreachable from current package location
                          return float('inf')
                     total_cost += 1 + drive_cost + 1 # pickup + drive + drop
                else:
                     # Locations not found in graph
                     return float('inf')

        return total_cost
