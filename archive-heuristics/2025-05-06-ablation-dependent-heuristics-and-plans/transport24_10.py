from fnmatch import fnmatch
from collections import deque # Used for BFS

# Assuming heuristics.heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact gracefully, though PDDL facts are structured.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    This heuristic estimates the required number of actions to move all packages
    to their goal locations. It sums the estimated cost for each package
    independently, ignoring vehicle capacity and coordination. The cost for
    a package includes pick-up (if on ground), drop-off (if not at goal),
    and the shortest path distance the package needs to be transported.

    # Assumptions
    - Actions (drive, pick-up, drop) have a cost of 1.
    - Vehicle capacity and coordination are ignored; any package can be moved
      between any two connected locations if a vehicle exists or can reach there.
    - The shortest path distance between locations represents the minimum
      number of drive actions required for transport between them.
    - Goals only involve packages being at specific locations.
    - Object names starting with 'p' are packages, and 'v' are vehicles.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds a graph of locations based on the static `road` facts.
    - Computes the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of every package and vehicle. A package can be
       on the ground at a location `(at package location)` or inside a
       vehicle `(in package vehicle)`. The vehicle's location is given by
       `(at vehicle location)`.
    2. Initialize the total heuristic cost to 0.
    3. For each package whose goal location is known (from the task goals):
       a. Check if the package is already at its goal location, i.e., if the fact
          `(at package goal_location)` is present in the current state. If yes,
          this package contributes 0 to the heuristic.
       b. If the package is not at its goal location:
          i. Determine the package's current physical location. If `(at package current_loc)`
             is in the state, the current location is `current_loc`. If `(in package vehicle)`
             is in the state, find the location of `vehicle` using `(at vehicle vehicle_loc)`
             in the state; the current location is `vehicle_loc`.
          ii. If the package's location cannot be determined (e.g., vehicle location unknown),
              return infinity as the state is likely invalid or leads to an unsolvable path.
          iii. Calculate the shortest path distance from the package's current physical
               location to its goal location using the precomputed distances. If the
               goal is unreachable, return infinity.
          iv. Estimate the actions needed for this package:
              - If the package is on the ground: 1 (pick-up) + distance (drive) + 1 (drop).
              - If the package is inside a vehicle: distance (drive) + 1 (drop).
          v. Add this estimated cost for the package to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and computing shortest paths.
        """
        super().__init__(task)

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                # Assuming goal is (at package location)
                if len(args) == 2:
                    package, location = args
                    # Only store goals for objects identified as packages (heuristic assumption)
                    # This is a heuristic assumption based on common transport domain goals.
                    # A more robust approach would parse object types from the problem file.
                    if package.startswith('p'):
                         self.goal_locations[package] = location

        # Build the location graph from road facts.
        self.graph = {}
        locations = set()
        for fact in self.static:
            predicate, *args = get_parts(fact)
            if predicate == "road":
                if len(args) == 2:
                    l1, l2 = args
                    locations.add(l1)
                    locations.add(l2)
                    self.graph.setdefault(l1, []).append(l2)
                    # Assuming roads are bidirectional based on example instance.
                    self.graph.setdefault(l2, []).append(l1)

        # Ensure all locations mentioned in goals are included in the graph keys,
        # even if they are isolated (no roads connected). This prevents errors
        # when looking up distances for these locations later.
        for loc in self.goal_locations.values():
             self.graph.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        all_locations = list(self.graph.keys())
        for start_node in all_locations:
            self._bfs(start_node)

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        queue = deque([(start_node, 0)])
        visited = {start_node: 0}
        self.distances[(start_node, start_node)] = 0

        while queue:
            (current_node, dist) = queue.popleft()

            for neighbor in self.graph.get(current_node, []):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    self.distances[(start_node, neighbor)] = dist + 1
                    queue.append((neighbor, dist + 1))

        # Distances for unreachable pairs remain unset in self.distances.
        # The .get() method in __call__ will handle this by returning the default (infinity).


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # If the goal is already reached, the heuristic is 0.
        if self.goals <= state:
             return 0

        # Track where packages and vehicles are currently located.
        package_locations_on_ground = {} # {package: location}
        package_in_vehicle = {}         # {package: vehicle}
        vehicle_locations = {}          # {vehicle: location}

        # Parse the state to populate location dictionaries
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate == "at":
                if len(args) == 2:
                    obj, loc = args
                    # Use naming convention assumption to distinguish packages and vehicles
                    if obj.startswith('p'):
                        package_locations_on_ground[obj] = loc
                    elif obj.startswith('v'):
                        vehicle_locations[obj] = loc
            elif predicate == "in":
                 if len(args) == 2:
                     pkg, veh = args
                     # Assuming 'in' is always (in package vehicle)
                     if pkg.startswith('p') and veh.startswith('v'):
                         package_in_vehicle[pkg] = veh

        total_cost = 0  # Initialize action cost counter.

        # Calculate cost for each package that is not at its goal location.
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location.
            # The goal is defined as (at package goal_location).
            if f"(at {package} {goal_location})" in state:
                 continue # Package is already at its goal location

            # Package is not at its goal. Determine its current physical location.
            current_location = None
            is_in_vehicle = False

            if package in package_locations_on_ground:
                current_location = package_locations_on_ground[package]
                is_in_vehicle = False
            elif package in package_in_vehicle:
                veh = package_in_vehicle[package]
                if veh in vehicle_locations:
                    current_location = vehicle_locations[veh]
                    is_in_vehicle = True
                else:
                    # Vehicle location unknown - state is likely invalid or leads to unsolvable path
                    # Returning infinity indicates this path is not viable.
                    return float('inf')
            else:
                 # Package location unknown (not 'at' ground, not 'in' vehicle) - invalid state?
                 # Treat as unreachable for goal.
                 return float('inf')

            # Calculate shortest path distance from current physical location to goal.
            # Use .get() with infinity default for unreachable locations.
            dist_to_goal = self.distances.get((current_location, goal_location), float('inf'))

            if dist_to_goal == float('inf'):
                 # Goal location is unreachable from the package's current location/vehicle location
                 return float('inf') # Indicate unsolvable problem from this state

            # Estimate actions needed for this package based on its state and distance.
            if is_in_vehicle:
                # Package is in a vehicle. Needs drive actions + drop action.
                # The drive actions move the vehicle (and package) from current_location to goal_location.
                cost_for_package = dist_to_goal + 1
            else:
                # Package is on the ground. Needs pick-up + drive actions + drop action.
                # A vehicle needs to come to current_location (cost ignored in this simple model),
                # pick up the package (1), drive to goal_location (dist_to_goal), and drop (1).
                # We only count the pick, the drive with the package, and the drop.
                cost_for_package = 1 + dist_to_goal + 1

            total_cost += cost_for_package

        # The heuristic only considers packages explicitly listed in the goal.
        # Packages not in the goal list do not affect the heuristic value.

        return total_cost
