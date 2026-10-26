from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections # Used for BFS queue

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    from its current location to its goal location, ignoring vehicle capacity
    and multi-package interactions. It sums the estimated costs for each package
    independently. The cost for a package includes loading, driving (shortest
    path distance), and unloading actions.

    # Assumptions
    - Each package can be moved independently by any available vehicle.
    - Vehicle capacity constraints are ignored.
    - The cost of a 'drive' action is 1, regardless of distance (shortest path
      distance in terms of number of roads is used as the number of drive actions).
    - 'load' and 'unload' actions cost 1.
    - All locations mentioned in 'road' facts are part of the road network.
    - The problem is solvable (goal locations are reachable from initial package locations).

    # Heuristic Initialization
    - Extract the goal location for each package from the task's goal conditions.
    - Identify all packages and vehicles based on initial/goal facts and static facts.
    - Build a graph representation of the road network from the static facts.
    - Precompute the shortest path distance (number of drive actions) between
      all pairs of locations using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is the goal state. If yes, the heuristic is 0.
    2. Initialize the total heuristic cost to 0.
    3. Determine the current physical location and state (on ground or in vehicle)
       for every package that has a goal location.
       - Iterate through the state facts.
       - If a fact is `(at package location)`, record the package's location.
       - If a fact is `(in package vehicle)`, record that the package is in a vehicle.
       - After finding which vehicle a package is in, find the vehicle's location
         `(at vehicle location)` to determine the package's physical location.
    4. For each package that is not yet satisfying its goal condition `(at package goal_loc)`:
       - Get the package's current physical location and its goal location.
       - Find the precomputed shortest path distance (`dist`) between the current
         location and the goal location. If no path exists, the state is likely
         unsolvable or represents a dead end, return infinity.
       - If the package is currently on the ground at `current_loc` (`(at package current_loc)`):
         - It needs to be loaded (1 action).
         - The vehicle needs to drive `dist` times (dist actions).
         - It needs to be unloaded (1 action).
         - Add `1 + dist + 1` to the total cost.
       - If the package is currently in a vehicle at `current_loc` (`(in package vehicle)` and `(at vehicle current_loc)`):
         - The vehicle needs to drive `dist` times (dist actions).
         - It needs to be unloaded (1 action).
         - Add `dist + 1` to the total cost.
       - Note: If the package is in a vehicle *at* the goal location, it only needs to be unloaded (dist=0, cost = 0 + 1 = 1). This is covered by the `dist + 1` case when `dist` is 0.
    5. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations, building the
        road graph, and precomputing shortest paths.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify all packages (those mentioned in goal 'at' facts)
        self.packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)

        # Identify all vehicles (those mentioned in static 'capacity' facts)
        self.vehicles = {get_parts(fact)[1] for fact in self.static if match(fact, "capacity", "*", "*")}

        # Build the road graph and identify all locations.
        all_locations = set()
        self.road_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                all_locations.add(loc1)
                all_locations.add(loc2)
                if loc1 not in self.road_graph:
                    self.road_graph[loc1] = set()
                self.road_graph[loc1].add(loc2)

        # Ensure all locations are keys in the graph, even if they have no outgoing roads
        for loc in all_locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = set()

        # Precompute shortest paths between all pairs of locations using BFS.
        self.shortest_paths = {}
        for start_loc in all_locations:
            distances = {loc: float('inf') for loc in all_locations}
            distances[start_loc] = 0
            queue = collections.deque([start_loc]) # Use deque for efficient BFS queue

            while queue:
                current_loc = queue.popleft()

                self.shortest_paths[(start_loc, current_loc)] = distances[current_loc]

                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        if distances[neighbor] == float('inf'): # Only visit unvisited nodes
                            distances[neighbor] = distances[current_loc] + 1
                            queue.append(neighbor)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # If the state is the goal state, the heuristic is 0.
        if self.goals <= state:
            return 0

        total_cost = 0

        # Map vehicles to their current physical location
        vehicle_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "at" and len(parts) == 3 and parts[1] in self.vehicles:
                vehicle_locations[parts[1]] = parts[2]

        # Determine the current state and physical location for each package
        package_state = {} # Maps package to 'at' or 'in'
        package_physical_loc = {} # Maps package to its physical location

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[1] in self.packages:
                if parts[0] == "at" and len(parts) == 3:
                    package_name, loc_name = parts[1], parts[2]
                    package_state[package_name] = 'at'
                    package_physical_loc[package_name] = loc_name
                elif parts[0] == "in" and len(parts) == 3:
                    package_name, vehicle_name = parts[1], parts[2]
                    package_state[package_name] = 'in'
                    # The physical location is the vehicle's location
                    if vehicle_name in vehicle_locations:
                         package_physical_loc[package_name] = vehicle_locations[vehicle_name]
                    # else: Vehicle not found or not at a location? Should not happen in valid states.
                    # If it happens, the package location is unknown, maybe return inf?
                    # For now, assume valid states.

        # Calculate cost for each package that is not yet satisfying its goal 'at' condition
        for package, goal_loc in self.goal_locations.items():
            # Check if the goal fact (at package goal_loc) is already true
            if f"(at {package} {goal_loc})" in state:
                continue # This package is already at its goal location on the ground

            # Get the package's current physical location
            current_loc = package_physical_loc.get(package)

            # If we couldn't determine the package's location, something is wrong or unreachable
            if current_loc is None:
                 # This package is not 'at' a location and not 'in' a located vehicle.
                 # This state is likely a dead end or invalid. Return infinity.
                 return float('inf')

            # Get the shortest distance to the goal location
            dist = self.shortest_paths.get((current_loc, goal_loc), float('inf'))

            # If the goal location is unreachable from the current location, return infinity
            if dist == float('inf'):
                 return float('inf')

            # Calculate cost based on the package's current state ('at' or 'in')
            if package_state.get(package) == 'at':
                # Package is on the ground at current_loc, needs load + drive + unload
                total_cost += 1      # load
                total_cost += dist   # drive actions
                total_cost += 1      # unload
            elif package_state.get(package) == 'in':
                # Package is in a vehicle at current_loc, needs drive + unload
                total_cost += dist   # drive actions
                total_cost += 1      # unload
            # else: package_state is None, handled by current_loc is None check

        return total_cost
