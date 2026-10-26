# Need to import deque for BFS
from collections import deque
# Assuming Heuristic base class is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define get_parts helper function
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Assume Heuristic base class is available and provides the expected interface.
# If running this code standalone, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

class transportHeuristic: # Renamed to match the request
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location, ignoring vehicle capacity
    and availability constraints. It sums the minimum actions needed for each
    package independently.

    # Assumptions
    - Each package can be transported independently by an available vehicle.
    - Vehicle capacity is not a bottleneck for calculating the minimum actions
      per package.
    - The cost of moving a package from location A to location B is the shortest
      path distance between A and B in the road network.
    - Actions have a cost of 1.
    - Goals are always of the form (at package location).
    - Package names and vehicle names are distinct and can be identified from
      initial state and goal facts. All locatables are either packages or vehicles.

    # Heuristic Initialization
    - Extract all locations from the problem definition (from static facts,
      initial state, and goals).
    - Build a graph representing the road network based on `road` predicates
      in the static facts.
    - Compute all-pairs shortest path distances between all locations using BFS.
      Store these distances.
    - Extract the goal location for each package from the task goals.
    - Identify all package and vehicle names from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Build a map of the current state, noting where each locatable object
       (package or vehicle) is located, or which vehicle a package is inside.
    2. Initialize total heuristic cost to 0.
    3. For each package `p` that has a goal location `g` defined in the task goals:
       a. Determine the package's current state: Is it at a location `l`
          (`(at p l)`) or inside a vehicle `v` (`(in p v)`)? This is found
          by looking up `p` in the state map.
       b. If the package is inside a vehicle `v`, find the current location `l`
          of that vehicle (`(at v l)`) by looking up `v` in the state map.
          The package's effective current location for transport purposes is `l`.
       c. If the package is currently at a location `l` (not inside a vehicle):
          - If `l` is the goal location `g`, the cost for this package is 0.
          - If `l` is not the goal location `g`: Needs pick-up (1), drive from `l` to `g` (distance), drop (1). Total: 2 + distance(l, g).
       d. If the package is currently inside a vehicle `v` which is at location `l`:
          - If `l` is the goal location `g`: Needs drop (1 action) to satisfy the `(at p g)` goal. Total cost: 1.
          - If `l` is not the goal location `g`: Needs drive from `l` to `g` (distance), then drop (1). Total cost: distance(l, g) (drive) + 1 (drop).
       e. If the required distance between locations `l` and `g` is infinite (meaning
          they are disconnected in the road network), the problem is likely unsolvable
          from this state, and the heuristic should return infinity immediately.
       f. If the package's state is not found or a vehicle's location is not found,
          return infinity (invalid state).
    4. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network graph, and precomputing distances.
        """
        # Store task components
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Collect all locations mentioned in the problem
        locations = set()
        # Also collect packages and vehicles to distinguish types later
        self.packages = set()
        self.vehicles = set()

        # Process static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "road":
                locations.add(parts[1])
                locations.add(parts[2])
            # capacity-predecessor, capacity facts are not needed for this heuristic

        # Process initial state facts
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "at":
                 # (at ?x - locatable ?v - location)
                 locatable, location = parts[1], parts[2]
                 locations.add(location)
                 # Cannot definitively type locatable from 'at' alone here without object types
             elif parts[0] == "in":
                 # (in ?x - package ?v - vehicle)
                 package, vehicle = parts[1], parts[2]
                 self.packages.add(package)
                 self.vehicles.add(vehicle)

        # Process goal facts
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             if parts[0] == "at":
                 # (at ?p - package ?l - location)
                 package, location = parts[1], parts[2]
                 locations.add(location)
                 self.packages.add(package) # Goals are for packages

        # Refine vehicles set: any locatable mentioned in initial 'at' facts
        # that is not identified as a package must be a vehicle.
        all_locatables_in_init_at = {get_parts(f)[1] for f in initial_state if get_parts(f) and get_parts(f)[0] == "at"}
        self.vehicles.update(all_locatables_in_init_at - self.packages)


        self.locations = list(locations) # Convert to list

        # 2. Build the road network graph (adjacency list)
        self.graph = {loc: [] for loc in self.locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                # Add edges assuming bidirectional roads
                if l1 in self.graph and l2 in self.graph:
                    self.graph[l1].append(l2)
                    self.graph[l2].append(l1)

        # Remove potential duplicate edges
        for loc in self.graph:
            self.graph[loc] = list(set(self.graph[loc]))

        # 3. Compute all-pairs shortest path distances using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = self._bfs(start_node)

        # 4. Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other types of goals if any exist, as per domain/examples

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all other nodes.
        Returns a dictionary mapping location nodes to their distance from start_node.
        Returns float('inf') for unreachable nodes.
        """
        dist = {node: float('inf') for node in self.graph}
        if start_node not in self.graph:
             # Should not happen if locations are collected correctly, but defensive
             return dist

        dist[start_node] = 0
        queue = deque([start_node])

        while queue:
            u = queue.popleft()
            # Check if u is a valid key in graph (should be)
            if u in self.graph:
                for v in self.graph[u]:
                    if dist[v] == float('inf'):
                        dist[v] = dist[u] + 1
                        queue.append(v)
        return dist

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Track current state of all locatables (packages and vehicles)
        # Maps object name to its location or vehicle name
        current_state_map = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                # (at ?x - locatable ?v - location)
                locatable, location = parts[1], parts[2]
                current_state_map[locatable] = location
            elif predicate == "in":
                # (in ?x - package ?v - vehicle)
                package, vehicle = parts[1], parts[2]
                current_state_map[package] = vehicle # Store the vehicle name as the package's state

        total_cost = 0  # Initialize action cost counter.

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_package_state = current_state_map.get(package)

            if current_package_state is None:
                 # Package state not found. This indicates an invalid state or problem definition.
                 return float('inf')

            # Check if the package is currently in a vehicle
            # A package is in a vehicle if its state maps to something that is a vehicle name
            if current_package_state in self.vehicles:
                 vehicle = current_package_state
                 current_location = current_state_map.get(vehicle) # Get vehicle's location

                 if current_location is None:
                     # Vehicle location not found. Invalid state.
                     return float('inf')

                 # Case: Package is in a vehicle at current_location
                 if current_location == goal_location:
                     # Package is in vehicle at goal location, needs 1 drop action
                     total_cost += 1
                 else:
                     # Package is in vehicle not at goal. Needs drive + drop.
                     # Distance from vehicle's current location to package's goal location
                     dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))
                     if dist == float('inf'):
                         return float('inf') # Unsolvable path
                     total_cost += dist + 1 # drive actions + 1 drop action

            else:
                # Case: Package is at a location (current_package_state is the location name)
                current_location = current_package_state # This is the location name

                if current_location != goal_location:
                    # Package is at a location, but not the goal. Needs pick-up + drive + drop.
                    dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))
                    if dist == float('inf'):
                        return float('inf') # Unsolvable path
                    total_cost += 1 + dist + 1 # 1 pick-up + drive actions + 1 drop action
                # If current_location == goal_location, cost for this package is 0, already handled.

        return total_cost
