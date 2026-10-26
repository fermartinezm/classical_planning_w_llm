from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this is available

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, ignoring vehicle capacity and sharing. It sums the
    estimated costs for each package independently. The cost for a package
    includes loading, driving (shortest path distance), and unloading actions.

    # Assumptions
    - The cost of any action (load, unload, drive one road segment) is 1.
    - Vehicle capacity constraints are ignored. Any package can be carried by any vehicle.
    - Vehicle availability is simplified: a vehicle is assumed to be available
      at a package's location when needed for pickup.
    - The heuristic is the sum of costs for each package independently reaching its goal.
    - The problem state representation includes all relevant 'at' and 'in' facts for packages and vehicles.
    - All locations mentioned in goals, initial state 'at' facts, and 'road' facts are part of the connected component relevant to the problem, or unreachable goals are handled by returning infinity.

    # Heuristic Initialization
    - Extracts goal locations for each package and identifies all packages.
    - Identifies all locations mentioned in the initial state, goals, and static facts ('road' facts).
    - Builds the road network graph from static facts.
    - Computes all-pairs shortest paths between all identified locations using BFS.
    - Identifies vehicles based on initial state/static facts (e.g., having capacity or being at a location and not being a package/location).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is a goal state. If yes, heuristic is 0.
    2. Identify the current location of every package. A package can be on the
       ground at a location `loc_p` or inside a vehicle `v`. This information
       is extracted from the state's `(at ?p ?loc)` and `(in ?p ?v)` facts.
    3. Identify the current location `loc_v` for every vehicle `v` from the
       state's `(at ?v ?loc)` facts.
    4. Initialize total heuristic cost to 0.
    5. For each package `p` that has a goal location `goal_p` (extracted during initialization):
       a. Get the current state of package `p` (its location or the vehicle it's in).
       b. If package `p` is already at `goal_p` (i.e., `(at p goal_p)` is in the state),
          this package contributes 0 to the heuristic.
       c. If package `p` is on the ground at `loc_p` (`loc_p` is a location and not `goal_p`):
          - The estimated cost for this package is 1 (load) + shortest_path_distance(`loc_p`, `goal_p`) (drive) + 1 (unload).
          - If `goal_p` is unreachable from `loc_p`, the heuristic is infinity.
       d. If package `p` is inside a vehicle `v`:
          - Find the current location `loc_v` of vehicle `v`.
          - The estimated cost for this package is shortest_path_distance(`loc_v`, `goal_p`) (drive) + 1 (unload).
          - If `goal_p` is unreachable from `loc_v`, the heuristic is infinity.
       e. Add the estimated cost for this package to the total heuristic cost.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, road network,
        and precomputing shortest paths.
        """
        # The base class Heuristic might have its own __init__, but it's not strictly
        # necessary to call super().__init__(task) based on the provided example base class.
        # We will initialize our own required attributes.

        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.goal_locations = {}
        self.packages = set()
        # Extract package goals and identify packages
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)

        self.locations = set()
        self.road_graph = {} # Adjacency list: {loc: {neighbor1, neighbor2, ...}}
        self.vehicles = set() # Identify vehicles

        # Extract locations, road network, and identify vehicles from initial state and static facts
        all_facts = static_facts | initial_state
        for fact in all_facts:
             parts = get_parts(fact)
             if not parts: continue

             predicate = parts[0]
             if predicate == "road":
                 loc1, loc2 = parts[1], parts[2]
                 self.locations.add(loc1)
                 self.locations.add(loc2)
                 if loc1 not in self.road_graph:
                     self.road_graph[loc1] = set()
                 self.road_graph[loc1].add(loc2)
             elif predicate == "at":
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 # Identify vehicles: objects at locations that are not packages or known locations
                 # This is a heuristic-specific simplification.
                 if obj not in self.packages and obj not in self.locations:
                      self.vehicles.add(obj)
             elif predicate == "capacity": # Vehicles also have capacity
                  vehicle = parts[1]
                  self.vehicles.add(vehicle)
             elif predicate == "in": # Vehicles can contain packages
                  # package = parts[1] # Not used for vehicle identification here, but good to parse
                  vehicle = parts[2]
                  self.vehicles.add(vehicle)


        # Ensure all locations from goals are included
        self.locations.update(self.goal_locations.values())

        # Ensure all locations in road_graph keys/values exist in locations set
        # Iterate over copies to allow adding elements during iteration
        for loc in list(self.road_graph.keys()):
             self.locations.add(loc)
             for neighbor in list(self.road_graph.get(loc, set())):
                 self.locations.add(neighbor)


        # Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_node in self.locations:
            self.shortest_paths[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_loc, dist = queue.popleft()
                self.shortest_paths[start_node][current_loc] = dist

                # Get neighbors, handle locations not in road_graph keys (isolated locations)
                neighbors = self.road_graph.get(current_loc, set())

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is already reached
        if self.task.goal_reached(state):
             return 0

        package_current_state = {} # Maps package to its location (string) or vehicle (string)
        vehicle_current_location = {} # Maps vehicle to its location (string)

        # Populate current locations from the state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_state[obj] = loc
                elif obj in self.vehicles:
                     vehicle_current_location[obj] = loc
            elif predicate == "in":
                p, v = parts[1], parts[2]
                if p in self.packages and v in self.vehicles:
                    package_current_state[p] = v

        total_cost = 0

        # Calculate cost for each package not at its goal
        for package, goal_location in self.goal_locations.items():
            current_state = package_current_state.get(package)

            # If package is not in the state, something is wrong or it's a goal fact already handled
            # Assuming valid states always contain facts about packages unless they are at goal
            # and the goal fact is present. The initial goal_reached check handles the latter.
            if current_state is None:
                 # This might happen if a package is not in the initial state facts,
                 # but is in the goal. Treat as unreachable.
                 return float('inf')

            # If package is on the ground at goal, cost is 0 for this package.
            if current_state == goal_location:
                 continue

            # Package is not (at p goal_location). Calculate cost.
            if current_state in self.locations: # Package is on the ground at current_state (loc_p) != goal_location
                loc_p = current_state
                # Cost = Load (1) + Drive (dist) + Unload (1)
                dist = self.shortest_paths.get(loc_p, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal location unreachable from package's current location
                total_cost += 2 + dist

            elif current_state in self.vehicles: # Package is inside vehicle (current_state is vehicle name v)
                v = current_state
                loc_v = vehicle_current_location.get(v)

                if loc_v is None:
                    # Vehicle containing package is not at any location? Invalid state.
                    return float('inf')

                # Cost = Drive (dist from vehicle's location) + Unload (1)
                dist = self.shortest_paths.get(loc_v, {}).get(goal_location, float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal location unreachable from vehicle's current location
                total_cost += 1 + dist
            else:
                # current_state is neither a location nor a known vehicle? Invalid state representation.
                return float('inf')


        return total_cost
