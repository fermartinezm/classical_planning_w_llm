import collections
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def bfs(start_node, graph):
    """Computes shortest path distances from start_node to all reachable nodes in a graph."""
    # Initialize distances for all nodes in the graph
    distances = {node: float('inf') for node in graph}

    # Distance from start_node to itself is 0
    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()
        current_dist = distances[current_node]

        # Iterate through neighbors if the current node has outgoing edges
        for neighbor in graph.get(current_node, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

    return distances


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum costs
    for each package that is not yet at its goal location. The cost for
    a package is estimated based on its current location (or the location
    of the vehicle carrying it) and its goal location, assuming unlimited
    vehicle capacity and ignoring the need for a vehicle to be present
    at the package's initial location for pick-up. The minimum travel
    cost between locations is the shortest path distance in the road network.

    Assumptions:
    - The road network is static and provides the only means of travel
      between locations for vehicles.
    - Vehicle capacity constraints are ignored.
    - The availability of a vehicle at a package's initial location for
      pick-up is ignored.
    - All locations mentioned in the problem (initial, goal, road network)
      are part of the road network graph (either as nodes or endpoints of edges).
    - The problem is solvable (i.e., goal locations are reachable from
      initial locations via the road network). Unreachable locations are
      assigned a large penalty in the heuristic.

    Heuristic Initialization:
    1. Parse the goal facts to identify the target location for each package.
       Store this in `self.goal_locations`.
    2. Parse the static facts to build the road network graph. Store this
       as an adjacency list in `self.road_graph`. Collect all unique locations.
    3. Compute shortest path distances between all pairs of locations using
       Breadth-First Search (BFS) on the road graph. Store these distances
       in `self.distances`.
    4. Identify all package and vehicle objects from the initial state,
       goals, and static facts for easier lookup in the heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the current state is the goal state. If yes, return 0.
    2. Initialize the total heuristic cost to 0.
    3. Determine the current location or carrier for every package and vehicle
       in the current state by parsing `(at ?x ?l)` and `(in ?p ?v)` facts.
       Store this in `current_locations`.
    4. Iterate through each package `p` and its goal location `goal_loc_p`
       stored during initialization.
    5. If package `p` is already at `goal_loc_p` (i.e., `(at p goal_loc_p)`
       is in the current state), add 0 cost for this package.
    6. If package `p` is not at its goal:
       a. Find its current status from `current_locations[p]`.
       b. If `p` is at a location `current_loc_p`:
          - The estimated cost for this package is 1 (pick-up) +
            shortest_distance(`current_loc_p`, `goal_loc_p`) (drive) +
            1 (drop).
          - Add this cost to the total heuristic.
       c. If `p` is in a vehicle `v`:
          - Find the vehicle's current location `current_loc_v` from
            `current_locations[v]`.
          - The estimated cost for this package is
            shortest_distance(`current_loc_v`, `goal_loc_p`) (drive) +
            1 (drop).
          - Add this cost to the total heuristic.
       d. If the goal location is unreachable from the current location
          (distance is infinity), add a large penalty to the total heuristic.
    7. Return the total heuristic cost.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals # Store goals for easy access
        self.goal_locations = {}
        self.road_graph = {}
        self.locations = set()
        self.packages = set()
        self.vehicles = set()

        # 1. Parse goals to get goal locations for packages
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at" and len(parts) == 3:
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)
                self.locations.add(location)

        # 2. Parse static facts to build road graph and collect locations/objects
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "road" and len(parts) == 3:
                # Fact is (road l1 l2)
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_graph:
                    self.road_graph[l1] = set()
                self.road_graph[l1].add(l2)
                self.locations.add(l1)
                self.locations.add(l2)
            # capacity-predecessor facts are ignored

        # Identify objects from initial state and goals if not already found
        # This ensures we have all packages, vehicles, and locations
        all_facts = set(task.initial_state) | set(task.goals)
        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] == "at" and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 # Simple inference: if it's a goal object or appears in 'in', it's a package
                 if obj in self.goal_locations or any(get_parts(f)[0] == 'in' and get_parts(f)[1] == obj for f in all_facts):
                     self.packages.add(obj)
                 else: # Otherwise, assume it's a vehicle
                     self.vehicles.add(obj)
             elif parts[0] == "in" and len(parts) == 3:
                 p, v = parts[1], parts[2]
                 self.packages.add(p)
                 self.vehicles.add(v)
             elif parts[0] == "capacity" and len(parts) == 2:
                 v = parts[1]
                 self.vehicles.add(v)
             # Add any locations mentioned in goals or initial state that weren't in road facts
             elif parts[0] == "at" and len(parts) == 3:
                 self.locations.add(parts[2])


        # Ensure all locations are in the graph keys for BFS, even if they have no outgoing edges
        for loc in self.locations:
            if loc not in self.road_graph:
                self.road_graph[loc] = set()


        # 3. Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
             self.distances[start_loc] = bfs(start_loc, self.road_graph)


    def __call__(self, node):
        state = node.state

        # 1. Check if goal reached (optional, but slightly faster)
        if self.task.goal_reached(state):
            return 0

        total_heuristic = 0
        current_locations = {} # Maps locatable object to its location or carrier

        # 3. Determine current locations/carriers
        # Iterate through all objects we know about (packages and vehicles)
        # and find their status in the current state.
        # This is more robust than iterating through state facts and guessing types.
        all_locatables = self.packages | self.vehicles
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in all_locatables:
                     current_locations[obj] = loc
            elif parts[0] == "in" and len(parts) == 3:
                p, v = parts[1], parts[2]
                if p in self.packages and v in self.vehicles:
                     current_locations[p] = v # Store carrier vehicle


        # 4. Iterate through misplaced packages
        for package, goal_loc_p in self.goal_locations.items():
            # 5. Check if package is already at goal
            # This check is necessary because current_locations might not contain
            # the package if it wasn't in the initial state, although our init
            # tries to prevent this for goal packages.
            if (f'(at {package} {goal_loc_p})' in state):
                continue # Package is at goal, cost is 0 for this package

            # 6. Package is not at goal, calculate cost
            current_status = current_locations.get(package)

            if current_status is None:
                 # This package's location/carrier is unknown in the current state.
                 # This indicates an issue or an unexpected state structure.
                 # Assign a large penalty.
                 total_heuristic += 1000000
                 continue

            if current_status in self.locations:
                # Package is at a location current_loc_p
                current_loc_p = current_status
                # Cost: pick-up (1) + drive (dist) + drop (1)
                dist = self.distances.get(current_loc_p, {}).get(goal_loc_p, float('inf'))
                if dist == float('inf'):
                    total_heuristic += 1000000 # Unreachable goal location
                else:
                    total_heuristic += 1 + dist + 1 # pick + drive + drop

            elif current_status in self.vehicles:
                # Package is in a vehicle v
                vehicle = current_status
                current_loc_v = current_locations.get(vehicle) # Get vehicle's location

                if current_loc_v is None or current_loc_v not in self.locations:
                    # Vehicle location unknown or invalid. Penalty.
                    total_heuristic += 1000000
                    continue

                # Cost: drive (dist) + drop (1)
                dist = self.distances.get(current_loc_v, {}).get(goal_loc_p, float('inf'))
                if dist == float('inf'):
                    total_heuristic += 1000000 # Unreachable goal location
                else:
                    total_heuristic += dist + 1 # drive + drop
            else:
                 # Status is neither a location nor a vehicle. Invalid state?
                 total_heuristic += 1000000


        return total_heuristic
