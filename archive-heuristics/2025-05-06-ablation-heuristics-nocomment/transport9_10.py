# Assuming the Heuristic base class is defined elsewhere and available for import
# from heuristics.heuristic_base import Heuristic

from collections import defaultdict, deque

# Helper function to parse PDDL fact strings
def _parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and arguments."""
    # Example: '(at p1 l1)' -> ('at', ['p1', 'l1'])
    # Handles potential extra spaces
    parts = fact_string.strip()[1:-1].split()
    if not parts:
        return None, []
    return parts[0], parts[1:]

class transportHeuristic: # Inherit from Heuristic if available
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal by summing up
    independent minimum costs for each package that is not yet at its
    goal location. The cost for a package is estimated based on whether
    it is currently at a location or inside a vehicle, and the shortest
    road distance between its current position (or the vehicle's position)
    and its goal location.

    Assumptions:
    - Roads defined by the 'road' predicate are bidirectional.
    - The shortest path distance between locations in the road network
      provides a reasonable lower bound for the number of 'drive' actions.
    - Vehicle capacity constraints are ignored.
    - The heuristic treats the transport of each package independently,
      ignoring potential efficiencies from carrying multiple packages
      or conflicts from competing for vehicles.
    - Objects in the goal state that are of type 'package' are assumed
      to be the only objects needing transport. Other objects (vehicles)
      are assumed to be means of transport.
    - Objects starting with 'p' are packages and objects starting with 'v'
      are vehicles. This is a simplification based on common PDDL conventions
      and example files, as object types are not readily available from the
      `Task` object's state/static facts alone.
    - The heuristic is 0 if and only if all packages specified in the
      goal are at their respective goal locations.
    - The heuristic value is finite if all goal locations are reachable
      from current package/vehicle locations via the road network.
      Unreachable goals result in a large heuristic value.
    - State representation is valid: packages are either '(at p l)' or
      '(in p v)', and vehicles are '(at v l)'.

    Heuristic Initialization:
    The `__init__` method performs the following steps:
    1. Parses the task's goal facts to create a dictionary mapping each
       package object (that needs to be at a specific location in the goal)
       to its target location. It assumes goal facts are of the form `(at package location)`.
    2. Iterates through the static facts to identify all 'road' predicates.
       These facts define the road network. An adjacency list representation
       of this network is built, assuming bidirectionality for all roads.
    3. Identifies all unique locations present in the road network.
    4. For each location in the network, performs a Breadth-First Search (BFS)
       to compute the shortest path distance (in terms of number of 'drive'
       actions) to every other reachable location. These distances are stored
       in a nested dictionary `self.distances[start_loc][end_loc]`.

    Step-By-Step Thinking for Computing Heuristic (`__call__`):
    1. The current state (a frozenset of facts) is received via the `node` object.
    2. The state is scanned to determine the current status of all relevant
       objects (packages specified in the goal and vehicles).
       - A dictionary `package_status` is populated, mapping each goal package
         to either `{'at': location}` or `{'in': vehicle}` based on the state facts.
       - A dictionary `vehicle_locations` is populated, mapping each vehicle
         (identified by starting with 'v', assuming naming convention) to its
         current location based on the state facts.
    3. A variable `total_cost` is initialized to 0.
    4. The heuristic iterates through each package and its goal location stored
       in `self.goal_locations`.
    5. For the current package `p` and its goal location `goal_loc`:
       - It retrieves the current status of `p` from `package_status`.
       - If `p` is `(at p current_loc)`:
         - If `current_loc` is the same as `goal_loc`, the package is done; cost is 0.
         - If `current_loc` is different from `goal_loc`, the package needs to be
           picked up, transported, and dropped. The estimated cost is 1 (pick-up)
           + `self.distances[current_loc][goal_loc]` (minimum drive actions) + 1 (drop).
           If `goal_loc` is unreachable from `current_loc` via roads, a large cost is assigned.
       - If `p` is `(in p v)`:
         - It finds the current location of vehicle `v` from `vehicle_locations`.
         - If vehicle `v` is at `vehicle_loc`:
           - If `vehicle_loc` is the same as `goal_loc`, the package just needs to be
             dropped; cost is 1 (drop).
           - If `vehicle_loc` is different from `goal_loc`, the vehicle needs to drive
             to `goal_loc`, and then the package needs to be dropped. The estimated
             cost is `self.distances[vehicle_loc][goal_loc]` (minimum drive actions) + 1 (drop).
             If `goal_loc` is unreachable from `vehicle_loc` via roads, a large cost is assigned.
         - If vehicle `v`'s location is not found (indicating a potentially invalid state),
           a large cost is assigned.
       - If the package's status is not found (indicating a potentially invalid state),
         a large cost is assigned.
       - The calculated cost for the current package is added to `total_cost`.
    6. The final `total_cost` is returned as the heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.goal_locations = {}
        # Assuming goal facts are always (at package location)
        for goal in self.goals:
            predicate, args = _parse_fact(goal)
            if predicate == "at":
                # Ensure it's a package based on naming convention or type if available
                # For now, assume anything in an (at ?obj ?loc) goal is a package
                package, location = args
                self.goal_locations[package] = location
            # Ignore other potential goal predicates if any

        self.location_graph = defaultdict(set)
        locations = set()
        for fact in static_facts:
            predicate, args = _parse_fact(fact)
            if predicate == "road":
                l1, l2 = args
                self.location_graph[l1].add(l2)
                self.location_graph[l2].add(l1) # Assume bidirectional roads
                locations.add(l1)
                locations.add(l2)

        self.distances = {}
        for start_loc in locations:
            self.distances[start_loc] = self._bfs(start_loc, locations)

    def _bfs(self, start_node, all_nodes):
        """
        Performs BFS from a start node to find distances to all reachable nodes.
        Returns a dictionary mapping reachable nodes to their distance from start_node.
        """
        distances = {node: float('inf') for node in all_nodes}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists as a key in the graph (it should if it's in all_nodes)
            # and iterate over its neighbors.
            # Use .get() with default empty list for safety, though graph should be complete for 'locations'
            for neighbor in self.location_graph.get(current_node, []):
                 if distances[neighbor] == float('inf'):
                     distances[neighbor] = distances[current_node] + 1
                     queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state

        # 1. Identify current locations/containment
        package_status = {} # package -> {'at': loc} or {'in': vehicle}
        vehicle_locations = {} # vehicle -> loc

        # Assuming objects starting with 'p' are packages and 'v' are vehicles
        # This is a simplification based on common PDDL conventions and example files.
        # A robust parser would get this from the :objects section with types.
        for fact in state:
            predicate, args = _parse_fact(fact)
            if predicate == "at":
                obj, loc = args
                # Check if the object is one of the packages we care about (i.e., in the goals)
                # Or if it's a vehicle (based on naming convention)
                if obj in self.goal_locations:
                    package_status[obj] = {'at': loc}
                elif obj.startswith('v'): # Assuming 'v' prefix for vehicles
                    vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = args
                # Check if the object is one of the packages we care about
                if package in self.goal_locations:
                     package_status[package] = {'in': vehicle}

        # 2. Initialize total cost
        total_cost = 0
        LARGE_COST = 1000000 # Used for unreachable goals or invalid states

        # 3. Iterate through packages with goals
        for package, goal_location in self.goal_locations.items():
            # 4. Compute cost for the current package
            status = package_status.get(package)

            if status is None:
                 # Package not found in state facts (neither 'at' nor 'in').
                 # This indicates an invalid state representation according to domain logic.
                 # Treat as unreachable goal for this package.
                 total_cost += LARGE_COST
                 continue

            if 'at' in status:
                current_location = status['at']
                if current_location == goal_location:
                    # Package is at its goal location
                    cost_for_package = 0
                else:
                    # Package is at a different location, needs pickup, drive, drop
                    # Need shortest distance from current_location to goal_location
                    dist = self.distances.get(current_location, {}).get(goal_location, float('inf'))
                    if dist == float('inf'):
                         # Goal location is unreachable from current location
                         cost_for_package = LARGE_COST
                    else:
                         # 1 (pick-up) + dist (drive) + 1 (drop)
                         cost_for_package = 2 + dist

            elif 'in' in status:
                vehicle = status['in']
                # Package is in a vehicle, needs drive (by vehicle) and drop
                vehicle_location = vehicle_locations.get(vehicle)
                if vehicle_location is None:
                    # Vehicle location is unknown. Invalid state?
                    total_cost += LARGE_COST
                    continue

                if vehicle_location == goal_location:
                     # Vehicle is at the goal location, just need to drop
                     cost_for_package = 1 # drop
                else:
                     # Vehicle needs to drive to goal location, then drop
                     # Need shortest distance from vehicle_location to goal_location
                     dist = self.distances.get(vehicle_location, {}).get(goal_location, float('inf'))
                     if dist == float('inf'):
                          # Goal location is unreachable from vehicle location
                          cost_for_package = LARGE_COST
                     else:
                          # dist (drive) + 1 (drop)
                          cost_for_package = 1 + dist
            else:
                 # Should not happen in valid states
                 total_cost += LARGE_COST
                 continue

            total_cost += cost_for_package

        # 5. Return total cost
        return total_cost
