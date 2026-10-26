from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque
import logging

# Configure logging if needed (optional)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.
    """

    def __init__(self, task: Task):
        """
        Initializes the transport heuristic.

        Heuristic Initialization:
        Parses the task definition to extract static information about
        the road network and package goals. It then precomputes the
        shortest path distances between all pairs of locations using BFS.
        It also identifies all packages, vehicles, and locations relevant
        to the problem by inspecting initial state, goals, and static facts.

        Args:
            task: An instance of the Task class representing the planning problem.
        """
        super().__init__()

        self.package_goals = {}
        self.locations = set()
        self.packages = set()
        self.vehicles = set()

        # 1. Extract package goals and relevant objects from goals
        for fact_str in task.goals:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'at':
                pkg, loc = parts[1], parts[2]
                self.packages.add(pkg)
                self.locations.add(loc)
                self.package_goals[pkg] = loc

        # 2. Extract locations and build road graph from static facts
        road_graph = {}
        for fact_str in task.static:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in road_graph:
                    road_graph[l1] = []
                road_graph[l1].append(l2)
            # Also extract vehicles from static facts like capacity
            elif parts[0] == 'capacity':
                 self.vehicles.add(parts[1])
            # Also extract packages/vehicles from static facts like 'in' (less common in static, but possible)
            elif parts[0] == 'in':
                 self.packages.add(parts[1])
                 self.vehicles.add(parts[2])


        # 3. Extract vehicles from initial state facts (e.g., vehicles' initial positions)
        # and ensure all packages/locations mentioned in init are included
        for fact_str in task.initial_state:
             parts = self._parse_fact(fact_str)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 # Assume if it's not a package we already found, it's a vehicle.
                 # This inference is based on typical transport domain structure.
                 if obj not in self.packages:
                      self.vehicles.add(obj)
                 self.locations.add(loc)
             elif parts[0] == 'in':
                  pkg, veh = parts[1], parts[2]
                  self.packages.add(pkg)
                  self.vehicles.add(veh)
             elif parts[0] == 'capacity':
                  self.vehicles.add(parts[1])
             # Locations from initial state facts
             if len(parts) > 2 and parts[0] in ['at']:
                  self.locations.add(parts[2])


        # Convert sets to lists for consistent iteration if needed (e.g., iterating through self.packages)
        self.locations = list(self.locations)
        self.packages = list(self.packages)
        self.vehicles = list(self.vehicles)


        # 4. Compute all-pairs shortest paths on the road graph
        self.distances = self._compute_distances(self.locations, road_graph)

    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
        return fact_str.strip('()').split()

    def _compute_distances(self, locations, graph):
        """Computes all-pairs shortest paths using BFS."""
        distances = {}
        for start_node in locations:
            distances[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            distances[start_node][start_node] = 0 # Distance to self is 0

            while q:
                current_node, dist = q.popleft()

                if current_node in graph:
                    for neighbor in graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_node][neighbor] = dist + 1
                            q.append((neighbor, dist + 1))

            # For any location not reached, distance remains effectively infinity
            # Ensure all locations are keys in the inner dictionary, even if unreachable
            for loc in locations:
                 if loc not in distances[start_node]:
                     distances[start_node][loc] = float('inf')

        return distances


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the heuristic value to 0.
        2. Create lookup dictionaries for 'at' and 'in' facts in the current state
           to quickly find the location of objects or which vehicle a package is in.
        3. Iterate through each package identified during initialization.
        4. For each package, check if it has a goal location defined. If not, skip it.
        5. Check if the package is currently at its goal location based on the 'at' lookup map.
           If it is, it contributes 0 to the heuristic, so continue to the next package.
        6. If the package is not at its goal:
           a. Check if the package is currently 'at' some location (which must be a wrong location).
              If yes, get the current location. The estimated cost for this package is
              1 (pick-up) + shortest_distance(current_location, goal_location) + 1 (drop).
              Add this cost to the total heuristic value.
           b. If the package is not 'at' any location, it must be 'in' a vehicle.
              Find the vehicle it is in using the 'in' lookup map.
              Find the current location of that vehicle using the 'at' lookup map.
              If the vehicle's location is found, the estimated cost for this package is
              shortest_distance(vehicle_location, goal_location) + 1 (drop).
              Add this cost to the total heuristic value.
              (Assumes valid states where a vehicle containing a package is always at a location).
        7. Return the total accumulated heuristic value.

        Assumptions:
        - The heuristic assumes that every package not at its goal location
          requires a sequence of actions (pick-up, drive, drop or drive, drop)
          independent of other packages or vehicle capacity constraints.
        - It assumes the road network is correctly defined by 'road' facts and
          distances are precomputed correctly.
        - It assumes valid state representation where packages are either 'at'
          a location or 'in' a vehicle, and vehicles are 'at' a location.
        - Object types (package, vehicle, location) are inferred based on
          their appearance in specific predicates ('at', 'in', 'capacity', 'road')
          in the initial state, goals, and static facts. This inference is
          a heuristic-specific assumption due to the lack of explicit object type
          information in the provided Task structure.

        Args:
            node: The current search node containing the state.

        Returns:
            An integer or float representing the estimated cost to reach the goal.
            Returns float('inf') if a required distance is infinite (e.g., unreachable location).
        """
        state = node.state
        h_value = 0

        # Create lookup maps for faster access to object locations and package contents
        at_map = {}
        in_map = {}
        for fact_str in state:
            parts = self._parse_fact(fact_str)
            if parts[0] == 'at':
                # parts[1] is the object (locatable), parts[2] is the location
                at_map[parts[1]] = parts[2]
            elif parts[0] == 'in':
                # parts[1] is the package, parts[2] is the vehicle
                in_map[parts[1]] = parts[2]

        # Iterate through all packages identified during initialization
        for pkg in self.packages:
            goal_l = self.package_goals.get(pkg)

            # Skip packages that don't have a specific goal location defined
            if goal_l is None:
                 continue

            # Check if the package is already at its goal location
            if pkg in at_map and at_map[pkg] == goal_l:
                continue # Package is at goal, cost is 0 for this package

            # Package is not at its goal. Calculate estimated cost.
            if pkg in at_map:
                # Package is at a location (which is not the goal)
                current_l = at_map[pkg]
                # Estimated cost: pick-up (1) + drive (distance) + drop (1)
                # Distance lookup: self.distances[start_loc][end_loc]
                # Handle potential unreachable locations
                distance = self.distances.get(current_l, {}).get(goal_l, float('inf'))
                if distance == float('inf'):
                    # If the goal location is unreachable from the package's current location,
                    # the state is likely a dead end or requires complex coordination not captured
                    # by this simple sum. Returning inf guides search away.
                    return float('inf')
                h_value += 2 + distance
            elif pkg in in_map:
                # Package is in a vehicle
                veh = in_map[pkg]
                # Find the vehicle's current location
                if veh in at_map:
                    v_l = at_map[veh]
                    # Estimated cost: drive (distance) + drop (1)
                    distance = self.distances.get(v_l, {}).get(goal_l, float('inf'))
                    if distance == float('inf'):
                         # If the goal location is unreachable from the vehicle's current location,
                         # treat as unreachable.
                         return float('inf')
                    h_value += 1 + distance
                else:
                    # This case indicates an invalid state where a vehicle containing
                    # a package is not located anywhere. Treat as unreachable.
                    # logger.warning(f"Vehicle {veh} containing package {pkg} is not at any location in state.")
                    return float('inf')
            else:
                # This case indicates an invalid state where a package is neither
                # at a location nor in a vehicle. Treat as unreachable.
                # logger.warning(f"Package {pkg} is neither at a location nor in a vehicle in state.")
                return float('inf')

        return h_value
