from collections import deque
import math
# Assuming Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """
    Parses a PDDL fact string into a list of its components.
    E.g., '(at p1 l1)' -> ['at', 'p1', 'l1']
    """
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

# BFS function for shortest path in unweighted graph
def bfs(graph, start_node):
    """
    Computes shortest path distances from start_node to all other nodes
    in an unweighted graph using Breadth-First Search.

    Args:
        graph: Adjacency list representation {node: set(neighbors)}.
        start_node: The starting node.

    Returns:
        A dictionary {node: distance}. Unreachable nodes have distance float('inf').
    """
    distances = {node: math.inf for node in graph}
    if start_node in distances: # Handle case where start_node might not be in graph nodes (e.g., isolated location)
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is in graph keys before iterating neighbors
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
    return distances


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum actions
        required for each misplaced package to reach its goal location,
        ignoring vehicle capacity constraints and potential synergies from
        carrying multiple packages. The cost for a package depends on whether
        it is currently at a location or inside a vehicle, and involves
        pick-up (if at location), driving (shortest path distance on the road
        network), and dropping off.

    Assumptions:
        - The road network defined by (road l1 l2) facts is static and bidirectional.
        - All locations relevant to package goals or initial positions of
          packages/vehicles are part of the road network graph or added to it.
        - The cost of each action (drive, pick-up, drop) is 1.
        - Vehicle capacity is sufficient whenever a pick-up is attempted
          (capacity constraints are ignored in the cost calculation).
        - Vehicle availability is not a bottleneck (cost of getting a vehicle
          to a package's location for pick-up is ignored).
        - Packages are only located at locations or inside vehicles.
        - Goal conditions only involve packages being at specific locations.

    Heuristic Initialization:
        - Parses goal facts to identify target locations for each package.
        - Parses static facts to build the road network graph.
        - Computes all-pairs shortest paths between locations using BFS, storing
          these distances.
        - Identifies packages and vehicles from initial state and goals
          (assuming all relevant objects appear in initial 'at' facts or goal 'at' facts).
        - Identifies the minimum capacity size (c0) from capacity-predecessor
          facts (though this is not used in the current heuristic calculation).

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize total heuristic cost to 0.
        2. Parse the current state to determine the current location or containing
           vehicle for each package, and the current location for each vehicle.
        3. Iterate through each package that has a goal location defined.
        4. For a package 'p' with goal 'loc_p_goal':
           a. Check if the state contains the fact '(at p loc_p_goal)'. If yes,
              the package is already at its goal; add 0 to the total cost and continue
              to the next package.
           b. If the package is not at its goal, find its current status:
              i. If the package 'p' is currently at a location 'loc_p_current'
                 (i.e., state contains '(at p loc_p_current)'):
                 - The package needs to be picked up (1 action).
                 - It needs to be transported from 'loc_p_current' to 'loc_p_goal'.
                   The minimum number of drive actions required is the shortest
                   path distance between these locations.
                 - It needs to be dropped off at 'loc_p_goal' (1 action).
                 - The estimated cost for this package is 1 (pick-up) +
                   distance(loc_p_current, loc_p_goal) + 1 (drop).
              ii. If the package 'p' is currently inside a vehicle 'v'
                  (i.e., state contains '(in p v)'):
                  - Find the current location 'loc_v_current' of vehicle 'v'
                    (i.e., state contains '(at v loc_v_current)').
                  - If 'loc_v_current' is the same as 'loc_p_goal':
                    - The package only needs to be dropped off (1 action).
                    - The estimated cost for this package is 1.
                  - If 'loc_v_current' is different from 'loc_p_goal':
                    - The package needs to be transported from 'loc_v_current'
                      to 'loc_p_goal'. The minimum number of drive actions is
                      the shortest path distance.
                    - It needs to be dropped off at 'loc_p_goal' (1 action).
                    - The estimated cost for this package is distance(loc_v_current, loc_p_goal) + 1 (drop).
           c. If at any point a required location is not found in the precomputed
              distances (meaning the locations are disconnected), return infinity.
           d. Add the estimated cost for package 'p' to the total cost.
        5. Return the total accumulated cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to identify vehicles

        # 1. Store goal locations for packages
        self.goal_locations = {}
        self.packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at':
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.packages.add(package)

        # 2. Identify vehicles (assume objects in initial 'at' or 'capacity' facts that are not packages)
        self.vehicles = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 obj, loc = parts[1], parts[2]
                 if obj not in self.packages:
                     self.vehicles.add(obj)
             elif parts[0] == 'capacity':
                  vehicle = parts[1]
                  self.vehicles.add(vehicle)

        # 3. Build road graph and identify all locations
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Roads are bidirectional

        # Add any locations mentioned in goals or initial state but not in road facts
        # This ensures BFS is run for all potentially relevant locations
        for loc in self.goal_locations.values():
             locations.add(loc)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                  loc = parts[2]
                  locations.add(loc)
        # Ensure all locations found are keys in the graph, even if isolated
        for loc in locations:
             self.road_graph.setdefault(loc, set())


        # 4. Compute all-pairs shortest paths
        self.distances = {}
        for loc in locations:
            self.distances[loc] = bfs(self.road_graph, loc)

        # 5. Identify minimum capacity size (optional for this heuristic)
        # Find sizes that are not the second argument of any capacity-predecessor
        first_args = set()
        second_args = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'capacity-predecessor':
                s1, s2 = parts[1], parts[2]
                first_args.add(s1)
                second_args.add(s2)
        min_capacities = first_args - second_args
        # Assuming there's exactly one minimum capacity 'c0'
        self.min_capacity = list(min_capacities)[0] if min_capacities else None


    def __call__(self, node):
        state = node.state

        # 1. Parse current state for package and vehicle locations/status
        package_current_status = {} # {package: ('at', location) or ('in', vehicle)}
        vehicle_current_locations = {} # {vehicle: location}

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == 'at':
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_status[obj] = ('at', loc)
                elif obj in self.vehicles:
                    vehicle_current_locations[obj] = loc
            elif pred == 'in':
                package, vehicle = parts[1], parts[2]
                if package in self.packages and vehicle in self.vehicles: # Ensure they are known objects
                     package_current_status[package] = ('in', vehicle)

        # 2. Calculate total cost based on misplaced packages
        total_cost = 0

        for package, goal_location in self.goal_locations.items():
            # Check if package is already at goal location
            if package in package_current_status and package_current_status[package] == ('at', goal_location):
                continue # Package is at goal, cost is 0 for this package

            # Package is misplaced
            if package not in package_current_status:
                 # Should not happen in valid states, but indicates goal is likely unreachable
                 return math.inf

            status, obj_or_loc = package_current_status[package]

            if status == 'at':
                loc_p_current = obj_or_loc
                # Cost: pick-up (1) + drive (dist) + drop (1)
                # Check if locations exist in our distance map
                if loc_p_current not in self.distances or goal_location not in self.distances.get(loc_p_current, {}):
                     return math.inf # Unreachable
                dist = self.distances[loc_p_current][goal_location]

                if dist == math.inf: return math.inf # Unreachable

                total_cost += 2 + dist

            elif status == 'in':
                vehicle = obj_or_loc
                if vehicle not in vehicle_current_locations:
                     # Vehicle location unknown? Should not happen in valid state.
                     return math.inf # Unreachable

                loc_v_current = vehicle_current_locations[vehicle]
                # Cost: drive (dist) + drop (1)
                # Check if locations exist in our distance map
                if loc_v_current not in self.distances or goal_location not in self.distances.get(loc_v_current, {}):
                     return math.inf # Unreachable
                dist = self.distances[loc_v_current][goal_location]

                if dist == math.inf: return math.inf # Unreachable

                total_cost += 1 + dist

        return total_cost
