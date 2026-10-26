from collections import deque
import math
# Assuming heuristics.heuristic_base.Heuristic is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper function to split a PDDL fact string into parts.
def get_parts(fact):
    """Helper function to split a PDDL fact string into parts."""
    # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    return fact[1:-1].split()

# Inherit from Heuristic base class
class transportHeuristic: # Replace with class transportHeuristic(Heuristic): if base class is available
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the cost to reach the goal by summing up
    independent costs for each package that is not yet in its final goal
    configuration. For a package currently at a location different from its goal,
    the cost is estimated as 2 (for pick-up and drop) plus the shortest path
    distance from its current location to its goal location. For a package
    currently inside a vehicle, the cost is estimated as 1 (for drop) plus the
    shortest path distance from the vehicle's current location to the package's
    goal location. If a package is in a vehicle that is already at the package's
    goal location, the cost is 1 (for the drop action). The shortest path
    distances are precomputed using BFS. This heuristic is non-admissible as it
    ignores vehicle capacity and assumes packages can be transported independently,
    potentially double-counting drive costs. It aims to guide a greedy best-first
    search efficiently by prioritizing states where packages are closer to their
    destinations or require fewer actions.

    Assumptions:
    - The road network is undirected (road l1 l2 implies road l2 l1). The PDDL
      domain confirms this with bidirectional road facts.
    - Vehicle capacity constraints are ignored for the heuristic calculation.
    - Any vehicle can pick up any package at the same location.
    - The state representation is a frozenset of strings like '(predicate arg1 arg2)'.
    - The task object provides 'goals' (frozenset of goal facts), 'static'
      (frozenset of static facts), and 'initial_state' (frozenset of initial facts).
    - The node object provides 'state' (frozenset of facts).
    - All locations and objects referenced in goals and initial state are
      present in the static facts or initial state facts in a way that allows
      identification (e.g., locations in road facts, vehicles in capacity/in/at facts,
      packages in goal/at/in facts).

    Heuristic Initialization:
    1. Parse goal facts to create a mapping from package objects to their goal
       locations. Also, identify the set of packages that are goals.
    2. Build an adjacency list representation of the road network graph from
       'road' static facts. Collect all unique locations.
    3. Identify vehicles by looking for objects appearing as the second argument
       in '(capacity ...)' static facts or as the second argument in '(in ...)'
       facts in the initial state. Also, include any object found at a location
       in the initial state that is not a goal package, as a potential vehicle.
    4. Compute all-pairs shortest path distances between all identified locations
       using Breadth-First Search (BFS) starting from each location. Store these
       distances in a dictionary of dictionaries. Unreachable locations are marked
       with infinity.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Create mappings for the current state: `current_status` (object -> location or vehicle)
       and `vehicle_locations` (vehicle -> location) by iterating through the facts
       in the current state.
    3. Iterate through the packages that are part of the goals (`self.goal_packages`).
    4. For each package `p`:
       a. Get its goal location `l_goal` from `self.goal_locations`.
       b. Find the current status of `p` using the `current_status` mapping.
       c. If `p` is not found in `current_status` (unexpected state), add infinity to `h`
          and continue to the next package.
       d. Let `current_loc_or_veh` be the current status of `p`.
       e. If `current_loc_or_veh` in self.locations: # Check if it's a location object
          i. If `l_current` is the same as `l_goal`, the package is already at its goal
             location. Add 0 to `h` for this package.
          ii. If `l_current` is different from `l_goal`:
              # Package is at the wrong location
              # Cost: 1 (pick-up) + 1 (drop) + drive from current location to goal
              if l_current in self.shortest_paths and l_goal in self.shortest_paths[l_current]:
                   h += 2 + self.shortest_paths[l_current][l_goal]
              else:
                   h += math.inf # Unreachable goal
       f. If `current_loc_or_veh` in self.vehicles: # Check if it's a vehicle object
           veh = current_loc_or_veh
           if veh in vehicle_locations: # Vehicle location known
               l_v = vehicle_locations[veh]
               if l_v != l_goal:
                   # Package is in vehicle, vehicle is not at goal location
                   # Cost: 1 (drop) + drive from vehicle location to goal
                   if l_v in self.shortest_paths and l_goal in self.shortest_paths[l_v]:
                        h += 1 + self.shortest_paths[l_v][l_goal]
                   else:
                        h += math.inf # Unreachable goal
               else: # l_v == l_goal
                   # Package is in vehicle, vehicle is at goal location
                   # Cost: 1 (drop)
                   h += 1
           else:
               # Vehicle location not found - state inconsistency?
               h += math.inf # Assume unreachable

            # Case 3: Package status is something else (should not happen in valid states)
            # This case is implicitly handled because if it's not a location or a vehicle,
            # it won't match the checks above. We could add an explicit else with math.inf
            # but the current structure covers the expected cases.

    5. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        # Assuming task object has attributes: goals, static, initial_state
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Parse goal facts and identify goal packages
        self.goal_locations = {}
        self.goal_packages = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                self.goal_packages.add(package)

        # 2. Build road network graph and identify all locations
        self.locations = set()
        self.road_graph = {} # Adjacency list: location -> [neighbor1, neighbor2, ...]
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Roads are bidirectional

        # Ensure all locations mentioned in goals are in the graph, even if isolated
        for loc in self.goal_locations.values():
             self.locations.add(loc)
             self.road_graph.setdefault(loc, [])

        # 3. Identify vehicles
        self.vehicles = set()
        # Vehicles appear in capacity facts (static)
        for fact in static_facts:
             parts = get_parts(fact)
             if parts[0] == "capacity":
                  self.vehicles.add(parts[1])
        # Vehicles appear as the container in 'in' facts (initial state)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "in":
                  package, vehicle = parts[1], parts[2]
                  self.vehicles.add(vehicle)
        # Vehicles appear at locations in initial state, and are not goal packages
        # This is a fallback/additional way to find vehicles
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj = parts[1]
                 # If the object is not a package we care about (a goal package), assume it's a vehicle
                 # This might include non-goal packages, but they won't affect the heuristic calculation
                 # which only considers goal_packages. It's safe to add them to self.vehicles set.
                 if obj not in self.goal_packages:
                      self.vehicles.add(obj)


        # 4. Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        for start_node in self.locations:
            self.shortest_paths[start_node] = {}
            queue = deque([(start_node, 0)])
            visited = {start_node: 0}

            while queue:
                current_loc, dist = queue.popleft()
                self.shortest_paths[start_node][current_loc] = dist

                # Handle locations not in road_graph (isolated nodes)
                neighbors = self.road_graph.get(current_loc, [])

                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

            # Mark unreachable locations with infinity
            for loc in self.locations:
                 if loc not in self.shortest_paths[start_node]:
                     self.shortest_paths[start_node][loc] = math.inf


    def __call__(self, node):
        state = node.state
        h = 0

        # Get current locations/status of all locatables (packages and vehicles)
        current_status = {} # obj -> location or vehicle
        vehicle_locations = {} # vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_status[obj] = loc
                if obj in self.vehicles:
                    vehicle_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle # Status of package is the vehicle it's in

        # Iterate through packages that have a goal location defined
        for package in self.goal_packages:
            l_goal = self.goal_locations[package]

            # Check if package is in the current state
            if package not in current_status:
                 # This package is not 'at' any location and not 'in' any vehicle.
                 # This indicates an invalid state or a package that disappeared.
                 # Treat as unreachable goal for this package.
                 h += math.inf
                 continue # Move to next package

            current_loc_or_veh = current_status[package]

            # Case 1: Package is at a location
            if current_loc_or_veh in self.locations: # Check if it's a location object
                l_current = current_loc_or_veh
                if l_current != l_goal:
                    # Package is at the wrong location
                    # Cost: 1 (pick-up) + 1 (drop) + drive from current location to goal
                    if l_current in self.shortest_paths and l_goal in self.shortest_paths[l_current]:
                         h += 2 + self.shortest_paths[l_current][l_goal]
                    else:
                         h += math.inf # Unreachable goal
                # Else: l_current == l_goal. Cost is 0 for this package.

            # Case 2: Package is in a vehicle
            elif current_loc_or_veh in self.vehicles: # Check if it's a vehicle object
                 veh = current_loc_or_veh
                 if veh in vehicle_locations: # Vehicle location known
                     l_v = vehicle_locations[veh]
                     if l_v != l_goal:
                         # Package is in vehicle, vehicle is not at goal location
                         # Cost: 1 (drop) + drive from vehicle location to goal
                         if l_v in self.shortest_paths and l_goal in self.shortest_paths[l_v]:
                              h += 1 + self.shortest_paths[l_v][l_goal]
                         else:
                              h += math.inf # Unreachable goal
                     else: # l_v == l_goal
                         # Package is in vehicle, vehicle is at goal location
                         # Cost: 1 (drop)
                         h += 1
                 else:
                     # Vehicle location not found - state inconsistency?
                     h += math.inf # Assume unreachable

            # Case 3: Package status is something else (should not happen in valid states)
            # This case is implicitly handled because if it's not a location or a vehicle,
            # it won't match the checks above. We could add an explicit else with math.inf
            # but the current structure covers the expected cases.

        return h
