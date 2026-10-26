from collections import deque
import math

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the Heuristic base class is not provided externally, you might need a placeholder:
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError


def get_parts(fact):
    """Splits a PDDL fact string into its predicate and arguments."""
    # Example: '(at p1 l1)' -> ['at', 'p1', 'l1']
    return fact[1:-1].split()


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the estimated costs
        for each package that is not yet at its goal location. The cost for
        a package is estimated as the sum of actions needed: pick-up (if not
        in a vehicle), drive (shortest path distance), and drop. Vehicle
        capacity and availability are ignored.

    Assumptions:
        - Goals are always of the form (at package location).
        - The road network is static and provided by (road l1 l2) facts.
        - All locations relevant to goals and initial package/vehicle positions
          are part of the road network graph.
        - The road network allows reaching any goal location from any relevant
          starting location (solvable problem). Unreachable locations are assigned
          infinite distance, resulting in infinite heuristic, which is acceptable
          for unsolvable states.

    Heuristic Initialization:
        - Parses static facts to build the road network graph (adjacency list).
        - Computes all-pairs shortest paths between locations using BFS.
          Stores distances in a dictionary `self.distances[(l_from, l_to)]`.
        - Stores the goal locations for each package in `self.goal_locations`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize total heuristic cost to 0.
        2. Determine the current location/container for all locatable objects
           (packages and vehicles) from the current state facts. Store this
           mapping (object -> location or vehicle) in `obj_location`.
        3. Iterate through each package that has a goal location defined in
           `self.goal_locations`.
        4. For the current package:
           a. Check if the goal predicate `(at package goal_location)` is already
              true in the current state. If it is, the package has reached its
              goal, and the cost for this package is 0. Continue to the next package.
           b. If the goal predicate is not true, find the package's current
              position (`p_current_pos`) from `obj_location`.
           c. If `p_current_pos` is a location `l_current` (meaning the package
              is currently at `l_current`):
              - The estimated cost for this package is 1 (pick-up) +
                shortest_path_distance(`l_current`, `goal_location`) + 1 (drop).
              - Add this cost to the total heuristic.
           d. If `p_current_pos` is a vehicle `v` (meaning the package is
              currently inside vehicle `v`):
              - Find the current location of vehicle `v` (`v_current_loc`)
                from `obj_location`.
              - If `v_current_loc` is the same as `goal_location`:
                - The estimated cost for this package is 1 (drop).
                - Add this cost to the total heuristic.
              - If `v_current_loc` is different from `goal_location`:
                - The estimated cost for this package is shortest_path_distance(`v_current_loc`, `goal_location`) + 1 (drop).
                - Add this cost to the total heuristic.
           e. If at any point a required location or vehicle is not found (e.g.,
              package or vehicle not in `obj_location`, or vehicle location not
              in the road graph), this implies an unreachable state or malformed
              problem instance. Return infinity.
        5. Return the total heuristic cost.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            # Goal is expected to be '(at package location)'
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
            # Ignore other potential goal types if any (based on domain, only 'at' is expected)

        # Build the road network graph and compute shortest paths
        self.road_graph = {}
        locations_from_roads = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                locations_from_roads.add(l1)
                locations_from_roads.add(l2)
                if l1 not in self.road_graph:
                    self.road_graph[l1] = []
                self.road_graph[l1].append(l2)

        # Include all locations mentioned in goals, even if isolated in the static facts
        all_relevant_locations = set(locations_from_roads) | set(self.goal_locations.values())
        for loc in all_relevant_locations:
             if loc not in self.road_graph:
                 self.road_graph[loc] = [] # Add isolated locations to graph structure

        all_locations_list = list(self.road_graph.keys()) # Use keys to get all locations including isolated ones

        # Compute all-pairs shortest paths using BFS
        self.distances = {}

        for start_node in all_locations_list:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                (curr_loc, dist) = q.popleft()

                # The distance from start_node to curr_loc is 'dist'.
                self.distances[(start_node, curr_loc)] = dist

                if curr_loc in self.road_graph: # Handle locations with no outgoing roads
                    for neighbor in self.road_graph[curr_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, dist + 1))

        # Fill in unreachable distances with infinity
        for l1 in all_locations_list:
            for l2 in all_locations_list:
                if (l1, l2) not in self.distances:
                    self.distances[(l1, l2)] = math.inf # Use infinity for unreachable

    def __call__(self, node):
        state = node.state

        # Check if goal is reached (heuristic is 0 iff goal is reached)
        if self.task.goal_reached(state):
             return 0

        # Determine current location/container for all locatable objects
        obj_location = {} # Maps object name (str) to its location (str) or vehicle (str)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                obj_location[obj] = loc
            elif parts[0] == 'in' and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                obj_location[pkg] = veh
            # Ignore other predicates like capacity, road, capacity-predecessor

        total_cost = 0

        # Calculate cost for each package that needs to reach a goal location
        for package, goal_location in self.goal_locations.items():
            # Check if the goal predicate (at package goal_location) is already true
            if '(at {} {})'.format(package, goal_location) in state:
                 continue # Package goal is satisfied, cost is 0 for this package

            # If we reach here, the package is NOT satisfying its (at package goal_location) goal.
            # It's either at a wrong location or inside a vehicle.

            # Find the package's current position (location or vehicle)
            p_current_pos = obj_location.get(package)

            if p_current_pos is None:
                 # This indicates a package from the goal is not in the state.
                 # This shouldn't happen in valid PDDL instances. Treat as unreachable.
                 return math.inf

            # Case 1: Package is at a location (not the goal location, handled by the continue above)
            # We check if the position is a known location from our graph.
            if p_current_pos in self.road_graph:
                l_current = p_current_pos
                # Need to pick up (1) + drive (dist) + drop (1)
                drive_cost = self.distances.get((l_current, goal_location), math.inf)
                cost_for_package = 1 + drive_cost + 1
                total_cost += cost_for_package

            # Case 2: Package is inside a vehicle
            else: # p_current_pos is assumed to be a vehicle name
                vehicle = p_current_pos
                # Need to find vehicle's location
                v_current_loc = obj_location.get(vehicle)

                if v_current_loc is None or v_current_loc not in self.road_graph:
                    # Vehicle location unknown or not in graph - implies unsolvable or malformed state
                    return math.inf

                # If vehicle is at the goal location
                if v_current_loc == goal_location:
                    # Need to drop (1)
                    cost_for_package = 1
                    total_cost += cost_for_package
                # If vehicle is not at the goal location
                else:
                    # Need to drive (dist) + drop (1)
                    drive_cost = self.distances.get((v_current_loc, goal_location), math.inf)
                    cost_for_package = drive_cost + 1
                    total_cost += cost_for_package

        return total_cost
