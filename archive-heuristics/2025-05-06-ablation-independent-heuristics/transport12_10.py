import collections
import re
import logging

from heuristics.heuristic_base import Heuristic
from task import Task

# Configure logging (optional, but helpful for debugging)
# logging.basicConfig(level=logging.INFO)

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to move all packages to their goal locations.
    The heuristic sums the estimated cost for each package that is not yet at its goal.
    The cost for a package is estimated based on its current location (at a location or in a vehicle)
    and the shortest path distance to its goal location in the road network.
    Vehicle capacity and availability are ignored, making this a relaxation.

    Assumptions:
    - The goal is primarily defined by '(at package location)' facts.
    - The road network defined by '(road l1 l2)' facts is static and provides the only means of travel for vehicles.
    - Roads are assumed to be bidirectional if a '(road l1 l2)' fact exists (i.e., if l1 can reach l2, l2 can reach l1).
    - Packages are either '(at package location)' or '(in package vehicle)' in any valid state.
    - All locations mentioned in goals, initial state, or road facts are relevant.
    - Object names do not contain spaces.
    - Objects appearing as the first argument of an '(in ...)' fact or in a goal '(at ...)' fact are packages. Other objects appearing as the first argument of an '(at ...)' fact are vehicles.

    Heuristic Initialization:
    1. Parses goal facts to create a mapping from package names to their goal location names (`self.package_goals`).
    2. Parses static facts to build an undirected graph representing the road network (`self.road_graph`).
    3. Identifies all relevant locations from road facts, goal facts, and initial state facts (`self.locations`).
    4. Computes all-pairs shortest paths between all identified locations using Breadth-First Search (BFS) and stores the distances (`self.shortest_paths`).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value `h` to 0.
    2. Get the current state as a set of facts (`node.state`).
    3. Create temporary dictionaries to quickly look up package and vehicle locations/status in the current state by iterating through the state facts:
       - `package_current_location`: maps package name to its current location string (if at a location).
       - `package_current_vehicle`: maps package name to the vehicle string it is currently in (if in a vehicle).
       - `vehicle_current_location`: maps vehicle name to its current location string.
       This involves two passes over the state facts to first identify packages (from 'in' facts and goal facts) and then classify 'at' facts based on this.
    4. Iterate through each package `p` that has a goal location `loc_p_goal` defined in `self.package_goals`.
    5. For the current package `p`:
       a. Check if `p` is already at its goal location in the current state (using the `package_current_location` lookup). If yes, the cost for this package is 0, continue to the next package.
       b. If `p` is not at its goal, determine its current status:
          i. If `p` is at a location `loc_p_current` (i.e., `p` is a key in `package_current_location`):
             - The estimated cost for this package is the sum of:
               - 1 action for `pick-up`.
               - The shortest path distance (`dist(loc_p_current, loc_p_goal)`) in number of `drive` actions.
               - 1 action for `drop`.
             - Total cost for this package: `2 + dist(loc_p_current, loc_p_goal)`.
             - If `loc_p_goal` is unreachable from `loc_p_current`, `dist` will be infinity, and the heuristic will correctly become infinity.
             - Add this cost to `h`.
          ii. If `p` is in a vehicle `v` (i.e., `p` is a key in `package_current_vehicle`):
              - Find the vehicle `v`'s current location `loc_v_current` using the `vehicle_current_location` lookup.
              - If `loc_v_current` is the same as `loc_p_goal`:
                - The estimated cost is 1 action for `drop`.
                - Add this cost to `h`.
              - If `loc_v_current` is different from `loc_p_goal`:
                - The estimated cost is the sum of:
                  - The shortest path distance (`dist(loc_v_current, loc_p_goal)`) in number of `drive` actions.
                  - 1 action for `drop`.
                - Total cost for this package: `dist(loc_v_current, loc_p_goal) + 1`.
                - If `loc_p_goal` is unreachable from `loc_v_current`, `dist` will be infinity, and the heuristic will correctly become infinity.
                - Add this cost to `h`.
          iii. If the package has a goal but is neither at a location nor in a vehicle according to the state facts (indicating an invalid or unexpected state representation):
              - Return `float('inf')` as the state is likely unreachable or problematic.
    6. Return the total accumulated heuristic value `h`. The heuristic is 0 if and only if all packages with goals are at their goal locations.
    """
    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = {}
        self.locations = set()
        self.road_graph = collections.defaultdict(set)
        self.shortest_paths = {} # Dict of dicts: shortest_paths[l1][l2] = distance

        # 1. Extract package goals and collect locations
        # Goal facts are like '(at p1 l2)'
        for goal_fact in self.task.goals:
            parsed = self._parse_fact(goal_fact)
            if parsed and parsed[0] == 'at' and len(parsed) == 3:
                package, location = parsed[1], parsed[2]
                # Assuming each package has only one goal location
                self.package_goals[package] = location
                self.locations.add(location) # Add goal location to known locations

        # 2. Build road network graph and collect locations
        # Static facts include '(road l1 l2)'
        for static_fact in self.task.static:
            parsed = self._parse_fact(static_fact)
            if parsed and parsed[0] == 'road' and len(parsed) == 3:
                loc1, loc2 = parsed[1], parsed[2]
                self.road_graph[loc1].add(loc2)
                self.road_graph[loc2].add(loc1) # Assuming bidirectional roads based on example
                self.locations.add(loc1)
                self.locations.add(loc2)

        # 3. Add locations from initial state (packages/vehicles) to ensure they are included in shortest path calculation
        for fact in self.task.initial_state:
             parsed = self._parse_fact(fact)
             if parsed and parsed[0] == 'at' and len(parsed) == 3:
                 locatable, location = parsed[1], parsed[2]
                 self.locations.add(location)

        # 4. Compute all-pairs shortest paths
        self._compute_all_pairs_shortest_paths()

    def _parse_fact(self, fact_string):
        """Parses a PDDL fact string into a list of strings."""
        # Remove surrounding parentheses
        fact_content = fact_string[1:-1]

        # Find the first space to separate predicate from arguments
        first_space_index = fact_content.find(' ')

        if first_space_index == -1:
             # Fact with no arguments (unlikely in this domain's relevant facts)
             return [fact_content] if fact_content else []
        else:
             predicate = fact_content[:first_space_index]
             args_string = fact_content[first_space_index:].strip()
             # Split arguments by space
             args = args_string.split()
             return [predicate] + args


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        # Ensure we only compute for locations that are actually in the graph or mentioned
        # in initial/goal states.
        relevant_locations = list(self.locations)

        for start_node in relevant_locations:
            self.shortest_paths[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            self.shortest_paths[start_node][start_node] = 0 # Distance to self is 0

            while queue:
                current_node, distance = queue.popleft()

                # Check if current_node is in road_graph keys before iterating neighbors
                # An isolated location might be in self.locations but not road_graph keys
                if current_node in self.road_graph:
                    for neighbor in self.road_graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.shortest_paths[start_node][neighbor] = distance + 1
                            queue.append((neighbor, distance + 1))

        # Ensure all relevant locations have an entry in shortest_paths, even if isolated
        # and only reachable from themselves.
        for loc in relevant_locations:
             if loc not in self.shortest_paths:
                  self.shortest_paths[loc] = {loc: 0}
             elif loc not in self.shortest_paths[loc]:
                   self.shortest_paths[loc][loc] = 0


    def get_distance(self, loc1, loc2):
        """Returns the shortest path distance between two locations."""
        # Return infinity if locations are not known or loc2 is unreachable from loc1
        # Unreachable locations will not have an entry in self.shortest_paths[loc1][loc2]
        # The .get() method handles missing keys gracefully.
        return self.shortest_paths.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        (Docstring content moved to class docstring as requested)
        """
        state = node.state

        # Create quick lookups for the current state
        package_current_location = {}
        package_current_vehicle = {}
        vehicle_current_location = {}
        all_packages_in_state = set()
        all_vehicles_in_state = set() # Keep track of vehicles too

        # First pass: Identify packages based on 'in' facts and goal facts
        for fact_string in state:
             parsed = self._parse_fact(fact_string)
             if not parsed: continue
             predicate = parsed[0]

             if predicate == 'in' and len(parsed) == 3:
                  package, vehicle = parsed[1], parsed[2]
                  all_packages_in_state.add(package)
                  all_vehicles_in_state.add(vehicle) # Vehicles are also mentioned in 'in' facts
                  package_current_vehicle[package] = vehicle
             # Also add packages from goals that might be in the state initially at a location
             # This is handled implicitly in the second pass when checking 'at' facts.

        # Second pass: Process 'at' facts and populate locations, classifying objects
        for fact_string in state:
             parsed = self._parse_fact(fact_string)
             if not parsed: continue
             predicate = parsed[0]

             if predicate == 'at' and len(parsed) == 3:
                  obj, loc = parsed[1], parsed[2]
                  # If the object is in our list of packages (either from goals or 'in' facts)
                  if obj in self.package_goals or obj in all_packages_in_state:
                       package_current_location[obj] = loc
                  # Otherwise, assume it's a vehicle
                  else:
                       vehicle_current_location[obj] = loc
                       all_vehicles_in_state.add(obj) # Add vehicles found only via 'at'

        # Now package_current_location, package_current_vehicle, vehicle_current_location are populated.

        h = 0
        # Iterate through packages that have a goal
        for package, goal_location in self.package_goals.items():
            # Check if package is already at goal
            if package in package_current_location and package_current_location[package] == goal_location:
                continue # Package is at goal, cost is 0

            # Package is not at goal. Find its current status.
            current_location = None
            current_vehicle = None

            if package in package_current_location:
                # Package is at a location
                current_location = package_current_location[package]
                # Cost: pick-up (1) + drive + drop (1)
                drive_cost = self.get_distance(current_location, goal_location)
                if drive_cost == float('inf'):
                    # If the goal location is unreachable from the package's current location,
                    # the problem is likely unsolvable from this state regarding this package.
                    return float('inf')
                h += 1 + drive_cost + 1 # pick-up + drive + drop

            elif package in package_current_vehicle:
                # Package is in a vehicle
                current_vehicle = package_current_vehicle[package]
                if current_vehicle in vehicle_current_location:
                    vehicle_location = vehicle_current_location[current_vehicle]
                    if vehicle_location == goal_location:
                        # Package is in vehicle at goal location
                        # Cost: drop (1)
                        h += 1
                    else:
                        # Package is in vehicle, vehicle is not at goal location
                        # Cost: drive + drop (1)
                        drive_cost = self.get_distance(vehicle_location, goal_location)
                        if drive_cost == float('inf'):
                            # If the goal location is unreachable from the vehicle's current location,
                            # the problem is likely unsolvable from this state regarding this package.
                            return float('inf')
                        h += drive_cost + 1 # drive + drop
                else:
                    # Package is in a vehicle, but vehicle location is unknown in the state.
                    # This indicates an inconsistent state representation.
                    logging.warning(f"Vehicle {current_vehicle} containing package {package} has no location in state.")
                    return float('inf')
            else:
                # Package has a goal but is neither at a location nor in a vehicle in the state.
                # This indicates an inconsistent state representation.
                logging.warning(f"Package {package} has goal {goal_location} but is neither at a location ({package in package_current_location}) nor in a vehicle ({package in package_current_vehicle}) in state.")
                return float('inf')

        # The heuristic value is the sum of estimated costs for all packages
        # that are not yet at their goal location.
        # If h is 0, it means all packages in self.package_goals are at their goals.
        # Assuming the task goal consists *only* of these package location facts,
        # h=0 implies the goal is reached. If the task goal includes other facts,
        # h=0 might occur in a non-goal state, which is acceptable for a non-admissible
        # heuristic guiding greedy search, as the search will verify the goal state.

        return h
