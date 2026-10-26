import collections
import logging

from heuristics.heuristic_base import Heuristic
from task import Task


# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """
    Parses a PDDL fact string into a tuple of strings.
    e.g., '(at p1 l1)' -> ('at', 'p1', 'l1')
    """
    # Remove outer parentheses and split by space
    # This assumes simple object names without spaces or internal parentheses
    content = fact_str[1:-1]
    parts = content.split()
    return tuple(parts)


class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum costs
        required to move each package not at its goal location to its goal
        location. The cost for a package depends on whether it is currently
        on the ground or in a vehicle. It includes costs for vehicle travel
        to the package (if on ground), pickup, driving the package to the
        goal location, and dropping the package. Shortest path distances
        between locations are precomputed using BFS. Capacity constraints
        are ignored.

    Assumptions:
        - Roads are bidirectional (handled by adding edges in both directions).
        - All locations mentioned in road facts are part of the connected graph.
        - All packages that need to be moved have a goal location specified
          by an '(at package location)' fact in the task goals.
        - Objects identified as packages are those appearing as the first
          argument of an '(at ...)' goal fact. Objects identified as vehicles
          are those appearing as the first argument of an '(at ...)' initial
          state fact that are not packages, or as the second argument of an
          '(in ...)' initial state fact.
        - The heuristic does not need to be admissible.
        - Action costs are 1.

    Heuristic Initialization:
        1. Parses all '(road l1 l2)' facts from the static information to build
           an undirected graph of locations. Collects all unique locations
           mentioned in road facts, initial state 'at' facts, and goal 'at' facts.
        2. Computes all-pairs shortest paths between all locations in the graph
           using BFS. Stores these distances in a dictionary `self.shortest_paths[loc1][loc2]`.
        3. Parses goal facts to identify which objects are packages requiring
           delivery and their respective goal locations. Stores these in
           `self.package_goals`.
        4. Identifies vehicle objects from the initial state facts. Stores
           these in `self.vehicles`.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic value `h` to 0.
        2. Parse the current state to determine:
           - The current location of each package (either on the ground or in a vehicle).
           - The current vehicle carrying a package, if applicable.
           - The current location of each vehicle present in the state.
        3. Get the list of vehicles currently located in the state.
        4. For each package that has a goal location defined in `self.package_goals`:
           a. Get the package's goal location (`loc_p_goal`).
           b. Determine the package's current status: its location (`loc_p_current`)
              and whether it is inside a vehicle (`is_in_vehicle`). Handle cases
              where the package or its carrying vehicle might not have a location
              listed in the state (though this indicates an issue).
           c. If the package is currently on the ground at its goal location
              (`loc_p_current == loc_p_goal` and `not is_in_vehicle`), it is
              considered delivered and contributes 0 to the heuristic. Continue
              to the next package.
           d. If the package is not delivered:
              i. Calculate the estimated minimum cost (`cost_p`) to move this
                 package to its goal:
                 - If the package is currently in a vehicle at location `loc_v_current`
                   (which is `loc_p_current`): The cost is the shortest distance
                   from `loc_v_current` to `loc_p_goal` (for the drive action),
                   plus 1 for the 'drop' action. Cost = `dist(loc_v_current, loc_p_goal) + 1`.
                 - If the package is currently on the ground at location `loc_p_current`:
                   A vehicle must first reach `loc_p_current`, pick up the package,
                   drive to `loc_p_goal`, and drop it. The cost is the minimum
                   shortest distance from any vehicle's current location (`loc_v_current`)
                   to `loc_p_current`, plus 1 for 'pick-up', plus the shortest
                   distance from `loc_p_current` to `loc_p_goal` (for the drive),
                   plus 1 for 'drop'. Cost = `min_v (dist(loc_v_current, loc_p_current)) + 1 + dist(loc_p_current, loc_p_goal) + 1`.
                   If no vehicles are in the state or reachable, the minimum
                   vehicle travel cost is infinity.
              ii. If `cost_p` is infinity (meaning a required location is unreachable
                  or no vehicles are available for a package on the ground), the
                  overall goal is likely unreachable. Set the total heuristic `h`
                  to infinity and break the loop (no need to sum further).
              iii. Add `cost_p` to the total heuristic value `h`.
        5. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = {}
        self.locations = set()
        self.graph = collections.defaultdict(list)
        self.shortest_paths = {}
        self.vehicles = set()

        # 1. Parse road facts and build graph, collect locations
        for fact_str in task.static:
            parts = parse_fact(fact_str)
            if parts[0] == 'road' and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.graph[l1].append(l2)
                self.graph[l2].append(l1) # Assuming bidirectional roads
                self.locations.add(l1)
                self.locations.add(l2)

        # Add any locations mentioned in initial state or goals but not roads
        # This ensures BFS is attempted from/to all relevant locations,
        # even if they are isolated (resulting in infinite distances).
        for fact_str in task.initial_state:
             parts = parse_fact(fact_str)
             # Check if the last part looks like a location (e.g., from (at obj loc))
             if len(parts) > 2 and parts[0] == 'at':
                 self.locations.add(parts[2])
        for fact_str in task.goals:
             parts = parse_fact(fact_str)
             # Check if the last part looks like a location (e.g., from (at obj loc))
             if len(parts) > 2 and parts[0] == 'at':
                 self.locations.add(parts[2])


        # 2. Compute all-pairs shortest paths using BFS
        for loc in self.locations:
            self.shortest_paths[loc] = self._bfs(loc)

        # 3. Parse goal facts to find package goals
        packages_with_goals = set()
        for goal_fact_str in task.goals:
            parts = parse_fact(goal_fact_str)
            if parts[0] == 'at' and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
                packages_with_goals.add(package)

        # 4. Identify vehicle objects from initial state
        for fact_str in task.initial_state:
            parts = parse_fact(fact_str)
            if parts[0] == 'at' and len(parts) == 3:
                obj = parts[1]
                # If it's not a package with a goal, assume it's a vehicle
                if obj not in packages_with_goals:
                    self.vehicles.add(obj)
            elif parts[0] == 'in' and len(parts) == 3:
                vehicle = parts[2]
                self.vehicles.add(vehicle)


    def _bfs(self, start_node):
        """Helper to compute shortest paths from a start_node using BFS."""
        distances = {loc: float('inf') for loc in self.locations}
        if start_node not in self.locations:
             # Start node is not in the set of known locations.
             # Distances from it will remain infinity.
             return distances

        # If start_node is in locations but not in graph keys, it's an isolated node.
        # BFS from it will only find itself with distance 0. This is correct.
        if start_node in self.graph:
            distances[start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                current_node = queue.popleft()
                current_dist = distances[current_node]

                # Check if current_node has neighbors in the graph
                if current_node in self.graph:
                    for neighbor in self.graph[current_node]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
        # If start_node is in locations but not in graph, distances remain inf except self=0 (handled above)
        # If start_node is not in locations, distances remain inf (handled above)
        return distances


    def get_distance(self, loc1, loc2):
        """Helper to retrieve precomputed shortest distance."""
        # Use .get() with a default empty dict to avoid KeyError if loc1 wasn't
        # a start node for BFS (e.g., if it wasn't in self.locations initially,
        # though the init logic tries to prevent this for relevant locations).
        # If loc2 is not in the inner dict, .get() returns default (None),
        # which we then map to float('inf').
        distance = self.shortest_paths.get(loc1, {}).get(loc2, float('inf'))
        if distance == float('inf'):
             logging.debug(f"Distance lookup failed for {loc1} -> {loc2}. Likely disconnected.")
        return distance


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 2. Parse current state
        package_current_loc = {} # package -> location (if on ground)
        package_in_vehicle = {}  # package -> vehicle (if in vehicle)
        vehicle_current_loc = {} # vehicle -> location

        for fact_str in state:
            parts = parse_fact(fact_str)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.package_goals: # It's a package we care about
                    package_current_loc[obj] = loc
                elif obj in self.vehicles: # It's a vehicle
                    vehicle_current_loc[obj] = loc
                # Ignore other 'at' facts if any
            elif parts[0] == 'in' and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                if package in self.package_goals: # It's a package we care about
                     package_in_vehicle[package] = vehicle
                # Ignore other 'in' facts if any
            # Ignore other predicates like 'capacity'

        # 3. Calculate heuristic
        h = 0
        vehicles_in_state = list(vehicle_current_loc.keys())

        for package, loc_p_goal in self.package_goals.items():
            loc_p_current = None
            is_in_vehicle = False
            # current_vehicle = None # Not strictly needed here

            if package in package_current_loc:
                loc_p_current = package_current_loc[package]
            elif package in package_in_vehicle:
                is_in_vehicle = True
                current_vehicle = package_in_vehicle[package]
                loc_p_current = vehicle_current_loc.get(current_vehicle)
                if loc_p_current is None:
                    # Vehicle carrying package is not located in the state?
                    # This indicates an inconsistent state representation or issue.
                    # Treat as unreachable for this package.
                    logging.warning(f"Vehicle {current_vehicle} carrying package {package} has no location in state.")
                    h = float('inf')
                    break # Goal unreachable

            if loc_p_current is None:
                 # Package is not 'at' a location and not 'in' a vehicle?
                 # This indicates an inconsistent state representation or issue.
                 # Treat as unreachable for this package.
                 logging.warning(f"Package {package} is neither at a location nor in a vehicle.")
                 h = float('inf')
                 break # Goal unreachable


            # Check if package is already at goal (and on the ground)
            if loc_p_current == loc_p_goal and not is_in_vehicle:
                continue # Package is done

            # Package is not at goal or is in a vehicle at the goal location

            cost_p = 0
            if is_in_vehicle:
                # Package is in vehicle at loc_p_current
                # Needs drive + drop
                drive_cost = self.get_distance(loc_p_current, loc_p_goal)
                if drive_cost == float('inf'):
                    h = float('inf') # Goal unreachable for this package
                    break
                cost_p = drive_cost + 1 # drive + drop
            else:
                # Package is on the ground at loc_p_current
                # Needs vehicle travel to package + pickup + drive to goal + drop

                # Find min cost for a vehicle to reach package location
                min_vehicle_travel_cost = float('inf')
                if not vehicles_in_state:
                    # No vehicles available in the current state to pick up package
                    min_vehicle_travel_cost = float('inf')
                else:
                    for v in vehicles_in_state:
                        loc_v_current = vehicle_current_loc.get(v)
                        if loc_v_current is not None:
                             travel_cost = self.get_distance(loc_v_current, loc_p_current)
                             min_vehicle_travel_cost = min(min_vehicle_travel_cost, travel_cost)
                        # else: Vehicle exists but has no location? Inconsistent state.
                        #      logging.warning(f"Vehicle {v} has no location in state.")


                if min_vehicle_travel_cost == float('inf'):
                     # No vehicle can reach the package
                     h = float('inf')
                     break

                # Cost to drive package from its current location to goal
                drive_to_goal_cost = self.get_distance(loc_p_current, loc_p_goal)
                if drive_to_goal_cost == float('inf'):
                     # Goal location unreachable from package location
                     h = float('inf')
                     break

                cost_p = min_vehicle_travel_cost + 1 + drive_to_goal_cost + 1 # vehicle_travel + pickup + drive_to_goal + drop

            h += cost_p

        return h
