import logging
from collections import deque
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available in the execution environment

# Helper function to parse PDDL facts
def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple."""
    if not fact_str or not fact_str.startswith('(') or not fact_str.endswith(')'):
        logging.debug(f"Malformed fact string: {fact_str}")
        return None
    content = fact_str[1:-1].strip()
    if not content:
        logging.debug(f"Empty fact content: {fact_str}")
        return None
    parts = content.split()
    return tuple(parts)

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing the
    estimated minimum number of actions required for each package that is not
    yet at its goal location. It considers the actions needed to move the package
    from its current location (or the location of the vehicle carrying it) to
    its goal location.

    Assumptions:
    - The road network defined by '(road l1 l2)' facts is bidirectional.
    - Package goal facts are exclusively of the form '(at package_name location_name)'.
    - Package names start with 'p' and vehicle names start with 'v' for simple identification.
    - State representation is consistent: a package is either '(at pX lY)' or '(in pX vY)',
      and a vehicle carrying a package '(in pX vY)' is also '(at vY lZ)' for some location lZ.
    - Action costs are uniform (1).
    - Vehicle capacity constraints are ignored for simplicity and efficiency.

    Heuristic Initialization:
    During initialization, the heuristic performs the following steps:
    1. It extracts all '(road l1 l2)' facts from the static information to build
       an undirected graph representing the road network.
    2. It identifies all unique locations mentioned in the road facts, initial state
       '(at ...)' facts, and goal '(at ...)' facts.
    3. It computes the shortest path distance (minimum number of drive actions)
       between all pairs of identified locations using Breadth-First Search (BFS).
       These distances are stored for quick lookup during heuristic computation.
    4. It extracts the target location for each package from the goal facts, storing
       them in a dictionary mapping package names to goal location names.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (a set of facts), the heuristic is computed as follows:
    1. Initialize the total heuristic value to 0.
    2. Identify the current status (location or inside a vehicle) of every package
       and the current location of every vehicle by iterating through the state facts.
       Store these in dictionaries for easy lookup.
    3. For each package that has a specific goal location defined in the task:
        a. Check if the package is already at its goal location (i.e., if the fact
           '(at package_name goal_location_name)' is present in the state). If it is,
           this package requires 0 additional actions and is skipped.
        b. If the package is not at its goal, determine its current status using the
           information gathered in step 2:
           - If the package is '(at package_name current_location)':
             Estimate the cost for this package as 1 (for the pick-up action) +
             the shortest distance (number of drive actions) from `current_location`
             to `goal_location` + 1 (for the drop action). This assumes a vehicle
             will become available at `current_location` and capacity is sufficient.
           - If the package is '(in package_name vehicle_name)':
             Find the current location of `vehicle_name` using the information
             gathered in step 2. Estimate the cost for this package as the shortest
             distance (number of drive actions) from the vehicle's location to
             `goal_location` + 1 (for the drop action). This assumes the vehicle
             will drive directly to `goal_location`.
           - If the package's status cannot be determined (e.g., not found in 'at'
             or 'in' facts, or the vehicle carrying it has no location), or if the
             required drive distance is infinite (locations are disconnected based
             on precomputed distances), the goal is considered unreachable from this
             state, and the heuristic computation stops, returning infinity.
        c. Add the estimated cost for this package to the total heuristic value.
    4. After processing all packages with goals, the total accumulated value is
       returned as the heuristic estimate. If any package goal was deemed unreachable,
       infinity is returned. The heuristic is 0 if and only if all package goals
       are already satisfied in the state.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task_goals = task.goals # Store the set of goal facts

        road_graph = {}
        locations = set()

        # Extract road network and locations from static facts
        for fact_str in task.static:
            parsed = parse_fact(fact_str)
            if parsed and parsed[0] == 'road':
                _, l1, l2 = parsed
                locations.add(l1)
                locations.add(l2)
                if l1 not in road_graph:
                    road_graph[l1] = []
                if l2 not in road_graph:
                    road_graph[l2] = []
                road_graph[l1].append(l2)
                road_graph[l2].append(l1) # Assuming bidirectional roads

        # Collect locations from initial state and goals
        for fact_str in task.initial_state | task.goals:
             parsed = parse_fact(fact_str)
             if parsed and parsed[0] == 'at':
                 _, obj, loc = parsed
                 locations.add(loc)

        # Compute all-pairs shortest paths
        self.distances = self._bfs_distances(road_graph, list(locations))

        # Extract package goals from goal facts
        self.package_goals = {} # { package_name: goal_location_name }
        for goal_fact_str in self.task_goals:
            parsed = parse_fact(goal_fact_str)
            if parsed and parsed[0] == 'at':
                _, obj, goal_loc = parsed
                # Assuming goals are only about packages being at locations
                # and package names start with 'p'
                if obj.startswith('p'):
                     self.package_goals[obj] = goal_loc
                # Note: Assumes each package has at most one (at pX lY) goal fact.


    @staticmethod
    def _bfs_distances(graph, locations):
        """Computes all-pairs shortest path distances using BFS."""
        distances = {loc: {other_loc: float('inf') for other_loc in locations} for loc in locations}

        for start_loc in locations:
            distances[start_loc][start_loc] = 0
            queue = deque([start_loc])
            visited = {start_loc}

            while queue:
                current_loc = queue.popleft()
                current_dist = distances[start_loc][current_loc]

                if current_loc in graph:
                    for neighbor in graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            distances[start_loc][neighbor] = current_dist + 1
                            queue.append(neighbor)

        # Ensure all locations are in the distances dict, even if isolated
        # And ensure distance to self is 0
        for loc in locations:
             if loc not in distances:
                 distances[loc] = {other_loc: float('inf') for other_loc in locations}
             distances[loc][loc] = 0 # Redundant if start_loc loop covers all locations, but safe.

        return distances


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.
        """
        state = node.state
        h_value = 0
        unreachable = False

        # Find current location/status of each package and vehicle locations
        package_status = {} # { package_name: ('at', location_name) or ('in', vehicle_name) }
        vehicle_locations = {} # { vehicle_name: location_name }

        for fact_str in state:
            parsed = parse_fact(fact_str)
            if parsed and parsed[0] == 'at':
                _, obj, loc = parsed
                # Simple check for package/vehicle based on name prefix
                if obj.startswith('p'):
                     package_status[obj] = ('at', loc)
                elif obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif parsed and parsed[0] == 'in':
                 _, package, vehicle = parsed
                 # Basic validation: check if they look like package/vehicle names
                 if package.startswith('p') and vehicle.startswith('v'):
                     package_status[package] = ('in', vehicle)
                 else:
                     logging.debug(f"Skipping 'in' fact with unexpected object names: {fact_str}")


        # Calculate heuristic for each package with a defined goal
        for package, goal_loc in self.package_goals.items():
            # Check if the package is already at its goal location
            goal_fact_str = f'(at {package} {goal_loc})'
            if goal_fact_str in state:
                continue # Package is already at goal, contributes 0

            # Package is not at goal. Find its current status.
            status = package_status.get(package)

            if status is None:
                # Package is not 'at' any location and not 'in' any vehicle.
                # This indicates an issue with the state or problem definition
                # where a package exists in the goal but not in the state.
                # Treat as unreachable.
                logging.warning(f"Package {package} from goals not found in state.")
                unreachable = True
                break

            status_type, current_info = status

            if status_type == 'at':
                current_loc = current_info
                # Package is (at package current_loc). Needs pickup, drive, drop.
                # Cost = 1 (pickup) + distance(current_loc, goal_loc) + 1 (drop)
                # Need to handle cases where current_loc or goal_loc might not be in distances map
                # (e.g., isolated locations not connected by roads, but present in 'at' facts)
                drive_cost = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if drive_cost == float('inf'):
                     unreachable = True
                     break
                h_value += 1 + drive_cost + 1
            elif status_type == 'in':
                vehicle_carrying = current_info
                # Package is (in package vehicle_carrying). Needs drive (by vehicle), drop.
                # Cost = distance(vehicle_loc, goal_loc) + 1 (drop)
                vehicle_loc = vehicle_locations.get(vehicle_carrying)
                if vehicle_loc is None:
                     # Vehicle carrying package is not at any location? Invalid state?
                     logging.warning(f"Vehicle {vehicle_carrying} carrying {package} has no location in state.")
                     unreachable = True
                     break
                # Need to handle cases where vehicle_loc or goal_loc might not be in distances map
                drive_cost = self.distances.get(vehicle_loc, {}).get(goal_loc, float('inf'))

                if drive_cost == float('inf'):
                     unreachable = True
                     break
                h_value += drive_cost + 1

        # Return infinity if any part was unreachable, otherwise return the sum
        return float('inf') if unreachable else h_value
