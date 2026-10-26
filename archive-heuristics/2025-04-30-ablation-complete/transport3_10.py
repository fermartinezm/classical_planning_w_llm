from collections import deque
from fnmatch import fnmatch

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is treated as a string and handle potential whitespace
    fact_str = str(fact).strip()
    if not fact_str.startswith('(') or not fact_str.endswith(')'):
         # Handle unexpected format - return empty list or raise error
         # Returning empty list allows calling code to check and skip invalid facts
         return []
    return fact_str[1:-1].split()


# Assuming Heuristic base class is defined elsewhere and available in the environment.
# If explicit inheritance is required, uncomment the line below and ensure Heuristic is imported.
# from heuristics.heuristic_base import Heuristic

class transportHeuristic:
# class transportHeuristic(Heuristic): # Uncomment this line if explicit inheritance from Heuristic is needed
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It sums the estimated cost for each
    misplaced package independently, assuming shortest path driving and
    necessary pick-up/drop actions. It ignores vehicle capacity constraints
    and the possibility of co-transporting multiple packages.

    # Assumptions
    - The road network is static and provides connections between locations.
    - Packages need to be picked up by a vehicle, driven to the destination,
      and dropped.
    - Vehicle capacity is not explicitly modeled in the heuristic calculation
      for simplicity and efficiency.
    - The cost of each action (drive, pick-up, drop) is 1.
    - All locations mentioned in road facts or goal facts are part of the
      connected road network or are intended destinations/origins.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task's goal conditions.
    - Builds a graph representation of the road network from static facts.
    - Identifies all relevant locations from road facts and goal locations.
    - Computes all-pairs shortest path distances between all identified locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package
       and the location for every vehicle.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a specified goal location:
       a. Determine the package's current status (on ground at location, or in vehicle).
       b. Determine the package's *actual* current location (either its ground location or its vehicle's location).
       c. Check if the package's actual current location is the same as its goal location. If yes, add 0 cost for this package and continue to the next package.
       d. If the package is currently on the ground at a location `current_loc`
          different from its goal location `goal_loc`:
          - Estimate the cost as 1 (pick-up) + shortest_distance(`current_loc`, `goal_loc`) (drive) + 1 (drop).
          - If `goal_loc` is unreachable from `current_loc`, the cost is infinity.
          - Add this cost to the total.
       e. If the package is currently inside a vehicle `v`:
          - Find the current location `loc_v` of vehicle `v`.
          - Estimate the cost as shortest_distance(`loc_v`, `goal_loc`) (drive) + 1 (drop).
          - If `goal_loc` is unreachable from `loc_v`, the cost is infinity.
          - Add this cost to the total.
       f. If at any point the cost for a package is infinity, the total heuristic
          cost for the state is infinity.
    4. Return the total accumulated cost (or infinity if any package's goal is unreachable).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph to compute shortest paths.
        """
        # Assuming task object has attributes goals and static
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        all_locations_set = set()
        for goal in self.goals:
            # Goal facts are typically (at package location)
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location
                all_locations_set.add(location)


        # Build the road network graph and collect all locations
        self.road_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                all_locations_set.add(l1)
                all_locations_set.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                # Assuming roads are bidirectional based on example
                self.road_graph.setdefault(l2, set()).add(l1)

        self.locations = list(all_locations_set) # Use all relevant locations for BFS

        # Compute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances from every location to every other
        location using BFS.
        Returns a dictionary distances[start_loc][end_loc] = distance.
        Unreachable locations will have distance float('inf').
        """
        distances = {}
        for start_node in self.locations:
            distances[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            distances[start_node][start_node] = 0 # Distance from self is 0

            while q:
                current_loc, dist = q.popleft()

                # Ensure current_loc is in the graph keys before iterating neighbors
                # Although all locations are added as keys, this is safer.
                for neighbor in self.road_graph.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[start_node][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

            # Mark locations not reached by BFS from start_node as unreachable
            for end_node in self.locations:
                 if end_node not in distances[start_node]:
                     distances[start_node][end_node] = float('inf')

        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Track where locatable objects (packages and vehicles) are.
        # Map object -> location or vehicle it's inside
        current_status = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                current_status[obj] = location
            elif predicate == "in" and len(parts) == 3:
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle # Package is inside vehicle

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            package_status = current_status.get(package)

            # If package is not found in the state facts, it's an issue.
            # This heuristic assumes all packages in goals are in the state.
            if package_status is None:
                 # Cannot determine package location, assume goal is unreachable
                 return float('inf')

            # Determine the package's actual current location
            actual_current_location = None
            if package_status in self.locations: # Package is on the ground
                 actual_current_location = package_status
            elif package_status in current_status: # Package_status is a vehicle name
                 vehicle = package_status
                 actual_current_location = current_status.get(vehicle) # Get vehicle's location
                 # If vehicle_location is None, the vehicle isn't 'at' anywhere, malformed state.
                 if actual_current_location is None:
                      return float('inf')
            # else: package_status is neither a location nor a known vehicle. Malformed state.
            # This case is implicitly handled if actual_current_location remains None.

            if actual_current_location is None:
                 # Malformed state where package_status was not a location or known vehicle
                 return float('inf')

            # Check if the package is already at its goal location
            if actual_current_location == goal_location:
                 continue # Package is already at its goal, cost is 0 for this package

            # Package is not at its goal. Calculate cost.
            if package_status in self.locations: # Package was on the ground at actual_current_location
                current_loc = actual_current_location # Same as package_status
                drive_cost = self.distances.get(current_loc, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from current location
                    return float('inf')

                # Cost: pick-up (1) + drive (distance) + drop (1)
                cost_for_package = 1 + drive_cost + 1
                total_cost += cost_for_package

            elif package_status in current_status: # Package was inside a vehicle at actual_current_location
                # actual_current_location is the vehicle's location
                loc_v = actual_current_location
                drive_cost = self.distances.get(loc_v, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from vehicle's current location
                    return float('inf')

                # Cost: drive (distance) + drop (1)
                cost_for_package = drive_cost + 1
                total_cost += cost_for_package
            # No need for else here, as malformed states were handled above.


        return total_cost
