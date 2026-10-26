from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys # Needed for float('inf')

# Define helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at package1 location1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    to its goal location, summing the individual estimates. It considers pick-up,
    drop, and driving actions, using shortest path distances on the road network
    as an estimate for driving costs.

    # Assumptions
    - The cost of each action (drive, pick-up, drop) is 1.
    - Vehicle capacity and availability are relaxed: it assumes a vehicle is
      available when needed to pick up a package on the ground, and ignores
      competition for vehicles or capacity limits during the plan execution
      (except implicitly by requiring a pick-up action).
    - The road network is undirected (if road A-B exists, road B-A exists).
      The PDDL shows `(road l1 l2)` and `(road l2 l1)` for connected locations,
      confirming this.
    - Objects appearing in goal conditions are packages. Any other object
      appearing in an `(at obj loc)` fact or as the second argument of an
      `(in pkg veh)` fact is considered a vehicle.

    # Heuristic Initialization
    - Parses goal conditions to map packages to their target locations.
    - Parses static facts to build the road network graph and identify all locations.
    - Computes all-pairs shortest paths on the road network using BFS.
    - Parses capacity hierarchy to understand size relationships (though this
      is not strictly used in the current simple version of the heuristic,
      it's good practice to extract static info).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location or containing vehicle for every package
       that has a goal location.
    2. Identify the current location for every vehicle present in the state.
    3. Initialize total heuristic cost to 0.
    4. For each package `p` whose goal is `(at p goal_loc)`:
        a. Check if `(at p goal_loc)` is true in the current state. If yes, the package is at its goal; add 0 to cost.
        b. If `(at p current_loc)` is true where `current_loc != goal_loc`:
            - The package needs to be picked up (1 action).
            - A vehicle needs to drive from `current_loc` to `goal_loc`. Estimate this cost as the shortest path distance between `current_loc` and `goal_loc` on the road network.
            - The package needs to be dropped (1 action).
            - Add 1 (pick-up) + shortest_path(current_loc, goal_loc) + 1 (drop) to the total cost.
            - This step assumes a vehicle is available at `current_loc` with capacity.
        c. If `(in p v)` is true (package is inside vehicle `v`):
            - Find the current location `vehicle_loc` of vehicle `v`.
            - If `vehicle_loc == goal_loc`: The package needs to be dropped (1 action). Add 1 (drop) to the total cost.
            - If `vehicle_loc != goal_loc`: Vehicle `v` needs to drive from `vehicle_loc` to `goal_loc`. Estimate this cost as the shortest path distance between `vehicle_loc` and `goal_loc`. Then the package needs to be dropped (1 action). Add shortest_path(vehicle_loc, goal_loc) + 1 (drop) to the total cost.
        d. If the package's location/status is not found in the state, or if a required vehicle location is unknown, the goal might be unreachable from this state; return infinity.
    5. Return the total calculated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the road network graph and collect all location names.
        self.road_graph = {}
        all_locations_set = set() # Collect all location names
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)
                self.road_graph.setdefault(loc1, []).append(loc2)
                # Assuming roads are bidirectional, add the reverse edge
                self.road_graph.setdefault(loc2, []).append(loc1)

        self.all_locations = frozenset(all_locations_set) # Store as frozenset

        # Compute all-pairs shortest paths.
        self.shortest_paths = {}
        # Use the collected set of locations for BFS starts
        for start_loc in self.all_locations:
            distances = self._bfs(start_loc)
            for end_loc, dist in distances.items():
                self.shortest_paths[(start_loc, end_loc)] = dist

        # Parse capacity hierarchy (optional for this simple heuristic, but good practice)
        # This part is included for completeness but not used in the current __call__ logic.
        self.capacity_predecessors = {}
        self.min_capacity = None
        capacity_levels = set()
        successor_levels = set()
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                s1, s2 = get_parts(fact)[1:]
                self.capacity_predecessors[s1] = s2
                capacity_levels.add(s1)
                capacity_levels.add(s2)
                successor_levels.add(s2)

        # Find the minimum capacity level (the one that is not a successor)
        possible_min_capacities = capacity_levels - successor_levels
        if len(possible_min_capacities) == 1:
             self.min_capacity = possible_min_capacities.pop()
        # Note: This heuristic doesn't currently use capacity info beyond parsing it.


    def _bfs(self, start_node):
        """Perform BFS from a start node to find shortest paths to all reachable nodes."""
        distances = {start_node: 0}
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current = queue.popleft()

            # Handle locations that might not have any roads connected (though unlikely for locations in road facts)
            # Ensure current node is in the graph before trying to access its neighbors
            if current not in self.road_graph:
                 continue # Should not happen if start_node is from self.all_locations

            for neighbor in self.road_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Track where packages and vehicles are currently located.
        package_locations = {} # package -> location or vehicle name
        vehicle_locations = {} # vehicle name -> location
        vehicles = set() # Set of vehicle names found in the state

        # First pass to identify vehicles and their locations, and package locations
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            if predicate == "at":
                obj, loc = parts[1:]
                # If obj is a package with a goal, store its location
                if obj in self.goal_locations:
                    package_locations[obj] = loc
                else:
                    # Assume any other object 'at' a location is a vehicle
                    vehicles.add(obj)
                    vehicle_locations[obj] = loc
            elif predicate == "in":
                 package, vehicle = parts[1:]
                 # If package has a goal, store that it's in this vehicle
                 if package in self.goal_locations:
                     package_locations[package] = vehicle # Store vehicle name
                 vehicles.add(vehicle) # Add vehicle to set (even if its location isn't in this fact)

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            current_status = package_locations.get(package)

            if current_status is None:
                 # Package with a goal is not found in the state (not at a location, not in a vehicle)
                 # This indicates an invalid state or an unreachable goal.
                 # Return infinity to prune this path.
                 return float('inf')

            # Check if current_status is a location string.
            # A status is a location if it is in the set of all known locations.
            is_location = current_status in self.all_locations

            if is_location:
                current_location = current_status
                if current_location == goal_location:
                    cost_for_package = 0
                else:
                    # Package needs pick-up, drive, drop
                    drive_cost = self.shortest_paths.get((current_location, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                         # Goal location is unreachable from the package's current location
                         return float('inf')
                    cost_for_package = 1 + drive_cost + 1 # pick-up + drive + drop
            else: # current_status is a vehicle name
                vehicle = current_status
                vehicle_location = vehicle_locations.get(vehicle)
                if vehicle_location is None:
                    # Vehicle location unknown? Should not happen in valid states generated by planner.
                    # Return infinity to prune this path.
                    return float('inf')

                if vehicle_location == goal_location:
                    # Vehicle is at the goal location, just need to drop
                    cost_for_package = 1 # drop
                else:
                    # Vehicle needs to drive to the goal location, then drop
                    drive_cost = self.shortest_paths.get((vehicle_location, goal_location), float('inf'))
                    if drive_cost == float('inf'):
                         # Goal location is unreachable from the vehicle's current location
                         return float('inf')
                    cost_for_package = drive_cost + 1 # drive + drop

            total_cost += cost_for_package

        return total_cost
