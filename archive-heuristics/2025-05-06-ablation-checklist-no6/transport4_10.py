from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (strings, '*' allowed as wildcard).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(arg == '*' or part == arg for part, arg in zip(parts, args))


# class transportHeuristic(Heuristic): # Uncomment this line in the actual environment
class transportHeuristic: # Using this for standalone code block
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move each misplaced package
    to its goal location, assuming each package is transported independently
    using the shortest path for driving. Capacity and vehicle availability
    are not fully modeled, providing a relaxed estimate.

    Heuristic = Sum over all misplaced packages:
        If package is on ground at L_current: 1 (pick-up) + shortest_path(L_current, L_goal) + 1 (drop)
        If package is in vehicle V at L_v: shortest_path(L_v, L_goal) + 1 (drop)
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the road network graph to precompute shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find vehicles

        # 1. Build the road network graph and collect all locations
        self.road_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                _, l1, l2 = get_parts(fact)
                locations.add(l1)
                locations.add(l2)
                self.road_graph.setdefault(l1, []).append(l2)
                self.road_graph.setdefault(l2, []).append(l1) # Assuming roads are bidirectional

        self.locations = list(locations) # Store list of locations

        # 2. Precompute all-pairs shortest paths using BFS
        self.shortest_distances = {}
        for start_loc in self.locations:
            self.shortest_distances[start_loc] = self._bfs(start_loc)

        # 3. Extract goal locations for each package and identify package names
        self.goal_locations = {}
        package_names = set()
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                _, package, location = get_parts(goal)
                self.goal_locations[package] = location
                package_names.add(package)

        # 4. Identify vehicle names
        self.vehicle_names = set()
        # Look in initial state and static facts for predicates involving vehicles
        # Vehicles appear as the second argument of 'in' or the first argument of 'capacity'
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts[0] == "capacity" and len(parts) == 3:
                 _, vehicle, _ = parts
                 self.vehicle_names.add(vehicle)
             elif parts[0] == "in" and len(parts) == 3:
                 _, package, vehicle = parts
                 # The domain definition says (in ?x - package ?v - vehicle)
                 # So the second argument of 'in' is always a vehicle.
                 self.vehicle_names.add(vehicle)

        # Ensure packages are not mistakenly identified as vehicles (e.g., if a package name starts with 'v')
        self.vehicle_names -= package_names


    def _bfs(self, start_node):
        """
        Perform Breadth-First Search from a start node to find shortest
        distances to all other nodes in the road graph.
        """
        distances = {node: float('inf') for node in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Check if current_node exists in the graph keys before accessing neighbors
            # This handles cases where a location might be mentioned but has no roads connected
            if current_node in self.road_graph:
                for neighbor in self.road_graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state
        total_cost = 0

        # Map current locations of all locatables (packages and vehicles)
        current_locatables_state = {} # object -> location or vehicle
        vehicle_locations = {} # vehicle -> location

        # Populate current locations and vehicle locations from the state
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                _, obj, loc = parts
                current_locatables_state[obj] = loc
                if obj in self.vehicle_names:
                     vehicle_locations[obj] = loc
            elif parts[0] == "in":
                _, package, vehicle = parts
                current_locatables_state[package] = vehicle # Package is inside vehicle

        # Iterate through packages and their goals
        for package, goal_location in self.goal_locations.items():
            # Check if the package is already at its goal location on the ground
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at goal, cost is 0 for this package

            # Package is not at goal. Find its current status.
            current_status = current_locatables_state.get(package)

            # Handle cases where package status isn't found (shouldn't happen in valid states)
            if current_status is None:
                 # If a goal package isn't 'at' or 'in' anywhere, it's likely an invalid state
                 # or indicates unsolvability from this point. Return infinity.
                 return float('inf')

            if current_status in self.locations: # Package is on the ground at current_status (a location)
                l_current = current_status
                # Cost: pick-up + drive + drop
                # Need to find shortest path from l_current to goal_location
                # Use .get() with default float('inf') to handle cases where goal_location
                # is not reachable from l_current (e.g., disconnected graph components)
                drive_cost = self.shortest_distances.get(l_current, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                    # Goal is unreachable from this location
                    return float('inf')

                total_cost += 1 + drive_cost + 1 # 1 for pick-up, drive_cost for driving, 1 for drop

            elif current_status in self.vehicle_names: # Package is inside a vehicle (current_status is the vehicle object name)
                vehicle = current_status
                # Find the location of the vehicle
                l_v = vehicle_locations.get(vehicle)

                if l_v is None:
                    # Vehicle location not found? Vehicle exists but isn't 'at' anywhere.
                    # Likely an invalid state or unreachable. Return infinity.
                    return float('inf')

                # Cost: drive + drop
                # Need to find shortest path from vehicle's location l_v to goal_location
                drive_cost = self.shortest_distances.get(l_v, {}).get(goal_location, float('inf'))

                if drive_cost == float('inf'):
                     return float('inf') # Goal is unreachable from vehicle's location

                total_cost += drive_cost + 1 # drive_cost for driving, 1 for drop
            else:
                 # current_status is neither a location nor a known vehicle? Invalid state?
                 # Return infinity.
                 return float('inf')


        return total_cost
