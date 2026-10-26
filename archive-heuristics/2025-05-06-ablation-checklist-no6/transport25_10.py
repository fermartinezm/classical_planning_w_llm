# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL fact strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and multiple spaces
    return fact.strip()[1:-1].split()

# Inherit from Heuristic base class in actual usage
# class transportHeuristic(Heuristic):
class transportHeuristic:
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the number of actions needed to move each package to its goal location.
    Assumes unit cost for all actions (drive, pick-up, drop).
    Ignores vehicle availability and capacity constraints for simplicity and speed.
    Calculates shortest path distances between locations using BFS.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, identifying
        object types, and building the road network graph to compute shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        self.packages = set()
        self.vehicles = set()
        self.locations = set()
        self.road_graph = {}

        # 1. Identify object types (packages, vehicles, locations)
        all_locatables_in_init_at = set()

        for fact in initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                all_locatables_in_init_at.add(obj)
                self.locations.add(loc) # Locations from initial 'at' facts
            elif parts[0] == "in":
                pkg, veh = parts[1], parts[2]
                self.packages.add(pkg) # Packages from 'in' facts
                self.vehicles.add(veh) # Vehicles from 'in' facts

        # Any locatable in an 'at' fact that wasn't identified as a package
        # must be a vehicle (assuming complete initial state description).
        for obj in all_locatables_in_init_at:
            if obj not in self.packages:
                self.vehicles.add(obj)
            # Any locatable in an 'at' fact that wasn't identified as a vehicle
            # must be a package. This covers packages not initially in vehicles.
            if obj not in self.vehicles:
                 self.packages.add(obj)


        # 2. Build road graph and collect locations from road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1) # Locations from road facts
                self.locations.add(loc2)
                self.road_graph.setdefault(loc1, []).append(loc2)
                self.road_graph.setdefault(loc2, []).append(loc1) # Roads are bidirectional

        # Ensure all identified locations are keys in the graph, even if they have no roads
        for loc in self.locations:
             self.road_graph.setdefault(loc, [])

        # 3. Extract goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                package, location = parts[1], parts[2]
                # Only consider goals for identified packages
                if package in self.packages:
                    self.goal_locations[package] = location
                # Ignore goals for vehicles or other objects

        # 4. Compute all-pairs shortest paths using BFS
        self.shortest_paths = {}
        location_list = list(self.locations)
        for start_node in location_list:
            distances = self._bfs(start_node)
            for end_node, dist in distances.items():
                 self.shortest_paths[(start_node, end_node)] = dist

    def _bfs(self, start_node):
        """Performs BFS from a start node to find distances to all reachable nodes."""
        distances = {node: float('inf') for node in self.locations}
        # Handle case where start_node might not be in self.locations (shouldn't happen with current logic)
        if start_node not in distances:
             # This indicates an issue during location identification or graph building
             # For robustness, return empty distances or raise an error.
             # Returning empty distances means any path from this node is inf.
             return {}

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Get neighbors safely
            neighbors = self.road_graph.get(current_node, [])

            for neighbor in neighbors:
                # Ensure neighbor is a known location before accessing distances
                if neighbor in distances and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Track current locations of packages and vehicles
        package_current_state = {} # Maps package -> location or vehicle
        vehicle_locations = {}     # Maps vehicle -> location

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj in self.packages:
                    package_current_state[obj] = loc
                elif obj in self.vehicles:
                    vehicle_locations[obj] = loc
                # Ignore other 'at' facts if any
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                # Only process 'in' facts for known packages and vehicles
                if package in self.packages and vehicle in self.vehicles:
                     package_current_state[package] = vehicle
                # Ignore other 'in' facts if any

        total_cost = 0

        # Iterate through packages that have a goal location
        for package, goal_location in self.goal_locations.items():
            # If package is not mentioned in the current state, it's an invalid state or something unexpected.
            # For heuristic, assume it needs infinite cost or skip. Skipping is safer.
            if package not in package_current_state:
                 # print(f"Warning: Package {package} not found in current state.") # Debugging
                 continue # Cannot estimate cost for a package whose state is unknown

            current_state_info = package_current_state[package]

            # Case 1: Package is on the ground
            if current_state_info in self.locations: # Check if it's a location name
                current_location = current_state_info

                # If package is already at the goal, cost is 0 for this package
                if current_location == goal_location:
                    continue # Package is done

                # Package is on the ground at current_location != goal_location
                # Needs pick-up (1), drive (distance), drop (1)
                # We need the distance from current_location to goal_location
                dist = self.shortest_paths.get((current_location, goal_location), float('inf'))

                # If goal is unreachable from current location, this package cannot reach its goal.
                # Heuristic should reflect this with infinite cost.
                if dist == float('inf'):
                    return float('inf') # Problem is likely unsolvable from here

                total_cost += 1 # pick-up action
                total_cost += dist # drive actions
                total_cost += 1 # drop action

            # Case 2: Package is inside a vehicle
            elif current_state_info in self.vehicles: # Check if it's a vehicle name
                vehicle = current_state_info

                # Find the vehicle's current location
                # If vehicle is not in vehicle_locations (e.g., invalid state), return inf.
                if vehicle not in vehicle_locations:
                     # print(f"Warning: Vehicle {vehicle} carrying {package} not found at a location.") # Debugging
                     return float('inf') # Vehicle location unknown, cannot estimate drive cost

                vehicle_location = vehicle_locations[vehicle]

                # If vehicle is at the goal location, package just needs to be dropped
                if vehicle_location == goal_location:
                    total_cost += 1 # drop action
                else:
                    # Vehicle needs to drive from vehicle_location to goal_location, then drop
                    dist = self.shortest_paths.get((vehicle_location, goal_location), float('inf'))

                    if dist == float('inf'):
                        return float('inf') # Goal location unreachable from vehicle's current location

                    total_cost += dist # drive actions
                    total_cost += 1 # drop action
            else:
                 # print(f"Warning: Package {package} in unexpected state: {current_state_info}") # Debugging
                 # Package is neither at a location nor in a known vehicle/object type.
                 # Treat as unreachable goal for this package.
                 return float('inf')


        return total_cost
