# Assuming heuristic_base.py provides the Heuristic base class
# from heuristics.heuristic_base import Heuristic

from collections import deque

# Dummy Heuristic base class if not provided by the environment
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the cost to move each package/vehicle to its goal location independently,
    ignoring vehicle capacity and assuming vehicles are available.

    Cost for a package not at its goal:
    - If on the ground: 1 (pick-up) + shortest_path_dist (drive) + 1 (drop)
    - If in a vehicle: shortest_path_dist (drive) + 1 (drop)

    Cost for a vehicle not at its goal:
    - shortest_path_dist (drive)

    The total heuristic is the sum of costs for all objects (packages/vehicles)
    that have a goal location and are not currently at that location.
    Shortest path distances are precomputed using BFS on the road network.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing shortest path distances.
        """
        # Call the base class constructor if needed, though it's empty here
        # super().__init__(task)

        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state

        self.road_graph = {}
        self.locations = set()
        self.packages = set()
        self.vehicles = set()

        # Process all facts (static and initial) to build graph and infer types
        all_facts = set(static_facts) | set(initial_state_facts)

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any
            predicate = parts[0]
            if predicate == "road" and len(parts) == 3:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph.setdefault(l1, set()).add(l2)
                self.road_graph.setdefault(l2, set()).add(l1) # Assuming bidirectional
            elif predicate == "capacity" and len(parts) == 3:
                vehicle = parts[1]
                self.vehicles.add(vehicle)
            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                self.packages.add(pkg)
                self.vehicles.add(veh)
            elif predicate == "at" and len(parts) == 3:
                 locatable_obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 # Type inference from 'at' is ambiguous, rely on 'in'/'capacity'

        # Add objects from goals if they weren't identified by 'in' or 'capacity'
        # This handles goals for vehicles that might not have capacity facts in init/static
        # or packages that are initially on the ground and not mentioned in 'in' facts.
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if parts and parts[0] == "at" and len(parts) == 3:
                 obj = parts[1]
                 # If obj is not already identified as a vehicle, assume it's a package
                 # (based on typical transport domain goals)
                 if obj not in self.vehicles:
                      self.packages.add(obj)
                 # If it is a vehicle, it's already in self.vehicles

        self.locations = list(self.locations) # Convert to list for consistent iteration/lookup if needed, dict is fine
        self.distances = self._compute_all_pairs_shortest_paths()

        # Extract goal locations for each object (package or vehicle)
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "at" and len(parts) == 3:
                obj, location = parts[1], parts[2]
                self.goal_locations[obj] = location
            # Ignore other types of goal facts if any exist (e.g., (in p v) goals are rare)


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS from each location.
        """
        distances = {}
        for start_loc in self.locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_loc):
        """
        Performs BFS starting from start_loc to find distances to all other locations.
        """
        dist = {loc: float('inf') for loc in self.locations}
        # Handle case where start_loc might not be in self.locations (e.g., inconsistent PDDL)
        if start_loc not in dist:
             # Cannot compute distances from an unknown location
             # Return a dict where all distances are inf
             return dist

        dist[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            current_loc = queue.popleft()

            # If current_loc is not in graph (isolated location), skip its neighbors
            if current_loc not in self.road_graph:
                continue

            for neighbor in self.road_graph[current_loc]:
                if dist[neighbor] == float('inf'):
                    dist[neighbor] = dist[current_loc] + 1
                    queue.append(neighbor)
        return dist

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Track current locations of all locatable objects (packages and vehicles)
        # and which package is inside which vehicle.
        obj_locations = {}
        package_in_vehicle = {}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts
            predicate = parts[0]
            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                obj_locations[obj] = loc
            elif predicate == "in" and len(parts) == 3:
                pkg, veh = parts[1], parts[2]
                package_in_vehicle[pkg] = veh

        total_cost = 0

        # Iterate through each object that has a goal location
        for obj, goal_location in self.goal_locations.items():

            current_location = None
            is_in_vehicle = False

            # Find current location and state (in vehicle or on ground)
            if obj in package_in_vehicle:
                # Object is a package inside a vehicle
                vehicle = package_in_vehicle[obj]
                current_location = obj_locations.get(vehicle)
                is_in_vehicle = True
            elif obj in obj_locations:
                 # Object is on the ground (package) or a vehicle
                 current_location = obj_locations[obj]
                 is_in_vehicle = False # Only packages can be 'in' vehicles

            # If the object's current location is unknown (e.g., vehicle not found for package),
            # this state might be invalid or unreachable. Return infinity.
            if current_location is None:
                 # This can happen if a package is 'in' a vehicle, but the vehicle
                 # is not 'at' any location in the state. Or if an object in the goal
                 # is neither 'at' a location nor 'in' a vehicle.
                 # This indicates an inconsistent state.
                 return float('inf')

            # If the object is already at its goal, no cost for this object.
            if current_location == goal_location:
                continue

            # Get the shortest path distance
            # Handle cases where current_location or goal_location might not be in self.locations
            # (e.g., if the state contains locations not mentioned in static/initial facts, unlikely but possible)
            if current_location not in self.distances or goal_location not in self.distances.get(current_location, {}):
                 # Goal location is unreachable from current location in the road network
                 return float('inf')

            distance = self.distances[current_location][goal_location]

            # If goal is unreachable, return infinity.
            if distance == float('inf'):
                return float('inf')

            # Calculate cost based on object type and state
            if obj in self.packages:
                if is_in_vehicle:
                    # Package in vehicle: drive + drop
                    cost_for_obj = distance + 1
                else:
                    # Package on ground: pick + drive + drop
                    cost_for_obj = 1 + distance + 1
            elif obj in self.vehicles:
                 # Vehicle: drive
                 cost_for_obj = distance
            else:
                 # Object in goal is neither identified as package nor vehicle.
                 # This shouldn't happen with the current inference logic if goals are well-formed.
                 # Treat as unreachable.
                 return float('inf')

            total_cost += cost_for_obj

        return total_cost
