from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict, deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL facts are structured.
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    Estimates the cost for each package to reach its goal location,
    summing up the minimum actions required (pick-up, drive, drop).
    Uses shortest path distances on the road network for drive costs.
    Ignores vehicle capacity and availability for simplicity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for packages
        and precomputing shortest path distances between locations.
        """
        # The base class constructor might need to be called depending on its definition.
        # Assuming it's not strictly necessary for this specific base class example,
        # or that the necessary attributes (goals, static) are accessed directly from task.
        # super().__init__(task)

        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find all locations

        # 1. Extract goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            # Goal facts are typically (at ?p ?l)
            parts = get_parts(goal)
            if parts and parts[0] == "at":
                # Assuming the goal is always (at package location)
                if len(parts) == 3:
                    package, location = parts[1], parts[2]
                    self.goal_locations[package] = location
                # Handle potential other goal types if necessary, but problem implies (at package location)

        # 2. Build the road network graph and collect all locations.
        self.road_graph = defaultdict(list)
        locations = set()

        # Add locations from static road facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "road":
                # Assuming road facts are always (road l1 l2)
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    self.road_graph[l1].append(l2)
                    locations.add(l1)
                    locations.add(l2)

        # Add locations from initial state (where objects are)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "at":
                 # Assuming 'at' facts are always (at object location)
                 if len(parts) == 3:
                     locations.add(parts[2]) # Add location of object
             elif parts and parts[0] == "in":
                 # If a package is initially 'in' a vehicle, we need the vehicle's location
                 # which should be covered by an 'at' fact for the vehicle.
                 # We don't add the vehicle name itself to locations.
                 pass # No location to add from 'in' fact itself

        # Add locations from goal state (where packages need to go)
        for loc in self.goal_locations.values():
             locations.add(loc)

        self.locations = list(locations) # Store locations list

        # 3. Compute all-pairs shortest paths.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Perform BFS from a start node to find shortest distances to all reachable nodes.
        Returns a dictionary {location: distance}.
        """
        distances = {loc: float('inf') for loc in self.locations}
        if start_node in distances: # Ensure start_node is one of the known locations
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_loc = queue.popleft()

                # If current_loc is not in road_graph, it's an isolated location
                # but might be a start/goal location. BFS from it can only reach itself.
                # We already initialized its distance to 0 if it was in self.locations.
                # Neighbors are only found if it's in the road_graph keys.
                if current_loc in self.road_graph:
                    for neighbor in self.road_graph[current_loc]:
                        # Ensure neighbor is a known location before accessing distances
                        if neighbor in distances and distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current_loc] + 1
                            queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Helper to get shortest distance between two locations."""
        # If either location is not in our precomputed distances,
        # it means it wasn't found in initial state, goals, or road facts.
        # This shouldn't happen for relevant locations in a solvable problem,
        # but returning infinity is a safe fallback for unreachable goals.
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf')

        # The inner dictionary self.distances[loc1] contains distances from loc1.
        # We need the distance to loc2.
        if loc2 not in self.distances[loc1]:
             return float('inf') # loc2 is unreachable from loc1

        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Track current locations of all locatables (packages and vehicles).
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "at":
                # Assuming 'at' facts are always (at object location)
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    current_locations[obj] = loc
            elif predicate == "in":
                # Assuming 'in' facts are always (in package vehicle)
                 if len(parts) == 3:
                    package, vehicle = parts[1], parts[2]
                    current_locations[package] = vehicle # Package is inside a vehicle

        total_cost = 0

        # Iterate through packages that have a goal location.
        for package, goal_location in self.goal_locations.items():
            # If package is not in current_locations, it's not in the state
            # (neither at a location nor in a vehicle). This would be an invalid state
            # for a package that needs to reach a goal.
            if package not in current_locations:
                 # This state is likely invalid or unreachable in a standard problem.
                 # Return infinity to prune.
                 return float('inf')

            package_current_state = current_locations[package]

            # Check if the package is already at its goal location.
            # This happens if package_current_state is a location and equals goal_location.
            if package_current_state == goal_location:
                continue # Package is at goal, cost is 0 for this package.

            # Package is not at the goal. It needs actions.
            # Minimum actions: pick-up, drive, drop.
            # The cost depends on whether it's on the ground or in a vehicle.

            # Case 1: Package is on the ground at package_current_state (which is a location).
            # We check if package_current_state is one of the known locations.
            if package_current_state in self.locations:
                 current_location = package_current_state # Rename for clarity
                 # Needs pick-up (1), drive (distance), drop (1).
                 # We need to drive from current_location to goal_location.
                 drive_cost = self.get_distance(current_location, goal_location)
                 if drive_cost == float('inf'):
                     # Goal is unreachable from current location.
                     # This state is likely part of an unsolvable path or problem.
                     # Return infinity to prune this branch.
                     return float('inf')
                 total_cost += 1 + drive_cost + 1 # pick-up + drive + drop

            # Case 2: Package is inside a vehicle (package_current_state is a vehicle name).
            else: # package_current_state is assumed to be a vehicle name
                 vehicle_name = package_current_state
                 # Find the location of the vehicle.
                 if vehicle_name not in current_locations:
                      # Vehicle location is missing from state? Should not happen in valid states.
                      # Treat as unreachable or error.
                      return float('inf') # Or handle error

                 vehicle_location = current_locations[vehicle_name]

                 # Needs drive (distance) + drop (1).
                 # We need to drive the vehicle from vehicle_location to goal_location.
                 drive_cost = self.get_distance(vehicle_location, goal_location)
                 if drive_cost == float('inf'):
                     # Goal is unreachable from vehicle's current location.
                     return float('inf')
                 total_cost += drive_cost + 1 # drive + drop

        return total_cost
