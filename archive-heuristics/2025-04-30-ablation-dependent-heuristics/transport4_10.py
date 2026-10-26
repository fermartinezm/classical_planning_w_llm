import collections
import math

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing
    the estimated costs for each package that is not yet at its goal location.
    For a package not at its goal, the estimated cost includes:
    1. If the package is at a location: 1 (pick-up) + shortest_path_distance(current_location, goal_location) + 1 (drop).
    2. If the package is inside a vehicle: shortest_path_distance(vehicle_location, goal_location) + 1 (drop).
    The shortest path distances between locations are precomputed using BFS on the road network.

    Assumptions:
    - The heuristic ignores vehicle capacity constraints.
    - The heuristic ignores vehicle availability (assumes any vehicle can be used).
    - The heuristic assumes that objects starting with 'p' are packages and objects starting with 'v' are vehicles, based on common PDDL naming conventions in this domain. Locations are identified from road facts.
    - The state representation is valid, meaning every package is either at a location or in a vehicle, and every vehicle is at a location.
    - Goal facts are primarily of the form (at package location).

    Heuristic Initialization:
    The constructor processes the static information from the task:
    1. It identifies all locations and the road network from '(road l1 l2)' facts.
    2. It computes the shortest path distance between all pairs of locations using Breadth-First Search (BFS). These distances are stored for quick lookup.
    3. It extracts the goal location for each package from the task's goal facts.

    Step-By-Step Thinking for Computing Heuristic:
    The __call__ method computes the heuristic value for a given state:
    1. Check if the state is the goal state. If yes, the heuristic is 0.
    2. Initialize the total heuristic value `h` to 0.
    3. Parse the current state to determine the location of each package (either at a location or in a vehicle) and the location of each vehicle. This information is stored in dictionaries (`package_location`, `vehicle_location`).
    4. Iterate through each package that has a goal location defined in the task.
    5. For each such package, find its current status (location or vehicle).
    6. If the package is already at its goal location, add 0 to `h`.
    7. If the package is at a location `loc_p_current` (identified as one of the known locations) and needs to go to `loc_p_goal`:
       - Add 2 (for pick-up and drop actions) to `h`.
       - Add the precomputed shortest distance from `loc_p_current` to `loc_p_goal` to `h`. If no path exists, the state is considered unsolvable from this perspective, and the heuristic returns infinity.
    8. If the package is inside a vehicle `v` (identified by 'v' prefix), find the vehicle's current location `loc_v_current`, and the package needs to go to `loc_p_goal`:
       - Add 1 (for the drop action) to `h`.
       - Add the precomputed shortest distance from `loc_v_current` to `loc_p_goal` to `h`. If no path exists, return infinity.
    9. If a package's status or a vehicle's location cannot be determined from the state (indicating an unexpected state structure), return infinity.
    10. Return the total computed value `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing static information.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.package_goals = {}
        self.locations = set()
        road_edges = []

        # Parse static facts
        for fact_str in task.static:
            # Extract predicate and arguments
            parts = fact_str.strip('()').split()
            if not parts: # Skip empty strings
                continue
            predicate = parts[0]

            if predicate == 'road':
                if len(parts) == 3:
                    l1, l2 = parts[1], parts[2]
                    self.locations.add(l1)
                    self.locations.add(l2)
                    road_edges.append((l1, l2))
                # Ignore malformed road facts
            # Ignore capacity-predecessor and other static facts for this heuristic

        # Build road graph
        self.graph = {loc: [] for loc in self.locations}
        for l1, l2 in road_edges:
            # Ensure l1 is in graph keys (it should be if collected from locations)
            if l1 in self.graph:
                 self.graph[l1].append(l2)


        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            q = collections.deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0

            while q:
                current_loc, dist = q.popleft()

                # Check if current_loc is a valid key in graph before accessing
                # This check is technically redundant if locations set is built correctly
                # from road facts, but harmless.
                if current_loc not in self.graph:
                     continue

                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_loc, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

        # Parse goal facts
        for fact_str in task.goals:
            parts = fact_str.strip('()').split()
            if not parts: # Skip empty strings
                continue
            predicate = parts[0]

            if predicate == 'at':
                if len(parts) == 3:
                    # Assuming goal facts of the form (at package location)
                    obj_name, goal_loc = parts[1], parts[2]
                    # Assuming objects starting with 'p' are packages
                    if obj_name.startswith('p'):
                         self.package_goals[obj_name] = goal_loc
                # Ignore malformed at facts in goals or goals for non-packages
            # Ignore other potential goal types if any

    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: The current state (frozenset of facts).

        Returns:
            The estimated number of actions to reach the goal, or infinity if
            the state is likely unsolvable.
        """
        if self.task.goal_reached(state):
            return 0

        package_location = {} # Maps package name to location string or vehicle name
        vehicle_location = {} # Maps vehicle name to location string

        # Parse current state facts
        for fact_str in state:
            parts = fact_str.strip('()').split()
            if not parts: # Skip empty strings
                continue
            predicate = parts[0]

            if predicate == 'at':
                if len(parts) == 3:
                    obj_name, loc_name = parts[1], parts[2]
                    # Assuming objects starting with 'p' are packages and 'v' are vehicles
                    if obj_name.startswith('p'):
                        package_location[obj_name] = loc_name
                    elif obj_name.startswith('v'):
                        vehicle_location[obj_name] = loc_name
                    # Ignore other types of objects at locations if any
                # Ignore malformed at facts
            elif predicate == 'in':
                if len(parts) == 3:
                    package_name, vehicle_name = parts[1], parts[2]
                    # Assuming object 1 is package and object 2 is vehicle in (in p v)
                    if package_name.startswith('p') and vehicle_name.startswith('v'):
                        package_location[package_name] = vehicle_name # Store vehicle name
                    else:
                        # Unexpected format for (in ...) fact
                        # This state might be invalid or unsolvable
                        return float('inf')
                # Ignore malformed in facts
            # Ignore other predicates like capacity

        h = 0
        # Calculate cost for each package that needs to reach its goal
        for package_name, goal_loc in self.package_goals.items():
            current_loc_or_vehicle = package_location.get(package_name)

            # Handle cases where package status is unknown (shouldn't happen in valid states)
            if current_loc_or_vehicle is None:
                 # Package not found in state facts - indicates an issue or unsolvable state
                 return float('inf')

            # If package is already at its goal, cost is 0 for this package
            if current_loc_or_vehicle == goal_loc:
                continue

            # If package is at a location
            if current_loc_or_vehicle in self.locations: # Check if it's a known location string
                current_loc = current_loc_or_vehicle
                # Cost: pick-up (1) + drive (distance) + drop (1)
                dist = self.distances.get((current_loc, goal_loc), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Goal location unreachable from current location
                h += 2 + dist
            # If package is in a vehicle
            elif current_loc_or_vehicle.startswith('v'): # Assuming it's a vehicle name
                vehicle_name = current_loc_or_vehicle
                current_loc_v = vehicle_location.get(vehicle_name)

                # Handle cases where vehicle location is unknown (shouldn't happen)
                if current_loc_v is None:
                    # Vehicle not found at a location - indicates an issue or unsolvable state
                    return float('inf')

                # Check if vehicle's current location is a known location
                if current_loc_v not in self.locations:
                     # Vehicle at an unknown location - indicates an issue or unsolvable state
                     return float('inf')

                # Cost: drive (distance) + drop (1)
                dist = self.distances.get((current_loc_v, goal_loc), float('inf'))

                if dist == float('inf'):
                    return float('inf') # Goal location unreachable from vehicle's location
                h += 1 + dist
            else:
                # current_loc_or_vehicle is neither a known location nor a vehicle name prefix 'v'
                # Indicates an unexpected state structure or object type
                return float('inf')


        return h
