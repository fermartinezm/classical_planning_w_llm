import collections
import math

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string like '(predicate arg1 arg2)' into a tuple."""
    # Remove leading/trailing parens and split by space
    parts = fact_string[1:-1].split()
    return tuple(parts)

class transportHeuristic:
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing
    the minimum estimated actions required for each package that is not
    at its goal location. For each such package, the estimated cost is
    calculated based on its current status (at a location or in a vehicle)
    and the shortest path distance in the road network between its current
    location and its goal location.

    Assumptions:
    - Roads are bidirectional (implied by example instance files).
    - All packages mentioned in the goal state exist in the initial state
      and subsequent states, and are either at a location or in a vehicle.
    - Vehicle capacity constraints are ignored for heuristic calculation.
    - The shortest path distance between locations is a reasonable estimate
      of the number of drive actions required.
    - The cost of pick-up and drop actions is 1 each.
    - Any object with an 'at' predicate that is not a package with a goal
      location is considered a vehicle for the purpose of finding its location.

    Heuristic Initialization:
    During initialization, the heuristic processes the static information
    from the task definition:
    1. It identifies the goal location for each package from the task's
       goal state definition. It also collects all unique locations mentioned
       in the goals and in the road network.
    2. It builds a graph representing the road network from the 'road' facts.
    3. It computes the shortest path distance (number of drive actions)
       between all pairs of locations using Breadth-First Search (BFS) on
       the road network graph. These distances are stored for quick lookup.
       Distances between locations not connected by roads (or isolated
       locations not on the road network) are recorded as infinity.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Check if the state is the goal state using `self.task.goal_reached(state)`.
       If yes, the heuristic value is 0.
    2. If not the goal state, initialize the total heuristic value `h` to 0.
    3. Iterate through the facts in the current state to determine the current
       location or status (in a vehicle) for every package that has a specified
       goal location, and also track the current location of each vehicle.
       - Store package status: map package name to ('at', location) or ('in', vehicle).
       - Store vehicle locations: map vehicle name to location.
    4. For each package that has a goal location (identified during initialization):
        a. Get the package's goal location.
        b. Determine the package's current location based on its status found in step 3.
           If the package is in a vehicle, its current location is the vehicle's location.
           If the package's status or the vehicle's location is unknown (e.g., not in state facts),
           the state is considered unsolvable from this point for this package.
        c. If the package's current location is the same as its goal location,
           it contributes 0 to the heuristic.
        d. If the package's current location is different from its goal location:
            i. Find the shortest path distance between the package's current
               location and its goal location using the precomputed distances.
               The precomputed distances handle cases where no path exists by storing `math.inf`.
            ii. If the distance is `math.inf` for the path between the package's
                current location and its goal location, it means the goal is
                unreachable for this package via the road network. In this case,
                the heuristic returns `math.inf` for the entire state.
            iii. If a finite path exists, estimate the minimum actions needed for this package:
                 - If the package is currently 'at' a location: It needs a pick-up
                   action (1), a sequence of drive actions (distance), and a drop
                   action (1). Estimated cost = 1 + distance + 1 = 2 + distance.
                 - If the package is currently 'in' a vehicle: It needs a sequence
                   of drive actions (distance) and a drop action (1). Estimated
                   cost = distance + 1.
            iv. Add this estimated cost for the package to the total heuristic value `h`.
    5. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Args:
            task: The planning task object (an instance of the Task class).
        """
        self.task = task
        self.goal_locations = {}
        self.locations = set()
        self.road_graph = collections.defaultdict(set)
        self.distances = {}

        # 1. Identify goal locations for packages and collect all locations
        for goal_fact_string in self.task.goals:
            goal_fact = parse_fact(goal_fact_string)
            if goal_fact[0] == 'at':
                # Assuming goal facts are only (at package location)
                package, location = goal_fact[1], goal_fact[2]
                self.goal_locations[package] = location
                self.locations.add(location) # Add goal location to known locations

        # 2. Build road network graph and collect locations from roads
        for fact_string in self.task.static:
            fact = parse_fact(fact_string)
            if fact[0] == 'road':
                l1, l2 = fact[1], fact[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.road_graph[l1].add(l2)
                self.road_graph[l2].add(l1) # Assuming roads are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        # Run BFS from all locations identified (from roads and goals)
        all_relevant_locations = list(self.locations) # Use a list for consistent iteration

        for start_node in all_relevant_locations:
            self.distances[start_node] = {}
            # BFS queue starts with the start node
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}

            while queue:
                current_node, dist = queue.popleft()
                self.distances[start_node][current_node] = dist

                # Only explore neighbors if the current node is part of the road graph
                if current_node in self.road_graph:
                    for neighbor in self.road_graph[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

        # Fill in unreachable distances with infinity for all pairs of known locations
        for l1 in all_relevant_locations:
             if l1 not in self.distances:
                 self.distances[l1] = {}
             for l2 in all_relevant_locations:
                 if l2 not in self.distances[l1]:
                     self.distances[l1][l2] = math.inf


    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal, or math.inf
            if the goal is unreachable from this state according to the
            relaxed model.
        """
        # 1. Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        h = 0
        current_package_status = {} # Maps package -> ('at', loc) or ('in', veh)
        current_vehicle_locations = {} # Maps vehicle -> loc

        # 3. Determine current locations/status of packages and vehicles
        # First pass: Collect all 'at' and 'in' facts
        all_at_facts = {} # Maps object -> location
        all_in_facts = {} # Maps package -> vehicle

        for fact_string in state:
             fact = parse_fact(fact_string)
             if fact[0] == 'at':
                 obj, loc = fact[1], fact[2]
                 all_at_facts[obj] = loc
             elif fact[0] == 'in':
                 package, vehicle = fact[1], fact[2]
                 all_in_facts[package] = vehicle

        # Second pass: Determine status for packages we care about (those with goals)
        for package in self.goal_locations:
            if package in all_in_facts:
                current_package_status[package] = ('in', all_in_facts[package])
            elif package in all_at_facts:
                 current_package_status[package] = ('at', all_at_facts[package])
            # If package is not in all_in_facts and not in all_at_facts,
            # it's not present in the state facts, which is an issue.
            # The check below will handle the case where current_status is None.


        # Populate vehicle locations from collected 'at' facts
        # Assume any object in all_at_facts that is not a goal package is a vehicle
        all_objects_at_locations = set(all_at_facts.keys())
        goal_packages = set(self.goal_locations.keys())
        # Objects that are 'locatable' but are not packages we need to deliver are assumed vehicles
        vehicle_objects = all_objects_at_locations - goal_packages

        for vehicle in vehicle_objects:
             current_vehicle_locations[vehicle] = all_at_facts[vehicle]


        # Now, iterate through packages that need to reach a goal
        for package, L_goal in self.goal_locations.items():
            current_status = current_package_status.get(package)

            if current_status is None:
                 # Package not found in state facts (neither 'at' nor 'in').
                 # This indicates an issue with the state representation or task definition.
                 # Treat as unsolvable.
                 # print(f"Warning: Package {package} with goal {L_goal} not found in state facts.")
                 return math.inf

            status_type, obj_or_loc = current_status

            L_current = None
            is_in_vehicle = False

            if status_type == 'at':
                L_current = obj_or_loc
                is_in_vehicle = False
            elif status_type == 'in':
                vehicle = obj_or_loc
                L_current = current_vehicle_locations.get(vehicle) # Get vehicle's location
                is_in_vehicle = True
                if L_current is None:
                    # Package is in a vehicle, but vehicle location is unknown.
                    # This also implies an invalid or unsolvable state.
                    # print(f"Warning: Vehicle {vehicle} carrying package {package} not found in state 'at' facts.")
                    return math.inf

            # If package is already at its goal, cost is 0 for this package
            if L_current == L_goal:
                continue

            # Package is not at its goal, calculate cost
            # Ensure L_current and L_goal are known locations
            if L_current not in self.locations or L_goal not in self.locations:
                 # This should ideally not happen if locations were collected correctly
                 # from roads and goals, and state facts use valid location names.
                 # print(f"Warning: Location {L_current} or {L_goal} from state/goal not found in known locations.")
                 return math.inf

            # Get distance, which is math.inf if unreachable
            distance = self.distances[L_current][L_goal]

            if distance == math.inf:
                 # Goal is unreachable from current location via road network
                 return math.inf

            if is_in_vehicle:
                # Needs drive + drop
                h += 1 + distance
            else:
                # Needs pick-up + drive + drop
                h += 2 + distance

        return h
