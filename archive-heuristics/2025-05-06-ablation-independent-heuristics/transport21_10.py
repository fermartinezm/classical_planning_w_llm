import collections
import re
from heuristics.heuristic_base import Heuristic
from task import Task

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    for each package that is not yet at its goal location. The estimated
    cost for a package depends on whether it is currently at a location
    or inside a vehicle, and involves the shortest path distance on the
    road network plus costs for pick-up and drop actions. This heuristic
    is non-admissible as it ignores vehicle capacity, availability, and
    potential synergies/conflicts when transporting multiple packages.

    Assumptions:
    - The road network defined by 'road' facts is static and bidirectional.
      Distances are calculated based on this assumption.
    - Vehicles are always located at some location in a valid state.
    - Packages are always either at a location or inside a vehicle in a valid state.
    - The primary goal facts are of the form (at package location). Other goal
      types are not explicitly handled in the cost calculation but the final
      goal state check ensures h=0 only for goal states.
    - Vehicle capacity and specific vehicle availability/assignment are ignored
      in the cost estimation for simplicity and efficiency.

    Heuristic Initialization:
    1. Parse the goal facts to identify the target location for each package
       that needs to be at a specific location. Store this mapping.
    2. Parse the static 'road' facts to build a graph representing the road network.
       Collect all unique locations.
    3. Compute all-pairs shortest paths between all identified locations
       using Breadth-First Search (BFS) on the road network graph. Store these
       distances in a dictionary where keys are (start_location, end_location) tuples.

    Step-By-Step Thinking for Computing Heuristic:
    1. Given a state, initialize the total heuristic value `h` to 0.
    2. Create temporary lookup dictionaries for the current state to quickly find:
       - The location of each package (`package_location`).
       - The vehicle containing each package (`package_in_vehicle`).
       - The location of each vehicle (`vehicle_location`).
       Populate these dictionaries by iterating through the facts in the current state.
       Objects in '(at ...)' facts that are not goal packages are assumed to be vehicles for location tracking.
    3. Iterate through each package that has a goal location defined (identified during initialization).
    4. For a package `p` with goal location `l_goal`:
       a. Check if `p` is already at `l_goal` in the current state using the lookup dictionaries. If yes, the estimated cost for this package is 0, and we proceed to the next package.
       b. If `p` is currently at location `l_current` (and `l_current != l_goal`):
          - Estimate the cost to move this package. This requires a vehicle to go to `l_current`, pick up `p`, drive to `l_goal`, and drop `p`.
          - The estimated cost is the shortest distance from `l_current` to `l_goal` (number of drive actions) plus 2 actions (one for pick-up, one for drop).
          - Add `dist(l_current, l_goal) + 2` to the total heuristic `h`. If `l_goal` is unreachable from `l_current`, the distance is infinite, and the heuristic becomes infinite.
       c. If `p` is currently inside a vehicle `v`:
          - Find the current location `l_v` of vehicle `v` using the lookup dictionaries.
          - Estimate the cost to move this package. This requires vehicle `v` to drive to `l_goal` and drop `p`.
          - The estimated cost is the shortest distance from `l_v` to `l_goal` (number of drive actions) plus 1 action (for drop).
          - Add `dist(l_v, l_goal) + 1` to the total heuristic `h`. If `l_goal` is unreachable from `l_v`, the distance is infinite, and the heuristic becomes infinite.
       d. If the package's location or containment cannot be determined from the state facts (which should not happen in valid states according to domain definition), return infinity as a fallback for potentially invalid states.
    5. After summing costs for all goal packages, perform a final check: if the current state is the actual goal state (using the task's `goal_reached` method), return 0. Otherwise, return the calculated total heuristic value `h` (which will be > 0 for non-goal states where packages need moving, or infinity for unreachable/invalid states). This ensures the heuristic is 0 if and only if the goal is reached.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.package_goals = {}
        self.locations = set()
        self.road_graph = collections.defaultdict(set)
        self.distances = {} # Stores shortest path distances: (loc1, loc2) -> dist

        # 1. Parse goal facts to find package goals
        # Assuming goals are primarily (at package location)
        for goal_fact in self.task.goals:
            match = re.match(r'\(at (\S+) (\S+)\)', goal_fact)
            if match:
                package, location = match.groups()
                self.package_goals[package] = location

        # 2. Parse static 'road' facts and build graph
        for static_fact in self.task.static:
            match = re.match(r'\(road (\S+) (\S+)\)', static_fact)
            if match:
                loc1, loc2 = match.groups()
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.road_graph[loc1].add(loc2)
                # Assuming roads are bidirectional based on examples
                self.road_graph[loc2].add(loc1)

        # 3. Compute all-pairs shortest paths using BFS
        self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS starting from each location.
        """
        for start_node in self.locations:
            queue = collections.deque([(start_node, 0)])
            visited = {start_node: 0}
            self.distances[(start_node, start_node)] = 0

            while queue:
                current_node, dist = queue.popleft()

                # Store distance from start_node to current_node
                self.distances[(start_node, current_node)] = dist

                # Explore neighbors
                for neighbor in self.road_graph.get(current_node, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
                        # Store distance as soon as it's found
                        self.distances[(start_node, neighbor)] = dist + 1


        # Ensure all pairs have a distance (infinity if unreachable)
        for loc1 in self.locations:
            for loc2 in self.locations:
                if (loc1, loc2) not in self.distances:
                    self.distances[(loc1, loc2)] = float('inf')


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Keyword arguments:
        node -- the current state node

        Returns:
        The estimated number of actions to reach a goal state.
        Returns float('inf') if the state is estimated to be a dead end.
        Returns 0 if the state is the goal state.
        """
        state = node.state

        # Check if it's the goal state first
        if self.task.goal_reached(state):
             return 0

        # Build lookup dictionaries for current state
        package_location = {}
        package_in_vehicle = {}
        vehicle_location = {}

        # Identify all objects to help distinguish packages from vehicles
        all_objects = set()
        for fact in self.task.initial_state | self.task.goals | self.task.static | state:
             # Simple regex to find objects within parentheses
             objs = re.findall(r'\(.*? (.*?)\)', fact) # Find objects after predicate
             all_objects.update(objs)
             objs = re.findall(r'\(.*? .*? (.*?)\)', fact) # Find second object
             all_objects.update(objs)
             objs = re.findall(r'\(.*? .*? .*? (.*?)\)', fact) # Find third object (for pick-up/drop)
             all_objects.update(objs)
             objs = re.findall(r'\(.*? .*? .*? .*? (.*?)\)', fact) # Find fourth object
             all_objects.update(objs)
             objs = re.findall(r'\(.*? .*? .*? .*? .*? (.*?)\)', fact) # Find fifth object
             all_objects.update(objs)


        # Identify potential vehicles: objects that are not goal packages
        # This is a simplification; a proper parser would provide types.
        potential_vehicles = all_objects - set(self.package_goals.keys())


        for fact in state:
            match_at = re.match(r'\(at (\S+) (\S+)\)', fact)
            if match_at:
                obj, loc = match_at.groups()
                if obj in self.package_goals:
                    package_location[obj] = loc
                elif obj in potential_vehicles: # Assume it's a vehicle if not a goal package
                     vehicle_location[obj] = loc
                # else: it's some other locatable we don't care about for this heuristic

            match_in = re.match(r'\(in (\S+) (\S+)\)', fact)
            if match_in:
                package, vehicle = match_in.groups()
                if package in self.package_goals: # Only track packages we care about
                    package_in_vehicle[package] = vehicle
                # else: it's some other package we don't care about

        h = 0
        # Iterate through packages that need to reach a goal location
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at the goal
            if package in package_location and package_location[package] == goal_location:
                continue # Goal satisfied for this package

            # Package is not at the goal. Where is it?
            if package in package_location:
                # Package is at a different location
                current_location = package_location[package]
                # Cost: drive vehicle to current_location, pick up, drive to goal_location, drop
                # Approximation: dist(current_location, goal_location) + pickup + drop
                dist = self.distances.get((current_location, goal_location), float('inf'))
                if dist == float('inf'):
                    return float('inf') # Unreachable goal location for this package
                h += dist + 2

            elif package in package_in_vehicle:
                # Package is inside a vehicle
                vehicle = package_in_vehicle[package]
                # Find vehicle's location
                if vehicle in vehicle_location:
                    vehicle_current_location = vehicle_location[vehicle]
                    # Cost: drive vehicle to goal_location, drop
                    # Approximation: dist(vehicle_current_location, goal_location) + drop
                    dist = self.distances.get((vehicle_current_location, goal_location), float('inf'))
                    if dist == float('inf'):
                        return float('inf') # Unreachable goal location for this package
                    h += dist + 1
                else:
                    # Vehicle location not found - implies invalid state or unhandled case
                    # Return inf as a safe fallback.
                    return float('inf')
            else:
                 # Package is neither at a location nor in a vehicle - implies invalid state
                 # Return inf as a safe fallback.
                 return float('inf')

        # If we reached here and h is 0, it means all goal packages are at their
        # goal locations. The initial check `self.task.goal_reached(state)`
        # already handled the case where the full goal is met.
        # If h > 0, it means some goal packages are not yet at their destination.
        # If h is inf, it means some goal package is in an unreachable location/vehicle.

        return h

