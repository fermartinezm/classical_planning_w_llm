from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a list of strings,
    removing the surrounding brackets.
    E.g., '(at p1 l1)' becomes ['at', 'p1', 'l1'].
    """
    # Remove leading/trailing brackets and split by space
    parts = fact_string.strip('()').split()
    return parts

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the transport domain.

    Summary:
        Estimates the cost to reach the goal by summing the minimum estimated
        actions required for each package to reach its goal location. The
        estimate for a single package is calculated as 2 (for pickup and drop)
        plus the shortest path distance (in drive actions) between the package's
        current location and its goal location in the road network. This
        heuristic is non-admissible and designed for greedy best-first search.

    Assumptions:
        - The heuristic is specific to the PDDL 'transport' domain.
        - The road network defined by 'road' predicates is static.
        - The goal is a conjunction of '(at package location)' facts.
        - The heuristic does not consider vehicle capacity or availability
          beyond the package's current location. It assumes a vehicle is
          available to pick up the package at its current location and
          transport it along the shortest path.

    Heuristic Initialization:
        1. Parses the goal facts to identify which packages need to be moved
           and their respective goal locations. Stores this in `self.package_goals`.
        2. Parses static 'road' facts to build an adjacency list representation
           of the road network graph. Identifies all unique locations mentioned
           in goals, initial state 'at' facts, and static 'road' facts.
        3. Computes all-pairs shortest paths on the road network graph using
           Breadth-First Search (BFS) starting from each location. Stores the
           distances in `self.shortest_paths` as a dictionary mapping
           (start_location, end_location) tuples to distances.

    Step-By-Step Thinking for Computing Heuristic:
        1. Given a state, check if it is the goal state using `task.goal_reached`.
           If it is, the heuristic value is 0.
        2. If not the goal state, parse the current state facts to determine
           the current location of each locatable object (packages and vehicles).
           This involves iterating through '(at object location)' facts.
           Then, iterate through '(in package vehicle)' facts and update the
           package's location to be the location of the vehicle it is in.
        3. Initialize the total heuristic value `h` to 0.
        4. Iterate through each package that needs to be delivered (identified
           during initialization from the goal facts stored in `self.package_goals`).
        5. For the current package, find its current location using the information
           parsed from the state.
        6. If the package's current location is different from its goal location:
            a. Look up the pre-calculated shortest path distance `d` between the
               current location and the goal location in the road network using
               `self.shortest_paths.get((current_location, goal_location))`.
            b. If no path exists (e.g., disconnected graph, or package location
               could not be determined), the state is likely unsolvable from here
               for this package, return `float('inf')` for the total heuristic.
            c. Add the estimated cost for this package (`2 + d`) to the total
               heuristic value `h`. The '2' represents the pickup and drop actions,
               and `d` represents the minimum number of drive actions.
        7. Return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task # Store task to use goal_reached
        self.package_goals = {}
        self.locations = set()
        self.adj = {}
        self.shortest_paths = {}

        # 1. Extract package goals and locations from goals
        for goal_fact_string in task.goals:
            parts = parse_fact(goal_fact_string)
            if parts[0] == 'at':
                # Assuming goals are always (at package location)
                package, location = parts[1], parts[2]
                self.package_goals[package] = location
                self.locations.add(location)
            # Ignore other potential goal types if any

        # 2. Extract locations and road network from initial state and static facts
        # Add locations from initial state 'at' facts
        for fact_string in task.initial_state:
             parts = parse_fact(fact_string)
             if parts[0] == 'at':
                 # arg1 is locatable, arg2 is location
                 self.locations.add(parts[2])

        # Build road graph and collect locations from static 'road' facts
        for fact_string in task.static:
            parts = parse_fact(fact_string)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                self.adj.setdefault(l1, []).append(l2)
                self.locations.add(l1)
                self.locations.add(l2)

        # 3. Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                curr_loc, curr_dist = q.popleft()
                self.shortest_paths[(start_loc, curr_loc)] = curr_dist
                for neighbor in self.adj.get(curr_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, curr_dist + 1))

    def __call__(self, node):
        state = node.state

        # 1. Check for goal state
        if self.task.goal_reached(state):
            return 0

        # 2. Parse current state to find locations
        current_locations = {} # {object_name: location_name}
        package_in_vehicle = {} # {package_name: vehicle_name}

        # First pass: get 'at' locations for everything
        for fact_string in state:
            parts = parse_fact(fact_string)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == 'in':
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        # Second pass: resolve locations for packages inside vehicles
        # Update current_locations for packages that are 'in' vehicles
        for package, vehicle in package_in_vehicle.items():
             if vehicle in current_locations:
                  current_locations[package] = current_locations[vehicle]
             # else: vehicle location unknown? This shouldn't happen in a valid state graph.
             # If it happens, the package's location remains unresolved, and the .get() below will return None.

        # 3. Initialize total heuristic
        h = 0

        # 4. Iterate through packages that need delivery
        for package, goal_location in self.package_goals.items():
            current_location = current_locations.get(package)

            # If package location is unknown (e.g., not in state or vehicle location unknown)
            # or it's already at the goal, skip
            if current_location is None or current_location == goal_location:
                continue

            # 6a. Look up shortest path distance
            distance = self.shortest_paths.get((current_location, goal_location))

            # 6b. If no path, return infinity
            if distance is None:
                # This means the current location cannot reach the goal location
                # via the road network. This state is likely a dead end for this package.
                return float('inf')

            # 6c. Add estimated cost (pickup + drive + drop)
            # This assumes a vehicle is available at current_location with capacity.
            # It also assumes the vehicle can reach goal_location.
            h += 2 + distance

        # 7. Return total heuristic
        return h
