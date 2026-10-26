from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic


class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move each package
    from its current location to its goal location. It sums the estimated costs
    for each package independently, ignoring vehicle capacity limits and
    potential conflicts. The cost for a package involves loading, driving, and unloading actions.

    # Assumptions
    - Each package needs to reach a specific goal location on the ground.
    - Any package can be transported by any vehicle that is available at the
      package's location and has sufficient capacity (size constraints are
      relaxed - we assume a suitable vehicle exists if needed).
    - Vehicles can move between locations connected by a road.
    - The cost of moving a vehicle between two locations is the shortest path
      distance in the road network.
    - The cost of loading a package is 1 action.
    - The cost of unloading a package is 1 action.
    - The heuristic ignores the initial location of vehicles if a package is
      on the ground and needs a vehicle; it assumes a vehicle is available
      at the package's location when needed (a common relaxation).

    # Heuristic Initialization
    - Extract the goal locations for each package from the task goals.
    - Build a graph representing the road network from the static facts.
    - Compute all-pairs shortest path distances between locations using BFS.
    - Identify vehicles from static facts (those with capacity).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the goal location for each package that needs to be transported.
    2. For each such package:
       a. Determine its current status: Is it on the ground at some location, or is it inside a vehicle?
       b. If it's inside a vehicle, find the current location of that vehicle. This vehicle's location is the package's effective current location.
       c. Let the package's effective current location be `current_loc`.
       d. Let the package's goal location be `goal_loc`.
       e. If the package is on the ground at `goal_loc`, it has reached its goal for this heuristic; its cost is 0.
       f. If the package is not on the ground at `goal_loc`:
          i. Calculate the shortest path distance `dist` between `current_loc` and `goal_loc` using the precomputed distances. If unreachable, the cost is infinite.
          ii. If the package is currently on the ground at `current_loc`: It needs to be loaded (1 action), the vehicle needs to drive (`dist` actions), and it needs to be unloaded (1 action). Estimated cost for this package: 1 + `dist` + 1 = 2 + `dist`.
          iii. If the package is currently inside a vehicle at `current_loc`: The vehicle needs to drive (`dist` actions), and it needs to be unloaded (1 action). Estimated cost for this package: `dist` + 1.
    3. The total heuristic value is the sum of the estimated costs for all packages that have not yet reached their goal location on the ground.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        building the road network graph, and computing shortest paths.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal locations for each package.
        self.goal_locations = {}
        # Identify packages that have goals
        self.goal_packages = set()
        for goal in self.goals:
            predicate, *args = self._get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location
                self.goal_packages.add(package)

        # Build the road network graph and collect all locations.
        self.graph = {}
        self.locations = set()
        for fact in self.static:
            if self._match(fact, "road", "*", "*"):
                _, loc1, loc2 = self._get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append(loc2)

        # Compute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify vehicles (those with capacity)
        self.vehicles = set()
        for fact in self.static:
            if self._match(fact, "capacity", "*", "*"):
                _, vehicle, _ = self._get_parts(fact)
                self.vehicles.add(vehicle)

    def _get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        # Handle potential leading/trailing whitespace and empty facts
        fact = fact.strip()
        if not fact or not fact.startswith('(') or not fact.endswith(')'):
            return []
        return fact[1:-1].split()

    def _match(self, fact, *args):
        """
        Check if a PDDL fact matches a given pattern.

        - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
        - `args`: The expected pattern (wildcards `*` allowed).
        - Returns `True` if the fact matches the pattern, else `False`.
        """
        parts = self._get_parts(fact)
        if len(parts) != len(args):
            return False
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def _bfs(self, start_node):
        """
        Performs Breadth-First Search from a start node to find distances
        to all reachable nodes in the road network graph.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.graph:
                for neighbor in self.graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to move all packages to their goal locations.
        """
        state = node.state

        # Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # Track current locations of vehicles
        vehicle_locations = {}
        for fact in state:
            if self._match(fact, "at", "*", "*"):
                 _, obj, loc = self._get_parts(fact)
                 if obj in self.vehicles:
                     vehicle_locations[obj] = loc

        # Track current status of packages we care about (those with goals)
        package_status = {}
        for package in self.goal_packages:
            package_status[package] = {'loc': None, 'in_vehicle': None} # Default status

        for fact in state:
            parts = self._get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]

            if predicate == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.goal_packages:
                    # Package is on the ground
                    package_status[obj] = {'loc': loc, 'in_vehicle': None}

            elif predicate == "in" and len(parts) == 3:
                 package, vehicle = parts[1], parts[2]
                 if package in self.goal_packages:
                     # Package is in a vehicle. Find vehicle's location.
                     loc_v = vehicle_locations.get(vehicle)
                     if loc_v is not None:
                         package_status[package] = {'loc': loc_v, 'in_vehicle': vehicle}
                     # else: # Vehicle location unknown - inconsistent state? Assume valid states.


        total_cost = 0

        for package, goal_location in self.goal_locations.items():
            status = package_status.get(package)

            # If package status is not found, it might mean the package
            # is not mentioned in 'at' or 'in' facts in this state,
            # which shouldn't happen in a valid state space traversal
            # starting from a valid initial state.
            # If status['loc'] is None, it means 'in' fact was found but vehicle location wasn't.
            # This indicates an issue with state representation or parsing.
            # Let's return infinity as a safe fallback for untracked packages or vehicles.
            if status is None or status['loc'] is None:
                 # print(f"Warning: Package {package} status or location unknown in state.")
                 return float('inf')


            current_loc = status['loc']
            in_vehicle = status['in_vehicle']

            # Check if the package is already at its goal location on the ground
            if current_loc == goal_location and in_vehicle is None:
                continue # This package is done

            # Calculate distance to goal location
            dist = self.distances.get(current_loc, {}).get(goal_location, float('inf'))

            if dist == float('inf'):
                # Goal location is unreachable from the package's current effective location
                # This state is likely not on a path to the goal, or the problem is unsolvable.
                # Return infinity to prune this branch.
                return float('inf')

            # Estimate cost for this package
            package_cost = 0
            if in_vehicle is None:
                # Package is on the ground. Needs load, drive, unload.
                package_cost += 1 # load
                package_cost += dist # drive
                package_cost += 1 # unload
            else:
                # Package is in a vehicle. Needs drive, unload.
                package_cost += dist # drive
                package_cost += 1 # unload

            total_cost += package_cost

        return total_cost
