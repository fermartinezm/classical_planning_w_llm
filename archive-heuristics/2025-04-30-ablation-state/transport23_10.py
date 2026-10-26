import math
from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the total number of actions required to move all
    packages to their goal locations. It sums the estimated cost for each
    package independently, ignoring vehicle capacity and availability constraints.
    The estimated cost for a single package includes the necessary pick-up/drop
    actions and the shortest path distance (in terms of drive actions) between
    its current effective location and its goal location.

    # Assumptions
    - Each package needs to reach a specific goal location.
    - Vehicles are available when needed (capacity and location permitting).
    - The cost of each action (drive, pick-up, drop) is 1.
    - The road network is static and bidirectional.
    - Vehicle names start with 'v'.

    # Heuristic Initialization
    - Extracts the goal location for each package from the task goals.
    - Builds the road network graph from static 'road' facts.
    - Computes all-pairs shortest paths on the road network using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current status (location or inside a vehicle) for every package and vehicle by examining the state facts.
    2. Initialize the total heuristic cost to 0.
    3. For each package that has a goal location defined in the task:
        a. Check if the package is already at its goal location (on the ground) in the current state. If the fact `(at package goal_location)` exists in the state, the cost for this package is 0, and we continue to the next package.
        b. If the package is not at its goal location:
            i. Determine the package's effective current location. This is the location where a vehicle would need to be to interact with the package. If the package is on the ground at location L (`(at package L)`), the effective location is L. If the package is inside a vehicle V (`(in package V)`) and the vehicle V is at location L (`(at V L)`), the effective location is L.
            ii. Calculate the minimum number of drive actions required for a vehicle to move from the effective current location to the package's goal location. This is the shortest path distance between these two locations in the precomputed road network graph. If the goal is unreachable, the heuristic returns infinity.
            iii. Add this drive cost to the total cost.
            iv. Add the cost of necessary pick-up and drop actions:
                - If the package is currently on the ground (not in a vehicle), it needs a pick-up action (cost +1).
                - The package always needs a drop action at the goal location (cost +1).
    4. The total heuristic value is the sum of costs calculated for all packages not yet at their final goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and computing
        shortest paths on the road network.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each package.
        self.package_goals = {}
        for goal in self.goals:
            # Goal facts are typically in the form (at package location)
            predicate, *args = self._get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

        # Build the road graph and collect all locations.
        self.road_graph = {}
        self.locations = set()
        for fact in static_facts:
            if self._match(fact, "road", "*", "*"):
                _, loc1, loc2 = self._get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.road_graph.setdefault(loc1, set()).add(loc2)
                self.road_graph.setdefault(loc2, set()).add(loc1) # Roads are bidirectional

        # Compute all-pairs shortest paths using BFS.
        self.shortest_paths = {}
        for start_loc in self.locations:
            self.shortest_paths[start_loc] = self._bfs(start_loc)

    @staticmethod
    def _get_parts(fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    @staticmethod
    def _match(fact, *args):
        """
        Check if a PDDL fact matches a given pattern.
        """
        parts = transportHeuristic._get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def _bfs(self, start_loc):
        """
        Performs BFS from a start location to find shortest paths to all other
        locations in the road graph.
        """
        distances = {loc: math.inf for loc in self.locations}
        distances[start_loc] = 0
        queue = deque([start_loc])

        while queue:
            u = queue.popleft()
            # Ensure u is in the graph keys before accessing
            if u in self.road_graph:
                for v in self.road_graph[u]:
                    if distances[v] == math.inf:
                        distances[v] = distances[u] + 1
                        queue.append(v)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state  # Current world state (frozenset of facts).

        # Track where packages and vehicles are currently located or contained.
        current_status = {} # Maps object name to its location or the vehicle it's in
        vehicle_locations = {} # Maps vehicle name to its location

        for fact in state:
            parts = self._get_parts(fact)
            predicate = parts[0]
            if predicate == "at":
                obj, loc = parts[1], parts[2]
                current_status[obj] = loc
                # Assuming vehicle names start with 'v' based on examples
                # This is a heuristic assumption based on common PDDL naming.
                # A more robust approach would involve parsing object types from the domain.
                if obj.startswith('v'):
                     vehicle_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = parts[1], parts[2]
                current_status[package] = vehicle

        total_cost = 0  # Initialize action cost counter.

        # Iterate through each package that needs to reach a goal location.
        for package, goal_location in self.package_goals.items():
            # Check if the package is already at its goal location on the ground.
            # This is the final state for the package.
            if f"(at {package} {goal_location})" in state:
                continue # Package is already at goal, cost is 0 for this package.

            # Package is not at its goal location. Calculate cost.
            current_status_pkg = current_status.get(package)

            # If package status is unknown, it might indicate an invalid state
            # or a package not relevant to the goals. Return inf.
            if current_status_pkg is None:
                 return math.inf

            # Determine the effective current location of the package.
            # This is the location where a vehicle would need to be to interact with it.
            L_eff_current = None
            needs_pickup = False # Flag to indicate if a pick-up action is needed

            if current_status_pkg in self.locations:
                # Package is on the ground at current_status_pkg
                L_eff_current = current_status_pkg
                needs_pickup = True
            else:
                # Package is inside a vehicle (current_status_pkg is the vehicle name)
                vehicle_name = current_status_pkg
                # Find the location of the vehicle
                L_eff_current = vehicle_locations.get(vehicle_name)
                # If vehicle location is unknown, this is an invalid state.
                if L_eff_current is None:
                    return math.inf

            # Add the cost of driving from the effective current location to the goal location.
            # Ensure both locations are in the shortest_paths dictionary.
            if L_eff_current in self.shortest_paths and goal_location in self.shortest_paths[L_eff_current]:
                 drive_cost = self.shortest_paths[L_eff_current][goal_location]
                 if drive_cost == math.inf:
                     # Goal location is unreachable from the current location.
                     return math.inf
                 total_cost += drive_cost
            else:
                 # Should not happen if locations were correctly extracted and BFS run
                 # on all extracted locations.
                 return math.inf # Indicate unreachable

            # Add the cost of necessary pick-up and drop actions.
            if needs_pickup:
                total_cost += 1 # pick-up action
            total_cost += 1 # drop action

        return total_cost
