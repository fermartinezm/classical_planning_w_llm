from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions for parsing PDDL facts represented as strings
def get_parts(fact):
    """Extract parts of a PDDL fact string."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Transport domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by summing up
    the minimum number of actions required for each package that is not yet
    at its goal location. It calculates this cost based on the shortest path
    distance between the package's current location (or the vehicle's location
    if the package is in a vehicle) and its goal location, plus a fixed cost
    for pick-up and drop actions.

    Assumptions:
    - The heuristic ignores vehicle availability and capacity constraints. It
      assumes that a suitable vehicle is always available where needed and
      has sufficient capacity to transport the package.
    - Roads are assumed to be bidirectional based on common PDDL conventions
      and example instances.
    - The cost of each action (drive, pick-up, drop) is assumed to be 1.

    Heuristic Initialization:
    In the constructor (`__init__`), the heuristic performs the following steps:
    1.  Collects all unique location objects mentioned in the static facts
        (specifically, in `road` predicates) and the goal facts (`at` predicates).
    2.  Builds an adjacency list representation of the road network graph
        from the `road` predicates, assuming bidirectional connections.
    3.  Computes the shortest path distance (in terms of drive actions)
        between all pairs of locations using Breadth-First Search (BFS).
        These distances are stored in a dictionary `self.dist`. Unreachable
        locations will effectively have an infinite distance.
    4.  Stores the goal location for each package by parsing the `at`
        predicates in the task's goal state.

    Step-By-Step Thinking for Computing Heuristic:
    In the `__call__` method for a given state:
    1.  It first identifies the current location of every locatable object
        (packages and vehicles) that is `at` a location, and which package
        is `in` which vehicle, by parsing the state facts.
    2.  It initializes a total cost `total_cost` to 0.
    3.  It iterates through each package that has a specified goal location.
    4.  For each package, it checks if the package is already at its goal location.
        If yes, it contributes 0 to the total cost.
    5.  If the package is not at its goal:
        a.  It determines the package's current status: either `at` a specific
            location or `in` a vehicle.
        b.  If the package is `at` a `current_location`:
            -   The estimated cost for this package is the shortest path distance
                from `current_location` to the package's `goal_location`, plus
                2 actions (1 for pick-up and 1 for drop).
            -   If the `goal_location` is unreachable from `current_location`
                via the road network, the heuristic returns `float('inf')`
                immediately, as the state is likely unsolvable.
        c.  If the package is `in` a `vehicle`:
            -   It finds the current `vehicle_location`.
            -   The estimated cost for this package is the shortest path distance
                from `vehicle_location` to the package's `goal_location`, plus
                1 action (for drop). The pick-up action is considered already done.
            -   If the `goal_location` is unreachable from `vehicle_location`,
                the heuristic returns `float('inf')`.
    6.  The estimated cost for the package is added to the `total_cost`.
    7.  After processing all packages with goals, the accumulated `total_cost`
        is returned as the heuristic value for the state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-calculating shortest paths
        between all locations and storing goal locations for packages.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        static_facts = task.static

        # Heuristic Initialization

        # 1. Collect all locations mentioned in the problem
        self.all_locations = set()
        self.road_network = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "road":
                l1, l2 = parts[1], parts[2]
                self.all_locations.add(l1)
                self.all_locations.add(l2)
                self.road_network.setdefault(l1, []).append(l2)
                self.road_network.setdefault(l2, []).append(l1) # Assuming bidirectional roads

        # Add locations from goals that might not be in road network (e.g., isolated)
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                # Goal is (at package location)
                goal_loc = parts[2]
                self.all_locations.add(goal_loc)
                self.road_network.setdefault(goal_loc, []) # Ensure goal loc is in network dict even if isolated

        # 2. Compute all-pairs shortest paths between locations using BFS
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, d = queue.popleft()
                self.dist[start_loc][current_loc] = d

                # Handle locations that might be in all_locations but not in road_network keys
                # (i.e., isolated locations)
                if current_loc in self.road_network:
                    for neighbor in self.road_network[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

        # 3. Store goal locations for packages
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "at":
                # Goal is (at package location)
                package, location = parts[1], parts[2]
                self.goal_locations[package] = location

        # Note: Capacity information is ignored in this heuristic for simplicity
        # and efficiency, assuming vehicles can carry packages needed for the goal.
        # This is a relaxation.

    def __call__(self, node):
        """
        Computes the heuristic value for a given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated number of actions to reach a goal state, or
            float('inf') if a package needs to reach an unreachable location.
        """
        state = node.state

        # Step-By-Step Thinking for Computing Heuristic

        # 1. Identify current locations of packages and vehicles
        current_locations = {} # Maps locatable (package or vehicle) to location
        package_in_vehicle = {} # Maps package to vehicle

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                # Fact is (at obj loc)
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in":
                # Fact is (in package vehicle)
                package, vehicle = parts[1], parts[2]
                package_in_vehicle[package] = vehicle

        # 2. Calculate the total heuristic cost
        total_cost = 0

        # Iterate through each package that has a goal location
        for package, goal_location in self.goal_locations.items():

            # Check if the package is already at its goal location
            if package in current_locations and current_locations[package] == goal_location:
                # Package is already at the goal, contributes 0 to heuristic
                continue

            # Package is not at the goal. Determine its current status.
            current_location = None
            is_in_vehicle = False
            vehicle_location = None

            if package in current_locations:
                # Package is at a location (at package current_loc)
                current_location = current_locations[package]
                is_in_vehicle = False
            elif package in package_in_vehicle:
                # Package is in a vehicle (in package vehicle)
                vehicle = package_in_vehicle[package]
                is_in_vehicle = True
                # Need the vehicle's location
                if vehicle in current_locations:
                     vehicle_location = current_locations[vehicle]
                else:
                     # This state shouldn't happen in a valid problem (vehicle must be somewhere)
                     # but handle defensively. If vehicle location is unknown, assume unreachable.
                     return float('inf')

            # If package location is still unknown, it's an invalid state for this heuristic
            # This case should ideally not be reached in a valid state representation
            if current_location is None and not is_in_vehicle:
                 return float('inf')


            # Estimate cost for this package

            if is_in_vehicle:
                # Package is in a vehicle at vehicle_location
                start_loc = vehicle_location
                # Cost = Drive from vehicle_location to goal_location + Drop
                # Drive cost is shortest path distance
                # Use .get() with default float('inf') to handle cases where start_loc or goal_location
                # are not in the pre-calculated distance map (e.g., isolated locations not connected
                # to anything else, although our BFS covers all known locations).
                distance = self.dist.get(start_loc, {}).get(goal_location, float('inf'))

                if distance == float('inf'):
                    # Vehicle cannot reach the goal location from its current location
                    return float('inf')

                # 1 action for drop
                cost_for_package = distance + 1
                total_cost += cost_for_package

            else:
                # Package is at current_location
                start_loc = current_location
                # Cost = Pick-up + Drive from current_location to goal_location + Drop
                # Drive cost is shortest path distance
                distance = self.dist.get(start_loc, {}).get(goal_location, float('inf'))

                if distance == float('inf'):
                    # Package's current location cannot reach the goal location
                    return float('inf')

                # 1 action for pick-up, 1 action for drop
                cost_for_package = distance + 2
                total_cost += cost_for_package

        # The total cost is the sum of estimated costs for all packages not at their goal.
        return total_cost
