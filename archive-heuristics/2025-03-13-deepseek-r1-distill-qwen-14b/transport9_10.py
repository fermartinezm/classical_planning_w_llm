<code-file-heuristic-transport>
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at p1 l1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move along roads between connected locations.
    - Each vehicle has a capacity that determines how many packages it can carry.
    - Packages must be picked up from their current location and dropped off at their goal location.

    # Heuristic Initialization
    - Extract the goal locations for each package from the task's goals.
    - Extract static facts including road connections and vehicle capacities.

    # Step-by-Step Thinking for Computing the Heuristic Value
    1. Identify the current location of each package and vehicle.
    2. For each package not yet at its goal location:
        a. If the package is in a vehicle, determine the vehicle's current location.
        b. Calculate the shortest path from the vehicle's location to the package's goal location using available roads.
        c. Estimate the number of actions required to move the package to its goal, considering pickup and drop actions.
    3. Sum the estimated actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections and vehicle capacities).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract road connections from static facts
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))
                self.roads.add((l2, l1))

        # Extract vehicle capacities and their predecessors
        self.capacity_info = {}
        for fact in static_facts:
            if match(fact, "capacity", "*", "*"):
                v, s = get_parts(fact)[1], get_parts(fact)[2]
                self.capacity_info[v] = s
            elif match(fact, "capacity-predecessor", "*", "*"):
                s1, s2 = get_parts(fact)[1], get_parts(fact)[2]
                if s2 not in self.capacity_info:
                    self.capacity_info[s2] = s1

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # Track where packages and vehicles are currently located
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, location = args
                current_locations[obj] = location

        total_cost = 0  # Initialize action cost counter

        # For each package, calculate the required actions
        for package, goal_location in self.goal_locations.items():
            # Skip if package is already at goal
            if package in current_locations and current_locations[package] == goal_location:
                continue

            # Determine if the package is inside a vehicle
            in_vehicle = False
            vehicle = None
            for obj, loc in current_locations.items():
                if match(obj, "*") and match(loc, "*"):
                    if loc.startswith("vehicle"):
                        vehicle = loc
                        in_vehicle = True
                        break

            # If the package is inside a vehicle, find its location
            if in_vehicle:
                vehicle_loc = current_locations.get(vehicle, None)
                if vehicle_loc is None:
                    vehicle_loc = "unknown"
            else:
                vehicle_loc = "unknown"

            # Calculate the shortest path from vehicle's location to goal location
            # (Simplified to count the number of steps as all roads are bidirectional)
            path_length = 0
            locations = [vehicle_loc] if vehicle_loc != "unknown" else [current_locations.get(package, "unknown")]
            visited = set()
            queue = [(current_locations.get(package, "unknown"), 0)]

            while queue:
                current_loc, steps = queue.pop(0)
                if current_loc == goal_location:
                    path_length = steps
                    break
                if current_loc in visited:
                    continue
                visited.add(current_loc)
                for neighbor in [loc for loc, loc2 in self.roads if loc == current_loc]:
                    if neighbor not in visited:
                        queue.append((neighbor, steps + 1))

            # Estimate the number of actions based on the path length
            # Each move between locations costs 1 action (drive)
            # Plus pickup and drop actions
            total_cost += 2 + path_length * 2  # pickup, drive actions, drop

        return total_cost
</code-file-heuristic-transport>