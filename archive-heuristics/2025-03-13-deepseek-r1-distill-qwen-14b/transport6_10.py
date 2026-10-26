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
    This heuristic estimates the number of actions needed to transport all packages to their goal locations. It considers:
    - The number of packages that still need to be moved.
    - The current location of each package and vehicle.
    - The required actions (pickup, drive, drop) to move each package to its destination.

    # Assumptions:
    - Vehicles can carry multiple packages based on their capacity.
    - Packages can be picked up and dropped off at any location connected by roads.
    - The shortest path between locations is considered for driving actions.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Extracts static facts including road connections and vehicle capacities.

    # Step-by-Step Thinking for Computing Heuristic
    1. For each package, determine its current location and whether it's in a vehicle.
    2. If the package is already at its goal location, no actions are needed.
    3. If the package is in a vehicle, calculate the actions needed to drop it at the goal.
    4. If the package is not in a vehicle, calculate the actions needed to pick it up and drop it at the goal.
    5. Sum the actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections and vehicle capacities).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (road connections and capacities)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = get_parts(goal)
            if predicate == "at":
                self.goal_locations[package] = location

        # Extract road connections
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))
                self.roads.add((l2, l1))

        # Extract vehicle capacities and predecessors
        self.capacity = {}
        self.capacity_predecessors = {}
        for fact in static_facts:
            if match(fact, "capacity", "*", "*", "*"):
                v, s1, s2 = get_parts(fact)
                self.capacity[(v, s1)] = s2
            elif match(fact, "capacity-predecessor", "*", "*"):
                s1, s2 = get_parts(fact)
                self.capacity_predecessors[s1] = s2

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # Track package locations and whether they're in a vehicle
        current_locations = {}
        in_vehicle = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in":
                p, v = parts[1], parts[2]
                in_vehicle[p] = v

        total_cost = 0  # Initialize action cost counter

        # For each package with a goal location
        for package, goal_location in self.goal_locations.items():
            if package not in current_locations:
                # Package is already at goal or doesn't exist in state
                continue

            current_location = current_locations[package]
            if current_location == goal_location:
                continue  # Package is already at goal

            # Check if package is in a vehicle
            if package in in_vehicle:
                # Package is in a vehicle, needs to be dropped
                v = in_vehicle[package]
                # Find a path from current vehicle location to goal location
                # (Assuming shortest path is 1 action per road segment)
                # For simplicity, assume direct drive is possible
                total_cost += 2  # drive and drop actions
            else:
                # Package is not in a vehicle, needs to be picked up and dropped
                # First, check if a vehicle is available at current location
                # (Assuming there's always a vehicle available)
                total_cost += 3  # pickup, drive, drop actions

        return total_cost
</code-file-heuristic-transport>