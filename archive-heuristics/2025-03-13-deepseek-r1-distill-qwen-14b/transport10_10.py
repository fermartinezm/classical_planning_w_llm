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
    - The number of packages that still need to be transported.
    - The current location of vehicles and packages.
    - The required driving actions between locations.

    # Assumptions:
    - Each vehicle can carry multiple packages, but capacity is limited.
    - Packages must be picked up and dropped off using vehicles.
    - The robot must drive between locations to transport packages.

    # Heuristic Initialization
    - Extracts goal locations for each package.
    - Maps each location to its connected roads.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all packages that are not yet at their goal locations.
    2. For each package:
       a. If it's in a vehicle, check if the vehicle is at the correct location to drop it off.
       b. If it's on the ground, plan to pick it up and drop it off.
    3. Calculate the number of required driving actions between locations.
    4. Sum the actions needed for all packages, considering shared vehicle trips.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (road connections).
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts (road connections)

        # Extract road connections
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                parts = get_parts(fact)
                self.roads.add((parts[1], parts[2]))

        # Map each location to its connected locations
        self.location_map = {}
        for loc1, loc2 in self.roads:
            if loc1 not in self.location_map:
                self.location_map[loc1] = []
            if loc2 not in self.location_map:
                self.location_map[loc2] = []
            self.location_map[loc1].append(loc2)
            self.location_map[loc2].append(loc1)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, package, location = get_parts(goal)
            if predicate == "at":
                self.goal_locations[package] = location

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state

        # Track package locations and vehicle locations
        package_locations = {}
        vehicle_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                if parts[1].startswith("p"):  # Package
                    package_locations[parts[1]] = parts[2]
                elif parts[1].startswith("v"):  # Vehicle
                    vehicle_locations[parts[1]] = parts[2]
            elif parts[0] == "in":  # Package is in a vehicle
                package_locations[parts[1]] = parts[2]

        total_actions = 0

        # For each package that needs to be transported
        for package, goal_loc in self.goal_locations.items():
            current_loc = package_locations.get(package, None)
            if current_loc == goal_loc:
                continue  # Package is already at goal

            # If package is in a vehicle, check if the vehicle is at the correct location
            if current_loc is None:
                # Package is not in any vehicle; need to pick it up
                total_actions += 1  # Pick up action
                total_actions += 1  # Drive to goal location
                total_actions += 1  # Drop off action
            else:
                # Package is in a vehicle; check if the vehicle needs to move
                vehicle = None
                for v, v_loc in vehicle_locations.items():
                    if v_loc == current_loc:
                        vehicle = v
                        break
                if vehicle:
                    # Vehicle is at current location; need to drive to goal
                    total_actions += 1  # Drive action
                    total_actions += 1  # Drop off action
                else:
                    # Need to pick up the package and drive to goal
                    total_actions += 1  # Pick up action
                    total_actions += 1  # Drive to goal location
                    total_actions += 1  # Drop off action

        return total_actions
</code-file-heuristic-transport>