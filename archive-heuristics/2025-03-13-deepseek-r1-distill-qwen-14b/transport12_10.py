from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations. It considers both intra-city and inter-city transport, including loading, unloading, and driving actions.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move along roads between connected locations.
    - The goal is to transport each package to a specific target location.
    - If a package is already at its goal location, no actions are needed.

    # Heuristic Initialization
    - Extracts goal locations for each package from the task's goals.
    - Constructs a graph of road connections from static facts.
    - Maps each vehicle to its current location.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, determine its current location and whether it's inside a vehicle.
    2. Identify the package's goal location.
    3. If the package is already at the goal, no actions are needed.
    4. If the package is in a vehicle, determine the vehicle's current location.
    5. Calculate the minimal steps required to move the package from its current location to the goal:
       a. If in the same city, estimate driving steps.
       b. If in different cities, estimate driving to an airport, flying, and then driving to the destination.
    6. Sum the estimated actions for all packages to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with task information and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build road graph from static facts
        self.roads = {}
        for fact in self.static:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 not in self.roads:
                    self.roads[l1] = []
                self.roads[l1].append(l2)
                if l2 not in self.roads:
                    self.roads[l2] = []
                self.roads[l2].append(l1)

        # Map packages to their goal locations
        self.package_goals = {}
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                package, loc = get_parts(goal)[1], get_parts(goal)[2]
                self.package_goals[package] = loc

    def __call__(self, node):
        """Estimate the minimal number of actions to reach the goal state."""
        state = node.state
        current_locations = {}
        in_vehicle = {}

        # Extract current locations of all objects
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                in_vehicle[package] = vehicle

        total_actions = 0

        # Process each package
        for package, goal_loc in self.package_goals.items():
            if package not in current_locations:
                continue  # Package is already at goal

            current_loc = current_locations[package]
            if current_loc == goal_loc:
                continue  # No action needed

            # Determine if package is in a vehicle
            if package in in_vehicle:
                vehicle = in_vehicle[package]
                # Find vehicle's current location
                for fact in state:
                    if match(fact, "at", vehicle, "*"):
                        vehicle_loc = get_parts(fact)[2]
                        break
                loc = vehicle_loc
            else:
                loc = current_loc

            # Check if in same city
            same_city = False
            if loc in self.roads and goal_loc in self.roads[loc]:
                same_city = True

            if same_city:
                # Intra-city transport: drive directly
                total_actions += 2  # drive to location and unload
            else:
                # Inter-city transport: requires driving to airport, flying, and driving to destination
                # Find nearest airport (simplified as one action)
                total_actions += 3  # drive to airport, fly, drive from airport

        return total_actions

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))
