from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class TransportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their target locations. It considers the shortest path between locations and the need for loading and unloading packages.

    # Assumptions:
    - Packages can be either on the ground or inside a vehicle.
    - Vehicles can move between connected locations.
    - Loading and unloading a package each require one action.
    - The shortest path between any two locations is used to minimize driving actions.

    # Heuristic Initialization
    - Extracts static facts to build a graph of road connections.
    - Maps each package to its target location.

    # Step-By-Step Thinking for Computing Heuristic
    1. Build a graph of road connections from static facts.
    2. For each package, determine its current location and whether it's in a vehicle.
    3. Calculate the shortest path from the package's current location to its target.
    4. Estimate the number of actions as:
       - 2 actions per segment of the path (one to load/unload, one to drive).
       - Adjust if the vehicle is already at the package's location.
    5. Sum the actions for all packages, considering shared vehicle trips.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts and goal locations."""
        self.goals = task.goals
        static_facts = task.static

        # Build road graph from static facts
        self.roads = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.roads:
                    self.roads[loc1] = []
                self.roads[loc1].append(loc2)
                if loc2 not in self.roads:
                    self.roads[loc2] = []
                self.roads[loc2].append(loc1)

        # Map each package to its target location
        self.package_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.package_goals[package] = location

    def __call__(self, node):
        """Compute the heuristic value for the current state."""
        state = node.state

        # Extract current locations of packages and vehicles
        current_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                current_locations[obj] = loc
            elif parts[0] == "in":
                package, vehicle = parts[1], parts[2]
                current_locations[package] = vehicle

        total_actions = 0

        # For each package, calculate required actions
        for package, goal_loc in self.package_goals.items():
            # Get current location of the package
            current_loc = current_locations.get(package, None)
            if current_loc is None:
                continue  # Package not present in state (shouldn't happen)

            # If package is already at goal, no actions needed
            if current_loc == goal_loc:
                continue

            # Determine if package is in a vehicle
            in_vehicle = current_loc not in self.roads

            # Get the vehicle's current location if in a vehicle
            if in_vehicle:
                vehicle = current_loc
                vehicle_loc = current_locations.get(vehicle, None)
                if vehicle_loc is None:
                    vehicle_loc = vehicle  # Assume vehicle is at its own location
            else:
                vehicle_loc = current_loc

            # Find the shortest path from vehicle's current location to goal location
            path = self.shortest_path(vehicle_loc, goal_loc)
            if path is None:
                continue  # No path exists (shouldn't happen in solvable instances)

            # Calculate the number of actions needed
            num_segments = len(path) - 1
            total_actions += 2 * num_segments  # 2 actions per segment (load/unload + drive)

            # If the vehicle is already at the package's location, save one action
            if vehicle_loc == current_loc:
                total_actions -= 1

        return total_actions

    def shortest_path(self, start, end):
        """Return the shortest path using BFS."""
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.roads.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return None  # No path found

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))
