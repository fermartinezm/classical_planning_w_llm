from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages
    from their current locations to the goal locations using vehicles.

    # Assumptions:
    - Vehicles can carry multiple packages based on their capacity.
    - Packages must be picked up and dropped off at specific locations.
    - Vehicles can move between connected locations via roads.

    # Heuristic Initialization
    - Extracts static facts to build a graph of connected locations.
    - Creates a hierarchy of vehicle capacities based on capacity-predecessor relationships.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each package, check if it is already at the goal location. If yes, no actions are needed.
    2. If a package is not at the goal, determine if it is currently in a vehicle:
       - If in a vehicle, check if the vehicle's capacity allows carrying more packages.
       - If not in a vehicle, the vehicle must drive to the package's location to pick it up.
    3. Calculate the driving actions required to move the vehicle to the package's location and then to the goal location.
    4. Sum the actions required for all packages, considering the vehicle's capacity and the need for multiple trips.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building necessary data structures."""
        # Extract static facts
        static_facts = task.static

        # Build road graph
        self.roads = {}
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if l1 not in self.roads:
                    self.roads[l1] = []
                self.roads[l1].append(l2)
                if l2 not in self.roads:
                    self.roads[l2] = []
                self.roads[l2].append(l1)

        # Build capacity hierarchy
        self.capacity_hierarchy = {}
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                s1, s2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
                if s1 not in self.capacity_hierarchy:
                    self.capacity_hierarchy[s1] = []
                self.capacity_hierarchy[s1].append(s2)

        # Store goal locations for each package
        self.goal_locations = {}
        for goal in task.goals:
            predicate, *args = goal[1:-1].split()
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        """Estimate the minimum number of actions needed to reach the goal state."""
        state = node.state

        def match(fact, *args):
            """Check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Track current locations of packages and vehicles
        current_locations = {}
        for fact in state:
            predicate, *args = fact[1:-1].split()
            if predicate == "at":
                obj, loc = args
                current_locations[obj] = loc
            elif predicate == "in":
                package, vehicle = args
                current_locations[package] = vehicle

        total_cost = 0

        # For each package, determine the required actions
        for package, goal_loc in self.goal_locations.items():
            current_loc = current_locations.get(package, None)
            if current_loc == goal_loc:
                continue  # Package is already at the goal

            # Determine if the package is in a vehicle
            in_vehicle = current_loc not in self.roads
            if in_vehicle:
                # Vehicle's current location
                vehicle = current_loc
                vehicle_loc = current_locations.get(vehicle, None)
                if vehicle_loc is None:
                    continue  # Vehicle not found, invalid state
            else:
                vehicle_loc = None

            # If not in a vehicle, vehicle must drive to pick up the package
            if not in_vehicle:
                # Find a vehicle that can carry the package
                vehicle_found = None
                for obj in current_locations:
                    if obj.startswith('v'):
                        vehicle_found = obj
                        break
                if vehicle_found is None:
                    continue  # No vehicle available, invalid state

                # Vehicle needs to drive to the package's location
                if vehicle_loc != current_loc:
                    # Calculate driving cost
                    if vehicle_loc not in self.roads or current_loc not in self.roads[vehicle_loc]:
                        continue  # No road, invalid state
                    total_cost += 2  # Drive to location and back if needed

                # Pick up the package
                total_cost += 1

                vehicle_loc = current_loc  # Vehicle is now at the package's location

            # Now, vehicle has the package, need to drive to goal location
            if vehicle_loc != goal_loc:
                # Check if vehicle can directly drive to goal
                if vehicle_loc in self.roads and goal_loc in self.roads[vehicle_loc]:
                    total_cost += 2  # Drive to goal and possibly back
                else:
                    # Need to find a path, assume worst case (might need multiple drives)
                    total_cost += 4  # Simplified estimate

            # Drop off the package
            total_cost += 1

        return total_cost
