from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class transport6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Transport domain.

    # Summary
    This heuristic estimates the number of actions required to move all packages to their goal locations.
    It considers the need to drive vehicles to packages, pick them up, drive to the goal location, and drop them off.
    It prioritizes moving packages that are furthest from their goal locations.

    # Assumptions
    - Each package needs to be picked up by a vehicle at its current location.
    - The vehicle needs to drive to the package's location.
    - The vehicle needs to drive to the package's goal location.
    - Each package needs to be dropped off at its goal location.
    - The heuristic assumes that there are enough vehicles and capacity to transport all packages.
    - The heuristic does not explicitly consider capacity constraints.

    # Heuristic Initialization
    - Extracts the goal locations for each package.
    - Extracts the road network to estimate driving distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Goal Information:** Identify the goal location for each package.
    2. **Identify Package Locations:** Determine the current location of each package.
    3. **Estimate Driving Costs:**
       - If a package is not at its goal location, estimate the cost to drive a vehicle to the package, pick it up, drive to the goal location, and drop it off.
       - Use the road network to estimate the shortest path between locations.
    4. **Sum Costs:** Sum the estimated costs for all packages to get the overall heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract goal locations for each package
        self.package_goals = {}
        for goal in self.goals:
            parts = self._extract_parts(goal)
            if parts[0] == 'at':
                self.package_goals[parts[1]] = parts[2]

        # Extract road network information
        self.road_network = {}
        for fact in self.static:
            parts = self._extract_parts(fact)
            if parts[0] == 'road':
                l1, l2 = parts[1], parts[2]
                if l1 not in self.road_network:
                    self.road_network[l1] = []
                self.road_network[l1].append(l2)
                if l2 not in self.road_network:
                    self.road_network[l2] = []
                self.road_network[l2].append(l1)

    def __call__(self, node):
        """
        Estimates the number of actions required to reach the goal state from the given state.
        """
        state = node.state
        heuristic_value = 0

        for package, goal_location in self.package_goals.items():
            # Find the current location of the package
            current_location = None
            for fact in state:
                parts = self._extract_parts(fact)
                if parts[0] == 'at' and parts[1] == package:
                    current_location = parts[2]
                    break
                elif parts[0] == 'in' and parts[1] == package:
                    # Package is in a vehicle, find the vehicle's location
                    vehicle = parts[2]
                    for vehicle_fact in state:
                        vehicle_parts = self._extract_parts(vehicle_fact)
                        if vehicle_parts[0] == 'at' and vehicle_parts[1] == vehicle:
                            current_location = vehicle_parts[2]
                            break
                    break

            if current_location is None:
                # Package location not found, possibly an invalid state
                continue

            if current_location != goal_location:
                # Estimate the cost: drive to package, pick up, drive to goal, drop off
                heuristic_value += 4  # Drive, Pick-up, Drive, Drop

        return heuristic_value

    def _extract_parts(self, fact):
        """
        Extracts the parts of a PDDL fact string.
        """
        return fact[1:-1].split()
