from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class LogisticsHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Logistics domain.

    # Summary
    The heuristic estimates the number of necessary actions (load, unload, and transport) in order to transport each package to its goal based on its current state.

    # Assumptions
    - Packages can be on the ground, in a truck, or in a plane.
    - Trucks move within a city, while planes move between cities.
    - If a package is already at the goal, no extra actions are needed.

    # Heuristic Initialization
    - Extract the goal locations for each package and the static facts (e.g., `in-city` relationships and airport locations) from the task.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information:
    - Identify the current location of every package.
    - Identify whether a package is inside a vehicle (truck or plane), and if so, find the physical location of that vehicle.

    2. Distinguish Between Intra-city and Inter-city Transport:
    - Determine the current city and goal city for each package by checking its location-to-city mapping.
    - If the current city is the same as the goal city, follow the intra-city package movement rules.
    - If the current city is different from the goal city, follow the inter-city package movement rules.

    3. Handle Intra-city Transport:
    - If the package is already at its goal location, no action is required.
    - If the package is in a plane, it must be unloaded.
    - If the package is not in a truck and not already at its goal, it must be loaded into a truck.
    - If the package is in a truck or not yet at its final location, it must be unloaded from the truck at the goal.

    4. Handle Inter-city Transport:
    - Step 1: Move the package to the airport in the current city.
        - If the package is not inside a truck and not at an airport, it must be loaded into a truck.
        - If the package is not at an airport or inside a truck, it must be unloaded from the truck at the airport.
    - Step 2: Fly the package to the destination city.
        - If the package is not in a plane, it must be loaded into a plane.
        - The package must always be unloaded from the plane at the airport of the destination city.
    - Step 3: Move the package from the airport to its final location.
        - If the goal location is not an airport, the package must be loaded into a truck at the airport.
        - Finally, the package must be unloaded from the truck at the goal location.

    5. Summing the Actions:
    - The total heuristic value is the sum of all necessary actions.
    - Loading and unloading costs are counted exactly based on the required actions.
    - Transport movements (trucks or planes) are counted only when necessary.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (`in-city` relationships and airport locations).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Map locations to their respective cities using "in-city" relationships.
        self.location_to_city = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, "in-city", "*", "*")
        }

        # Identify all airport locations.
        self.airports = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "airport", "*")
        }

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:  # Track both direct location and inside vehicle.
                obj, location = args
                current_locations[obj] = location

        total_cost = 0  # Initialize action cost counter.

        for package, goal_location in self.goal_locations.items():
            # Get the current location of the package (could be a city location, truck or plane).
            current_location = current_locations[package]

            # Check if the package is inside a vehicle.
            in_vehicle = current_location not in self.location_to_city

            if in_vehicle:
                # Identify type of vehicle (truck or plane).
                in_plane = current_location.startswith("plane")
                in_truck = current_location.startswith("truck")
                assert in_plane ^ in_truck, f"Invalid state: {current_location}"

                # Retrieve the physical location of the vehicle.
                current_location = current_locations[current_location]
            else:
                in_plane = False
                in_truck = False

            # Get the city of the package's current location and goal.
            current_city = self.location_to_city[current_location]
            goal_city = self.location_to_city[goal_location]

            # Intra-city Transport (Same City)
            if current_city == goal_city:
                if in_plane:
                    total_cost += 1  # Unload from the plane.

                if current_location != goal_location and not in_truck:
                    total_cost += 1  # Load into a truck.

                if current_location != goal_location or in_truck:
                    total_cost += 1  # Unload from the truck.

            # Inter-city Transport (Different Cities)
            else:
                # Step 1: Move to the airport in the current city.
                if current_location not in self.airports and not in_truck:
                    total_cost += 1  # Load into a truck.

                if current_location not in self.airports or in_truck:
                    total_cost += 1  # Unload from the truck at the airport.

                # Step 2: Fly to the destination city.
                if not in_plane:
                    total_cost += 1  # Load into a plane.

                total_cost += 1  # Unload from the plane.

                # Step 3: Transport from airport to the goal (if required).
                if goal_location not in self.airports:
                    total_cost += 1  # Load into a truck at destination airport.
                    total_cost += 1  # Unload at the destination.

        return total_cost
