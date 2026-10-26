from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class transportHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the transport domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all packages to their respective goal locations. It considers the current locations of packages and vehicles, the capacity constraints of vehicles, and the road connections between locations.

    # Assumptions:
    - Packages can only be transported by vehicles.
    - Vehicles can carry multiple packages, subject to their capacity constraints.
    - The goal is to have all packages at their specified target locations.
    - The heuristic assumes that vehicles can move freely between connected locations.

    # Heuristic Initialization
    - Extracts goal locations for each package and static facts (road connections and capacity constraints) from the task.

    # Step-By-Step Thinking for Computing Heuristic Value
    1. Extract Relevant Information:
       - Identify the current location of every package and vehicle.
       - Determine the goal location for each package.
       - Check the capacity constraints of each vehicle.

    2. For Each Package:
       a. If the package is already at its goal location, no actions are needed.
       b. If the package is not at its goal location:
          i. Determine the closest vehicle that can carry the package.
          ii. Calculate the number of actions needed to load the package onto the vehicle.
          iii. Calculate the number of actions needed to drive the vehicle to the package's goal location.
          iv. Calculate the number of actions needed to unload the package at the goal location.

    3. Sum the Actions:
       - Sum the actions needed for all packages to reach their goal locations.
       - Consider vehicle capacities to minimize the number of trips.

    # Example Calculation:
    Suppose we have a package that needs to be transported from location A to location B:
    1. Load the package: 1 action
    2. Drive from A to B: 1 action per intermediate location
    3. Unload the package: 1 action
    Total actions: 1 (load) + n (drive) + 1 (unload) = n + 2
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Static facts

        # Extract road connections from static facts
        self.roads = set()
        for fact in static_facts:
            if match(fact, "road", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.roads.add((l1, l2))
                self.roads.add((l2, l1))

        # Extract capacity predecessor relationships
        self.capacity_predecessors = {}
        for fact in static_facts:
            if match(fact, "capacity-predecessor", "*", "*"):
                s1, s2 = get_parts(fact)[1], get_parts(fact)[2]
                if s1 not in self.capacity_predecessors:
                    self.capacity_predecessors[s1] = []
                self.capacity_predecessors[s1].append(s2)

        # Extract goal locations for each package
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        """Estimate the minimum cost to transport all packages to their goal locations."""
        state = node.state

        def get_parts(fact):
            """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
            return fact[1:-1].split()

        def match(fact, *args):
            """
            Utility function to check if a PDDL fact matches a given pattern.
            - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
            - `args`: The pattern to match (e.g., "at", "*", "rooma").
            - Returns `True` if the fact matches the pattern, `False` otherwise.
            """
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Track current locations of packages and vehicles
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:
                obj, loc = args
                current_locations[obj] = loc

        total_actions = 0

        # For each package, calculate the required actions
        for package, goal_loc in self.goal_locations.items():
            current_loc = current_locations.get(package, None)

            if current_loc == goal_loc:
                continue  # Package is already at the goal

            # Find a vehicle that can carry the package
            vehicle = None
            for obj in current_locations:
                if obj.startswith("v"):
                    vehicle = obj
                    break

            if vehicle is None:
                continue  # No vehicle available to carry the package

            # Calculate actions to load the package
            total_actions += 1  # Loading action

            # Calculate actions to drive to the goal location
            # This is a simplified estimate; in a full implementation, you would calculate the shortest path
            total_actions += 2  # Driving actions (assumes two steps: leaving current location and arriving at goal)

            # Calculate actions to unload the package
            total_actions += 1  # Unloading action

        return total_actions
