from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class spanner2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the cost
    of walking, picking up a spanner, and tightening nuts.

    # Assumptions:
    - The man can only carry one spanner at a time.
    - A usable spanner must be available to tighten a nut.
    - The heuristic assumes the shortest path between locations is always taken.

    # Heuristic Initialization
    - Identify all loose nuts from the goal conditions.
    - Create a dictionary mapping locations to adjacent locations based on the 'link' predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened.
    2. For each loose nut:
       a. Find the closest usable spanner. This involves:
          i. Calculating the distance (number of 'walk' actions) from the man's current location to the nut's location.
          ii. Calculating the distance from the man's current location to each usable spanner's location.
          iii. Selecting the closest spanner.
       b. Estimate the cost of picking up the spanner (1 action).
       c. Estimate the cost of walking to the nut's location.
       d. Estimate the cost of tightening the nut (1 action).
    3. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Identify loose nuts from the goal conditions.
        self.loose_nuts = set()
        for goal in self.goals:
            if "tightened" in goal:
                nut = goal.split()[1][:-1]  # Extract nut name
                self.loose_nuts.add(nut)

        # Create a dictionary mapping locations to adjacent locations.
        self.links = {}
        for fact in static_facts:
            if "link" in fact:
                parts = fact[1:-1].split()
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 not in self.links:
                    self.links[loc1] = []
                if loc2 not in self.links:
                    self.links[loc2] = []
                self.links[loc1].append(loc2)
                self.links[loc2].append(loc1)

    def __call__(self, node):
        """Estimate the number of actions needed to tighten all loose nuts."""
        state = node.state

        # Check if the goal is already reached
        if self.goal_reached(state):
            return 0

        def match(fact, *args):
            """
            Utility function to check if a PDDL fact matches a given pattern.
            """
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Get the man's current location.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*") and "bob" in fact:
                man_location = fact.split()[2][:-1]
                break

        # Get the spanners' locations and usability.
        spanner_locations = {}
        usable_spanners = set()
        for fact in state:
            if match(fact, "at", "*", "*") and "spanner" in fact:
                spanner = fact.split()[1]
                location = fact.split()[2][:-1]
                spanner_locations[spanner] = location
            if match(fact, "usable", "*"):
                spanner = fact.split()[1][:-1]
                usable_spanners.add(spanner)

        # Get the nuts' locations and status.
        nut_locations = {}
        loose_nuts_in_state = set()
        for fact in state:
            if match(fact, "at", "*", "*") and "nut" in fact:
                nut = fact.split()[1]
                location = fact.split()[2][:-1]
                nut_locations[nut] = location
            if match(fact, "loose", "*"):
                nut = fact.split()[1][:-1]
                loose_nuts_in_state.add(nut)

        # Filter out already tightened nuts from the initial loose_nuts set
        nuts_to_tighten = loose_nuts_in_state

        total_cost = 0
        for nut in nuts_to_tighten:
            nut_location = nut_locations[nut]

            # Find the closest usable spanner.
            closest_spanner = None
            min_distance = float('inf')
            for spanner in usable_spanners:
                if spanner in spanner_locations:
                    spanner_location = spanner_locations[spanner]
                    distance = self.shortest_path_length(man_location, spanner_location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner

            # Estimate the cost of picking up the spanner and walking to the nut.
            if closest_spanner:
                spanner_location = spanner_locations[closest_spanner]
                distance_to_spanner = self.shortest_path_length(man_location, spanner_location)
                distance_to_nut = self.shortest_path_length(spanner_location, nut_location)

                total_cost += 1  # Pickup spanner
                total_cost += distance_to_spanner # Walk to spanner
                total_cost += distance_to_nut # Walk to nut
                total_cost += 1  # Tighten nut
            else:
                # No usable spanner available, return a high cost
                return 1000

        return total_cost

    def shortest_path_length(self, start, end):
        """Calculate the shortest path length between two locations."""
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)
            if location == end:
                return distance

            if location in self.links:
                for neighbor in self.links[location]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))

        return float('inf')  # Return infinity if no path exists

    def goal_reached(self, state):
        """Check if the goal is reached in the given state."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True
