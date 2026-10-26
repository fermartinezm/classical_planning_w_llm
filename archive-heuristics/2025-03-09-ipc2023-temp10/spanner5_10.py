from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(link loc1 loc2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the cost
    of walking, picking up a spanner, and tightening nuts.

    # Assumptions
    - The heuristic assumes that the man needs to be at the same location as the nut
      and have a usable spanner to tighten it.
    - It assumes that the man can carry only one spanner at a time.
    - It prioritizes picking up a spanner before moving to tighten nuts.

    # Heuristic Initialization
    - Extract the link information between locations from the static facts.
    - Store the locations of usable spanners.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts and their locations.
    2. Determine the man's current location and whether he is carrying a spanner.
    3. If the man is not carrying a spanner, find the closest usable spanner and
       estimate the cost of walking to it and picking it up.
    4. For each loose nut:
       - Estimate the cost of walking from the man's current location to the nut's location.
       - Add the cost of tightening the nut.
    5. Sum up the costs for all loose nuts and the cost of acquiring a spanner (if needed).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Link information between locations.
        - Locations of usable spanners.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract link information
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                loc1 = parts[1]
                loc2 = parts[2]
                if loc1 not in self.links:
                    self.links[loc1] = []
                self.links[loc1].append(loc2)
                if loc2 not in self.links:
                    self.links[loc2] = []
                self.links[loc2].append(loc1)

        # Extract locations of usable spanners
        self.usable_spanner_locations = {}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loose_nuts.append(nut)
                for f in state:
                    if match(f, "at", nut, "*"):
                        nut_locations[nut] = get_parts(f)[2]
                        break

        # Determine the man's current location and whether he is carrying a spanner
        man_location = None
        carrying_spanner = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":
                    man_location = parts[2]
            if match(fact, "carrying", "bob", "*"):
                carrying_spanner = get_parts(fact)[2]

        # Get usable spanners and their locations
        usable_spanners = []
        spanner_locations = {}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.append(spanner)
                for f in state:
                    if match(f, "at", spanner, "*"):
                        spanner_locations[spanner] = get_parts(f)[2]
                        break

        # If already in goal state, return 0
        goal_achieved = True
        for goal in self.goals:
            if goal not in state:
                goal_achieved = False
                break
        if goal_achieved:
            return 0

        total_cost = 0

        # If not carrying a spanner, find the closest usable spanner and pick it up
        if carrying_spanner is None:
            min_distance = float('inf')
            closest_spanner = None
            for spanner in usable_spanners:
                if spanner in spanner_locations:
                    spanner_location = spanner_locations[spanner]
                    distance = self.estimate_distance(man_location, spanner_location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner
            if closest_spanner:
                total_cost += min_distance + 1  # Walk + pickup
                man_location = spanner_locations[closest_spanner] # update man location

        # For each loose nut, estimate the cost of walking and tightening
        for nut in loose_nuts:
            nut_location = nut_locations[nut]
            distance = self.estimate_distance(man_location, nut_location)
            total_cost += distance + 1  # Walk + tighten

        return total_cost

    def estimate_distance(self, start, end):
        """
        Estimate the distance between two locations based on the link information.
        This is a simple breadth-first search to find the shortest path.
        """
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

        return float('inf')  # No path found
