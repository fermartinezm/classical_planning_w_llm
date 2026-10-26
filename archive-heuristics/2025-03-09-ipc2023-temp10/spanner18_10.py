from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at nut1 gate)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner18Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the cost
    of walking, picking up a spanner, and tightening nuts.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner must be at the same location as the man and the nut to tighten it.
    - A usable spanner must be carried by the man to tighten a nut.
    - The heuristic assumes that each tightening action makes the spanner unusable.

    # Heuristic Initialization
    - Identify the locations of all spanners and nuts.
    - Determine which spanners are usable.
    - Store the link information between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state. If so, return 0.
    2. Identify the man's current location.
    3. Identify all loose nuts.
    4. For each loose nut:
       a. Find the closest usable spanner. This involves calculating the shortest path
          from the man's current location to the spanner's location and then from the
          spanner's location to the nut's location.
       b. Estimate the cost of picking up the spanner (1 action).
       c. Estimate the cost of tightening the nut (1 action).
       d. Sum the costs for all loose nuts.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting relevant information from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information to calculate shortest paths.
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 not in self.links:
                    self.links[l1] = []
                self.links[l1].append(l2)
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l2].append(l1)

        # Identify all locations. This is needed for the shortest path calculation.
        self.locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                self.locations.add(get_parts(fact)[1])
                self.locations.add(get_parts(fact)[2])

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the given state."""
        state = node.state

        # If the goal is reached, the heuristic value is 0.
        if self.goal_reached(state):
            return 0

        # Identify the man's location.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*") and "bob" in fact:
                man_location = get_parts(fact)[2]
                break

        # Identify loose nuts.
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.append(get_parts(fact)[1])

        # Identify usable spanners and their locations.
        usable_spanners = {}
        for fact in state:
            if match(fact, "at", "*", "*") and "spanner" in fact and match(fact, "usable", get_parts(fact)[1]):
                spanner = get_parts(fact)[1]
                if match(fact, "usable", spanner):
                    usable_spanners[spanner] = get_parts(fact)[2]

        total_cost = 0
        for nut in loose_nuts:
            # Find the closest usable spanner.
            closest_spanner = None
            min_distance = float('inf')

            for spanner, spanner_location in usable_spanners.items():
                # Find the location of the nut
                nut_location = None
                for fact in state:
                    if match(fact, "at", nut, "*"):
                        nut_location = get_parts(fact)[2]
                        break

                # Calculate the distance from the man to the spanner and then to the nut.
                distance = self.shortest_path(man_location, spanner_location) + self.shortest_path(spanner_location, nut_location)

                if distance < min_distance:
                    min_distance = distance
                    closest_spanner = spanner

            # Estimate the cost of picking up the spanner and tightening the nut.
            if closest_spanner:
                total_cost += min_distance + 2  # Walk + pickup + tighten

        return total_cost

    def shortest_path(self, start, end):
        """Calculate the shortest path between two locations using BFS."""
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

        return float('inf')  # Return infinity if no path is found

    def goal_reached(self, state):
        """Check if the goal state has been reached."""
        for goal in self.goals:
            if goal not in state:
                return False
        return True
