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


class spanner11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the cost
    of walking, picking up a spanner, and tightening nuts.

    # Assumptions
    - The man can carry only one spanner at a time.
    - A spanner must be usable to tighten a nut.
    - The heuristic assumes that the shortest path to a spanner and then to a nut is always taken.

    # Heuristic Initialization
    - Extract the link information to calculate distances between locations.
    - Identify usable spanners.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. For each loose nut:
       a. Find the closest usable spanner.
       b. Estimate the cost to reach the spanner (walk).
       c. Estimate the cost to pick up the spanner.
       d. Estimate the cost to reach the nut (walk).
       e. Estimate the cost to tighten the nut.
    4. Sum the costs for all loose nuts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Link information for distance calculation.
        - Usable spanners.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract link information to build a location graph.
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

        # Identify usable spanners.
        self.usable_spanners = {get_parts(fact)[1] for fact in static_facts if match(fact, "usable", "*")}

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if the goal is already reached.
        if node.task.goal_reached(state):
            return 0

        # Identify the man's current location.
        man_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", "*", "bob")), None)

        # Identify loose nuts and their locations.
        loose_nuts = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                nut_location = next((get_parts(f)[2] for f in state if match(f, "at", nut, "*")), None)
                loose_nuts[nut] = nut_location

        # Identify carried spanners
        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", "bob", "*")}

        total_cost = 0
        for nut, nut_location in loose_nuts.items():
            # Find the closest usable spanner.
            closest_spanner = None
            min_distance = float('inf')

            for fact in state:
                if match(fact, "at", "*", "*"):
                    spanner = get_parts(fact)[1]
                    if spanner in self.usable_spanners:
                        spanner_location = get_parts(fact)[2]
                        distance = self.calculate_distance(man_location, spanner_location)
                        if distance < min_distance:
                            min_distance = distance
                            closest_spanner = spanner
                            closest_spanner_location = spanner_location

            # If the man is not carrying a spanner, estimate the cost to reach and pick up the spanner.
            if not carried_spanners:
                total_cost += min_distance  # Walk to spanner
                total_cost += 1  # Pick up spanner
                man_location = closest_spanner_location
            else:
                # If the man is carrying a spanner, no need to pick up another one
                pass

            # Estimate the cost to reach the nut.
            distance_to_nut = self.calculate_distance(man_location, nut_location)
            total_cost += distance_to_nut  # Walk to nut

            # Estimate the cost to tighten the nut.
            total_cost += 1  # Tighten nut

        return total_cost

    def calculate_distance(self, start, end):
        """Calculate the shortest path distance between two locations using BFS."""
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)

            if location == end:
                return distance

            for neighbor in self.links.get(location, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # Return infinity if no path is found
