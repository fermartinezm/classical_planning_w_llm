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


class spanner14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the spanner's location, and the nuts' locations,
    estimating the cost of walking, picking up the spanner, and tightening the nuts.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A usable spanner must be carried to tighten a nut.
    - The heuristic assumes that the shortest path to a location is always used.

    # Heuristic Initialization
    - Extract the links between locations from the static facts to build a simple connectivity graph.
    - Identify all nuts that need to be tightened from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state. If so, return 0.
    2. Identify the man's current location.
    3. Identify all loose nuts and their locations.
    4. If the man is not carrying a usable spanner, find the closest usable spanner and estimate the cost to pick it up.
    5. For each loose nut:
       - Calculate the cost to move to the nut's location.
       - Calculate the cost to tighten the nut (which includes the action cost).
    6. Sum up the costs for all loose nuts and the cost to pick up the spanner (if needed).
    7. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build a connectivity graph from the static facts.
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                if l1 not in self.links:
                    self.links[l1] = []
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l1].append(l2)
                self.links[l2].append(l1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the goal is already reached.
        if self.goals <= state:
            return 0

        # Extract information from the current state.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*") and "bob" in fact:
                man_location = get_parts(fact)[2]
                break

        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                nut_location = None
                for f in state:
                    if match(f, "at", "*", "*") and nut in f:
                        nut_location = get_parts(f)[2]
                        break
                loose_nuts.append((nut, nut_location))

        carrying_spanner = False
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                spanner = get_parts(fact)[2]
                for f in state:
                    if match(f, "usable", spanner):
                        carrying_spanner = True
                        break
                break

        # Estimate the cost.
        cost = 0

        # If not carrying a usable spanner, find one and pick it up.
        if not carrying_spanner:
            closest_spanner_location = None
            min_dist = float('inf')
            for fact in state:
                if match(fact, "at", "*", "*") and "spanner" in fact:
                    spanner = get_parts(fact)[1]
                    for f in state:
                        if match(f, "usable", spanner):
                            spanner_location = get_parts(fact)[2]
                            dist = self.shortest_path(man_location, spanner_location)
                            if dist < min_dist:
                                min_dist = dist
                                closest_spanner_location = spanner_location
            if closest_spanner_location:
                cost += min_dist  # Walk to spanner
                cost += 1  # Pick up spanner

        # Tighten each loose nut.
        for nut, nut_location in loose_nuts:
            dist = self.shortest_path(man_location, nut_location)
            cost += dist  # Walk to nut
            cost += 1  # Tighten nut
            man_location = nut_location  # Update man's location

        return cost

    def shortest_path(self, start, end):
        """Compute the shortest path between two locations using BFS."""
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
