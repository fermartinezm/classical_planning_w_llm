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


class spanner22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the location of the nuts, the location of the spanners,
    and whether the man is carrying a usable spanner.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner must be picked up before it can be used.
    - The heuristic assumes that the agent will always pick up the closest available spanner.
    - The heuristic does not consider the "usable" status of the spanner.
    - The heuristic assumes that there is always a path between any two locations.

    # Heuristic Initialization
    - Extract the links between locations from the static facts to build a simple connectivity graph.
    - Identify all nuts and their initial locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened.
    2. Determine the man's current location.
    3. If the man is not carrying a spanner, find the closest spanner and estimate the cost to pick it up.
    4. For each loose nut:
       - Calculate the distance from the man's current location to the nut's location.
       - Add the cost of tightening the nut (1 action).
    5. Return the sum of these costs as the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The connectivity graph of locations from static facts.
        - The initial locations of all nuts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build a connectivity graph from the static facts.
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.links:
                    self.links[loc1] = []
                if loc2 not in self.links:
                    self.links[loc2] = []
                self.links[loc1].append(loc2)
                self.links[loc2].append(loc1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify loose nuts.
        loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}

        # Identify the man's location.
        man_location = next((get_parts(fact)[1] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] == 'bob'), None)

        # Check if the man is carrying a spanner.
        carrying_spanner = any(match(fact, "carrying", "bob", "*") for fact in state)

        # If all goals are reached, return 0.
        if all(goal in state for goal in self.goals):
            return 0

        total_cost = 0

        # If not carrying a spanner, estimate the cost to pick one up.
        if not carrying_spanner:
            # Find the closest spanner.
            spanner_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[0] == 'at' and get_parts(fact)[1] != 'bob' and get_parts(fact)[1].startswith('spanner')}
            if spanner_locations:
                # Estimate the cost to reach the closest spanner.
                min_distance = float('inf')
                for spanner, spanner_location in spanner_locations.items():
                    distance = self.shortest_path_length(man_location, spanner_location)
                    if distance < min_distance:
                        min_distance = distance
                total_cost += min_distance + 1  # Walk to spanner + pick up spanner
            else:
                total_cost += 10 #High cost if no spanner available

        # Estimate the cost to tighten each loose nut.
        for nut in loose_nuts:
            nut_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", nut, "*")), None)
            distance = self.shortest_path_length(man_location, nut_location)
            total_cost += distance + 1  # Walk to nut + tighten nut

        return total_cost

    def shortest_path_length(self, start, end):
        """
        Compute the shortest path length between two locations using a simple breadth-first search.
        """
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)
            for neighbor in self.links.get(location, []):
                if neighbor == end:
                    return distance + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # Return infinity if no path exists
