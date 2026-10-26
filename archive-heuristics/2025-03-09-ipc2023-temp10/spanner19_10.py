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


class spanner19Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the spanner locations, the nut locations,
    and whether the man is carrying a spanner. It prioritizes moving the man to
    a location with both a nut and a spanner, picking up the spanner, and then
    tightening the nut.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner can only be used once.
    - The heuristic assumes that the shortest path to a location containing both
      a nut and a spanner is the best approach.

    # Heuristic Initialization
    - Extract the locations of all nuts, spanners, and the man from the initial state.
    - Identify the links between locations from the static facts.
    - Store the goal conditions (tightened nuts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts in the current state.
    2. Determine the man's current location and whether he is carrying a spanner.
    3. For each loose nut:
       a. Find a usable spanner.
       b. Calculate the cost of moving the man to the nut's location.
       c. If the man is not carrying a spanner, calculate the cost of moving to a
          spanner location and picking it up.
       d. Estimate the cost of tightening the nut (1 action).
    4. Sum the costs for all loose nuts to get the total heuristic value.
    5. If the goal is already reached, return 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                if l1 not in self.links:
                    self.links[l1] = []
                self.links[l1].append(l2)
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l2].append(l1)

    def __call__(self, node):
        """Estimate the number of actions needed to tighten all loose nuts."""
        state = node.state

        # Check if the goal is already reached
        if all(goal in state for goal in self.goals):
            return 0

        # Identify loose nuts
        loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.add(get_parts(fact)[1])

        # Determine man's location and whether he is carrying a spanner
        man_location = None
        carrying_spanner = None
        for fact in state:
            if match(fact, "at", "*", "*") and "bob" in fact:
                man_location = get_parts(fact)[2]
            if match(fact, "carrying", "*", "*") and "bob" in fact:
                carrying_spanner = get_parts(fact)[2]

        total_cost = 0
        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            # Find a usable spanner
            usable_spanner = None
            for fact in state:
                if match(fact, "usable", "*"):
                    usable_spanner = get_parts(fact)[1]
                    break

            if usable_spanner is None:
                # No usable spanners available, problem is unsolvable
                return float('inf')

            spanner_location = None
            for fact in state:
                if match(fact, "at", usable_spanner, "*"):
                    spanner_location = get_parts(fact)[2]
                    break

            # Estimate cost of moving the man to the nut's location
            if man_location != nut_location:
                # Simple cost: 1 action per link
                cost_to_nut = self.shortest_path_cost(man_location, nut_location)
                total_cost += cost_to_nut

            # Estimate cost of picking up the spanner if not carrying one
            if carrying_spanner is None:
                if man_location != spanner_location:
                    cost_to_spanner = self.shortest_path_cost(man_location, spanner_location)
                    total_cost += cost_to_spanner
                total_cost += 1  # Pickup spanner

            total_cost += 1  # Tighten nut

        return total_cost

    def shortest_path_cost(self, start, end):
        """
        Calculate the shortest path cost between two locations using BFS.
        """
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, cost = queue.pop(0)
            if location == end:
                return cost

            if location in self.links:
                for neighbor in self.links[location]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, cost + 1))

        return float('inf')  # No path found
