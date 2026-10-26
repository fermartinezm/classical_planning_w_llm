from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(link location1 location2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the location of the nuts, the location of the spanners,
    and whether the man is carrying a usable spanner. It also takes into account the links
    between locations to estimate the walking cost.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A tightened nut remains tightened.
    - A spanner can only be used once.

    # Heuristic Initialization
    - Extract the links between locations from the static facts.
    - Identify all nuts from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened.
    2. Determine the man's current location.
    3. Determine if the man is carrying a usable spanner.
    4. If the man is not carrying a usable spanner, find the closest usable spanner and estimate the cost to pick it up.
    5. For each loose nut, estimate the cost to reach the nut's location, tighten it, and potentially find another spanner.
    6. The total heuristic value is the sum of the costs for picking up a spanner (if needed) and tightening all loose nuts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Links between locations from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract links between locations.
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

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        if self.goal_reached(state):
            return 0

        # Identify loose nuts.
        loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.add(get_parts(fact)[1])

        # Determine the man's location.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*") and "bob" in fact:
                man_location = get_parts(fact)[2]
                break

        # Check if the man is carrying a usable spanner.
        carrying_usable_spanner = False
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                spanner = get_parts(fact)[2]
                if f"(usable {spanner})" in state:
                    carrying_usable_spanner = True
                    break

        total_cost = 0

        # If not carrying a usable spanner, estimate the cost to pick one up.
        if not carrying_usable_spanner:
            closest_spanner_cost = float('inf')
            for fact in state:
                if match(fact, "at", "*", man_location) and "spanner" in fact:
                    spanner = get_parts(fact)[1]
                    if f"(usable {spanner})" in state:
                        closest_spanner_cost = 1  # Picking up the spanner
                        break
                elif match(fact, "at", "*", "*") and "spanner" in fact:
                    spanner_location = get_parts(fact)[2]
                    if f"(usable {get_parts(fact)[1]})" in state:
                        # Estimate walking cost to the spanner
                        if man_location in self.links and spanner_location in self.links:
                            walk_cost = self.shortest_path(man_location, spanner_location)
                            if walk_cost is not None:
                                closest_spanner_cost = min(closest_spanner_cost, walk_cost + 1)  # Walk + pickup
            total_cost += closest_spanner_cost if closest_spanner_cost != float('inf') else 1000

        # Estimate the cost to tighten each loose nut.
        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            if nut_location:
                # Estimate walking cost to the nut
                if man_location in self.links and nut_location in self.links:
                    walk_cost = self.shortest_path(man_location, nut_location)
                    if walk_cost is not None:
                        total_cost += walk_cost + 1  # Walk + tighten
                else:
                    total_cost += 1000  # High cost if no path

        return total_cost

    def shortest_path(self, start, end):
        """
        Finds the shortest path between two locations using BFS.
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

        return None

    def goal_reached(self, state):
        """
        Check if the goal state has been reached.
        """
        for goal in self.goals:
            if goal not in state:
                return False
        return True
