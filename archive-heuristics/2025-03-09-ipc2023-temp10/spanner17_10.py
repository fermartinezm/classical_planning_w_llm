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


class spanner17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the location of the nuts, the location of the spanners,
    and whether the man is carrying a usable spanner.

    # Assumptions
    - The heuristic assumes that the agent always picks up the closest spanner.
    - The heuristic assumes that the agent always tightens the nuts in the order of their appearance in the state.
    - The heuristic assumes that the agent can only carry one spanner at a time.

    # Heuristic Initialization
    - The heuristic initializes the links between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current state information: man's location, carrying spanner, usable spanners, loose nuts, and tightened nuts.
    2. If all nuts are tightened, the heuristic value is 0.
    3. Calculate the cost for each loose nut:
        a. If the man is not carrying a usable spanner:
            i. Find the closest usable spanner.
            ii. Calculate the cost to walk to the spanner, pick it up.
        b. Calculate the cost to walk to the nut and tighten it.
    4. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information from static facts
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                l1 = parts[1]
                l2 = parts[2]
                if l1 not in self.links:
                    self.links[l1] = []
                self.links[l1].append(l2)
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l2].append(l1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Extract information from the current state
        man_location = None
        carrying_spanner = None
        usable_spanners = set()
        loose_nuts = set()
        tightened_nuts = set()

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":
                    man_location = parts[2]
            elif match(fact, "carrying", "*", "*"):
                parts = get_parts(fact)
                carrying_spanner = parts[2]
            elif match(fact, "usable", "*"):
                parts = get_parts(fact)
                usable_spanners.add(parts[1])
            elif match(fact, "loose", "*"):
                parts = get_parts(fact)
                loose_nuts.add(parts[1])
            elif match(fact, "tightened", "*"):
                parts = get_parts(fact)
                tightened_nuts.add(parts[1])

        # Check if all goals are achieved
        all_goals_achieved = True
        for goal in self.goals:
            if goal not in state:
                all_goals_achieved = False
                break

        if all_goals_achieved:
            return 0

        total_cost = 0

        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            # If not carrying a usable spanner, find the closest one and pick it up
            if carrying_spanner not in usable_spanners:
                closest_spanner = None
                min_distance = float('inf')
                for spanner in usable_spanners:
                    spanner_location = None
                    for fact in state:
                        if match(fact, "at", spanner, "*"):
                            spanner_location = get_parts(fact)[2]
                            break
                    if spanner_location:
                        distance = self.calculate_distance(man_location, spanner_location)
                        if distance < min_distance:
                            min_distance = distance
                            closest_spanner = spanner
                            closest_spanner_location = spanner_location

                if closest_spanner:
                    total_cost += self.calculate_distance(man_location, closest_spanner_location)  # Walk to spanner
                    total_cost += 1  # Pick up spanner
                    man_location = closest_spanner_location
                    carrying_spanner = closest_spanner

            # Walk to the nut and tighten it
            total_cost += self.calculate_distance(man_location, nut_location)  # Walk to nut
            total_cost += 1  # Tighten nut
            man_location = nut_location

        return total_cost

    def calculate_distance(self, start, end):
        """Calculate the distance between two locations using a simple BFS."""
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
