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


class spanner3Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the location of the nuts, the location of the spanners,
    and whether the man is carrying a usable spanner.

    # Assumptions
    - The heuristic assumes that the agent needs to pick up a spanner, move to the nut, and tighten it.
    - It also assumes that the agent needs to move between locations.
    - It assumes that the agent can only carry one spanner at a time.
    - It assumes that a spanner becomes unusable after tightening a nut.

    # Heuristic Initialization
    - Extract the locations of all nuts, spanners, and the man from the initial state.
    - Store the link information between locations from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened.
    2. Determine the man's current location and whether he is carrying a usable spanner.
    3. For each loose nut:
       - If the man is not at the nut's location, estimate the cost to move to the nut's location.
       - If the man is not carrying a usable spanner, estimate the cost to pick up a usable spanner and carry it to the nut's location.
       - Estimate the cost to tighten the nut.
    4. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal conditions (tightened nuts).
        - Static facts (link information).
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract link information from static facts.
        self.links = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                self.links.add(tuple(get_parts(fact)[1:]))
                self.links.add((get_parts(fact)[2], get_parts(fact)[1]))  # Add reverse link

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Identify loose nuts.
        loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.add(get_parts(fact)[1])

        # Identify the man's location.
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
                if "(usable {})".format(spanner) in state:
                    carrying_usable_spanner = True
                break

        # Calculate the heuristic value.
        total_cost = 0
        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            if man_location != nut_location:
                # Estimate the cost to move to the nut's location.
                if (man_location, nut_location) in self.links:
                    total_cost += 1  # Walk action
                else:
                    total_cost += 2 # Assume we can reach the nut in two steps.

            if not carrying_usable_spanner:
                # Estimate the cost to pick up a usable spanner.
                spanner_location = None
                for fact in state:
                    if match(fact, "at", "*", "*") and "spanner" in fact and "(usable {})".format(get_parts(fact)[1]) in state:
                        spanner_location = get_parts(fact)[2]
                        break

                if spanner_location:
                    if man_location != spanner_location:
                        # Estimate the cost to move to the spanner's location.
                        if (man_location, spanner_location) in self.links:
                            total_cost += 1  # Walk action
                        else:
                            total_cost += 2 # Assume we can reach the spanner in two steps.
                    total_cost += 1  # Pickup spanner action
                    #Move to the nut location
                    if spanner_location != nut_location:
                        if (spanner_location, nut_location) in self.links:
                            total_cost += 1  # Walk action
                        else:
                            total_cost += 2 # Assume we can reach the nut in two steps.
                else:
                    return float('inf') #No usable spanner

            total_cost += 1  # Tighten nut action

        # Check if the goal is reached
        if self.goal_reached(state):
            return 0

        return total_cost
