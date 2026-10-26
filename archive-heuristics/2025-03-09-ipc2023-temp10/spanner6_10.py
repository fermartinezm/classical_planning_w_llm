from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the
    number of walk, pickup_spanner, and tighten_nut actions required.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner must be picked up before it can be used to tighten a nut.
    - The heuristic assumes that the agent will always pick up the closest available spanner.
    - The heuristic assumes that the agent will always tighten the closest available loose nut.

    # Heuristic Initialization
    - Extract the link information between locations from the static facts.
    - No precomputation of shortest paths is performed for efficiency.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners and their locations.
    4. For each loose nut:
       a. Find the closest usable spanner.
       b. Estimate the cost to walk to the spanner, pick it up, walk to the nut, and tighten it.
    5. Sum the costs for all loose nuts to get the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information from static facts.
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
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the goal is already reached.
        if self.goal_reached(state):
            return 0

        # Identify the man's location.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":  # Assuming the man is always named "bob"
                    man_location = parts[2]
                    break

        # Identify loose nuts and their locations.
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                parts = get_parts(fact)
                nut = parts[1]
                loose_nuts.append(nut)
                for f in state:
                    if match(f, "at", nut, "*"):
                        nut_locations[nut] = get_parts(f)[2]
                        break

        # Identify usable spanners and their locations.
        usable_spanners = []
        spanner_locations = {}
        for fact in state:
            if match(fact, "usable", "*"):
                parts = get_parts(fact)
                spanner = parts[1]
                usable_spanners.append(spanner)
                for f in state:
                    if match(f, "at", spanner, "*"):
                        spanner_locations[spanner] = get_parts(f)[2]
                        break

        # Check if the man is carrying a spanner
        carrying_spanner = None
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                carrying_spanner = get_parts(fact)[2]
                break

        total_cost = 0
        for nut in loose_nuts:
            nut_location = nut_locations[nut]

            # If already carrying a usable spanner, check if it can be used at the nut's location
            if carrying_spanner is not None:
                spanner = carrying_spanner
                if spanner in usable_spanners and man_location == nut_location:
                    total_cost += 1  # Tighten nut
                    continue
                else:
                    # Need to drop the current spanner and find a usable one
                    total_cost += 1 # Drop the current spanner

            # Find the closest usable spanner
            closest_spanner = None
            min_distance = float('inf')
            for spanner in usable_spanners:
                if spanner in spanner_locations:
                    spanner_location = spanner_locations[spanner]
                    # Estimate the distance (number of links) between man and spanner
                    distance = self.estimate_distance(man_location, spanner_location)
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner

            if closest_spanner is None:
                # No usable spanner available
                return float('inf')

            spanner_location = spanner_locations[closest_spanner]

            # Estimate the cost to walk to the spanner, pick it up, walk to the nut, and tighten it.
            total_cost += self.estimate_distance(man_location, spanner_location)  # Walk to spanner
            total_cost += 1  # Pick up spanner
            total_cost += self.estimate_distance(spanner_location, nut_location)  # Walk to nut
            total_cost += 1  # Tighten nut

        return total_cost

    def estimate_distance(self, start, end):
        """Estimates the distance between two locations based on the number of links."""
        if start == end:
            return 0

        if start not in self.links or end not in self.links:
            return float('inf')

        # Simple heuristic: return 1 if they are linked, otherwise 2
        if end in self.links[start]:
            return 1
        else:
            return 2

    def goal_reached(self, state):
        """Check if all goal conditions are met in the given state."""
        return self.goals <= state
