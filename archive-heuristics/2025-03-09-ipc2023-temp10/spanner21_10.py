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


class spanner21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the cost
    of walking, picking up a spanner, and tightening nuts.

    # Assumptions
    - The man can carry multiple spanners, but only one can be used at a time.
    - The heuristic assumes that the man will always pick up the closest available spanner.
    - The heuristic ignores the 'usable' predicate after tightening a nut, assuming the spanner becomes unusable.

    # Heuristic Initialization
    - Extract the link information between locations from the static facts.
    - Identify all loose nuts from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the loose nuts that need to be tightened based on the goal state.
    2.  Determine the man's current location.
    3.  For each loose nut:
        a. Find the closest usable spanner.
        b. Estimate the cost of walking to the spanner, picking it up, walking to the nut, and tightening it.
    4.  Sum the costs for all loose nuts to get the final heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Link information between locations.
        - Loose nuts from the goal conditions.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract link information: link[location1] = location2
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                self.links[loc1] = loc2

        # Identify loose nuts from the goal conditions.
        self.loose_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                self.loose_nuts.add(nut)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break
        if goal_reached:
            return 0

        # Find the man's location.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":  # Assuming the man is always named 'bob'
                    man_location = parts[2]
                    break

        # Find the nuts that are still loose.
        loose_nuts_in_state = set()
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                loose_nuts_in_state.add(nut)

        # Find the spanners that are usable.
        usable_spanners = set()
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)

        # Find the location of each spanner.
        spanner_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in usable_spanners:
                    spanner_locations[parts[1]] = parts[2]

        # Find the location of each nut.
        nut_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in loose_nuts_in_state:
                    nut_locations[parts[1]] = parts[2]

        # Find the spanners carried by the man
        spanners_carried = set()
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":
                    spanners_carried.add(parts[2])

        total_cost = 0
        for nut in loose_nuts_in_state:
            nut_location = nut_locations.get(nut)
            if not nut_location:
                continue

            # Find the closest usable spanner.
            closest_spanner = None
            min_distance = float('inf')
            for spanner in usable_spanners:
                if spanner in spanners_carried:
                    closest_spanner = spanner
                    min_distance = 0
                    break
                spanner_location = spanner_locations.get(spanner)
                if not spanner_location:
                    continue

                # Estimate the distance (number of walk actions) from the man to the spanner.
                distance_to_spanner = 0
                current_location = man_location
                while current_location != spanner_location:
                    if current_location not in self.links:
                        distance_to_spanner = float('inf')
                        break
                    current_location = self.links[current_location]
                    distance_to_spanner += 1

                if distance_to_spanner < min_distance:
                    min_distance = distance_to_spanner
                    closest_spanner = spanner

            if closest_spanner:
                # Estimate the cost of walking to the spanner, picking it up, walking to the nut, and tightening it.
                cost = 0

                # Walking to the spanner.
                if closest_spanner not in spanners_carried:
                    cost += min_distance

                    # Picking up the spanner.
                    cost += 1

                    # Update man location
                    man_location = spanner_locations[closest_spanner]

                # Walking to the nut.
                distance_to_nut = 0
                current_location = man_location
                while current_location != nut_location:
                    if current_location not in self.links:
                        distance_to_nut = float('inf')
                        break
                    current_location = self.links[current_location]
                    distance_to_nut += 1
                cost += distance_to_nut

                # Tightening the nut.
                cost += 1

                total_cost += cost

                # Remove the tightened nut and unusable spanner from the sets.
                usable_spanners.discard(closest_spanner)

        return total_cost
