from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class spanner7Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the
    number of walk, pickup_spanner, and tighten_nut actions required.

    # Assumptions:
    - The man can only carry one spanner at a time.
    - A usable spanner must be carried to tighten a loose nut.
    - The heuristic assumes the existence of a path between any two locations.

    # Heuristic Initialization
    - Extract the locations of all objects (man, spanners, nuts) from the initial state.
    - Identify usable spanners.
    - Identify loose nuts.
    - Extract link information to estimate walking costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of loose nuts that need to be tightened.
    2. Determine if the man is carrying a usable spanner.
    3. For each loose nut:
       a. If the man is not at the nut's location, estimate the cost to walk to the nut.
       b. If the man is not carrying a usable spanner:
          i. Find the closest usable spanner.
          ii. Estimate the cost to walk to the spanner.
          iii. Add the cost to pick up the spanner.
          iv. Estimate the cost to walk back to the nut.
       c. Add the cost to tighten the nut.
    4. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract link information
        self.links = set()
        for fact in self.static_facts:
            if self.match(fact, "link", "*", "*"):
                parts = self.get_parts(fact)
                self.links.add((parts[1], parts[2]))
                self.links.add((parts[2], parts[1]))

    def __call__(self, node):
        """Estimate the number of actions needed to tighten all loose nuts."""
        state = node.state

        # Extract information from the state
        man_location = None
        carrying_spanner = None
        usable_spanners = set()
        loose_nuts = set()
        object_locations = {}

        for fact in state:
            if self.match(fact, "at", "bob", "*"):
                man_location = self.get_parts(fact)[2]
                object_locations['bob'] = man_location
            elif self.match(fact, "carrying", "bob", "*"):
                carrying_spanner = self.get_parts(fact)[2]
            elif self.match(fact, "usable", "*"):
                usable_spanners.add(self.get_parts(fact)[1])
            elif self.match(fact, "loose", "*"):
                loose_nuts.add(self.get_parts(fact)[1])
            elif self.match(fact, "at", "*", "*"):
                parts = self.get_parts(fact)
                if parts[1] != 'bob':
                    object_locations[parts[1]] = parts[2]

        # Check if the goal is already reached
        if self.goal_reached(state):
            return 0

        # Estimate the cost to achieve the goal
        total_cost = 0
        for nut in loose_nuts:
            nut_location = object_locations[nut]

            # Cost to walk to the nut if not already there
            if man_location != nut_location:
                total_cost += 1  # walking cost

            # If not carrying a usable spanner, find the closest one and pick it up
            if carrying_spanner is None or carrying_spanner not in usable_spanners:
                closest_spanner = None
                min_distance = float('inf')
                for spanner in usable_spanners:
                    if spanner in object_locations:
                        spanner_location = object_locations[spanner]
                        distance = self.estimate_distance(man_location, spanner_location)
                        if distance < min_distance:
                            min_distance = distance
                            closest_spanner = spanner

                if closest_spanner:
                    spanner_location = object_locations[closest_spanner]
                    if man_location != spanner_location:
                        total_cost += 1 # walking cost
                    total_cost += 1  # pickup_spanner cost
                    man_location = spanner_location
                    if man_location != nut_location:
                        total_cost += 1 # walking cost

            total_cost += 1  # tighten_nut cost

        return total_cost

    def goal_reached(self, state):
        """Check if all goal conditions are satisfied in the given state."""
        return self.goals <= state

    def get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    def match(self, fact, *args):
        """Check if a PDDL fact matches a given pattern."""
        parts = self.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def estimate_distance(self, start, end):
        """Estimates the distance between two locations.  In this simple
        implementation, it returns 0 if the locations are the same, 1 if
        they are linked, and 2 otherwise."""
        if start == end:
            return 0
        if (start, end) in self.links:
            return 1
        return 2
