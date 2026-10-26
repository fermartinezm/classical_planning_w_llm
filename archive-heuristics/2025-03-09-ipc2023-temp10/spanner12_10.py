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


class spanner12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the need to walk to the nut, pick up a spanner if not carrying one,
    and then tighten the nut.

    # Assumptions
    - The agent can only carry one spanner at a time.
    - A spanner must be usable to tighten a nut.
    - The heuristic assumes the shortest path to the nut and spanner.

    # Heuristic Initialization
    - Extract the locations of all nuts, spanners, and the man from the initial state.
    - Identify the links between locations from the static facts.
    - Store the goal conditions (tightened nuts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened based on the goal state.
    2. Determine if the man is carrying a usable spanner.
    3. For each loose nut:
       - Calculate the cost to reach the nut's location from the man's current location (walk actions).
       - If the man is not carrying a usable spanner:
         - Find the closest usable spanner.
         - Calculate the cost to reach the closest usable spanner from the man's current location (walk actions).
         - Add the cost of picking up the spanner.
         - Update the man's location to the spanner's location.
       - Add the cost of tightening the nut.
    4. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information
        self.links = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                self.links.add((l1, l2))
                self.links.add((l2, l1))  # Assuming links are bidirectional

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Extract information from the current state
        man_location = None
        carried_spanner = None
        usable_spanners = set()
        loose_nuts = set()

        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":
                    man_location = parts[2]
            elif match(fact, "carrying", "*", "*"):
                carried_spanner = get_parts(fact)[2]
            elif match(fact, "usable", "*"):
                usable_spanners.add(get_parts(fact)[1])
            elif match(fact, "loose", "*"):
                loose_nuts.add(get_parts(fact)[1])

        # Identify tightened nuts from the goal state
        tightened_nuts_goal = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Filter loose nuts to only include those that are goals
        loose_nuts = {nut for nut in loose_nuts if nut in tightened_nuts_goal}

        # If all goal nuts are tightened, return 0
        if not loose_nuts:
            return 0

        total_cost = 0

        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            # Check if a usable spanner is being carried
            if carried_spanner is None or carried_spanner not in usable_spanners:
                # Find the closest usable spanner
                closest_spanner = None
                min_distance = float('inf')

                for spanner in usable_spanners:
                    spanner_location = None
                    for fact in state:
                        if match(fact, "at", spanner, "*"):
                            spanner_location = get_parts(fact)[2]
                            break

                    if spanner_location:
                        distance = self.shortest_path(man_location, spanner_location)
                        if distance < min_distance:
                            min_distance = distance
                            closest_spanner = spanner
                            closest_spanner_location = spanner_location

                if closest_spanner:
                    total_cost += min_distance  # Cost to walk to the spanner
                    total_cost += 1  # Cost to pick up the spanner
                    man_location = closest_spanner_location  # Update man's location
                else:
                    # No usable spanner available, the problem is unsolvable
                    return float('inf')

            # Calculate the cost to walk to the nut and tighten it
            distance_to_nut = self.shortest_path(man_location, nut_location)
            total_cost += distance_to_nut  # Cost to walk to the nut
            total_cost += 1  # Cost to tighten the nut

        return total_cost

    def shortest_path(self, start, end):
        """Calculate the shortest path between two locations using BFS."""
        if start == end:
            return 0

        queue = [(start, 0)]
        visited = {start}

        while queue:
            location, distance = queue.pop(0)
            for l1, l2 in self.links:
                if l1 == location and l2 == end:
                    return distance + 1
                elif l1 == location and l2 not in visited:
                    queue.append((l2, distance + 1))
                    visited.add(l2)

        return float('inf')  # No path found
