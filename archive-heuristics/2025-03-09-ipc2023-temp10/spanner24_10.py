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


class spanner24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the spanner's location, the nut's location,
    and whether the man is carrying a usable spanner. It also accounts for the
    need to walk between locations.

    # Assumptions
    - The agent can only carry one spanner at a time.
    - A spanner can only be used once.
    - The heuristic assumes the shortest path between locations is always used.

    # Heuristic Initialization
    - Extract the link information between locations from the static facts.
    - Create a dictionary representing the connectivity graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts.
    2. For each loose nut, estimate the cost to tighten it:
       a. If the man is not at the nut's location, estimate the cost to walk there.
       b. If the man is not carrying a usable spanner, estimate the cost to:
          i. Walk to a location with a usable spanner.
          ii. Pick up the spanner.
          iii. Walk to the nut's location.
       c. Tighten the nut (1 action).
    3. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting link information from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build a connectivity graph from the link facts.
        self.connectivity = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 not in self.connectivity:
                    self.connectivity[l1] = []
                if l2 not in self.connectivity:
                    self.connectivity[l2] = []
                self.connectivity[l1].append(l2)
                self.connectivity[l2].append(l1)

    def __call__(self, node):
        """
        Estimate the number of actions needed to tighten all loose nuts.
        """
        state = node.state

        # Check if the goal is already reached.
        if self.goal_reached(state):
            return 0

        # Helper function to find the shortest path between two locations using BFS.
        def shortest_path_length(start, end):
            if start == end:
                return 0
            queue = [(start, 0)]
            visited = {start}
            while queue:
                location, distance = queue.pop(0)
                if location == end:
                    return distance
                for neighbor in self.connectivity.get(location, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
            return float('inf')  # Return infinity if no path exists.

        # Find the man's location.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*") and get_parts(fact)[1] == "bob":
                man_location = get_parts(fact)[2]
                break

        # Find the spanner being carried by the man.
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                carried_spanner = get_parts(fact)[2]
                break

        # Check if the carried spanner is usable.
        spanner_usable = False
        if carried_spanner:
            for fact in state:
                if match(fact, "usable", carried_spanner):
                    spanner_usable = True
                    break

        # Identify loose nuts.
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.append(get_parts(fact)[1])

        total_cost = 0
        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            # If the man is not at the nut's location, estimate the cost to walk there.
            if man_location != nut_location:
                total_cost += shortest_path_length(man_location, nut_location)

            # If the man is not carrying a usable spanner, estimate the cost to get one.
            if not spanner_usable:
                # Find a usable spanner.
                spanner_location = None
                best_spanner_location = None
                min_distance = float('inf')
                for fact in state:
                    if match(fact, "at", "*", "*") and match(fact, "usable", get_parts(fact)[1]):
                        spanner = get_parts(fact)[1]
                        for fact2 in state:
                            if match(fact2, "at", spanner, "*"):
                                spanner_location = get_parts(fact2)[2]
                                distance = shortest_path_length(man_location, spanner_location)
                                if distance < min_distance:
                                    min_distance = distance
                                    best_spanner_location = spanner_location
                                break

                if best_spanner_location is not None:
                    # Walk to the spanner.
                    total_cost += shortest_path_length(man_location, best_spanner_location)
                    # Pick up the spanner.
                    total_cost += 1
                    # Walk to the nut.
                    total_cost += shortest_path_length(best_spanner_location, nut_location)
                else:
                    return float('inf')

            # Tighten the nut.
            total_cost += 1

        return total_cost

    def goal_reached(self, state):
        """
        Check if all goal conditions are satisfied in the given state.
        """
        return self.goals <= state
