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


class spanner9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the need to walk to the nut, pick up a usable spanner, and then tighten the nut.

    # Assumptions
    - The agent can only carry one spanner at a time.
    - A spanner becomes unusable after tightening a nut.
    - The heuristic assumes the shortest path to a nut or spanner.

    # Heuristic Initialization
    - Extract the locations of all nuts, spanners, and the man from the initial state.
    - Identify the links between locations to calculate shortest paths.
    - Identify usable spanners.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened.
    2. For each loose nut:
       a. Determine if the agent is at the nut's location. If not, estimate the cost to walk to the nut.
       b. Determine if the agent is carrying a usable spanner. If not:
          i. Find the closest usable spanner.
          ii. Estimate the cost to walk to the spanner and pick it up.
       c. Estimate the cost to tighten the nut (1 action).
    3. Sum the costs for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract link information for path calculation
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1], get_parts(fact)[2]
                if loc1 not in self.links:
                    self.links[loc1] = []
                self.links[loc1].append(loc2)
                if loc2 not in self.links:
                    self.links[loc2] = []
                self.links[loc2].append(loc1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if the goal is already reached
        if self.goal_reached(state):
            return 0

        # Extract the location of the man
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                if parts[1] == "bob":
                    man_location = parts[2]
                    break

        # Extract the spanner being carried by the man
        carrying_spanner = None
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                carrying_spanner = get_parts(fact)[2]
                break

        # Extract usable spanners
        usable_spanners = set()
        for fact in state:
            if match(fact, "usable", "*"):
                usable_spanners.add(get_parts(fact)[1])

        # Extract loose nuts
        loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                loose_nuts.add(get_parts(fact)[1])

        total_cost = 0
        for nut in loose_nuts:
            # Find the location of the nut
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            # Cost to walk to the nut if not already there
            if man_location != nut_location:
                total_cost += self.shortest_path(man_location, nut_location)

            # Check if carrying a usable spanner
            if carrying_spanner is None or carrying_spanner not in usable_spanners:
                # Find the closest usable spanner
                closest_spanner = None
                min_distance = float('inf')
                for spanner in usable_spanners:
                    # Find the location of the spanner
                    spanner_location = None
                    for fact in state:
                        if match(fact, "at", spanner, "*"):
                            spanner_location = get_parts(fact)[2]
                            break

                    if spanner_location is not None:
                        distance = self.shortest_path(man_location, spanner_location)
                        if distance < min_distance:
                            min_distance = distance
                            closest_spanner = spanner

                if closest_spanner is not None:
                    # Cost to walk to the spanner and pick it up
                    spanner_location = None
                    for fact in state:
                        if match(fact, "at", closest_spanner, "*"):
                            spanner_location = get_parts(fact)[2]
                            break
                    if man_location != spanner_location:
                        total_cost += self.shortest_path(man_location, spanner_location)
                    total_cost += 1  # Pickup spanner

            # Cost to tighten the nut
            total_cost += 1

        return total_cost

    def shortest_path(self, start, end):
        """Calculate the shortest path between two locations using BFS."""
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

    def goal_reached(self, state):
        """Check if all goal conditions are met in the given state."""
        return self.goals <= state
