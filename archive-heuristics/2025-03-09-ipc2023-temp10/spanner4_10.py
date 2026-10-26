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


class spanner4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's location, the location of the nuts, the location of the spanners,
    and whether the man is carrying a usable spanner.

    # Assumptions
    - The heuristic assumes that the agent always has to pick up a spanner before tightening a nut.
    - The heuristic assumes that each spanner can only tighten one nut.
    - The heuristic assumes that the agent can only carry one spanner at a time.

    # Heuristic Initialization
    - The heuristic initializes by extracting the link information between locations from the static facts.
    - It also identifies all nuts that need to be tightened based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the loose nuts that need to be tightened based on the goal state and the current state.
    2. Determine the man's current location.
    3. Determine if the man is carrying a usable spanner.
    4. For each loose nut:
       a. If the man is not carrying a usable spanner:
          i. Find the closest usable spanner.
          ii. Estimate the cost to walk to the spanner, pick it up.
       b. Estimate the cost to walk to the nut and tighten it.
    5. Sum the costs for all loose nuts to get the final heuristic value.
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
                l1 = parts[1]
                l2 = parts[2]
                if l1 not in self.links:
                    self.links[l1] = []
                self.links[l1].append(l2)
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l2].append(l1)

        # Identify nuts that need to be tightened.
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to tighten all remaining loose nuts."""
        state = node.state

        # Identify loose nuts that are goals but not yet tightened.
        loose_nuts = set()
        for nut in self.goal_nuts:
            if f"(loose {nut})" in state:
                loose_nuts.add(nut)

        if not loose_nuts:
            return 0  # Goal state reached

        # Determine the man's current location.
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*") and "bob" in fact:
                man_location = get_parts(fact)[2]
                break

        # Determine if the man is carrying a usable spanner.
        carrying_usable_spanner = False
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                spanner = get_parts(fact)[2]
                if f"(usable {spanner})" in state:
                    carrying_usable_spanner = True
                break

        total_cost = 0
        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break

            if not carrying_usable_spanner:
                # Find the closest usable spanner.
                closest_spanner = None
                closest_spanner_distance = float('inf')
                for fact in state:
                    if match(fact, "at", "*", "*") and "spanner" in fact and f"(usable {get_parts(fact)[1]})" in state:
                        spanner = get_parts(fact)[1]
                        spanner_location = get_parts(fact)[2]
                        distance = self.shortest_path_distance(man_location, spanner_location)
                        if distance < closest_spanner_distance:
                            closest_spanner_distance = distance
                            closest_spanner = spanner
                            closest_spanner_location = spanner_location

                # Estimate the cost to walk to the spanner and pick it up.
                total_cost += closest_spanner_distance + 1  # Walk + pickup

                #Update man location
                man_location = closest_spanner_location
                carrying_usable_spanner = True


            # Estimate the cost to walk to the nut and tighten it.
            distance_to_nut = self.shortest_path_distance(man_location, nut_location)
            total_cost += distance_to_nut + 1  # Walk + tighten

        return total_cost

    def shortest_path_distance(self, start, end):
        """Calculate the shortest path distance between two locations using BFS."""
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

        return float('inf')  # No path found
