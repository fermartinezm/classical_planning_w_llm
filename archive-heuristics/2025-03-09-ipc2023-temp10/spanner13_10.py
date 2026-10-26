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


class spanner13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the location of the man, spanners, and nuts, and estimates the
    number of walk, pickup_spanner, and tighten_nut actions required.

    # Assumptions
    - The man can only carry one spanner at a time.
    - A spanner must be at the same location as the man and the nut to tighten it.
    - A usable spanner becomes unusable after tightening a nut.
    - The heuristic assumes that the agent will always pick up the closest spanner.

    # Heuristic Initialization
    - Extract the link information to calculate shortest paths between locations.
    - Identify usable spanners.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts and their locations.
    2. Determine the man's current location.
    3. Check if the man is carrying a spanner.
    4. If not carrying a spanner, find the closest usable spanner:
       - Calculate the shortest path from the man's location to each usable spanner's location.
       - Select the closest spanner.
       - Add the cost of walking to the spanner and picking it up.
    5. For each loose nut:
       - Calculate the shortest path from the man's current location to the nut's location.
       - Add the cost of walking to the nut and tightening it.
       - After tightening a nut, the carried spanner becomes unusable.
    6. Sum up the costs to get the estimated number of actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract link information for path calculation.
        self.links = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                if l1 not in self.links:
                    self.links[l1] = []
                if l2 not in self.links:
                    self.links[l2] = []
                self.links[l1].append(l2)
                self.links[l2].append(l1)

        # Identify usable spanners.
        self.usable_spanners = {get_parts(fact)[1] for fact in static_facts if match(fact, "usable", "*")}

    def __call__(self, node):
        """Estimate the minimum cost to tighten all loose nuts."""
        state = node.state

        # Identify loose nuts and their locations.
        loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        nut_locations = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] in loose_nuts
        }

        # Determine the man's current location.
        man_location = next(get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] == 'bob')

        # Check if the man is carrying a spanner.
        carrying_spanner = next((get_parts(fact)[2] for fact in state if match(fact, "carrying", "bob", "*")), None)

        total_cost = 0

        # If not carrying a spanner, find the closest usable spanner.
        if not carrying_spanner:
            spanner_locations = {
                get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.usable_spanners
            }
            closest_spanner = None
            min_distance = float('inf')

            for spanner, location in spanner_locations.items():
                distance = self.shortest_path(man_location, location)
                if distance < min_distance:
                    min_distance = distance
                    closest_spanner = spanner
                    closest_spanner_location = location

            if closest_spanner:
                total_cost += min_distance  # Walk to the spanner.
                total_cost += 1  # Pick up the spanner.
                carrying_spanner = closest_spanner
                man_location = closest_spanner_location

        # Tighten each loose nut.
        for nut, location in nut_locations.items():
            distance = self.shortest_path(man_location, location)
            total_cost += distance  # Walk to the nut.
            total_cost += 1  # Tighten the nut.
            man_location = location
            carrying_spanner = None
            break # Only tighten one nut per state

        # Check if all goals are reached
        goal_reached = True
        for goal in self.goals:
            if goal not in state:
                goal_reached = False
                break

        if goal_reached:
            return 0

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

            for neighbor in self.links.get(location, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # Return infinity if no path is found.
