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


class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts in the goal.
    It considers:
    - The man's current location and carried spanners
    - The locations of loose nuts and usable spanners
    - The path needed to collect spanners and reach nuts

    # Assumptions:
    - Each nut requires exactly one spanner to be tightened
    - A spanner becomes unusable after tightening one nut
    - The man can carry multiple spanners at once
    - The path between locations is always the shortest possible

    # Heuristic Initialization
    - Extract the link graph between locations from static facts
    - Identify goal nuts (those that need to be tightened)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (goal nuts)
    2. For each loose nut:
       a. Find the shortest path from man's current location to the nut's location
       b. Find the nearest usable spanner to the nut's location
       c. Calculate path from man's current position to spanner to nut
    3. If man is already carrying usable spanners:
       a. Consider using them first before collecting new ones
    4. Sum:
       a. Walk actions to collect spanners and reach nuts
       b. Pickup actions for spanners
       c. Tighten actions for nuts
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build graph of location links
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)

        # Identify which nuts need to be tightened (goal nuts)
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def _shortest_path_length(self, start, end):
        """BFS to find shortest path length between two locations."""
        if start == end:
            return 0

        visited = set()
        queue = [(start, 0)]
        while queue:
            loc, dist = queue.pop(0)
            if loc == end:
                return dist
            if loc in visited:
                continue
            visited.add(loc)
            for neighbor in self.links.get(loc, set()):
                queue.append((neighbor, dist + 1))
        return float('inf')  # No path exists

    def __call__(self, node):
        """Estimate the minimum cost to tighten all remaining loose nuts."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Extract current state information
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        spanner_locations = {}
        loose_nuts = set()
        nut_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("spanner"):
                    spanner_locations[obj] = loc
                elif obj.startswith("nut"):
                    nut_locations[obj] = loc
            elif match(fact, "loose", "*"):
                loose_nuts.add(parts[1])

        # Only consider goal nuts that are still loose
        remaining_nuts = self.goal_nuts & loose_nuts
        if not remaining_nuts:
            return 0

        total_cost = 0

        # Calculate cost for each remaining nut
        for nut in remaining_nuts:
            nut_loc = nut_locations[nut]
            
            # Find best spanner to use (either carried or in world)
            best_cost = float('inf')
            
            # Check carried spanners first
            for spanner in carried_spanners:
                if spanner in usable_spanners:
                    # Just need to walk to nut
                    path_cost = self._shortest_path_length(man_location, nut_loc)
                    best_cost = min(best_cost, path_cost + 1)  # +1 for tighten action
                    break  # Can use first available spanner
            
            if best_cost == float('inf'):
                # Need to find and pickup a spanner
                for spanner in usable_spanners:
                    if spanner not in carried_spanners:
                        spanner_loc = spanner_locations[spanner]
                        # Cost to go from man -> spanner -> nut
                        path1 = self._shortest_path_length(man_location, spanner_loc)
                        path2 = self._shortest_path_length(spanner_loc, nut_loc)
                        total_path = path1 + path2
                        best_cost = min(best_cost, total_path + 2)  # +1 pickup, +1 tighten
            
            total_cost += best_cost if best_cost != float('inf') else 0

        return total_cost
