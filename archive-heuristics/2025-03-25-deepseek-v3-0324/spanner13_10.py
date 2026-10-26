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
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the man's current location, carried spanners, and the locations of
    loose nuts and usable spanners.

    # Assumptions:
    - The man can carry multiple spanners at once.
    - Each spanner can only be used once (becomes unusable after tightening a nut).
    - The man must be at the nut's location to tighten it.
    - The man must be at a spanner's location to pick it up.

    # Heuristic Initialization
    - Extract link information from static facts to compute distances between locations.
    - Identify goal nuts (those that need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (goal nuts not yet tightened).
    2. For each loose nut:
       a. If the man is already at the nut's location with a usable spanner:
          - Add 1 action (tighten).
       b. Else:
          - If not carrying a usable spanner:
            - Find nearest usable spanner (using link distances).
            - Add distance to spanner + 1 action (pickup).
          - Add distance from spanner to nut.
          - Add 1 action (tighten).
    3. If multiple loose nuts exist at the same location, the man can tighten them
       consecutively if he has enough usable spanners.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build graph of locations from link facts
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)

        # Precompute shortest paths between all locations
        self.distances = {}
        locations = set(self.links.keys())
        for loc in locations:
            self.distances[loc] = self._bfs(loc)

    def _bfs(self, start):
        """Compute shortest distances from start location to all others using BFS."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in self.links.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Identify loose nuts that need to be tightened (goals not yet satisfied)
        loose_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                if f"(loose {nut})" in state:
                    loose_nuts.add(nut)

        if not loose_nuts:
            return 0  # All nuts tightened

        # Get man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
                break

        # Get carried usable spanners
        usable_spanners = set()
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                spanner = get_parts(fact)[2]
                if f"(usable {spanner})" in state:
                    usable_spanners.add(spanner)

        # Get locations of loose nuts
        nut_locations = {}
        for nut in loose_nuts:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_locations[nut] = get_parts(fact)[2]
                    break

        total_cost = 0

        for nut in loose_nuts:
            nut_loc = nut_locations[nut]

            if man_loc == nut_loc and usable_spanners:
                # Can tighten immediately
                total_cost += 1
                usable_spanners.pop()  # Use one spanner
                continue

            # Need to get to nut location with usable spanner
            if not usable_spanners:
                # Find nearest usable spanner
                min_dist = float('inf')
                nearest_spanner_loc = None
                for fact in state:
                    if match(fact, "at", "*", "*") and match(fact, "at", "spanner*", "*"):
                        spanner = get_parts(fact)[1]
                        if f"(usable {spanner})" in state:
                            spanner_loc = get_parts(fact)[2]
                            dist = self.distances[man_loc].get(spanner_loc, float('inf'))
                            if dist < min_dist:
                                min_dist = dist
                                nearest_spanner_loc = spanner_loc

                if nearest_spanner_loc is None:
                    return float('inf')  # No usable spanners left

                total_cost += min_dist + 1  # Walk to spanner + pickup
                man_loc = nearest_spanner_loc
                usable_spanners.add("dummy")  # Count as carrying one now

            # Walk to nut location
            dist_to_nut = self.distances[man_loc].get(nut_loc, float('inf'))
            if dist_to_nut == float('inf'):
                return float('inf')  # Unreachable

            total_cost += dist_to_nut + 1  # Walk to nut + tighten
            man_loc = nut_loc
            usable_spanners.pop()  # Use one spanner

        return total_cost
