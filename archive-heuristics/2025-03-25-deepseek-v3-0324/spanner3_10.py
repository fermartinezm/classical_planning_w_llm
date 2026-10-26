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
    It considers the following factors:
    - Distance the man needs to walk to collect spanners and reach nuts
    - Whether the man needs to pick up additional spanners
    - Whether spanners need to be used to tighten nuts

    # Assumptions:
    - The man can carry multiple spanners at once.
    - Each spanner can only be used once (becomes unusable after tightening a nut).
    - The man must be at the nut's location to tighten it.
    - The path between locations is always the shortest possible.

    # Heuristic Initialization
    - Extract the link graph between locations from static facts.
    - Identify all goal nuts that need to be tightened.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts that still need to be tightened (not in goal state yet).
    2. For each loose nut:
       a. Find the man's current location and the nut's location.
       b. If the man isn't carrying any usable spanners:
          i. Find the nearest usable spanner to the man's current location.
          ii. Add the distance to reach that spanner.
          iii. Mark the spanner as picked up (but not yet used).
       c. Add the distance from current location (or spanner location) to the nut.
       d. Add 1 action for tightening the nut (using a spanner).
    3. The total heuristic is the sum of all walking distances plus tightening actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build a graph of location links for pathfinding
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)

        # Precompute goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

    def _shortest_path(self, start, end, visited=None):
        """BFS to find shortest path length between two locations."""
        if start == end:
            return 0
        if visited is None:
            visited = set()
        visited.add(start)
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.links.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found

    def __call__(self, node):
        """Estimate the minimum cost to tighten all remaining loose nuts."""
        state = node.state
        total_cost = 0

        # Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
                break
        if not man_loc:
            return float('inf')  # Invalid state

        # Find all loose nuts that still need to be tightened
        loose_nuts = set()
        tightened_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.goal_nuts:
                    loose_nuts.add(nut)
            elif match(fact, "tightened", "*"):
                nut = get_parts(fact)[1]
                tightened_nuts.add(nut)

        remaining_nuts = self.goal_nuts - tightened_nuts
        if not remaining_nuts:
            return 0  # Goal already reached

        # Find usable spanners the man is carrying
        carried_spanners = set()
        usable_carried = 0
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                spanner = get_parts(fact)[2]
                carried_spanners.add(spanner)
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                if spanner in carried_spanners:
                    usable_carried += 1

        # Find all usable spanners in the world (including carried ones)
        usable_spanners = set()
        spanner_locations = {}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)
            if match(fact, "at", "spanner*", "*"):
                spanner, loc = get_parts(fact)[1], get_parts(fact)[2]
                spanner_locations[spanner] = loc

        # For each remaining nut to tighten
        for nut in remaining_nuts:
            # Find nut location
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]
                    break
            if not nut_loc:
                return float('inf')  # Invalid state

            # If no usable spanners carried, find nearest one
            if usable_carried <= 0:
                nearest_spanner_dist = float('inf')
                nearest_spanner_loc = None
                for spanner in usable_spanners:
                    if spanner in carried_spanners:
                        continue  # Already considered these
                    spanner_loc = spanner_locations.get(spanner)
                    if not spanner_loc:
                        continue  # Spanner not found (invalid state)
                    dist = self._shortest_path(man_loc, spanner_loc)
                    if dist < nearest_spanner_dist:
                        nearest_spanner_dist = dist
                        nearest_spanner_loc = spanner_loc

                if nearest_spanner_dist == float('inf'):
                    return float('inf')  # No usable spanners available

                total_cost += nearest_spanner_dist + 1  # walk + pickup
                man_loc = nearest_spanner_loc  # update man's location
                usable_carried += 1  # now carrying one more usable spanner

            # Walk to nut location
            walk_dist = self._shortest_path(man_loc, nut_loc)
            if walk_dist == float('inf'):
                return float('inf')  # No path to nut
            total_cost += walk_dist
            man_loc = nut_loc  # update man's location

            # Tighten the nut
            total_cost += 1  # tighten action
            usable_carried -= 1  # used a spanner

        return total_cost
