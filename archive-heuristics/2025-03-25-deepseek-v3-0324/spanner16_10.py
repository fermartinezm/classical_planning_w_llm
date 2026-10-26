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
    - Identify goal nuts that need to be tightened.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many loose nuts still need to be tightened (not yet in goal state).
    2. Check if the man is carrying any usable spanners:
       - If yes, these can be used immediately for tightening.
       - If no, find the nearest usable spanner that needs to be picked up.
    3. For each loose nut:
       - If the man has a usable spanner:
         - Add distance from current location to nut's location.
         - Add 1 action for tightening.
       - If no usable spanner is carried:
         - Find nearest usable spanner.
         - Add distance to spanner's location + distance to nut's location.
         - Add 1 action for pickup and 1 for tightening.
    4. The total heuristic is the sum of all required movement and action steps.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build a graph of locations from link facts
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Precompute shortest paths between all locations
        self.distances = {}
        locations = set(self.location_graph.keys())
        for loc in locations:
            self.distances[loc] = self._bfs(loc)

    def _bfs(self, start):
        """Compute shortest path distances from start location using BFS."""
        distances = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in self.location_graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Extract current state information
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        loose_nuts = set()
        spanner_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "loose", "*"):
                loose_nuts.add(parts[1])
            elif match(fact, "at", "*", "*") and parts[1] not in ['bob']:
                obj, loc = parts[1], parts[2]
                spanner_locations[obj] = loc

        # Filter out already tightened nuts from goals
        remaining_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                if f"(loose {nut})" in state:
                    remaining_nuts.add(nut)

        if not remaining_nuts:
            return 0

        total_cost = 0

        # Find usable spanners that are either carried or on the ground
        available_spanners = []
        for spanner in usable_spanners:
            if spanner in carried_spanners:
                available_spanners.append((0, spanner))  # Already carried
            elif spanner in spanner_locations:
                dist = self.distances[man_location].get(spanner_locations[spanner], float('inf'))
                available_spanners.append((dist, spanner))

        if not available_spanners:
            return float('inf')  # No usable spanners left

        # Sort spanners by distance (carried spanners first)
        available_spanners.sort()

        for nut in remaining_nuts:
            # Find nut location
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]
                    break

            if not nut_loc:
                continue  # Nut not found in state

            if available_spanners:
                dist_to_spanner, spanner = available_spanners[0]
                if dist_to_spanner == 0:  # Already carrying this spanner
                    dist_to_nut = self.distances[man_location].get(nut_loc, float('inf'))
                    total_cost += dist_to_nut + 1  # move + tighten
                else:
                    spanner_loc = spanner_locations[spanner]
                    dist_spanner_to_nut = self.distances[spanner_loc].get(nut_loc, float('inf'))
                    total_cost += dist_to_spanner + dist_spanner_to_nut + 2  # move to spanner + pickup + move to nut + tighten
                
                # Remove this spanner from available list (can only use once)
                available_spanners.pop(0)
            else:
                return float('inf')  # No more usable spanners

        return total_cost
