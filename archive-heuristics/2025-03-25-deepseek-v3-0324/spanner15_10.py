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
    It considers the man's current location, carried spanners, and the locations of loose nuts
    and usable spanners.

    # Assumptions:
    - The man can carry multiple spanners at once.
    - Each spanner can only be used once (becomes unusable after tightening a nut).
    - The man must be at the same location as a nut to tighten it.
    - The man must be carrying a usable spanner to tighten a nut.

    # Heuristic Initialization
    - Extract the goal conditions (which nuts need to be tightened).
    - Parse static facts to build a graph of connected locations for path planning.
    - Identify all spanners and nuts in the domain.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many nuts still need to be tightened (from goal conditions).
    2. Check if the man is already carrying usable spanners:
       - If not, find the nearest usable spanner and add the path cost to reach it.
    3. For each nut that needs tightening:
       a. Calculate the path cost from current location to the nut's location.
       b. If no usable spanner is currently carried, add the path cost to the nearest spanner.
       c. Add 1 action for the tighten operation.
    4. The total heuristic is the sum of:
       - Path costs to collect spanners (if needed)
       - Path costs to reach nuts
       - Tighten actions for each nut
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build a graph of connected locations from static 'link' facts
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        """Estimate the minimum cost to tighten all required nuts."""
        state = node.state

        # Extract goal nuts that need to be tightened
        goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Get current state information
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        nut_locations = {}
        spanner_locations = {}

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
                if obj.startswith("nut"):
                    nut_locations[obj] = loc
                elif obj.startswith("spanner"):
                    spanner_locations[obj] = loc

        # Calculate which nuts still need tightening
        loose_nuts = {nut for nut in goal_nuts if f"(tightened {nut})" not in state}

        if not loose_nuts:
            return 0  # All goals satisfied

        total_cost = 0
        current_loc = man_location
        remaining_spanners = {s for s in carried_spanners if s in usable_spanners}

        # If no usable spanners are carried, find the nearest one
        if not remaining_spanners:
            nearest_spanner_loc, dist = self._find_nearest_spanner(current_loc, spanner_locations, usable_spanners)
            if nearest_spanner_loc is None:
                return float('inf')  # No usable spanners available
            total_cost += dist + 1  # walk to spanner + pickup action
            current_loc = nearest_spanner_loc
            # After pickup, we have at least one usable spanner

        # For each loose nut, calculate path to it
        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if nut_loc is None:
                return float('inf')  # Nut doesn't exist in state

            if current_loc != nut_loc:
                path_cost = self._shortest_path_length(current_loc, nut_loc)
                if path_cost == float('inf'):
                    return float('inf')  # No path to nut
                total_cost += path_cost
                current_loc = nut_loc

            total_cost += 1  # tighten action
            remaining_spanners.pop()  # use one spanner

            # If we run out of spanners, get another one
            if not remaining_spanners and len(loose_nuts) > 1:
                nearest_spanner_loc, dist = self._find_nearest_spanner(current_loc, spanner_locations, usable_spanners)
                if nearest_spanner_loc is None:
                    return float('inf')  # No more usable spanners
                total_cost += dist + 1  # walk to spanner + pickup action
                current_loc = nearest_spanner_loc
                remaining_spanners.add(next(iter(usable_spanners - carried_spanners), None))

        return total_cost

    def _shortest_path_length(self, start, end):
        """Calculate the shortest path length between two locations using BFS."""
        if start == end:
            return 0

        visited = {start}
        queue = [(start, 0)]

        while queue:
            current, dist = queue.pop(0)
            for neighbor in self.location_graph.get(current, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf')  # No path found

    def _find_nearest_spanner(self, start_loc, spanner_locations, usable_spanners):
        """Find the nearest usable spanner using BFS."""
        min_dist = float('inf')
        nearest_loc = None

        for spanner, loc in spanner_locations.items():
            if spanner in usable_spanners:
                dist = self._shortest_path_length(start_loc, loc)
                if dist < min_dist:
                    min_dist = dist
                    nearest_loc = loc

        return nearest_loc, min_dist
