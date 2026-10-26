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
    - The man must be carrying a usable spanner to tighten a nut.

    # Heuristic Initialization
    - Extract static link information between locations for path planning.
    - Store goal conditions (which nuts need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (from goals).
    2. Check if man is already carrying usable spanners:
       - If yes, these can be used immediately for tightening.
       - If no, need to collect usable spanners.
    3. For each loose nut:
       a. Calculate path distance from man's current location to nut's location.
       b. If no usable spanner is carried:
          - Find nearest usable spanner.
          - Add distance to spanner's location.
          - Add 1 action for picking up the spanner.
       c. Add distance from spanner location to nut location.
       d. Add 1 action for tightening the nut.
    4. Optimize by considering that:
       - Multiple nuts at same location can be tightened with one trip.
       - Multiple spanners can be picked up along the way.
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

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Extract goal nuts that need to be tightened
        goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Count loose nuts that still need tightening
        loose_nuts = set()
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in goal_nuts:
                    loose_nuts.add(nut)

        if not loose_nuts:
            return 0  # All nuts are already tightened

        # Get man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
                break
        if not man_loc:
            return float('inf')  # Invalid state

        # Get locations of loose nuts
        nut_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                if obj in loose_nuts:
                    nut_locations[obj] = loc

        # Get usable spanners not being carried
        usable_spanners = set()
        spanner_locations = {}
        carrying_spanners = set()
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                if obj in usable_spanners:
                    spanner_locations[obj] = loc
            elif match(fact, "carrying", "*", "*"):
                _, man, spanner = get_parts(fact)
                if spanner in usable_spanners:
                    carrying_spanners.add(spanner)

        # Calculate heuristic value
        total_cost = 0

        # Group nuts by location
        nuts_by_location = {}
        for nut, loc in nut_locations.items():
            nuts_by_location.setdefault(loc, set()).add(nut)

        # BFS function to find shortest path between locations
        def bfs(start, end):
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

        current_loc = man_loc
        remaining_spanners = carrying_spanners.copy()
        
        for loc, nuts in nuts_by_location.items():
            # If no usable spanners left, find nearest one
            if not remaining_spanners and usable_spanners:
                min_dist = float('inf')
                nearest_spanner = None
                for spanner, spanner_loc in spanner_locations.items():
                    dist = bfs(current_loc, spanner_loc)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spanner = spanner
                
                if nearest_spanner:
                    total_cost += min_dist  # Walk to spanner
                    total_cost += 1  # Pick up action
                    remaining_spanners.add(nearest_spanner)
                    current_loc = spanner_locations[nearest_spanner]
                    usable_spanners.remove(nearest_spanner)
                    del spanner_locations[nearest_spanner]
                else:
                    return float('inf')  # No reachable spanners
            
            if remaining_spanners:
                # Walk to nut location
                dist = bfs(current_loc, loc)
                if dist == float('inf'):
                    return float('inf')  # Unreachable
                total_cost += dist
                current_loc = loc
                
                # Tighten all nuts at this location
                for _ in nuts:
                    if remaining_spanners:
                        total_cost += 1  # Tighten action
                        remaining_spanners.pop()
                    else:
                        return float('inf')  # Not enough spanners
            else:
                return float('inf')  # No spanners available

        return total_cost
