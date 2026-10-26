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
    It considers:
    - The man's current location
    - Whether he's carrying usable spanners
    - The locations of loose nuts and usable spanners
    - The path needed to collect spanners and reach nuts

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can only be used once (becomes unusable after tightening)
    - The man must be at the nut's location to tighten it
    - The man must be carrying a usable spanner to tighten a nut

    # Heuristic Initialization
    - Extract link information from static facts to build a graph of locations
    - Identify goal nuts (those that need to be tightened)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (from goal conditions)
    2. Check if man is already carrying usable spanners
    3. For each loose nut:
       a. Calculate distance from man's current position to nut's location
       b. If no usable spanners are carried, find nearest usable spanner:
          - Calculate distance from man's position to spanner
          - Add distance from spanner to nut
       c. Add 1 action for tightening the nut
    4. The heuristic is the sum of:
       - Walking distances (each step = 1 action)
       - 1 action per spanner pickup
       - 1 action per nut tightening
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

        # Identify which nuts need to be tightened (from goals)
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def _bfs_distance(self, start, end):
        """Calculate the shortest path distance between two locations using BFS."""
        if start == end:
            return 0

        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == end:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.links.get(current, set()):
                queue.append((neighbor, dist + 1))
        
        return float('inf')  # No path exists

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
        spanner_locations = {}
        nut_locations = {}
        loose_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "at", "spanner*", "*"):
                spanner_locations[parts[1]] = parts[2]
            elif match(fact, "at", "nut*", "*"):
                nut_locations[parts[1]] = parts[2]
            elif match(fact, "loose", "*"):
                if parts[1] in self.goal_nuts:  # Only count nuts that need tightening
                    loose_nuts.add(parts[1])

        # Calculate required actions
        total_cost = 0
        remaining_spanners = usable_spanners & set(spanner_locations.keys())
        carried_usable = carried_spanners & usable_spanners

        for nut in loose_nuts:
            nut_loc = nut_locations[nut]
            
            if not carried_usable:
                # Need to pick up a spanner first
                if not remaining_spanners:
                    return float('inf')  # No usable spanners left
                
                # Find nearest usable spanner
                min_spanner_dist = float('inf')
                best_spanner = None
                for spanner in remaining_spanners:
                    spanner_loc = spanner_locations[spanner]
                    dist = self._bfs_distance(man_location, spanner_loc)
                    if dist < min_spanner_dist:
                        min_spanner_dist = dist
                        best_spanner = spanner
                
                if best_spanner is None:
                    return float('inf')  # No reachable spanners
                
                total_cost += min_spanner_dist  # Walk to spanner
                total_cost += 1  # Pick up spanner
                man_location = spanner_locations[best_spanner]
                remaining_spanners.remove(best_spanner)
                carried_usable.add(best_spanner)
            
            # Walk to nut location
            walk_dist = self._bfs_distance(man_location, nut_loc)
            total_cost += walk_dist
            man_location = nut_loc
            
            # Tighten nut
            total_cost += 1
            carried_usable.pop()  # Spanner becomes unusable after tightening

        return total_cost
