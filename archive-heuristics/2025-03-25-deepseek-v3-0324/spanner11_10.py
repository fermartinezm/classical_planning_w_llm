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
    - The man's current location and carried spanners
    - Locations of loose nuts and usable spanners
    - The path needed to collect spanners and reach nuts

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can only be used once (becomes unusable after tightening)
    - The man must be at the nut's location to tighten it
    - The man must be carrying a usable spanner to tighten a nut

    # Heuristic Initialization
    - Extract link information from static facts to build a graph of locations
    - Identify goal nuts from the task goals

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need tightening (from state and goals)
    2. Check if man is carrying usable spanners (if not, need to collect some)
    3. For each loose nut:
       a. Calculate distance from man's current position to nut's location
       b. If no usable spanner is carried, add distance to nearest usable spanner
    4. For each needed spanner (if not already carried):
       a. Add 1 action to pick it up
    5. For each nut:
       a. Add 1 action to tighten it (if preconditions are met)
    6. Sum all movement actions (walking) and other actions (pickup, tighten)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build graph of locations from link facts
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Identify which nuts need to be tightened (from goals)
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        total_cost = 0

        # Get man's current location
        man_loc = None
        carrying = set()
        usable_spanners = set()
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
            elif match(fact, "carrying", "bob", "*"):
                carrying.add(get_parts(fact)[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(get_parts(fact)[1])

        # Find all loose nuts that need tightening
        loose_nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.goal_nuts:
                    loose_nuts.add(nut)
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj.startswith("nut"):
                    nut_locations[obj] = loc

        # If no loose nuts left, return 0 (goal reached)
        if not loose_nuts:
            return 0

        # Find usable spanners not being carried
        available_spanners = usable_spanners - carrying

        # Find locations of available spanners
        spanner_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj in available_spanners:
                    spanner_locations[obj] = loc

        # Calculate movement costs
        movement_cost = 0

        # If not carrying usable spanners, go get one
        if not (carrying & usable_spanners) and available_spanners:
            # Find nearest spanner
            min_dist = float('inf')
            for spanner, loc in spanner_locations.items():
                dist = self._shortest_path_length(man_loc, loc)
                if dist < min_dist:
                    min_dist = dist
            movement_cost += min_dist
            # Add pickup action
            total_cost += 1

        # Calculate cost to reach each nut
        for nut in loose_nuts:
            nut_loc = nut_locations[nut]
            dist = self._shortest_path_length(man_loc, nut_loc)
            movement_cost += dist
            # Add tighten action
            total_cost += 1

        total_cost += movement_cost

        return total_cost

    def _shortest_path_length(self, start, end):
        """Calculate shortest path length between two locations using BFS."""
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
            for neighbor in self.location_graph.get(current, []):
                queue.append((neighbor, dist + 1))
        return float('inf')  # No path found (shouldn't happen in valid problems)
