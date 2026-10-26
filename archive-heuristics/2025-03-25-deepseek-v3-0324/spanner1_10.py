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
    It considers the following factors:
    - Distance the man needs to walk to collect spanners and reach nuts
    - Whether the man needs to pick up spanners
    - Whether spanners need to be used to tighten nuts

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can only be used once (becomes unusable after tightening)
    - The man must be at the nut's location to tighten it
    - The man must be carrying a usable spanner to tighten a nut

    # Heuristic Initialization
    - Extract link information between locations for path planning
    - Identify all spanners and nuts from static facts
    - Store goal conditions (which nuts need to be tightened)

    # Step-By-Step Thinking for Computing Heuristic
    1. For each loose nut that needs to be tightened:
        a. If the man is not at the nut's location:
            - Calculate the shortest path distance from current location to nut
            - Add walking actions needed
        b. If the man doesn't have a usable spanner:
            - Find the nearest usable spanner
            - Add walking actions to reach it and pickup action
            - If no usable spanners are available, the state is unsolvable
    2. For each tightening action needed:
        - Add the tightening action
    3. Sum all these actions to get the total heuristic estimate
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

        # Check if goal is already satisfied
        if all(goal in state for goal in self.goals):
            return 0

        # Extract current state information
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
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
            elif match(fact, "at", "*", "*") and parts[1].startswith("nut"):
                nut_locations[parts[1]] = parts[2]
            elif match(fact, "loose", "*"):
                loose_nuts.add(parts[1])

        # Identify which nuts still need to be tightened
        nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                if nut in loose_nuts:
                    nuts_to_tighten.add(nut)

        if not nuts_to_tighten:
            return 0

        total_cost = 0

        # BFS function to find shortest path between locations
        def bfs(start, end):
            if start == end:
                return 0
            visited = {start}
            queue = [(start, 0)]
            while queue:
                loc, dist = queue.pop(0)
                for neighbor in self.location_graph.get(loc, []):
                    if neighbor == end:
                        return dist + 1
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return float('inf')  # No path found

        current_loc = man_location
        available_spanners = carried_spanners & usable_spanners

        for nut in nuts_to_tighten:
            nut_loc = nut_locations[nut]

            # If we don't have a usable spanner, find the nearest one
            if not available_spanners:
                # Find all usable spanners not being carried
                unused_spanners = []
                for fact in state:
                    if match(fact, "at", "spanner*", "*") and match(fact, "usable", "*"):
                        spanner = get_parts(fact)[1]
                        if spanner not in carried_spanners:
                            loc = get_parts(fact)[2]
                            unused_spanners.append((spanner, loc))

                if not unused_spanners:
                    return float('inf')  # No usable spanners left

                # Find nearest spanner
                min_dist = float('inf')
                nearest_spanner = None
                nearest_loc = None
                for spanner, loc in unused_spanners:
                    dist = bfs(current_loc, loc)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spanner = spanner
                        nearest_loc = loc

                if min_dist == float('inf'):
                    return float('inf')  # Unreachable spanner

                # Add cost to get spanner
                total_cost += min_dist  # Walk to spanner
                total_cost += 1  # Pickup action
                current_loc = nearest_loc
                available_spanners.add(nearest_spanner)

            # Go to nut location if not already there
            if current_loc != nut_loc:
                dist = bfs(current_loc, nut_loc)
                if dist == float('inf'):
                    return float('inf')  # Unreachable nut
                total_cost += dist
                current_loc = nut_loc

            # Tighten the nut
            total_cost += 1
            available_spanners.pop()  # Spanner becomes unusable

        return total_cost
