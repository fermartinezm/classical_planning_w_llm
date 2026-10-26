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
    - The locations of loose nuts and usable spanners
    - The path distances between locations

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can only be used once (becomes unusable after tightening)
    - The path between locations is always the shortest possible
    - The man must be at the nut's location to tighten it

    # Heuristic Initialization
    - Extract link information between locations to build a graph for path finding
    - Identify all spanners and nuts from the static facts
    - Store goal conditions (which nuts need to be tightened)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need tightening (from goal conditions)
    2. Check if man is carrying any usable spanners:
       - If not, find the closest usable spanner and add path cost to reach it
    3. For each loose nut:
       - Calculate path cost from man's current location to nut's location
       - If no usable spanner is carried, add path cost to nearest spanner
    4. Add 1 action for each nut that needs tightening (tighten_nut action)
    5. If multiple spanners are needed (more nuts than carried spanners), 
       add additional spanner collection trips
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build graph of location links for path finding
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)

    def _shortest_path(self, start, end, visited=None):
        """BFS to find shortest path between two locations."""
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
        """Estimate the minimum cost to tighten all required nuts."""
        state = node.state
        total_cost = 0

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

        # Determine which nuts still need tightening (from goals)
        nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nuts_to_tighten.add(get_parts(goal)[1])

        # If no nuts left to tighten, return 0
        if not nuts_to_tighten:
            return 0

        # Find usable spanners not being carried
        available_spanners = [
            s for s in usable_spanners 
            if s not in carried_spanners and s in spanner_locations
        ]

        current_loc = man_location
        remaining_nuts = list(nuts_to_tighten)
        remaining_spanners = list(carried_spanners & usable_spanners)

        # While there are nuts to tighten
        while remaining_nuts:
            nut = remaining_nuts.pop()
            nut_loc = nut_locations[nut]

            # If we need a spanner and don't have one, go get one
            if not remaining_spanners and available_spanners:
                # Find closest available spanner
                closest_dist = float('inf')
                closest_spanner = None
                for spanner in available_spanners:
                    spanner_loc = spanner_locations[spanner]
                    dist = self._shortest_path(current_loc, spanner_loc)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_spanner = spanner

                if closest_spanner:
                    total_cost += closest_dist  # walk to spanner
                    total_cost += 1  # pickup_spanner action
                    remaining_spanners.append(closest_spanner)
                    available_spanners.remove(closest_spanner)
                    current_loc = spanner_locations[closest_spanner]

            # Go to nut location if not already there
            if current_loc != nut_loc:
                dist = self._shortest_path(current_loc, nut_loc)
                if dist == float('inf'):
                    return float('inf')  # unreachable
                total_cost += dist
                current_loc = nut_loc

            # Tighten the nut (if we have a spanner)
            if remaining_spanners:
                total_cost += 1  # tighten_nut action
                remaining_spanners.pop()
            else:
                return float('inf')  # no spanners left

        return total_cost
