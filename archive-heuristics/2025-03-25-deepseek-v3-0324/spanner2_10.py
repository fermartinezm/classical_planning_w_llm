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
    - Distance to the nearest usable spanner
    - Whether the man is already carrying a usable spanner
    - Distance from current position to each loose nut
    - Number of nuts that still need to be tightened

    # Assumptions:
    - The man can carry multiple spanners at once
    - Only one nut can be tightened per action (even if multiple spanners are carried)
    - Spanners become unusable after tightening a nut
    - The man must be at the same location as the nut to tighten it

    # Heuristic Initialization
    - Extract link information from static facts to build a graph of locations
    - Identify all nuts that need to be tightened from the goal conditions

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many nuts still need to be tightened (from goal conditions)
    2. Check if the man is carrying any usable spanners:
       - If yes, calculate distance to nearest loose nut
       - If no, find nearest usable spanner and calculate distance to it
    3. For each loose nut:
       - Calculate distance from man's current position to nut's location
       - If carrying a usable spanner, add distance to nut
       - If not, add distance to nearest spanner plus distance from spanner to nut
    4. The heuristic value is:
       - Number of tighten actions needed (one per nut)
       - Plus walking actions needed to reach spanners/nuts
       - Plus pickup actions if spanners need to be collected
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
        """Estimate the minimum cost to tighten all nuts that are still loose."""
        state = node.state

        # Get all nuts that need to be tightened (from goals)
        nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nuts_to_tighten.add(get_parts(goal)[1])

        # Get current state information
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        spanner_locations = {}
        nut_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("spanner"):
                    spanner_locations[obj] = loc
                elif obj.startswith("nut"):
                    nut_locations[obj] = loc
                elif obj.startswith("bob"):  # Assuming the man is named 'bob'
                    man_location = loc
            elif match(fact, "carrying", "*", "*"):
                _, man, spanner = parts
                carried_spanners.add(spanner)
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])

        # Remove already tightened nuts
        for fact in state:
            if match(fact, "tightened", "*"):
                nut = get_parts(fact)[1]
                nuts_to_tighten.discard(nut)

        if not nuts_to_tighten:
            return 0  # All nuts are already tightened

        # Calculate distances between locations using BFS
        def bfs_distance(start, end):
            if start == end:
                return 0
            visited = set()
            queue = [(start, 0)]
            while queue:
                loc, dist = queue.pop(0)
                if loc == end:
                    return dist
                if loc not in visited:
                    visited.add(loc)
                    for neighbor in self.location_graph.get(loc, []):
                        queue.append((neighbor, dist + 1))
            return float('inf')  # No path found

        # Find usable spanners not being carried
        available_spanners = []
        for spanner in usable_spanners:
            if spanner not in carried_spanners:
                available_spanners.append(spanner)

        # Calculate minimal cost
        total_cost = 0

        # If not carrying any usable spanner, find and pick up one
        usable_carried = [s for s in carried_spanners if s in usable_spanners]
        if not usable_carried and available_spanners:
            # Find nearest usable spanner
            min_dist = float('inf')
            nearest_spanner_loc = None
            for spanner in available_spanners:
                spanner_loc = spanner_locations[spanner]
                dist = bfs_distance(man_location, spanner_loc)
                if dist < min_dist:
                    min_dist = dist
                    nearest_spanner_loc = spanner_loc
            
            if nearest_spanner_loc is not None:
                total_cost += min_dist + 1  # walk + pickup
                man_location = nearest_spanner_loc  # update position after pickup

        # For each nut to tighten
        for nut in nuts_to_tighten:
            nut_loc = nut_locations[nut]
            dist_to_nut = bfs_distance(man_location, nut_loc)
            total_cost += dist_to_nut + 1  # walk + tighten
            man_location = nut_loc  # update position after tightening

        return total_cost
