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
    - Only one nut can be tightened per action
    - Spanners become unusable after tightening a nut
    - The man must be at the same location as the nut to tighten it

    # Heuristic Initialization
    - Extract link information to build a graph of locations
    - Identify all spanners and nuts in the problem
    - Store goal conditions (which nuts need to be tightened)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many nuts still need to be tightened (from goal conditions)
    2. Check if the man is carrying any usable spanners
    3. If not carrying usable spanners:
       a. Find the nearest usable spanner
       b. Add distance to reach it
       c. Add 1 action to pick it up
    4. For each loose nut:
       a. Add distance from current position (or spanner location) to nut
       b. Add 1 action to tighten it
    5. If multiple nuts need tightening, account for possible need to:
       a. Find additional spanners when current ones are used up
       b. Move between nut locations
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build a graph of connected locations
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state

        # Extract goal nuts that need to be tightened
        goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Count how many nuts still need to be tightened
        tightened_nuts = {get_parts(fact)[1] for fact in state if match(fact, "tightened", "*")}
        remaining_nuts = goal_nuts - tightened_nuts
        if not remaining_nuts:
            return 0  # Goal reached

        # Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
                break

        # Find all usable spanners not being carried
        usable_spanners = set()
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Check if spanner is not being carried
                carrying = any(match(f, "carrying", "bob", spanner) for f in state)
                if not carrying:
                    # Find spanner's location
                    for loc_fact in state:
                        if match(loc_fact, "at", spanner, "*"):
                            usable_spanners.add((spanner, get_parts(loc_fact)[2]))
                            break

        # Check if man is carrying any usable spanners
        carrying_usable = any(
            match(fact, "carrying", "bob", "*") and 
            any(match(usable_fact, "usable", get_parts(fact)[2]) for usable_fact in state)
            for fact in state
        )

        # BFS function to find shortest path between locations
        def bfs(start, end):
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
                    for neighbor in self.links.get(loc, []):
                        queue.append((neighbor, dist + 1))
            return float('inf')  # No path found

        total_cost = 0

        if not carrying_usable and usable_spanners:
            # Need to get a usable spanner first
            closest_spanner_dist = float('inf')
            closest_spanner_loc = None
            for spanner, loc in usable_spanners:
                dist = bfs(man_loc, loc)
                if dist < closest_spanner_dist:
                    closest_spanner_dist = dist
                    closest_spanner_loc = loc

            if closest_spanner_loc is not None:
                total_cost += closest_spanner_dist  # Walk to spanner
                total_cost += 1  # Pick up spanner
                man_loc = closest_spanner_loc  # Update man's location

        # For each remaining nut, find its location and distance
        nut_locations = {}
        for nut in remaining_nuts:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_locations[nut] = get_parts(fact)[2]
                    break

        # Calculate cost to tighten each nut
        for nut, nut_loc in nut_locations.items():
            dist = bfs(man_loc, nut_loc)
            total_cost += dist  # Walk to nut
            total_cost += 1  # Tighten nut
            man_loc = nut_loc  # Update man's location

            # Spanner becomes unusable after tightening
            carrying_usable = False

            # If there are more nuts, we might need another spanner
            if len(remaining_nuts) > 1 and not carrying_usable and usable_spanners:
                # Find next closest usable spanner
                closest_spanner_dist = float('inf')
                closest_spanner_loc = None
                for spanner, loc in usable_spanners:
                    dist = bfs(man_loc, loc)
                    if dist < closest_spanner_dist:
                        closest_spanner_dist = dist
                        closest_spanner_loc = loc

                if closest_spanner_loc is not None:
                    total_cost += closest_spanner_dist  # Walk to spanner
                    total_cost += 1  # Pick up spanner
                    man_loc = closest_spanner_loc  # Update man's location

        return total_cost
