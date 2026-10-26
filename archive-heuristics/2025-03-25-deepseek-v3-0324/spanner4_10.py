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
    It considers the following aspects:
    - The man's current location and whether he needs to move to pick up spanners or reach nuts
    - Whether the man is carrying usable spanners
    - The locations of loose nuts and available spanners
    - The path distances between locations in the environment

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can only be used once (becomes unusable after tightening a nut)
    - The environment is static (links between locations don't change)
    - The shortest path between locations is always used for movement

    # Heuristic Initialization
    - Extract the link information between locations to build a graph for pathfinding
    - Identify all spanners and nuts in the problem
    - Store goal conditions (which nuts need to be tightened)

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many loose nuts still need to be tightened (from goal conditions)
    2. Check if the man is carrying any usable spanners:
       - If not, find the nearest usable spanner and add the path distance to reach it
    3. For each loose nut:
       - Calculate the distance from man's current location (or spanner pickup location) to the nut
       - Add 1 action for tightening (if we have a usable spanner)
    4. If we need more spanners than currently carried:
       - Add actions to pick up additional spanners along the way
    5. The total heuristic is the sum of:
       - Movement actions (walking between locations)
       - Pickup actions (for spanners)
       - Tightening actions (for nuts)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build a graph of location connections from static 'link' facts
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

    def _shortest_path_distance(self, start, end):
        """Calculate the shortest path distance between two locations using BFS."""
        if start == end:
            return 0

        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            if current == end:
                return distance
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.location_graph.get(current, set()):
                queue.append((neighbor, distance + 1))
        
        return float('inf')  # No path exists

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Extract current state information
        man_location = None
        carrying_spanners = set()
        usable_spanners = set()
        spanner_locations = {}
        loose_nuts = set()
        nut_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carrying_spanners.add(parts[2])
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

        # Determine which nuts still need to be tightened (from goals)
        nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nuts_to_tighten.add(get_parts(goal)[1])

        # If all goal nuts are already tightened, heuristic is 0
        if not nuts_to_tighten:
            return 0

        total_cost = 0
        current_location = man_location
        available_spanners = carrying_spanners & usable_spanners

        # If no usable spanners are being carried, find the nearest one
        if not available_spanners:
            min_distance = float('inf')
            nearest_spanner_loc = None
            
            for spanner in usable_spanners:
                if spanner in spanner_locations:
                    distance = self._shortest_path_distance(current_location, spanner_locations[spanner])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_spanner_loc = spanner_locations[spanner]
            
            if nearest_spanner_loc is not None:
                total_cost += min_distance
                current_location = nearest_spanner_loc
                # After picking up, we have at least one usable spanner
                available_spanners = set(next(iter(usable_spanners)))

        # Calculate cost to tighten each remaining nut
        for nut in nuts_to_tighten:
            if nut in nut_locations:
                nut_loc = nut_locations[nut]
                distance = self._shortest_path_distance(current_location, nut_loc)
                total_cost += distance + 1  # +1 for tighten action
                current_location = nut_loc
                
                # Spanner becomes unusable after tightening
                if available_spanners:
                    available_spanners.pop()
                else:
                    # Need to get another spanner
                    min_distance = float('inf')
                    nearest_spanner_loc = None
                    
                    for spanner in usable_spanners:
                        if spanner in spanner_locations:
                            distance = self._shortest_path_distance(current_location, spanner_locations[spanner])
                            if distance < min_distance:
                                min_distance = distance
                                nearest_spanner_loc = spanner_locations[spanner]
                    
                    if nearest_spanner_loc is not None:
                        total_cost += min_distance + 1  # +1 for pickup action
                        current_location = nearest_spanner_loc
                        available_spanners = set(next(iter(usable_spanners)))

        return total_cost
