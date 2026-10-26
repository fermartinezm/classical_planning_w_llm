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
    - The path needed to collect spanners and reach nuts

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can only be used once (becomes unusable after tightening)
    - The path between locations is always the shortest possible
    - The man must be at the nut's location to tighten it

    # Heuristic Initialization
    - Extract the link graph between locations from static facts
    - Identify goal nuts that need to be tightened

    # Step-By-Step Thinking for Computing Heuristic
    1. For each loose nut that needs tightening:
       a. If the man is already at the nut's location with a usable spanner:
          - Just need to perform the tighten action (cost: 1)
       b. Else:
          - Need to collect a usable spanner (if not already carrying one)
          - Need to walk to the nut's location
          - Perform the tighten action
    2. For collecting spanners:
       a. If already carrying usable spanners, no additional cost
       b. Else, find the nearest usable spanner and calculate walking distance
    3. For walking to nuts:
       a. Calculate shortest path from current location (or spanner location) to nut
    4. Sum all required actions:
       - Pickup actions for spanners
       - Walk actions for movement
       - Tighten actions for nuts
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Build a graph of location links for pathfinding
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)  # links are bidirectional

    def __call__(self, node):
        """Estimate the minimum cost to tighten all required nuts."""
        state = node.state
        
        # Check if we're already in a goal state
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
            elif match(fact, "at", "*", "*") and parts[1].startswith("spanner"):
                spanner_locations[parts[1]] = parts[2]
            elif match(fact, "at", "*", "*") and parts[1].startswith("nut"):
                pass  # nut locations are only needed if they're loose
        
        # Only consider nuts that are both loose and need tightening (in goals)
        nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                if f"(loose {nut})" in state:
                    nuts_to_tighten.add(nut)
        
        if not nuts_to_tighten:
            return 0  # all required nuts are already tightened
        
        total_cost = 0
        
        # Check if we need to collect spanners
        usable_carried = carried_spanners & usable_spanners
        if not usable_carried:
            # Find nearest usable spanner not being carried
            nearest_spanner_dist = float('inf')
            for spanner in usable_spanners - carried_spanners:
                spanner_loc = spanner_locations.get(spanner)
                if spanner_loc:
                    dist = self._shortest_path_distance(man_location, spanner_loc)
                    if dist < nearest_spanner_dist:
                        nearest_spanner_dist = dist
            
            if nearest_spanner_dist != float('inf'):
                total_cost += nearest_spanner_dist  # walk to spanner
                total_cost += 1  # pickup action
                # Update man's location to spanner location after walking
                man_location = spanner_loc
        
        # For each nut to tighten, calculate path from current location
        for nut in nuts_to_tighten:
            # Find nut location
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]
                    break
            
            if nut_loc:
                if man_location != nut_loc:
                    dist = self._shortest_path_distance(man_location, nut_loc)
                    total_cost += dist  # walk to nut
                    man_location = nut_loc  # update location after walking
                
                total_cost += 1  # tighten action
        
        return total_cost
    
    def _shortest_path_distance(self, start, end):
        """Calculate shortest path distance between two locations using BFS."""
        if start == end:
            return 0
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == end:
                return dist
                
            if current not in visited:
                visited.add(current)
                for neighbor in self.links.get(current, []):
                    queue.append((neighbor, dist + 1))
        
        return float('inf')  # no path found (shouldn't happen in valid problems)
