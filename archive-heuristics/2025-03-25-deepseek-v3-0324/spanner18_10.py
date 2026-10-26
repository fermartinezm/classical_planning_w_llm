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
    - Only one man exists in the problem

    # Heuristic Initialization
    - Extract link information between locations to compute distances
    - Identify goal nuts that need to be tightened

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (from goals)
    2. For each loose nut:
       a. If already at nut's location with usable spanner: cost = 1 (just tighten)
       b. Else:
          i. Need to get a usable spanner (if not already carrying one)
             - Find nearest usable spanner
             - Add distance to spanner and back to nut's location
          ii. Need to reach nut's location (if not already there)
              - Add distance from current/spanner location to nut
       c. Add 1 for the tighten action
    3. Sum all costs for each nut
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
        
        # Extract goal nuts that need to be tightened
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if goal is already reached
        if self.goals <= state:
            return 0
            
        # Find man's current location
        man_loc = None
        carrying = set()
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
            elif match(fact, "carrying", "bob", "*"):
                carrying.add(get_parts(fact)[2])
        
        # Find all usable spanners (not in carrying implies they're at some location)
        usable_spanners = set()
        spanner_locations = {}
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in usable_spanners and obj not in carrying:
                    spanner_locations[obj] = loc
        
        # Find loose nuts and their locations
        loose_nuts = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.goal_nuts:  # Only count nuts that are in goals
                    # Find nut's location
                    for loc_fact in state:
                        if match(loc_fact, "at", nut, "*"):
                            loose_nuts[nut] = get_parts(loc_fact)[2]
                            break
        
        if not loose_nuts:
            return 0  # All required nuts are tightened
        
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
            return float('inf')  # No path found (shouldn't happen in valid problems)
        
        for nut, nut_loc in loose_nuts.items():
            # Check if we're already at the nut's location with a usable spanner
            if man_loc == nut_loc and carrying:
                total_cost += 1  # Just need to tighten
                continue
            
            # Need to get a usable spanner if not carrying one
            if not carrying:
                # Find nearest usable spanner
                min_spanner_dist = float('inf')
                nearest_spanner_loc = None
                for spanner, spanner_loc in spanner_locations.items():
                    dist = bfs(man_loc, spanner_loc)
                    if dist < min_spanner_dist:
                        min_spanner_dist = dist
                        nearest_spanner_loc = spanner_loc
                
                if min_spanner_dist != float('inf'):
                    total_cost += min_spanner_dist  # Go to spanner
                    man_loc = nearest_spanner_loc  # Update man's location
                    # Pick up is free since we'll pass through the location anyway
                
            # Now go to nut's location
            dist_to_nut = bfs(man_loc, nut_loc)
            total_cost += dist_to_nut
            man_loc = nut_loc  # Update man's location
            
            # Tighten the nut
            total_cost += 1
            
            # Spanner becomes unusable after tightening
            if carrying:
                carrying.pop()  # Remove one spanner
        
        return total_cost
