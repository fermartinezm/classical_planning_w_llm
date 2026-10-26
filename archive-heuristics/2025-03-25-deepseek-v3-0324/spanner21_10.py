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
    - The path between locations is always bidirectional (link implies both directions)
    - The goal is always to tighten all nuts (no partial goals)

    # Heuristic Initialization
    - Extract the link graph from static facts to compute distances between locations
    - Identify all nuts that need to be tightened from the goal conditions

    # Step-By-Step Thinking for Computing Heuristic
    1. For each loose nut that needs tightening:
        a. If there's no usable spanner left, return infinity (unsolvable)
        b. If the man is at the nut's location with a usable spanner, cost is 1 (tighten)
        c. Otherwise:
            i. Find the nearest usable spanner to the nut's location
            ii. Calculate path distance from man's current location to spanner to nut
            iii. Add cost for picking up spanner (1) and tightening (1)
    2. Sum costs for all loose nuts, optimizing for shared paths where possible
    3. If man is carrying usable spanners, consider them as being at current location
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        
        # Build graph of location links for pathfinding
        self.links = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1)  # assume bidirectional

    def _bfs_distance(self, start, goal):
        """Compute shortest path distance between two locations using BFS."""
        if start == goal:
            return 0
            
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, dist = queue.pop(0)
            if current == goal:
                return dist
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self.links.get(current, set()):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        return float('inf')  # no path exists

    def __call__(self, node):
        """Estimate the minimum cost to tighten all loose nuts."""
        state = node.state
        
        # Extract current state information
        man_location = None
        carried_spanners = set()
        spanner_locations = {}
        nut_locations = set()
        usable_spanners = set()
        
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanners.add(parts[2])
            elif match(fact, "at", "spanner*", "*"):
                spanner_locations[parts[1]] = parts[2]
            elif match(fact, "at", "nut*", "*"):
                nut_locations.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "loose", "*"):
                pass  # we'll check goals for which nuts need tightening
        
        # Determine which nuts still need tightening
        nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nuts_to_tighten.add(get_parts(goal)[1])
        
        if not nuts_to_tighten:
            return 0  # all nuts already tightened
        
        total_cost = 0
        
        # For each nut that needs tightening
        for nut in nuts_to_tighten:
            # Find nut location (assuming it's in the state)
            nut_loc = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_loc = get_parts(fact)[2]
                    break
            
            if not nut_loc:
                continue  # nut not found in state
            
            # Find usable spanners (including carried ones)
            available_spanners = []
            for spanner in usable_spanners:
                if spanner in carried_spanners:
                    available_spanners.append((spanner, man_location))
                elif spanner in spanner_locations:
                    available_spanners.append((spanner, spanner_locations[spanner]))
            
            if not available_spanners:
                return float('inf')  # no usable spanners left
            
            # Find nearest spanner to the nut
            min_spanner_cost = float('inf')
            for spanner, spanner_loc in available_spanners:
                # Cost to get spanner (if not already carried)
                if spanner in carried_spanners:
                    get_spanner_cost = 0
                else:
                    # Walk to spanner and pick it up
                    walk_cost = self._bfs_distance(man_location, spanner_loc)
                    get_spanner_cost = walk_cost + 1  # +1 for pickup
                
                # Cost to go from spanner location to nut
                walk_to_nut_cost = self._bfs_distance(spanner_loc, nut_loc)
                
                total_spanner_cost = get_spanner_cost + walk_to_nut_cost + 1  # +1 for tighten
                
                if total_spanner_cost < min_spanner_cost:
                    min_spanner_cost = total_spanner_cost
            
            total_cost += min_spanner_cost
        
        return total_cost
