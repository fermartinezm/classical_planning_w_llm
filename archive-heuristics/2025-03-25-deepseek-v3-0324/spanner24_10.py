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
    It considers the man's current location, carried spanners, and the locations of
    loose nuts and usable spanners.

    # Assumptions:
    - The man can carry multiple spanners at once.
    - Each spanner can only be used once (becomes unusable after tightening a nut).
    - The man must be at the same location as a nut to tighten it.
    - The man must be carrying a usable spanner to tighten a nut.

    # Heuristic Initialization
    - Extract the link information between locations from static facts.
    - Identify goal conditions (which nuts need to be tightened).

    # Step-By-Step Thinking for Computing Heuristic
    1. Count how many loose nuts still need to be tightened (not in goal state).
    2. Check if the man is carrying any usable spanners.
    3. If not carrying usable spanners:
       - Find the nearest usable spanner to the man's current location.
       - Add the path length to reach that spanner.
       - Add 1 action to pick up the spanner.
    4. For each loose nut:
       - Calculate the path length from man's current location to the nut.
       - Add 1 action to tighten the nut (if carrying usable spanner).
    5. If multiple loose nuts exist at the same location, the man can tighten them
       consecutively without moving (1 action per nut).
    6. The total heuristic is the sum of all movement and action costs.
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

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
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
        nut_locations = {}
        
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
            elif match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("spanner"):
                    spanner_locations[obj] = loc
                elif obj.startswith("nut"):
                    nut_locations[obj] = loc
        
        # Calculate the shortest path between two locations using BFS
        def shortest_path(start, end):
            if start == end:
                return 0
            visited = set()
            queue = [(start, 0)]
            while queue:
                loc, dist = queue.pop(0)
                if loc == end:
                    return dist
                if loc in visited:
                    continue
                visited.add(loc)
                for neighbor in self.location_graph.get(loc, []):
                    queue.append((neighbor, dist + 1))
            return float('inf')  # unreachable
        
        total_cost = 0
        
        # Count how many nuts still need to be tightened
        nuts_to_tighten = [nut for nut in loose_nuts if f"(tightened {nut})" not in self.goals]
        if not nuts_to_tighten:
            return 0
            
        # Check usable spanners being carried
        usable_carried = [s for s in carried_spanners if s in usable_spanners]
        
        # If no usable spanners are being carried, we need to get one
        if not usable_carried:
            # Find the nearest usable spanner not being carried
            nearest_spanner_dist = float('inf')
            nearest_spanner_loc = None
            for spanner in usable_spanners:
                if spanner not in carried_spanners and spanner in spanner_locations:
                    dist = shortest_path(man_location, spanner_locations[spanner])
                    if dist < nearest_spanner_dist:
                        nearest_spanner_dist = dist
                        nearest_spanner_loc = spanner_locations[spanner]
            
            if nearest_spanner_loc is not None:
                total_cost += nearest_spanner_dist  # walk to spanner
                total_cost += 1  # pickup action
                man_location = nearest_spanner_loc  # update man's location
                # Assume we pick up one usable spanner
                usable_carried = [spanner for spanner in usable_spanners 
                                if spanner_locations.get(spanner, None) == nearest_spanner_loc][:1]
        
        # If we still don't have usable spanners, problem is unsolvable
        if not usable_carried:
            return float('inf')
            
        # Group nuts by location
        nuts_by_location = {}
        for nut in nuts_to_tighten:
            loc = nut_locations[nut]
            nuts_by_location.setdefault(loc, []).append(nut)
        
        # Calculate cost to tighten all nuts
        for loc, nuts in nuts_by_location.items():
            if loc != man_location:
                total_cost += shortest_path(man_location, loc)
                man_location = loc
            # Each nut takes 1 action to tighten (if we have spanners)
            total_cost += min(len(nuts), len(usable_carried))
            # Update usable spanners count (each can only be used once)
            usable_carried = usable_carried[min(len(nuts), len(usable_carried)):]
        
        return total_cost
