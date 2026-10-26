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
    - Only one man exists in the problem (as per the examples)
    - Spanners become unusable after tightening a nut
    - The man can carry multiple spanners at once
    - The path between locations is always the shortest possible

    # Heuristic Initialization
    - Extract link information between locations to compute distances
    - Identify goal nuts that need to be tightened

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts that still need to be tightened (not in goal state)
    2. For each loose nut:
       a. If the man is at the nut's location with a usable spanner:
          - Add 1 action (tighten)
       b. Else:
          - Find the nearest usable spanner (considering already carried ones)
          - Compute path cost to:
            * Pick up the spanner (if not already carried)
            * Walk to the nut's location
            * Tighten the nut
    3. Sum the actions required for all loose nuts
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
        
        # Extract goal nuts that need to be tightened
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state
        
        # Check if we're already in a goal state
        if self.goals <= state:
            return 0
            
        # Track current state information
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
            elif match(fact, "carrying", "*", "*"):
                carried_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "at", "spanner*", "*"):
                spanner_locations[parts[1]] = parts[2]
            elif match(fact, "loose", "*"):
                loose_nuts.add(parts[1])
            elif match(fact, "at", "nut*", "*"):
                nut_locations[parts[1]] = parts[2]
        
        # Only consider nuts that are both loose and in goals
        target_nuts = loose_nuts & self.goal_nuts
        if not target_nuts:
            return 0
            
        total_cost = 0
        
        # For each nut that needs tightening
        for nut in target_nuts:
            nut_loc = nut_locations[nut]
            
            # Check if we're already at the nut's location with a usable spanner
            if man_location == nut_loc:
                usable_carried = [s for s in carried_spanners if s in usable_spanners]
                if usable_carried:
                    total_cost += 1  # Just need to tighten
                    usable_spanners.remove(usable_carried[0])  # Spanner becomes unusable
                    continue
            
            # Need to get to the nut with a usable spanner
            # Find the closest usable spanner (either carried or on the ground)
            closest_spanner = None
            min_distance = float('inf')
            
            # Check carried spanners first
            for spanner in carried_spanners:
                if spanner in usable_spanners:
                    distance = self._path_length(man_location, nut_loc)
                    if distance < min_distance:
                        min_distance = distance
                        closest_spanner = spanner
            
            # Check spanners on the ground
            for spanner, loc in spanner_locations.items():
                if spanner in usable_spanners:
                    # Distance to pick up spanner then go to nut
                    pickup_dist = self._path_length(man_location, loc)
                    nut_dist = self._path_length(loc, nut_loc)
                    total_dist = pickup_dist + nut_dist
                    if total_dist < min_distance:
                        min_distance = total_dist
                        closest_spanner = spanner
            
            if closest_spanner is None:
                continue  # No usable spanners left for this nut
                
            # Calculate actions needed:
            # 1. If spanner needs to be picked up (not carried)
            if closest_spanner not in carried_spanners:
                total_cost += 1  # pickup action
                # Update man's location to spanner's location
                man_location = spanner_locations[closest_spanner]
            
            # 2. Walk to nut location (if not already there)
            if man_location != nut_loc:
                total_cost += min_distance  # walk actions
            
            # 3. Tighten the nut
            total_cost += 1
            
            # Spanner becomes unusable
            usable_spanners.discard(closest_spanner)
        
        return total_cost
    
    def _path_length(self, start, end):
        """Compute the shortest path length between two locations using BFS."""
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
            
            for neighbor in self.location_graph.get(current, []):
                queue.append((neighbor, distance + 1))
        
        return float('inf')  # No path exists (shouldn't happen in valid problems)
