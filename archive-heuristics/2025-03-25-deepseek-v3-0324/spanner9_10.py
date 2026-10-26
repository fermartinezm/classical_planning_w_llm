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
    This heuristic estimates the number of actions needed to tighten all loose nuts
    in the goal state. It considers:
    - The man's current location and carried spanners
    - Locations of loose nuts and usable spanners
    - The path needed to collect spanners and reach nuts

    # Assumptions:
    - Each nut requires exactly one spanner to be tightened
    - A spanner becomes unusable after tightening one nut
    - The man can carry multiple spanners at once
    - The path between locations is always the shortest possible

    # Heuristic Initialization
    - Extract link information between locations from static facts
    - Identify goal nuts that need to be tightened

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (goal nuts not yet tightened)
    2. For each loose nut:
       a. If no usable spanner is being carried:
          i. Find nearest usable spanner
          ii. Add cost to walk to spanner and pick it up
       b. Add cost to walk to nut's location
       c. Add cost to tighten the nut (1 action)
    3. If multiple spanners are needed (more than carried), repeat step 2a
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
        cost = 0

        # Check if we're already in a goal state
        if all(goal in state for goal in self.goals):
            return 0

        # Get current man's location
        man_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_loc = get_parts(fact)[2]
                break

        # Get currently carried spanners
        carried_spanners = set()
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                carried_spanners.add(get_parts(fact)[2])

        # Get usable spanners (either carried or on the ground)
        usable_spanners = set()
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Check if it's carried or on the ground
                if (f"(carrying bob {spanner})" in state or 
                    any(match(f, "at", spanner, "*") for f in state)):
                    usable_spanners.add(spanner)

        # Get locations of loose nuts that need tightening
        loose_nuts = []
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.goal_nuts:
                    # Find nut's location
                    for loc_fact in state:
                        if match(loc_fact, "at", nut, "*"):
                            loose_nuts.append((nut, get_parts(loc_fact)[2]))
                            break

        # For each loose nut, estimate actions needed
        for nut, nut_loc in loose_nuts:
            # If no usable spanner is carried, get one
            if not any(spanner in usable_spanners for spanner in carried_spanners):
                # Find nearest usable spanner not carried
                nearest_spanner = None
                min_distance = float('inf')
                
                for spanner in usable_spanners:
                    if spanner in carried_spanners:
                        continue
                    # Find spanner's location
                    for fact in state:
                        if match(fact, "at", spanner, "*"):
                            spanner_loc = get_parts(fact)[2]
                            # Calculate distance from current location to spanner
                            distance = self._bfs_distance(man_loc, spanner_loc)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_spanner = spanner
                            break
                
                if nearest_spanner:
                    # Add cost to walk to spanner and pick it up
                    cost += min_distance + 1
                    # Update man's location to spanner's location
                    man_loc = None
                    for fact in state:
                        if match(fact, "at", nearest_spanner, "*"):
                            man_loc = get_parts(fact)[2]
                            break
                    # Mark spanner as carried (for subsequent nuts)
                    carried_spanners.add(nearest_spanner)

            # Add cost to walk to nut and tighten it
            distance_to_nut = self._bfs_distance(man_loc, nut_loc)
            cost += distance_to_nut + 1
            # Update man's location to nut's location
            man_loc = nut_loc
            # Spanner becomes unusable after tightening
            usable_spanners.discard(next(iter(carried_spanners), None))

        return cost

    def _bfs_distance(self, start, end):
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
        
        # If no path found (shouldn't happen in valid problems)
        return float('inf')
