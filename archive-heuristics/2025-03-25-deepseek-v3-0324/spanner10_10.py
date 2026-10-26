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
    - The man must be at the nut's location to tighten it.
    - The man must be at a spanner's location to pick it up.

    # Heuristic Initialization
    - Extract static link information between locations to compute distances.
    - Identify goal nuts that need to be tightened.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count remaining loose nuts that need to be tightened (from goals).
    2. For each loose nut:
       a. If already at nut's location with usable spanner:
          - Just need to tighten (1 action)
       b. Else:
          - Need to get to nut's location (walk actions)
          - May need to pick up usable spanner first (pickup action)
          - May need to walk to spanner location first
    3. For spanner collection:
       - Find nearest usable spanner to current position
       - Calculate walking distance to spanner
       - Add pickup action
    4. Sum all required actions:
       - Walking to spanners
       - Picking up spanners
       - Walking to nuts
       - Tightening nuts
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Build graph of locations from static link facts
        self.location_graph = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1)

        # Precompute shortest paths between all locations
        self.shortest_paths = self._compute_shortest_paths()

    def _compute_shortest_paths(self):
        """Compute shortest paths between all locations using BFS."""
        paths = {}
        locations = list(self.location_graph.keys())
        
        for start in locations:
            paths[start] = {}
            queue = [(start, [])]
            visited = set()
            
            while queue:
                current, path = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                
                paths[start][current] = len(path)
                
                for neighbor in self.location_graph[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return paths

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        total_cost = 0

        # Get man's current location
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

        # Get usable spanners (both carried and on the ground)
        usable_spanners = set()
        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                # Check if spanner is carried or on the ground
                if (f"(carrying bob {spanner})" in state or 
                    (f"(at {spanner} *)" in state and 
                     not any(match(f, "carrying", "*", spanner) for f in state))):
                    usable_spanners.add(spanner)

        # Get locations of usable spanners not carried
        spanner_locations = {}
        for fact in state:
            if match(fact, "at", "spanner*", "*"):
                spanner, loc = get_parts(fact)[1], get_parts(fact)[2]
                if spanner in usable_spanners and spanner not in carried_spanners:
                    spanner_locations[spanner] = loc

        # Get loose nuts and their locations
        loose_nuts = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                # Find nut's location
                for loc_fact in state:
                    if match(loc_fact, "at", nut, "*"):
                        loose_nuts[nut] = get_parts(loc_fact)[2]
                        break

        # Filter only nuts that are in goals and still loose
        goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                goal_nuts.add(get_parts(goal)[1])
        loose_nuts = {k: v for k, v in loose_nuts.items() if k in goal_nuts}

        if not loose_nuts:
            return 0  # All goals satisfied

        # Calculate costs for each nut
        for nut, nut_loc in loose_nuts.items():
            if man_loc == nut_loc and usable_spanners:
                # Can tighten immediately (1 action)
                total_cost += 1
                # Mark one spanner as used
                usable_spanners.pop()
                continue

            # Need to get to nut location
            walk_cost = self.shortest_paths[man_loc][nut_loc] if man_loc != nut_loc else 0

            # Need usable spanner - check if we have one or need to get one
            if not usable_spanners:
                # No spanners left - problem unsolvable
                return float('inf')

            if carried_spanners:
                # Already have usable spanner
                spanner_cost = 0
            else:
                # Need to get nearest usable spanner
                nearest_spanner = None
                min_dist = float('inf')
                for spanner, loc in spanner_locations.items():
                    dist = self.shortest_paths[man_loc][loc]
                    if dist < min_dist:
                        min_dist = dist
                        nearest_spanner = spanner

                if nearest_spanner is None:
                    return float('inf')  # No reachable spanners

                spanner_cost = min_dist + 1  # walk + pickup
                man_loc = spanner_locations[nearest_spanner]  # update position after pickup
                walk_cost = self.shortest_paths[man_loc][nut_loc]  # recalculate walk to nut

                # Remove this spanner from available ones
                del spanner_locations[nearest_spanner]
                usable_spanners.remove(nearest_spanner)

            total_cost += spanner_cost + walk_cost + 1  # +1 for tighten action
            man_loc = nut_loc  # update position after tightening

        return total_cost
