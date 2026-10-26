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
    This heuristic estimates the number of actions needed to tighten all loose nuts in the goal.
    It considers the man's current location, carried spanners, and the locations of loose nuts and usable spanners.

    # Assumptions:
    - The man can carry multiple spanners at once.
    - Each spanner can only be used to tighten one nut (becomes unusable after use).
    - The man must be at the nut's location to tighten it.
    - The man must be at a spanner's location to pick it up.

    # Heuristic Initialization
    - Extract static link information to compute distances between locations.
    - Extract goal conditions to know which nuts need tightening.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts that need tightening (from goals).
    2. For each loose nut:
       a. If already tightened, skip.
       b. Else, calculate the minimal path to tighten it:
          i. If man is carrying usable spanners:
             - Add distance from current location to nut's location.
             - Add 1 action to tighten the nut (consuming one spanner).
          ii. If no usable spanners are carried:
              - Find nearest usable spanner.
              - Add distance to spanner's location + 1 action to pick it up.
              - Add distance from spanner to nut's location + 1 action to tighten.
    3. Sum all required actions for all loose nuts.
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

        # Extract which nuts need to be tightened from goals
        self.nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.nuts_to_tighten.add(get_parts(goal)[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal state."""
        state = node.state

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Extract current state information
        man_location = None
        carried_spanners = set()
        usable_spanners = set()
        spanner_locations = {}
        nut_locations = {}
        tightened_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carried_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                if obj.startswith("spanner"):
                    spanner_locations[obj] = loc
                elif obj.startswith("nut"):
                    nut_locations[obj] = loc
            elif match(fact, "tightened", "*"):
                tightened_nuts.add(parts[1])

        # Only consider nuts that need tightening and aren't already tightened
        nuts_to_handle = self.nuts_to_tighten - tightened_nuts

        if not nuts_to_handle:
            return 0

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
            return float('inf')  # unreachable

        for nut in nuts_to_handle:
            nut_loc = nut_locations[nut]
            
            # Check if we have usable spanners already
            usable_carried = [s for s in carried_spanners if s in usable_spanners]
            
            if usable_carried:
                # Just need to go to nut and tighten
                distance = bfs(man_location, nut_loc)
                total_cost += distance + 1  # walk + tighten
            else:
                # Need to find nearest usable spanner first
                min_spanner_cost = float('inf')
                
                for spanner in usable_spanners:
                    if spanner in spanner_locations:
                        spanner_loc = spanner_locations[spanner]
                        # Cost to get spanner: walk to it + pick up
                        cost_to_spanner = bfs(man_location, spanner_loc) + 1
                        # Cost from spanner to nut: walk + tighten
                        cost_to_nut = bfs(spanner_loc, nut_loc) + 1
                        total_spanner_cost = cost_to_spanner + cost_to_nut
                        
                        if total_spanner_cost < min_spanner_cost:
                            min_spanner_cost = total_spanner_cost
                
                if min_spanner_cost == float('inf'):
                    return float('inf')  # no usable spanners available
                
                total_cost += min_spanner_cost

        return total_cost
