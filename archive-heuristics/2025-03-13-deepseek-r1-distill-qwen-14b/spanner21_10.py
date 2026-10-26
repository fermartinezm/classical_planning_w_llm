from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.

    # Assumptions:
    - The man can carry one spanner at a time.
    - The man must be at the same location as a nut to tighten it.
    - If the man is not carrying a spanner, he must pick one up before tightening a nut.
    - The man can move between connected locations.

    # Heuristic Initialization
    - Extracts goal conditions and static facts (location links) from the task.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of loose nuts that need to be tightened.
    2. For each loose nut:
       a. Determine the man's current location.
       b. Calculate the shortest path from the man's location to the nut's location.
       c. If the man is not carrying a spanner, find the nearest spanner and calculate the path to it.
       d. Sum the total steps required, including moving to the spanner (if needed) and then to the nut.
    3. If the man is already carrying a spanner, subtract one step for each nut since he doesn't need to pick up a new spanner.
    4. Return the total estimated steps.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Build the graph of connected locations
        self.graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = fact[5:-1].split()
                if loc1 not in self.graph:
                    self.graph[loc1] = []
                if loc2 not in self.graph:
                    self.graph[loc2] = []
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

    def __call__(self, node):
        """Estimate the minimum number of actions to tighten all loose nuts."""
        state = node.state

        def match(fact, *args):
            """Check if a PDDL fact matches a given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        def get_parts(fact):
            """Extract the components of a PDDL fact."""
            return fact[1:-1].split()

        # Extract current state information
        man_location = None
        carrying_spanner = None
        spanner_locations = {}
        nut_states = {}

        for fact in state:
            if match(fact, "at", "*", "location"):
                obj, loc = get_parts(fact)
                if obj == "bob":
                    man_location = loc
                elif obj.startswith("spanner"):
                    spanner_locations[obj] = loc
            elif match(fact, "carrying", "bob", "*"):
                _, s = get_parts(fact)
                carrying_spanner = s
            elif match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                nut_states[nut] = False
            elif match(fact, "tightened", "*"):
                nut = get_parts(fact)[1]
                nut_states[nut] = True

        # Count loose nuts
        loose_nuts = [nut for nut, state in nut_states.items() if not state]
        if not loose_nuts:
            return 0

        # If no spanners are available, the problem is unsolvable
        if not spanner_locations:
            return float('inf')

        total_cost = 0

        for nut in loose_nuts:
            nut_location = None
            for fact in state:
                if match(fact, "at", nut, "*"):
                    nut_location = get_parts(fact)[2]
                    break
            if not nut_location:
                continue  # Nut's location not found

            # Find the shortest path from man's location to nut's location
            def bfs(start, end):
                visited = set()
                queue = deque([(start, 0)])
                while queue:
                    current, steps = queue.popleft()
                    if current == end:
                        return steps
                    if current in visited:
                        continue
                    visited.add(current)
                    for neighbor in self.graph.get(current, []):
                        if neighbor not in visited:
                            queue.append((neighbor, steps + 1))
                return float('inf')  # No path found

            distance_to_nut = bfs(man_location, nut_location)
            if distance_to_nut == float('inf'):
                return float('inf')  # Problem unsolvable

            # If not carrying a spanner, find the nearest spanner
            if not carrying_spanner:
                nearest_spanner = None
                nearest_distance = float('inf')
                for spanner, loc in spanner_locations.items():
                    distance = bfs(man_location, loc)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_spanner = spanner
                if nearest_spanner is None:
                    return float('inf')  # No spanner available
                total_cost += nearest_distance  # Walk to spanner
                total_cost += 1  # Pick up spanner
                carrying_spanner = nearest_spanner

            # Walk to nut's location
            total_cost += distance_to_nut
            # Tighten the nut
            total_cost += 1

            # If carrying a spanner, no need to pick up again
            if carrying_spanner:
                # Subtract one step since we're already carrying a spanner
                total_cost -= 1

        return total_cost
