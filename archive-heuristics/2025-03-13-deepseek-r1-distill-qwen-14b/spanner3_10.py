from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts. Each nut requires:
    1. Moving to its location if not already there.
    2. Picking up a spanner if not already carrying one.
    3. Tightening the nut.

    # Assumptions:
    - The man starts at the shed.
    - Each nut must be tightened once.
    - A spanner is required to tighten a nut.
    - The man can carry multiple spanners but only one is needed at a time.

    # Heuristic Initialization
    - Extract static facts (location links) to compute distances between locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of loose nuts.
    2. For each loose nut:
       a. Find the shortest path from the man's current location to the nut's location.
       b. If the man isn't carrying a spanner, add the cost to pick one up.
       c. Add the cost to tighten the nut.
    3. Sum the costs for all nuts, considering the worst-case scenario for each.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Build a graph of connected locations from static facts
        self.location_graph = {}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = fact[5:-1].split()
                if loc1 not in self.location_graph:
                    self.location_graph[loc1] = []
                if loc2 not in self.location_graph:
                    self.location_graph[loc2] = []
                self.location_graph[loc1].append(loc2)
                self.location_graph[loc2].append(loc1)

    def __call__(self, node):
        """Estimate the minimal number of actions to tighten all loose nuts."""
        state = node.state

        def match(fact, *args):
            """Check if a fact matches the given pattern."""
            parts = fact[1:-1].split()
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        def get_parts(fact):
            """Extract components of a fact."""
            return fact[1:-1].split()

        # Extract current state information
        man_location = None
        nuts = {}
        spanners = {}
        carrying_spanner = False

        for fact in state:
            if match(fact, "at", "*", "location?"):
                obj, loc = get_parts(fact)
                if obj == "bob":
                    man_location = loc
                elif obj == "nut":
                    nuts[loc] = False  # Assume loose unless stated otherwise
                elif obj == "spanner":
                    spanners[loc] = True  # Assume usable unless stated otherwise
            elif match(fact, "carrying", "bob", "*"):
                carrying_spanner = True
            elif match(fact, "tightened", "*"):
                nut = get_parts(fact)[1]
                nuts[nut] = True  # Mark nut as tightened
            elif match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                nuts[nut] = False  # Mark nut as loose

        # Count loose nuts
        loose_nuts = [loc for loc, status in nuts.items() if not status]
        if not loose_nuts:
            return 0  # All nuts are already tightened

        # Calculate cost for each loose nut
        total_cost = 0

        for nut_loc in loose_nuts:
            # Find the man's current location
            if man_location is None:
                man_location = "shed"  # Default starting point

            # Find the shortest path from man's location to nut's location
            path = self.breadth_first_search(man_location, nut_loc)
            distance = len(path) if path else 0

            # Add cost for moving to nut's location
            total_cost += distance

            # Add cost for picking up a spanner if not carrying one
            if not carrying_spanner:
                # Find the closest spanner
                closest_spanner = None
                min_spanner_distance = float('inf')
                for spanner_loc in spanners:
                    if spanner_loc == nut_loc:
                        closest_spanner = spanner_loc
                        break
                    spanner_path = self.breadth_first_search(man_location, spanner_loc)
                    if spanner_path:
                        spanner_distance = len(spanner_path)
                        if spanner_distance < min_spanner_distance:
                            min_spanner_distance = spanner_distance
                            closest_spanner = spanner_loc
                if closest_spanner is None:
                    # No spanner available, cannot proceed
                    return float('inf')
                # Move to spanner's location
                spanner_path = self.breadth_first_search(man_location, closest_spanner)
                if spanner_path:
                    total_cost += len(spanner_path)
                else:
                    return float('inf')  # No path to spanner
                carrying_spanner = True

            # Add cost for tightening the nut
            total_cost += 1  # One action to tighten

            # Update man's location to the nut's location
            man_location = nut_loc

        return total_cost

    def breadth_first_search(self, start, goal):
        """Find the shortest path in an unweighted graph using BFS."""
        visited = set()
        queue = deque([(start, [])])

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.location_graph.get(current, []):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return None  # No path found
