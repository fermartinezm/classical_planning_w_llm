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
    It considers the following aspects:
    - The man's current location and whether he needs to move to pick up spanners or reach nuts
    - Whether the man is carrying usable spanners
    - The locations of loose nuts and available spanners
    - The path distances between locations in the environment

    # Assumptions:
    - The man can carry multiple spanners at once
    - Each spanner can only be used once (becomes unusable after tightening a nut)
    - The environment is connected via links (no unreachable locations)
    - The goal is to tighten all nuts that are initially loose

    # Heuristic Initialization
    - Extract the link information from static facts to build a graph of locations
    - Identify all goal nuts (those that need to be tightened)
    - Precompute shortest paths between all locations for distance calculations

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if all goal nuts are already tightened (heuristic = 0)
    2. For each loose nut that needs tightening:
        a. If the man is at the nut's location with a usable spanner:
           - Add 1 action (tighten)
        b. Else:
           - Calculate the minimal path to:
              i. Pick up a usable spanner (if not already carrying one)
              ii. Move to the nut's location
           - Add actions for pickup (if needed) and movement
    3. Optimize the order of operations to minimize total movement
    4. Sum all required actions
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

        # Precompute shortest paths between all locations
        self.shortest_paths = self._compute_shortest_paths()

    def _compute_shortest_paths(self):
        """Compute shortest paths between all locations using BFS."""
        shortest_paths = {}
        locations = set(self.location_graph.keys())
        
        for start in locations:
            shortest_paths[start] = {}
            visited = {start: 0}
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                for neighbor in self.location_graph[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            
            for loc in locations:
                shortest_paths[start][loc] = visited.get(loc, float('inf'))
        
        return shortest_paths

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state

        # Check if all goals are already satisfied
        if self.goals <= state:
            return 0

        # Extract current state information
        man_location = None
        carrying_spanners = set()
        usable_spanners = set()
        spanner_locations = {}
        loose_nuts = set()
        nut_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "bob", "*"):
                man_location = parts[2]
            elif match(fact, "carrying", "bob", "*"):
                carrying_spanners.add(parts[2])
            elif match(fact, "usable", "*"):
                usable_spanners.add(parts[1])
            elif match(fact, "at", "spanner*", "*"):
                spanner_locations[parts[1]] = parts[2]
            elif match(fact, "loose", "*"):
                loose_nuts.add(parts[1])
            elif match(fact, "at", "nut*", "*"):
                nut_locations[parts[1]] = parts[2]

        # Identify which nuts still need to be tightened
        nuts_to_tighten = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nuts_to_tighten.add(get_parts(goal)[1])

        total_cost = 0

        # For each nut that needs tightening
        for nut in nuts_to_tighten:
            if nut not in loose_nuts:
                continue  # Already tightened

            nut_loc = nut_locations[nut]
            best_cost = float('inf')

            # Option 1: Use a spanner we're already carrying
            for spanner in carrying_spanners:
                if spanner in usable_spanners:
                    # Need to move to nut location if not already there
                    move_cost = 0 if man_location == nut_loc else 1
                    cost = move_cost + 1  # tighten action
                    best_cost = min(best_cost, cost)

            # Option 2: Pick up a new usable spanner
            for spanner, spanner_loc in spanner_locations.items():
                if spanner in usable_spanners and spanner not in carrying_spanners:
                    # Cost to get to spanner, pick it up, then go to nut
                    pickup_cost = (self.shortest_paths[man_location][spanner_loc] + 
                                 1 +  # pickup action
                                 self.shortest_paths[spanner_loc][nut_loc])
                    cost = pickup_cost + 1  # tighten action
                    best_cost = min(best_cost, cost)

            # If no usable spanners available, heuristic can't find solution
            if best_cost == float('inf'):
                return float('inf')

            total_cost += best_cost

            # Update man's location and carrying status for subsequent nuts
            man_location = nut_loc
            # Assume we used a spanner (mark as not usable)
            if carrying_spanners:
                usable_spanners.discard(next(iter(carrying_spanners)))

        return total_cost
