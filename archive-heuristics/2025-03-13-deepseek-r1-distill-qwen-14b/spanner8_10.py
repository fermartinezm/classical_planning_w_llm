from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by calculating the minimal steps required for each nut, considering whether the man has a usable spanner.

    # Assumptions:
    - The man can carry only one spanner at a time.
    - Each spanner can be used only once.
    - The man starts with no spanner unless he has picked one up.
    - The graph of locations is connected and unweighted.

    # Heuristic Initialization
    - Extract the initial locations of all spanners from the task's initial state.
    - Build a graph of locations from the static facts.
    - Precompute the shortest paths between all pairs of locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine:
       a. The man's current location.
       b. Whether the man is carrying a usable spanner.
       c. The locations of all loose nuts.
       d. The locations of all available spanners (at their initial locations and usable).

    2. For each loose nut:
       a. If the man has a usable spanner, calculate the distance from his current location to the nut's location and add the tightening action.
       b. If the man does not have a usable spanner, find the nearest available spanner, calculate the distance to it, add the pickup action, then calculate the distance from the spanner's location to the nut's location and add the tightening action.

    3. Sum the actions for all nuts to get the total heuristic value.
    """

    def __init__(self, task):
        # Extract initial locations of spanners from the task's initial state
        self.initial_spanner_locations = {}
        for fact in task.initial_state:
            if fact.startswith('(at spanner'):
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1].startswith('spanner'):
                    spanner = parts[1]
                    location = parts[2]
                    self.initial_spanner_locations[spanner] = location

        # Build the graph from static facts
        self.graph = {}
        for fact in task.static:
            if fact.startswith('(link'):
                parts = get_parts(fact)
                if len(parts) >= 3 and parts[0] == 'link':
                    l1, l2 = parts[1], parts[2]
                    if l1 not in self.graph:
                        self.graph[l1] = []
                    self.graph[l1].append(l2)
                    if l2 not in self.graph:
                        self.graph[l2] = []
                    self.graph[l2].append(l1)

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for loc in self.graph:
            self.distances[loc] = {}
            visited = {loc}
            queue = deque()
            queue.append((loc, 0))
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[loc][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state

        # Parse the current state
        man_location = None
        for fact in state:
            if fact.startswith('(at bob '):
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[1] == 'bob':
                    man_location = parts[2]
                    break

        has_spanner = False
        for fact in state:
            if fact.startswith('(carrying bob spanner'):
                parts = get_parts(fact)
                if parts[0] == 'carrying' and parts[1] == 'bob' and parts[2].startswith('spanner'):
                    spanner = parts[2]
                    if f'(usable {spanner})' in state:
                        has_spanner = True
                        break

        loose_nuts = []
        for fact in state:
            if fact.startswith('(loose '):
                parts = get_parts(fact)
                if parts[0] == 'loose':
                    nut = parts[1]
                    loc = parts[2]
                    loose_nuts.append((nut, loc))

        available_spanners = []
        for spanner, initial_loc in self.initial_spanner_locations.items():
            if f'(at {spanner} {initial_loc})' in state and f'(usable {spanner})' in state:
                available_spanners.append(initial_loc)

        total_cost = 0

        for nut, nut_loc in loose_nuts:
            if has_spanner:
                if man_location in self.distances and nut_loc in self.distances[man_location]:
                    distance = self.distances[man_location][nut_loc]
                    total_cost += distance + 1  # walk + tighten
                else:
                    # If no path exists, assume it's unsolvable (though problem assumes solvable)
                    pass
            else:
                if not available_spanners:
                    # No available spanners, but problem is solvable
                    continue
                min_distance = float('inf')
                nearest_spanner = None
                for spanner_loc in available_spanners:
                    if man_location in self.distances and spanner_loc in self.distances[man_location]:
                        d = self.distances[man_location][spanner_loc]
                        if d < min_distance:
                            min_distance = d
                            nearest_spanner = spanner_loc
                if nearest_spanner is None:
                    continue
                if nearest_spanner in self.distances and nut_loc in self.distances[nearest_spanner]:
                    d1 = min_distance
                    d2 = self.distances[nearest_spanner][nut_loc]
                    total_cost += d1 + 1 + d2 + 1  # walk to spanner + pickup + walk to nut + tighten
                else:
                    # If no path exists, assume it's unsolvable (though problem assumes solvable)
                    pass

        return total_cost
