from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spanner16Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions required to tighten all loose nuts by calculating the minimal path for each nut, considering the need to collect usable spanners and travel to nut locations.

    # Assumptions:
    - The man can carry multiple spanners, but each can be used only once.
    - Links between locations are static and form a directed graph.
    - There are enough usable spanners to tighten all loose nuts.

    # Heuristic Initialization
    - Extract link facts from static information to build a location graph.
    - Precompute shortest path distances between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State Information**:
       - Find the man's current location.
       - Identify all loose nuts and their locations.
       - Collect all usable spanners (carried or on ground).
    2. **Check Solvability**:
       - If fewer usable spanners than loose nuts, return infinity.
    3. **Assign Spanners to Nuts**:
       - For each nut, compute minimal cost to use the closest spanner (carried or on ground).
       - Sum costs, ensuring each spanner is used once.
    """

    def __init__(self, task):
        self.static = task.static
        self.location_graph = {}
        self.distances = {}

        # Build location graph from static links
        for fact in self.static:
            if match(fact, 'link', '*', '*'):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                if l1 not in self.location_graph:
                    self.location_graph[l1] = []
                self.location_graph[l1].append(l2)

        # Collect all unique locations
        locations = set()
        for l1 in self.location_graph:
            locations.add(l1)
            for l2 in self.location_graph[l1]:
                locations.add(l2)
        self.locations = list(locations)

        # Precompute shortest paths using BFS for each location
        self.distances = {}
        for source in self.locations:
            self.distances[source] = {}
            queue = deque([(source, 0)])
            visited = set()
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                self.distances[source][current] = dist
                if current in self.location_graph:
                    for neighbor in self.location_graph[current]:
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state
        man_loc = None
        # Extract man's current location
        for fact in state:
            if match(fact, 'at', 'bob', '*'):
                man_loc = get_parts(fact)[2]
                break
        if not man_loc:
            return float('inf')

        # Collect loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if match(fact, 'loose', '*'):
                nut = get_parts(fact)[1]
                for f in state:
                    if match(f, 'at', nut, '*'):
                        nut_loc = get_parts(f)[2]
                        loose_nuts.append((nut, nut_loc))
                        break

        # Collect usable spanners (carried or on ground)
        usable_spanners = []
        for fact in state:
            if match(fact, 'usable', '*'):
                spanner = get_parts(fact)[1]
                carried = any(match(f, 'carrying', 'bob', spanner) for f in state)
                if carried:
                    usable_spanners.append((spanner, man_loc))
                else:
                    for f in state:
                        if match(f, 'at', spanner, '*'):
                            spanner_loc = get_parts(f)[2]
                            usable_spanners.append((spanner, spanner_loc))
                            break

        if len(usable_spanners) < len(loose_nuts):
            return float('inf')

        # Greedily assign closest spanner to each nut
        total_cost = 0
        available_spanners = usable_spanners.copy()

        for nut, nut_loc in loose_nuts:
            min_cost = float('inf')
            best_idx = -1

            for idx, (spanner, s_loc) in enumerate(available_spanners):
                if s_loc == man_loc:
                    dist = self.distances.get(man_loc, {}).get(nut_loc, float('inf'))
                    cost = dist + 1 if dist != float('inf') else float('inf')
                else:
                    d1 = self.distances.get(man_loc, {}).get(s_loc, float('inf'))
                    d2 = self.distances.get(s_loc, {}).get(nut_loc, float('inf'))
                    cost = d1 + d2 + 2 if d1 != float('inf') and d2 != float('inf') else float('inf')

                if cost < min_cost:
                    min_cost = cost
                    best_idx = idx

            if best_idx == -1:
                return float('inf')

            total_cost += min_cost
            available_spanners.pop(best_idx)

        return total_cost
