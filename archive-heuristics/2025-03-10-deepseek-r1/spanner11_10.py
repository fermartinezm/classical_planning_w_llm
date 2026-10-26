from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class spanner11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions needed to tighten all loose nuts by:
    1. Using carried usable spanners first.
    2. Collecting the closest available spanners for remaining nuts.
    Costs include walking, picking up spanners, and tightening actions.

    # Assumptions:
    - Each nut requires a separate usable spanner.
    - The man can carry multiple spanners but must pick them up one by one.
    - Links between locations are directed; shortest paths are precomputed.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using BFS based on static links.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Collect all loose nuts and their locations.
    3. Identify all usable spanners (carried and available).
    4. Check if there are enough spanners to tighten all loose nuts.
    5. For each loose nut:
        a. Use a carried spanner if available, adding walking and tightening cost.
        b. Otherwise, use the closest available spanner, adding pickup, walking, and tightening cost.
    6. Sum all costs for the total heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build directed graph from static links
        self.graph = defaultdict(list)
        self.all_locations = set()
        for fact in self.static:
            if self._match(fact, 'link', '*', '*'):
                parts = self._get_parts(fact)
                from_loc, to_loc = parts[1], parts[2]
                self.graph[from_loc].append(to_loc)
                self.all_locations.update([from_loc, to_loc])

        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        for loc in self.all_locations:
            self.shortest_paths[loc] = self._bfs(loc)

    def _get_parts(self, fact):
        return fact[1:-1].split()

    def _match(self, fact, *args):
        parts = self._get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def _bfs(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def _get_distance(self, from_loc, to_loc):
        if from_loc not in self.shortest_paths:
            return float('inf')
        return self.shortest_paths[from_loc].get(to_loc, float('inf'))

    def __call__(self, node):
        state = node.state
        man_name, man_location = None, None

        # Determine man's name and location
        for fact in state:
            if self._match(fact, 'carrying', '*', '*'):
                man_name = self._get_parts(fact)[1]
                break
        if not man_name:
            for fact in state:
                if self._match(fact, 'at', '*', '*'):
                    obj = self._get_parts(fact)[1]
                    if 'spanner' not in obj and 'nut' not in obj:
                        man_name = obj
                        break
        for fact in state:
            if self._match(fact, 'at', man_name, '*'):
                man_location = self._get_parts(fact)[2]
                break

        # Collect loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if self._match(fact, 'loose', '*'):
                nut = self._get_parts(fact)[1]
                for loc_fact in state:
                    if self._match(loc_fact, 'at', nut, '*'):
                        nut_loc = self._get_parts(loc_fact)[2]
                        loose_nuts.append((nut, nut_loc))
                        break

        if not loose_nuts:
            return 0

        # Collect usable spanners
        carried_spanners = []
        available_spanners = []
        for fact in state:
            if self._match(fact, 'carrying', man_name, '*'):
                spanner = self._get_parts(fact)[2]
                if f'(usable {spanner})' in state:
                    carried_spanners.append(spanner)
            elif self._match(fact, 'at', '*', '*'):
                obj = self._get_parts(fact)[1]
                if 'spanner' in obj and f'(usable {obj})' in state:
                    loc = self._get_parts(fact)[2]
                    available_spanners.append((obj, loc))

        # Check if enough spanners
        required = len(loose_nuts)
        available = len(carried_spanners) + len(available_spanners)
        if available < required:
            return float('inf')

        # Sort available spanners by distance from man's location
        available_spanners.sort(key=lambda x: self._get_distance(man_location, x[1]))

        total_cost = 0
        for nut, nut_loc in loose_nuts:
            if carried_spanners:
                distance = self._get_distance(man_location, nut_loc)
                if distance == float('inf'):
                    return float('inf')
                total_cost += distance + 1
                carried_spanners.pop()
            else:
                if not available_spanners:
                    return float('inf')
                spanner, spanner_loc = available_spanners.pop(0)
                dist_to_spanner = self._get_distance(man_location, spanner_loc)
                dist_to_nut = self._get_distance(spanner_loc, nut_loc)
                if dist_to_spanner == float('inf') or dist_to_nut == float('inf'):
                    return float('inf')
                total_cost += dist_to_spanner + 1 + dist_to_nut + 1

        return total_cost
