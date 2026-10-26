from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

class spanner24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions needed to tighten all loose nuts by considering the shortest paths to collect spanners and reach nuts. The heuristic sums the minimal steps for each nut, considering both carried and on-ground spanners.

    # Assumptions
    - The man can carry multiple spanners, but each tighten action uses one.
    - Spanners are usable until used, after which they become unusable.
    - The links between locations form a directed graph for movement.

    # Heuristic Initialization
    - Extracts static 'link' facts to build a directed graph.
    - Precomputes shortest path distances between all locations using BFS.

    # Step-By-Step Thinking
    1. Identify the man's current location and all loose nuts.
    2. For each nut, calculate the minimal cost to tighten it using either:
       a. A carried usable spanner: distance from man's location to nut + tighten.
       b. A usable spanner on the ground: distance to spanner + pickup + distance to nut + tighten.
    3. Sum the minimal costs for all nuts to get the heuristic value.
    """

    def __init__(self, task):
        self.static = task.static
        self.graph = defaultdict(list)
        for fact in self.static:
            parts = self._get_parts(fact)
            if parts[0] == 'link':
                from_loc, to_loc = parts[1], parts[2]
                self.graph[from_loc].append(to_loc)
        self.distances = self._compute_distances()

    def _get_parts(self, fact):
        return fact.strip('()').split()

    def _compute_distances(self):
        distances = {}
        locations = set(self.graph.keys())
        for links in self.graph.values():
            locations.update(links)
        for start in locations:
            distances[start] = {}
            queue = deque([(start, 0)])
            visited = set()
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                distances[start][current] = dist
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
        return distances

    def __call__(self, node):
        state = node.state
        man_loc = None
        carrying = set()
        usable = set()
        loose_nuts = set()
        nut_locs = {}
        spanner_locs = {}

        # Extract man's location and carrying spanners
        man = next((p[1] for fact in state if (p := self._get_parts(fact))[0] == 'carrying'), None)
        if man:
            for fact in state:
                parts = self._get_parts(fact)
                if parts[0] == 'at' and parts[1] == man:
                    man_loc = parts[2]
                    break
            carrying = {p[2] for fact in state if (p := self._get_parts(fact))[0] == 'carrying' and p[1] == man}

        # Extract usable spanners and their locations
        usable = {p[1] for fact in state if (p := self._get_parts(fact))[0] == 'usable'}
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'at' and parts[1] in usable and parts[1] not in carrying:
                spanner_locs[parts[1]] = parts[2]

        # Extract loose nuts and their locations
        loose_nuts = {p[1] for fact in state if (p := self._get_parts(fact))[0] == 'loose'}
        for fact in state:
            parts = self._get_parts(fact)
            if parts[0] == 'at' and parts[1] in loose_nuts:
                nut_locs[parts[1]] = parts[2]

        total = 0
        for nut in loose_nuts:
            nut_loc = nut_locs.get(nut)
            if not nut_loc or not man_loc:
                continue

            min_cost = float('inf')
            # Case 1: Use carried usable spanner
            if carrying & usable:
                dist = self.distances[man_loc].get(nut_loc, float('inf'))
                if dist != float('inf'):
                    min_cost = min(min_cost, dist + 1)

            # Case 2: Use spanner on ground
            for s, s_loc in spanner_locs.items():
                if s not in usable:
                    continue
                to_s = self.distances[man_loc].get(s_loc, float('inf'))
                s_to_nut = self.distances[s_loc].get(nut_loc, float('inf'))
                if to_s != float('inf') and s_to_nut != float('inf'):
                    cost = to_s + 1 + s_to_nut + 1
                    min_cost = min(min_cost, cost)

            if min_cost != float('inf'):
                total += min_cost

        return total
