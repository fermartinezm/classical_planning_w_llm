from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

class spanner25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the number of actions required to tighten all loose nuts by calculating the minimal path to collect necessary spanners and move to each nut's location.

    Assumptions:
    - The man can carry multiple spanners but each spanner can be used only once.
    - Spanners must be picked up before use and walking is allowed only between linked locations.
    - The problem is solvable (i.e., there are enough usable spanners for all loose nuts).

    Heuristic Initialization:
    - Precomputes shortest paths between all locations using static link information to enable efficient distance lookup.

    Step-By-Step Thinking for Computing Heuristic:
    1. Extract the man's current location, loose nuts, and usable spanners (carried or available).
    2. For each loose nut:
        a. If carrying a usable spanner, add the distance from current position to the nut's location plus tightening action.
        b. If no carried spanners, find the closest available spanner, add distance to collect it, then move to the nut's location and tighten.
    3. Track the man's updated position after each action to ensure accurate path calculation.
    4. Sum all costs to get the total estimated actions.
    """

    def __init__(self, task):
        # Build directed graph from static links
        self.graph = {}
        self.locations = set()
        for fact in task.static:
            if fact.startswith('(link '):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                if l1 not in self.graph:
                    self.graph[l1] = []
                self.graph[l1].append(l2)
                self.locations.update({l1, l2})

        # Precompute shortest paths between all locations using BFS
        self.shortest_paths = {}
        for start in self.locations:
            self.shortest_paths[start] = {}
            queue = deque([(start, 0)])
            visited = {start: 0}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            for loc in self.locations:
                self.shortest_paths[start][loc] = visited.get(loc, float('inf'))

    def __call__(self, node):
        state = node.state
        man_loc = None
        carried = []
        available = []
        loose_nuts = []
        nut_locs = {}

        # Extract man's location
        for fact in state:
            if fact.startswith('(at bob '):
                man_loc = fact[1:-1].split()[2]
                break
        if not man_loc:
            return float('inf')

        # Extract loose nuts and their locations
        for fact in state:
            if fact.startswith('(loose '):
                nut = fact[1:-1].split()[1]
                loose_nuts.append(nut)
            elif fact.startswith('(at ') and 'nut' in fact:
                parts = fact[1:-1].split()
                nut_locs[parts[1]] = parts[2]

        # Extract carried usable spanners
        carried = [
            parts[2] for fact in state
            if fact.startswith('(carrying bob ') and '(usable ' + parts[2] + ')' in state
            for parts in [fact[1:-1].split()]
        ]

        # Extract available usable spanners
        available = [
            (parts[1], parts[2]) for fact in state
            if fact.startswith('(at ') and 'spanner' in fact
            for parts in [fact[1:-1].split()]
            if f'(usable {parts[1]})' in state and parts[1] not in carried
        ]

        if not loose_nuts:
            return 0

        total_cost = 0
        current_pos = man_loc
        remaining_carried = list(carried)
        remaining_available = list(available)

        for nut in loose_nuts:
            nut_loc = nut_locs.get(nut)
            if not nut_loc:
                return float('inf')

            if remaining_carried:
                # Use carried spanner
                dist = self.shortest_paths[current_pos].get(nut_loc, float('inf'))
                if dist == float('inf'):
                    return float('inf')
                total_cost += dist + 1
                current_pos = nut_loc
                remaining_carried.pop(0)
            else:
                if not remaining_available:
                    return float('inf')

                # Find closest available spanner
                min_dist = float('inf')
                closest = None
                for s, s_loc in remaining_available:
                    d = self.shortest_paths[current_pos].get(s_loc, float('inf'))
                    if d < min_dist:
                        min_dist = d
                        closest = (s, s_loc)
                if min_dist == float('inf'):
                    return float('inf')

                total_cost += min_dist + 1
                current_pos = closest[1]

                # Move to nut and tighten
                dist_to_nut = self.shortest_paths[current_pos].get(nut_loc, float('inf'))
                if dist_to_nut == float('inf'):
                    return float('inf')
                total_cost += dist_to_nut + 1
                current_pos = nut_loc
                remaining_available.remove(closest)

        return total_cost
