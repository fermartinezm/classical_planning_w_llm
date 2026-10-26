from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    return fact[1:-1].split()

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions needed to tighten all loose nuts. For each nut, it calculates the minimal steps required to either use a carried usable spanner or pick up a new spanner and then tighten the nut.

    # Assumptions
    - The man can carry multiple spanners.
    - Each spanner can be used only once.
    - The shortest path between locations is used for movement.

    # Heuristic Initialization
    - Extracts spanners, nuts, and the man from the initial state.
    - Builds a graph of location links from static facts.
    - Precomputes shortest path distances between all pairs of locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each loose nut:
        a. Find the nut's location.
        b. Check if the man is carrying any usable spanner. If so, compute the distance from the man's current location to the nut's location + 1 (tighten).
        c. Otherwise, for each usable spanner not carried:
            i. Compute the distance from man's current location to the spanner's location.
            ii. Compute the distance from the spanner's location to the nut's location.
            iii. Add 2 actions (pickup and tighten).
            iv. Take the minimal such cost.
        d. Sum the minimal cost for each nut.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract spanners, nuts, and man from initial state
        self.spanners = set()
        self.nuts = set()
        self.man = None

        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'usable':
                self.spanners.add(parts[1])
            elif parts[0] in ['loose', 'tightened']:
                self.nuts.add(parts[1])

        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj = parts[1]
                if obj not in self.spanners and obj not in self.nuts:
                    self.man = obj
                    break

        # Build graph from static links
        self.graph = defaultdict(list)
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.graph[l1].append(l2)

        # Precompute all pairs shortest paths
        self.distances = {}
        all_locations = set()
        for l1 in self.graph:
            all_locations.add(l1)
            all_locations.update(self.graph[l1])
        all_locations = list(all_locations)

        for start in all_locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for end in all_locations:
                self.distances[(start, end)] = visited.get(end, float('inf'))

    def __call__(self, node):
        state = node.state
        total_cost = 0

        # Find man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man:
                man_location = parts[2]
                break
        if man_location is None:
            return 0  # Should not happen

        # Find loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for nut in self.nuts:
            if f'(loose {nut})' in state:
                loose_nuts.append(nut)
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'at' and parts[1] == nut:
                        nut_locations[nut] = parts[2]
                        break

        if not loose_nuts:
            return 0  # All nuts tightened

        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if not nut_loc:
                continue  # Nut not placed anywhere? Shouldn't happen

            # Check carried usable spanners
            carried_usable = []
            for fact in state:
                parts = get_parts(fact)
                if parts[0] == 'carrying' and parts[1] == self.man:
                    spanner = parts[2]
                    if f'(usable {spanner})' in state:
                        carried_usable.append(spanner)

            cost_a = float('inf')
            if carried_usable:
                distance = self.distances.get((man_location, nut_loc), float('inf'))
                cost_a = distance + 1  # walk + tighten

            # Check available spanners (not carried, usable, and have location)
            available_spanners = []
            for s in self.spanners:
                if f'(usable {s})' not in state:
                    continue
                # Check if spanner is carried
                carried = any(fact == f'(carrying {self.man} {s})' for fact in state)
                if carried:
                    continue
                # Find spanner's location
                s_loc = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'at' and parts[1] == s:
                        s_loc = parts[2]
                        break
                if s_loc:
                    available_spanners.append((s, s_loc))

            min_cost_b = float('inf')
            for s, s_loc in available_spanners:
                d1 = self.distances.get((man_location, s_loc), float('inf'))
                d2 = self.distances.get((s_loc, nut_loc), float('inf'))
                total = d1 + d2 + 2  # pickup + tighten
                if total < min_cost_b:
                    min_cost_b = total

            min_cost = min(cost_a, min_cost_b)
            if min_cost == float('inf'):
                min_cost = 1000  # Penalty for unreachable
            total_cost += min_cost

        return total_cost
