from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class spanner5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions required to tighten all loose nuts by calculating the minimal steps needed to collect spanners and reach each nut's location.

    # Assumptions:
    - The man can carry multiple spanners.
    - Each spanner can be used only once.
    - Walking between locations uses the shortest path based on directed links.
    - Spanners not carried are at their initial locations unless picked up.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using BFS based on static links.
    - Extract static links and build a directed graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify available spanners (carried or on ground, usable).
    4. For each loose nut:
        a. If the man is carrying a usable spanner, cost is steps to nut + 1.
        b. Else, compute cost for each available spanner: steps to spanner + pickup + steps to nut + tighten.
        c. Select the minimal cost and mark the spanner as used.
    5. Sum all minimal costs for loose nuts.
    """

    def __init__(self, task):
        self.locations = set()
        self.graph = defaultdict(list)
        for fact in task.static:
            if fact.startswith('(link '):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                self.graph[l1].append(l2)
                self.locations.update([l1, l2])

        # Precompute shortest paths using BFS for each location
        self.shortest_paths = defaultdict(dict)
        for start in self.locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                current_steps = visited[current]
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited or current_steps + 1 < visited.get(neighbor, float('inf')):
                        visited[neighbor] = current_steps + 1
                        queue.append(neighbor)
            self.shortest_paths[start] = visited

    def __call__(self, node):
        state = node.state
        man_loc = None
        loose_nuts_locs = []
        available_spanners = []
        carried_spanners = set()

        # Extract man's current location
        for fact in state:
            if fnmatch(fact, '(at bob *'):
                parts = fact[1:-1].split()
                man_loc = parts[2]
                break
        if not man_loc:
            return 0

        # Extract locations of loose nuts
        nut_locations = {}
        for fact in state:
            if fnmatch(fact, '(loose *'):
                nut = fact.split()[1]
                for loc_fact in state:
                    if fnmatch(loc_fact, f'(at {nut} *'):
                        loc = loc_fact.split()[2]
                        loose_nuts_locs.append(loc)
                        break

        # Extract available spanners (carried and usable, or on ground and usable)
        carried_spanners = set()
        for fact in state:
            if fnmatch(fact, '(carrying bob *'):
                spanner = fact.split()[2]
                if any(fnmatch(f, f'(usable {spanner})') for f in state):
                    carried_spanners.add(spanner)
        for spanner in carried_spanners:
            available_spanners.append((spanner, man_loc, True))  # (spanner, location, is_carried)

        # Spanners on the ground
        for fact in state:
            if fnmatch(fact, '(at spanner* *'):
                parts = fact[1:-1].split()
                spanner, loc = parts[1], parts[2]
                if spanner not in carried_spanners and any(fnmatch(f, f'(usable {spanner})') for f in state):
                    available_spanners.append((spanner, loc, False))

        total_cost = 0
        used_spanners = set()

        for nut_loc in loose_nuts_locs:
            min_cost = float('inf')
            best_spanner = None
            for spanner_info in available_spanners:
                spanner, s_loc, is_carried = spanner_info
                if spanner in used_spanners:
                    continue

                if is_carried:
                    steps_to_s = 0
                    pickup_cost = 0
                else:
                    steps_to_s = self.shortest_paths[man_loc].get(s_loc, float('inf'))
                    pickup_cost = 1

                steps_to_nut = self.shortest_paths[s_loc].get(nut_loc, float('inf'))
                if steps_to_s == float('inf') or steps_to_nut == float('inf'):
                    continue

                cost = steps_to_s + pickup_cost + steps_to_nut + 1  # +1 for tighten
                if cost < min_cost:
                    min_cost = cost
                    best_spanner = spanner_info

            if best_spanner:
                total_cost += min_cost
                used_spanners.add(best_spanner[0])

        return total_cost
