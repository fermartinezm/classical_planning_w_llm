from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class spanner6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions needed to tighten all loose nuts by considering the minimal path to collect usable spanners and reach each nut's location.

    # Assumptions
    - Each spanner can be used only once (as per the domain's 'usable' effect).
    - The man can carry multiple spanners, but each pickup requires visiting the spanner's location.
    - The shortest path between locations is precomputed using static link information.

    # Heuristic Initialization
    - Precompute the shortest paths between all locations using the static 'link' facts.
    - Extract static links to build a directed graph for path calculation.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. For each loose nut:
        a. If the man is carrying a usable spanner, add the distance to the nut's location plus tighten action.
        b. Otherwise, find the nearest usable spanner, compute the cost to collect it and reach the nut.
    3. Sum the minimal costs for all loose nuts.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build the graph from static links
        self.static_links = set()
        for fact in self.static:
            if fact.startswith('(link '):
                parts = fact[1:-1].split()
                l1, l2 = parts[1], parts[2]
                self.static_links.add((l1, l2))

        # Precompute shortest paths between all locations
        self.graph = defaultdict(list)
        for l1, l2 in self.static_links:
            self.graph[l1].append(l2)

        all_locations = {l1 for l1, _ in self.static_links} | {l2 for _, l2 in self.static_links}
        self.shortest_path = defaultdict(dict)

        for start in all_locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            # Set shortest paths for reachable and unreachable locations
            for loc in all_locations:
                self.shortest_path[start][loc] = visited.get(loc, float('inf'))

    def __call__(self, node):
        state = node.state
        man = None
        man_location = None

        # Determine the man's identity and location
        for fact in state:
            if fact.startswith('(carrying '):
                parts = fact[1:-1].split()
                man = parts[1]
                break
        if man is None:
            for fact in state:
                if fact.startswith('(at ') and 'spanner' not in fact and 'nut' not in fact:
                    parts = fact[1:-1].split()
                    man = parts[1]
                    man_location = parts[2]
                    break
        else:
            for fact in state:
                if fact.startswith(f'(at {man} '):
                    parts = fact[1:-1].split()
                    man_location = parts[2]
                    break

        if man is None or man_location is None:
            return 0  # Fallback if no man found (unlikely in valid states)

        # Collect loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if fact.startswith('(loose '):
                nut = fact[1:-1].split()[1]
                loose_nuts.append(nut)
            elif fact.startswith('(at nut'):
                parts = fact[1:-1].split()
                nut_locations[parts[1]] = parts[2]

        total_cost = 0

        # Check carried usable spanners
        usable_carried = []
        for fact in state:
            if fact.startswith(f'(carrying {man} '):
                spanner = fact[1:-1].split()[2]
                if f'(usable {spanner})' in state:
                    usable_carried.append(spanner)

        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if not nut_loc:
                continue

            min_cost = float('inf')

            # Case 1: Use already carried usable spanner
            if usable_carried:
                distance = self.shortest_path[man_location].get(nut_loc, float('inf'))
                if distance != float('inf'):
                    min_cost = distance + 1  # walk + tighten

            # Case 2: Find best spanner to pick up
            for fact in state:
                if fact.startswith('(usable '):
                    spanner = fact[1:-1].split()[1]
                    if f'(carrying {man} {spanner})' in state:
                        continue  # Skip already carried
                    # Find spanner location
                    spanner_loc = None
                    for f in state:
                        if f.startswith(f'(at {spanner} '):
                            spanner_loc = f[1:-1].split()[2]
                            break
                    if not spanner_loc:
                        continue
                    # Calculate path costs
                    d1 = self.shortest_path[man_location].get(spanner_loc, float('inf'))
                    d2 = self.shortest_path[spanner_loc].get(nut_loc, float('inf'))
                    if d1 != float('inf') and d2 != float('inf'):
                        cost = d1 + 1 + d2 + 1  # pickup + walk + tighten
                        min_cost = min(min_cost, cost)

            total_cost += min_cost if min_cost != float('inf') else 0

        return total_cost
