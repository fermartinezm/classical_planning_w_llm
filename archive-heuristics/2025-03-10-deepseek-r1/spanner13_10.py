from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class Spanner13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions required to tighten all loose nuts by considering the shortest paths to collect spanners and reach each nut's location. The heuristic uses a greedy assignment of spanners to nuts to minimize total steps.

    # Assumptions:
    - The man can carry multiple spanners, but each spanner can be used only once.
    - Links between locations are directed, and shortest paths are precomputed.
    - The problem is solvable (sufficient usable spanners are available).

    # Heuristic Initialization
    - Precomputes shortest paths between all pairs of locations using BFS based on static link facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners (carried or in the world) and their locations.
    4. For each loose nut, calculate the minimal cost to tighten it using the closest available spanner.
    5. Greedily assign spanners to nuts to minimize the total cost, ensuring each spanner is used once.
    6. Sum the costs for all assigned spanner-nut pairs.
    """

    def __init__(self, task):
        self.static_links = {}
        self.locations = set()

        # Extract static links from the task's static facts
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'link':
                from_loc, to_loc = parts[1], parts[2]
                if from_loc not in self.static_links:
                    self.static_links[from_loc] = []
                self.static_links[from_loc].append(to_loc)
                self.locations.update([from_loc, to_loc])

        # Precompute shortest paths using BFS for each location
        self.shortest_paths = {}
        for start in self.locations:
            self.shortest_paths[start] = {}
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.static_links.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for loc in self.locations:
                self.shortest_paths[start][loc] = visited.get(loc, float('inf'))

    def __call__(self, node):
        state = node.state
        man_name = None
        man_location = None
        loose_nuts = {}  # {nut: location}
        usable_spanners = {}  # {spanner: (location, is_carried)}

        # Extract man's name and location
        for fact in state:
            if fact.startswith('(at '):
                parts = fact[1:-1].split()
                if parts[0] == 'at' and len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    # Check if it's the man (heuristic: assume only one 'man' object)
                    if man_name is None:
                        man_name = obj
                        man_location = loc
            elif fact.startswith('(carrying '):
                parts = fact[1:-1].split()
                if parts[0] == 'carrying':
                    man_name = parts[1]

        # If man's location not found, search again using man's name
        if man_location is None and man_name is not None:
            for fact in state:
                if fact.startswith(f'(at {man_name} '):
                    parts = fact[1:-1].split()
                    man_location = parts[2]
                    break

        # Extract loose nuts and their locations
        for fact in state:
            if fact.startswith('(loose '):
                nut = fact[1:-1].split()[1]
                for f in state:
                    if f.startswith(f'(at {nut} '):
                        loc = f[1:-1].split()[2]
                        loose_nuts[nut] = loc
                        break

        # Extract usable spanners and their locations
        for fact in state:
            if fact.startswith('(usable '):
                spanner = fact[1:-1].split()[1]
                carried = f'(carrying {man_name} {spanner})' in state if man_name else False
                if carried:
                    usable_spanners[spanner] = (man_location, True)
                else:
                    for f in state:
                        if f.startswith(f'(at {spanner} '):
                            loc = f[1:-1].split()[2]
                            usable_spanners[spanner] = (loc, False)
                            break

        if not loose_nuts:
            return 0  # All nuts are tightened

        if len(usable_spanners) < len(loose_nuts):
            return float('inf')  # Not enough spanners

        # Generate all possible (nut, spanner) costs
        assignments = []
        for nut, nut_loc in loose_nuts.items():
            for spanner, (spanner_loc, is_carried) in usable_spanners.items():
                if is_carried:
                    dist = self.shortest_paths.get(man_location, {}).get(nut_loc, float('inf'))
                    cost = dist + 1  # walk to nut + tighten
                else:
                    dist1 = self.shortest_paths.get(man_location, {}).get(spanner_loc, float('inf'))
                    dist2 = self.shortest_paths.get(spanner_loc, {}).get(nut_loc, float('inf'))
                    cost = dist1 + 1 + dist2 + 1  # walk to spanner + pickup + walk to nut + tighten
                assignments.append((nut, spanner, cost))

        # Sort by cost and assign greedily
        assignments.sort(key=lambda x: x[2])
        used_spanners = set()
        used_nuts = set()
        total_cost = 0

        for nut, spanner, cost in assignments:
            if nut not in used_nuts and spanner not in used_spanners:
                used_nuts.add(nut)
                used_spanners.add(spanner)
                total_cost += cost
                if len(used_nuts) == len(loose_nuts):
                    break

        return total_cost if total_cost != 0 else 0
