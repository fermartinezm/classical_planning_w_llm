from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class spanner4Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts. It considers the man's current location, the locations of usable spanners (carried or on the ground), and the shortest path distances between locations.

    # Assumptions
    - Each spanner can be used only once.
    - The man can carry multiple spanners but must pick up each one individually.
    - The shortest path between locations is precomputed using BFS based on static link facts.

    # Heuristic Initialization
    - Precompute the shortest path distances between all locations using BFS based on the static link facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Collect all loose nuts and their locations.
    3. Collect all usable spanners and determine if they are carried or their current location.
    4. For each loose nut, find the minimal cost spanner (not yet used) considering:
        a. If the spanner is carried, cost is distance from man's location to nut's location + 1 (tighten).
        b. If the spanner is on the ground, cost is distance to spanner + 1 (pickup) + distance to nut + 1 (tighten).
    5. Sum the minimal costs for all nuts, ensuring each spanner is used at most once.
    """

    def __init__(self, task):
        """Precompute shortest paths between all locations using static link facts."""
        self.distance = defaultdict(dict)
        links = [fact for fact in task.static if fnmatch(fact, '(link * *')]

        # Build adjacency list
        adj = defaultdict(list)
        for fact in links:
            parts = fact[1:-1].split()
            l1, l2 = parts[1], parts[2]
            adj[l1].append(l2)

        # Collect all unique locations
        all_locations = set()
        for l1 in adj:
            all_locations.add(l1)
            for l2 in adj[l1]:
                all_locations.add(l2)
        all_locations = list(all_locations)

        # Precompute shortest paths using BFS for each location
        for start in all_locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                current_dist = visited[current]
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        queue.append(neighbor)
            # Update distance dictionary for this start location
            for loc in all_locations:
                self.distance[start][loc] = visited.get(loc, float('inf'))

    def __call__(self, node):
        state = node.state
        man_name = None
        man_loc = None

        # Determine man's name and location
        for fact in state:
            if fnmatch(fact, '(at * *'):
                parts = fact[1:-1].split()
                possible_man = parts[1]
                if any(fnmatch(f, f'(carrying {possible_man} *') for f in state):
                    man_name = possible_man
                    man_loc = parts[2]
                    break
        if not man_name:
            # Fallback: check for 'at' facts not corresponding to spanner or nut
            spanners = {parts[1] for fact in state if fnmatch(fact, '(at spanner* *')}
            nuts = {parts[1] for fact in state if fnmatch(fact, '(at nut* *')}
            for fact in state:
                if fnmatch(fact, '(at * *'):
                    parts = fact[1:-1].split()
                    obj = parts[1]
                    if obj not in spanners and obj not in nuts:
                        man_name = obj
                        man_loc = parts[2]
                        break
        if not man_name:  # Final fallback
            man_name = 'bob'
            for fact in state:
                if fnmatch(fact, '(at bob *'):
                    parts = fact[1:-1].split()
                    man_loc = parts[2]
                    break

        # Collect loose nuts and their locations
        loose_nuts = []
        for fact in state:
            if fnmatch(fact, '(loose *'):
                nut = fact[1:-1].split()[1]
                for loc_fact in state:
                    if fnmatch(loc_fact, f'(at {nut} *'):
                        nut_loc = loc_fact[1:-1].split()[2]
                        loose_nuts.append((nut, nut_loc))
                        break

        # Collect usable spanners and their info
        usable_spanners = []
        for fact in state:
            if fnmatch(fact, '(usable *'):
                spanner = fact[1:-1].split()[1]
                carried = f'(carrying {man_name} {spanner})' in state
                spanner_loc = man_loc if carried else None
                if not carried:
                    for loc_fact in state:
                        if fnmatch(loc_fact, f'(at {spanner} *'):
                            spanner_loc = loc_fact[1:-1].split()[2]
                            break
                if spanner_loc is not None:
                    usable_spanners.append((spanner, spanner_loc, carried))

        total_cost = 0
        used_spanners = set()

        for nut, nut_loc in loose_nuts:
            min_cost = float('inf')
            best_spanner = None
            for s_info in usable_spanners:
                s, s_loc, carried = s_info
                if s in used_spanners:
                    continue
                if carried:
                    dist = self.distance[man_loc].get(nut_loc, float('inf'))
                    cost = dist + 1  # walk + tighten
                else:
                    dist_to_s = self.distance[man_loc].get(s_loc, float('inf'))
                    dist_s_to_nut = self.distance[s_loc].get(nut_loc, float('inf'))
                    cost = dist_to_s + 1 + dist_s_to_nut + 1  # walk, pickup, walk, tighten
                if cost < min_cost:
                    min_cost = cost
                    best_spanner = s
            if best_spanner is not None:
                total_cost += min_cost
                used_spanners.add(best_spanner)

        return total_cost
