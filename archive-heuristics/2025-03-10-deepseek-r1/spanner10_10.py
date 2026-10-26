from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class spanner10Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions required to tighten all loose nuts by considering the minimal path to collect usable spanners and reach each nut's location.

    # Assumptions
    - The man (bob) can carry multiple spanners, but each spanner can be used only once.
    - The shortest path between locations is precomputed using static link information.
    - The man's name is 'bob' as per the problem examples.

    # Heuristic Initialization
    - Extracts static link facts to build a directed graph of locations.
    - Precomputes shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location.
    2. Collect all loose nuts and their locations.
    3. Identify all usable spanners (carried or on the ground).
    4. For each loose nut, compute the minimal cost to tighten it using the best available spanner.
    5. Assign spanners to nuts greedily, marking used spanners to avoid reuse.
    6. Sum the minimal costs for all nuts to get the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static_links = defaultdict(list)
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'link':
                start, end = parts[1], parts[2]
                self.static_links[start].append(end)
        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        all_locations = set(self.static_links.keys())
        for ends in self.static_links.values():
            all_locations.update(ends)
        all_locations = list(all_locations)
        for loc in all_locations:
            distances = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.static_links.get(current, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            self.shortest_paths[loc] = distances

    def __call__(self, node):
        state = node.state
        # Find man's location (assumed to be 'bob')
        man_location = None
        for fact in state:
            if fact.startswith('(at bob '):
                parts = fact[1:-1].split()
                man_location = parts[2]
                break
        if not man_location:
            return float('inf')  # Invalid state
        
        # Collect loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if fact.startswith('(loose '):
                parts = fact[1:-1].split()
                loose_nuts.append(parts[1])
            elif fact.startswith('(at ') and 'nut' in fact:
                parts = fact[1:-1].split()
                obj, loc = parts[1], parts[2]
                nut_locations[obj] = loc
        
        # Collect usable spanners and their locations
        usable_spanners = []
        carried_spanners = []
        spanner_locations = {}
        for fact in state:
            if fact.startswith('(usable '):
                parts = fact[1:-1].split()
                usable_spanners.append(parts[1])
            if fact.startswith('(carrying bob '):
                parts = fact[1:-1].split()
                carried_spanners.append(parts[2])
            if fact.startswith('(at ') and 'spanner' in fact:
                parts = fact[1:-1].split()
                spanner_locations[parts[1]] = parts[2]
        
        # Prepare available spanners (carried or on ground)
        available_spanners = []
        for spanner in usable_spanners:
            if spanner in carried_spanners:
                available_spanners.append((spanner, man_location, True))
            elif spanner in spanner_locations:
                available_spanners.append((spanner, spanner_locations[spanner], False))
        
        # Assign spanners to nuts greedily
        total_cost = 0
        used_spanners = set()
        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if not nut_loc:
                continue  # Skip if nut location not found
            
            min_cost = float('inf')
            best_spanner = None
            for spanner_info in available_spanners:
                spanner, s_loc, is_carried = spanner_info
                if spanner in used_spanners:
                    continue
                
                if is_carried:
                    distance = self.shortest_paths[man_location].get(nut_loc, float('inf'))
                    cost = distance + 1
                else:
                    d1 = self.shortest_paths[man_location].get(s_loc, float('inf'))
                    d2 = self.shortest_paths[s_loc].get(nut_loc, float('inf'))
                    cost = d1 + 1 + d2 + 1 if d1 != float('inf') and d2 != float('inf') else float('inf')
                
                if cost < min_cost:
                    min_cost = cost
                    best_spanner = spanner
            
            if best_spanner is not None:
                total_cost += min_cost
                used_spanners.add(best_spanner)
            else:
                total_cost += 1000000  # Penalize for missing spanner
        
        return total_cost
