from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spanner12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by calculating the minimal path to collect spanners and reach each nut's location.

    # Assumptions
    - Each spanner can be used only once.
    - The man can carry multiple spanners but must pick up each one individually.
    - The shortest path between locations is precomputed using static link information.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using BFS based on static link facts.
    - Extract and store static information about links to build a directed graph.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the man's current location from the state.
    2. Identify all loose nuts and their respective locations.
    3. Identify all usable spanners and their locations (carried or not).
    4. For each loose nut, calculate the minimal cost to tighten it using the closest usable spanner.
    5. Sum the minimal costs for all loose nuts to get the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build the directed graph from link facts
        self.graph = defaultdict(list)
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.graph[l1].append(l2)

        # Precompute shortest paths between all locations using BFS
        self.shortest_paths = {}
        all_locations = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_locations.update(neighbors)
        all_locations = list(all_locations)

        for loc in all_locations:
            self.shortest_paths[loc] = self.bfs(loc)

    def bfs(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def shortest_path(self, from_loc, to_loc):
        if from_loc == to_loc:
            return 0
        return self.shortest_paths.get(from_loc, {}).get(to_loc, float('inf'))

    def __call__(self, node):
        state = node.state

        # Extract man's name and location
        man_name, man_location = None, None
        for fact in state:
            if match(fact, 'at', '*', '*'):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                # Check if the object is not a spanner or nut
                is_spanner_or_nut = False
                for f in state:
                    if match(f, 'at', obj, '*') and (match(f, 'at', '*', '*') and (match(f, 'carrying', '*', obj) or match(f, 'loose', obj) or match(f, 'tightened', obj))):
                        is_spanner_or_nut = True
                        break
                if not is_spanner_or_nut:
                    man_name, man_location = obj, loc
                    break

        if not man_location:
            return 0 if self.goals <= state else float('inf')

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

        if not loose_nuts:
            return 0

        # Collect usable spanners and their locations (carried or not)
        usable_spanners = []
        for fact in state:
            if match(fact, 'usable', '*'):
                spanner = get_parts(fact)[1]
                carried = any(match(f, 'carrying', man_name, spanner) for f in state)
                if carried:
                    usable_spanners.append((spanner, man_location, True))
                else:
                    for f in state:
                        if match(f, 'at', spanner, '*'):
                            spanner_loc = get_parts(f)[2]
                            usable_spanners.append((spanner, spanner_loc, False))
                            break

        total_cost = 0
        for nut, nut_loc in loose_nuts:
            min_cost = float('inf')
            for s, s_loc, is_carried in usable_spanners:
                if is_carried:
                    path = self.shortest_path(man_location, nut_loc)
                    cost = path + 1  # walk steps + tighten
                else:
                    path_to_spanner = self.shortest_path(man_location, s_loc)
                    path_to_nut = self.shortest_path(s_loc, nut_loc)
                    cost = path_to_spanner + 1 + path_to_nut + 1  # pickup and tighten
                if cost < min_cost:
                    min_cost = cost
            total_cost += min_cost if min_cost != float('inf') else 1e9

        return total_cost
