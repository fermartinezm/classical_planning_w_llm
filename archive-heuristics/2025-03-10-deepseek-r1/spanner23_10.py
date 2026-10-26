from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    return fact.strip('()').split()

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions needed to tighten all loose nuts by calculating the minimal steps for each nut. 
    Considers the need to pick up spanners and move to nut locations.

    # Assumptions
    - The man can carry multiple spanners, but each can be used only once.
    - Links between locations are directed, and shortest paths are precomputed.
    - All spanners on the map are usable unless picked up and used.

    # Heuristic Initialization
    - Precomputes shortest paths between all locations using BFS based on static link facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location and loose nuts.
    2. For each loose nut:
        a. If the man has a usable spanner, calculate the distance to the nut's location and add tighten action.
        b. If no usable spanner is carried, find the closest available spanner, calculate the cost to pick it up, move to the nut, and tighten.
    3. Sum the minimal costs for all loose nuts.
    """

    def __init__(self, task):
        self.static = task.static
        self.location_graph = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                from_loc, to_loc = parts[1], parts[2]
                if from_loc not in self.location_graph:
                    self.location_graph[from_loc] = []
                self.location_graph[from_loc].append(to_loc)
        self.shortest_paths = {}
        locations = set()
        for from_loc in self.location_graph:
            locations.add(from_loc)
            for to_loc in self.location_graph[from_loc]:
                locations.add(to_loc)
        locations = list(locations)
        for loc in locations:
            self.shortest_paths[loc] = self.bfs(loc)

    def bfs(self, start):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.location_graph.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        state = node.state
        man_name = None
        spanner_names = set()
        nut_names = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'usable':
                spanner_names.add(parts[1])

        for fact in state:
            parts = get_parts(fact)
            if parts[0] in ['loose', 'tightened']:
                nut_names.add(parts[1])

        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj = parts[1]
                if obj not in spanner_names and obj not in nut_names:
                    man_name = obj
                    man_location = parts[2]
                    break

        if man_name is None or man_location is None:
            return float('inf')

        loose_nuts = [get_parts(fact)[1] for fact in state if fact.startswith('(loose')]

        nut_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in nut_names:
                nut_locations[parts[1]] = parts[2]

        carried_spanners = [get_parts(fact)[2] for fact in state if 
                            get_parts(fact)[0] == 'carrying' and get_parts(fact)[1] == man_name]
        usable_carried = [s for s in carried_spanners if f'(usable {s})' in state]

        total_cost = 0
        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if not nut_loc:
                return float('inf')

            distance_man_to_nut = self.shortest_paths.get(man_location, {}).get(nut_loc, float('inf'))
            cost_carried = distance_man_to_nut + 1 if usable_carried else float('inf')
            min_cost = cost_carried

            available_spanners = []
            for s in spanner_names:
                at_fact = any(fact.startswith(f'(at {s} ') for fact in state)
                usable = f'(usable {s})' in state
                if at_fact and usable:
                    available_spanners.append(s)

            min_spanner_cost = float('inf')
            for s in available_spanners:
                s_loc = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'at' and parts[1] == s:
                        s_loc = parts[2]
                        break
                if not s_loc:
                    continue

                d_man_to_s = self.shortest_paths.get(man_location, {}).get(s_loc, float('inf'))
                d_s_to_nut = self.shortest_paths.get(s_loc, {}).get(nut_loc, float('inf'))
                total_d = d_man_to_s + d_s_to_nut
                if total_d + 2 < min_spanner_cost:
                    min_spanner_cost = total_d + 2

            min_cost = min(min_cost, min_spanner_cost)
            if min_cost == float('inf'):
                return float('inf')
            total_cost += min_cost

        return total_cost
