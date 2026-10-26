from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import heapq

class spanner15Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts. It considers the distance the man must walk to collect spanners and reach nuts, as well as the necessary pickup and tighten actions.

    # Assumptions
    - The man can carry multiple spanners but must pick up each one individually.
    - Each tighten action consumes one usable spanner.
    - The links between locations are static and directed.

    # Heuristic Initialization
    - Extracts static links to build a location graph and precompute shortest paths.
    - Identifies spanner and nut objects from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Precompute shortest paths between all locations using BFS.
    2. Determine the man's current location.
    3. Identify all loose nuts and their locations.
    4. Identify usable spanners carried by the man and available in the world.
    5. For each loose nut:
        a. Use a carried spanner if available, adding walk and tighten costs.
        b. Otherwise, find the closest available spanner, add costs for pickup, walk to spanner, walk to nut, and tighten.
    6. Sum all costs for an estimated total action count.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build the location graph from static links
        self.graph = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'link':
                from_loc, to_loc = parts[1], parts[2]
                if from_loc not in self.graph:
                    self.graph[from_loc] = []
                self.graph[from_loc].append(to_loc)

        # Precompute shortest paths between all locations
        self.locations = set(self.graph.keys())
        for links in self.graph.values():
            self.locations.update(links)
        self.shortest_paths = {loc: {} for loc in self.locations}

        for source in self.locations:
            queue = [(source, 0)]
            visited = set()
            while queue:
                current, dist = heapq.heappop(queue)
                if current in visited:
                    continue
                visited.add(current)
                self.shortest_paths[source][current] = dist
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        heapq.heappush(queue, (neighbor, dist + 1))
            # Set unreachable locations to infinity
            for loc in self.locations:
                if loc not in self.shortest_paths[source]:
                    self.shortest_paths[source][loc] = float('inf')

        # Extract spanners and nuts
        self.spanners = set()
        self.nuts = set()

        # Extract nuts from goals
        for goal in self.goals:
            parts = goal[1:-1].split()
            if parts[0] == 'tightened':
                self.nuts.add(parts[1])

        # Extract spanners from initial state
        initial_spanners = set()
        for fact in task.initial_state:
            parts = fact[1:-1].split()
            if parts[0] == 'carrying':
                initial_spanners.add(parts[2])
            elif parts[0] == 'at' and '(usable {})'.format(parts[1]) in task.initial_state:
                initial_spanners.add(parts[1])
        self.spanners = initial_spanners

    def get_distance(self, from_loc, to_loc):
        return self.shortest_paths.get(from_loc, {}).get(to_loc, float('inf'))

    def __call__(self, node):
        state = node.state
        man_location = None
        man_name = None

        # Find man's location
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] not in self.spanners and parts[1] not in self.nuts:
                man_name = parts[1]
                man_location = parts[2]
                break
        if not man_location:
            return float('inf')

        # Find loose nuts
        loose_nuts = set()
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'loose':
                loose_nuts.add(parts[1])

        # Carried usable spanners
        carried_spanners = set()
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'carrying' and parts[1] == man_name:
                spanner = parts[2]
                if f'(usable {spanner})' in state:
                    carried_spanners.add(spanner)
        num_carried = len(carried_spanners)

        # World usable spanners not carried
        world_spanners = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] in self.spanners:
                spanner = parts[1]
                if spanner in carried_spanners:
                    continue
                if f'(usable {spanner})' in state:
                    world_spanners[spanner] = parts[2]

        # Prepare sorted world spanners by distance
        world_sp_list = []
        for spanner, loc in world_spanners.items():
            dist = self.get_distance(man_location, loc)
            world_sp_list.append((dist, spanner, loc))
        world_sp_list.sort()

        total_cost = 0
        remaining_carried = num_carried
        available_world = world_sp_list.copy()

        for nut in loose_nuts:
            # Find nut's location
            nut_loc = None
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == 'at' and parts[1] == nut:
                    nut_loc = parts[2]
                    break
            if not nut_loc:
                continue

            if remaining_carried > 0:
                dist = self.get_distance(man_location, nut_loc)
                total_cost += dist + 1  # walk and tighten
                remaining_carried -= 1
            else:
                if not available_world:
                    return float('inf')
                dist_to_spanner, spanner, spanner_loc = available_world.pop(0)
                dist_to_nut = self.get_distance(spanner_loc, nut_loc)
                total_cost += dist_to_spanner + 1  # walk and pickup
                total_cost += dist_to_nut + 1  # walk and tighten

        return total_cost
