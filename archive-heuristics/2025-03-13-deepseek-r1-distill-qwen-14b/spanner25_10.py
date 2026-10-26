from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts. It considers the man's current location, whether he is carrying a spanner, and the locations of the nuts and spanners.

    # Assumptions:
    - The man can walk between connected locations.
    - A spanner is required to tighten a nut.
    - Each nut must be moved to if the man isn't already at its location.

    # Heuristic Initialization
    - Extracts static facts to build a graph of locations and precomputes shortest paths between all pairs of locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of loose nuts that need to be tightened.
    2. Determine if the man is currently carrying a spanner.
    3. If the man is not carrying a spanner, find the nearest spanner to his current location and calculate the cost to pick it up.
    4. For each loose nut, calculate the shortest path distance from the man's current location (or the nearest spanner's location if he isn't carrying one) to the nut's location.
    5. Sum the distances and add the number of actions required to tighten each nut.
    """

    def __init__(self, task):
        super().__init__(task)
        # Build the graph from static facts
        self.graph = {}
        static_facts = task.static
        for fact in static_facts:
            if fact.startswith('(link '):
                loc1, loc2 = self._get_link_locations(fact)
                self._add_edge(loc1, loc2)
                self._add_edge(loc2, loc1)
        # Precompute shortest paths using BFS for each node
        self.distances = {}
        for loc in self.graph:
            self.distances[loc] = self._compute_shortest_paths(loc)

    def _get_link_locations(self, fact):
        parts = fact[1:-1].split()
        return parts[1], parts[2]

    def _add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def _compute_shortest_paths(self, start):
        visited = {}
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            if node in visited:
                continue
            visited[node] = dist
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return visited

    def __call__(self, node):
        state = node.state
        # Extract man's location
        man_loc = None
        for fact in state:
            if fact.startswith('(at ') and ' - man' in fact:
                man_loc = fact.split()[-2]
                break
        if not man_loc:
            return 0  # shouldn't happen

        # Check if carrying a spanner
        carrying_spanner = any(fact.startswith('(carrying {})'.format(man_loc)) for fact in state)

        # Count loose nuts and get their locations
        loose_nuts = []
        for fact in state:
            if fact.startswith('(loose '):
                nut = fact.split()[1]
                loose_nuts.append(nut)
        if not loose_nuts:
            return 0

        # If carrying a spanner, calculate distances from man's location
        if carrying_spanner:
            total = 0
            for nut in loose_nuts:
                nut_loc = None
                for fact in state:
                    if fact.startswith('(at ' + nut + ' ') and fact.endswith(' - nut)'):
                        nut_loc = fact.split()[-2]
                        break
                if nut_loc:
                    distance = self.distances[man_loc].get(nut_loc, float('inf'))
                    total += distance + 1  # +1 for tightening
            return total
        else:
            # Find all spanner locations
            spanner_locs = []
            for fact in state:
                if fact.startswith('(at ') and ' - spanner' in fact:
                    spanner_locs.append(fact.split()[-2])
            if not spanner_locs:
                return float('inf')  # no spanner available

            # Find the nearest spanner to the man's location
            min_distance = float('inf')
            nearest_spanner = None
            for spanner_loc in spanner_locs:
                distance = self.distances[man_loc].get(spanner_loc, float('inf'))
                if distance < min_distance:
                    min_distance = distance
                    nearest_spanner = spanner_loc

            # Cost to pick up the nearest spanner
            cost = min_distance + 1  # distance + 1 action to pick up

            # Calculate distances from the nearest spanner to each nut
            total_nut_distance = 0
            for nut in loose_nuts:
                nut_loc = None
                for fact in state:
                    if fact.startswith('(at ' + nut + ' ') and fact.endswith(' - nut)'):
                        nut_loc = fact.split()[-2]
                        break
                if nut_loc:
                    distance = self.distances[nearest_spanner].get(nut_loc, float('inf'))
                    total_nut_distance += distance + 1  # +1 for tightening

            total = cost + total_nut_distance
            return total
