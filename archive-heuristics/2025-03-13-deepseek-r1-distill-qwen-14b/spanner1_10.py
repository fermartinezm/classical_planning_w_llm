from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by calculating the minimal steps required for the man to reach each nut with a spanner.

    # Assumptions:
    - The man can carry only one spanner at a time.
    - If the man doesn't have a spanner, he must pick one up before tightening a nut.
    - The shortest path between locations is used to minimize travel distance.

    # Heuristic Initialization
    - Extract static facts to build a graph of locations and compute shortest paths between all pairs of locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the man.
    2. Determine if the man is carrying a spanner.
    3. For each loose nut:
       a. If the man has a spanner:
          i. If he is at the nut's location, 1 action is needed.
          ii. Otherwise, the distance from his current location to the nut's location plus 1 action.
       b. If the man doesn't have a spanner:
          i. Find the closest spanner's location.
          ii. Calculate the distance from the man's current location to the spanner's location plus 1 action to pick it up.
          iii. Then, calculate the distance from the spanner's location to the nut's location plus 1 action to tighten it.
          iv. Sum these distances and actions for the nut.
    4. Sum the actions for all loose nuts to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building the location graph."""
        super().__init__(task)
        self.location_graph = self.build_location_graph(task.static)
        self.distances = self.compute_all_pairs_shortest_paths(self.location_graph)

    def build_location_graph(self, static_facts):
        """Build an adjacency list representing the graph of locations."""
        graph = {}
        for fact in static_facts:
            if fact.startswith('(link'):
                parts = fact[5:-1].split()
                loc1, loc2 = parts[0], parts[1]
                if loc1 not in graph:
                    graph[loc1] = []
                if loc2 not in graph:
                    graph[loc2] = []
                graph[loc1].append(loc2)
                graph[loc2].append(loc1)
        return graph

    def compute_all_pairs_shortest_paths(self, graph):
        """Compute shortest paths between all pairs of locations using BFS."""
        distances = {}
        for source in graph:
            distances[source] = {}
            queue = deque()
            queue.append((source, 0))
            visited = {source}
            while queue:
                current, dist = queue.popleft()
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[source][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            # For locations not reachable, set distance to infinity
            for loc in graph:
                if loc not in distances[source]:
                    distances[source][loc] = float('inf')
        return distances

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state
        man_location = self.get_man_location(state)
        has_spanner = self.has_spanner(state)
        loose_nuts = self.get_loose_nuts(state)
        total_actions = 0

        for nut in loose_nuts:
            nut_location = self.get_nut_location(nut)
            if has_spanner:
                if man_location == nut_location:
                    total_actions += 1
                else:
                    distance = self.distances[man_location].get(nut_location, float('inf'))
                    if distance == float('inf'):
                        return float('inf')
                    total_actions += distance + 1
            else:
                spanner_locations = self.get_spanner_locations(state)
                min_actions = float('inf')
                for spanner_loc in spanner_locations:
                    d1 = self.distances[man_location].get(spanner_loc, float('inf'))
                    d2 = self.distances[spanner_loc].get(nut_location, float('inf'))
                    if d1 == float('inf') or d2 == float('inf'):
                        continue
                    actions = d1 + 1 + d2 + 1
                    if actions < min_actions:
                        min_actions = actions
                if min_actions == float('inf'):
                    return float('inf')
                total_actions += min_actions

        return total_actions

    def get_man_location(self, state):
        """Extract the current location of the man."""
        for fact in state:
            if fact.startswith('(at ') and ' - man' in fact:
                parts = fact.split()
                return parts[-1]
        return None

    def has_spanner(self, state):
        """Check if the man is carrying a spanner."""
        for fact in state:
            if fact.startswith('(carrying ') and ' - spanner' in fact:
                return True
        return False

    def get_loose_nuts(self, state):
        """Extract all loose nuts from the state."""
        return [fact for fact in state if fact.startswith('(loose ')]

    def get_nut_location(self, nut_fact):
        """Extract the location of a nut from its fact."""
        parts = nut_fact.split()
        return parts[-1]

    def get_spanner_locations(self, state):
        """Extract the current locations of all spanners."""
        spanner_locs = []
        for fact in state:
            if fact.startswith('(at ') and ' - spanner' in fact:
                parts = fact.split()
                spanner_locs.append(parts[-1])
        return spanner_locs
