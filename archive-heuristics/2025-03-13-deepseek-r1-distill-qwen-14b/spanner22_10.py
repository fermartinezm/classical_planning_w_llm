from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from heapq import heappush, heappop

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed for the man to tighten all loose nuts by considering the minimal movements required to pick up spanners and transport them to the nuts.

    # Assumptions:
    - Each nut requires a separate spanner to be tightened.
    - The man can carry only one spanner at a time and must drop it before picking up another.
    - The man must move between locations connected by links, with each move counting as one action.

    # Heuristic Initialization
    - Extract static facts to build a graph of locations and precompute the shortest paths between all pairs of locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the man's current location and whether he is carrying a spanner.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners and their locations.
    4. For each loose nut:
       a. Find the closest usable spanner to the man's current location.
       b. Calculate the distance from the man's location to the spanner's location.
       c. Add actions for moving to the spanner, picking it up, moving to the nut, and tightening it.
    5. If the man is carrying a spanner, add an action to drop it before proceeding.
    6. Sum all the calculated actions to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building the location graph."""
        static_facts = task.static

        # Build the location graph
        self.location_graph = {}
        for fact in static_facts:
            if fact.startswith('(link '):
                loc1, loc2 = self._parse_link(fact)
                self._add_edge(loc1, loc2)
                self._add_edge(loc2, loc1)

        # Precompute shortest paths between all pairs of locations
        self.shortest_path = {}
        for loc in self.location_graph:
            self._compute_shortest_paths(loc)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        man_location = self._get_man_location(state)
        carrying_spanner = self._is_carrying_spanner(state)
        nuts = self._get_loose_nuts(state)
        usable_spanners = self._get_usable_spanners(state)

        total_cost = 0

        if carrying_spanner:
            total_cost += 1  # Drop the current spanner

        for nut_loc in nuts:
            min_distance = float('inf')
            closest_spanner = None
            for spanner_loc in usable_spanners:
                distance = self.shortest_path.get(man_location, {}).get(spanner_loc, float('inf'))
                if distance < min_distance:
                    min_distance = distance
                    closest_spanner = spanner_loc
            if closest_spanner is None:
                continue  # No spanner available, but problem is solvable
            # Move to spanner and pick it up
            total_cost += min_distance + 1
            # Move to nut and tighten it
            distance_spanner_to_nut = self.shortest_path.get(closest_spanner, {}).get(nut_loc, float('inf'))
            total_cost += distance_spanner_to_nut + 1

        return total_cost

    def _parse_link(self, fact):
        """Parse a link fact and return the two connected locations."""
        parts = fact[5:-1].split()
        return parts[0], parts[1]

    def _add_edge(self, loc1, loc2):
        """Add an edge between two locations in the graph."""
        if loc1 not in self.location_graph:
            self.location_graph[loc1] = {}
        self.location_graph[loc1][loc2] = 1

    def _compute_shortest_paths(self, start_loc):
        """Compute shortest paths from start_loc to all other locations using BFS."""
        visited = {start_loc: 0}
        queue = deque([(start_loc, 0)])
        while queue:
            current, dist = queue.popleft()
            for neighbor in self.location_graph.get(current, {}):
                if neighbor not in visited or visited[neighbor] > dist + 1:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        self.shortest_path[start_loc] = visited

    def _get_man_location(self, state):
        """Extract the man's current location from the state."""
        for fact in state:
            if fact.startswith('(at ') and ' - man' in fact:
                return fact.split(')')[0].split(' (at ')[1].split(' ')[1]
        return None

    def _is_carrying_spanner(self, state):
        """Check if the man is carrying a spanner."""
        for fact in state:
            if fact.startswith('(carrying ') and ' - spanner' in fact:
                return True
        return False

    def _get_loose_nuts(self, state):
        """Extract the locations of all loose nuts."""
        nuts = []
        for fact in state:
            if fact.startswith('(loose ') and ' - nut' in fact:
                nut = fact.split(')')[0].split(' (loose ')[1].split(' ')[1]
                for loc_fact in state:
                    if loc_fact.startswith('(at ') and nut in loc_fact:
                        nuts.append(loc_fact.split(')')[0].split(' (at ')[1].split(' ')[2])
                        break
        return nuts

    def _get_usable_spanners(self, state):
        """Extract the locations of all usable spanners."""
        spanners = []
        for fact in state:
            if fact.startswith('(at ') and ' - spanner' in fact:
                spanner = fact.split(')')[0].split(' (at ')[1].split(' ')[1]
                location = fact.split(')')[0].split(' (at ')[1].split(' ')[2]
                spanners.append(location)
        return spanners
