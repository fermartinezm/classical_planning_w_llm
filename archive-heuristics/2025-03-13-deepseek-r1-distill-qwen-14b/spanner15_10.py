from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by considering:
    - The distance the man needs to travel to reach the nuts.
    - The number of loose nuts.
    - Whether the man needs to pick up a spanner first.

    # Assumptions:
    - The man can carry multiple spanners, each usable for one nut.
    - The man must be at the same location as a nut to tighten it.
    - The man must carry a spanner to tighten a nut.

    # Heuristic Initialization
    - Extracts goal conditions (all nuts must be tightened).
    - Builds a graph of locations from static facts (links between locations).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the man.
    2. Identify the locations of all loose nuts.
    3. If there are no loose nuts, return 0.
    4. Calculate the shortest distance from the man's location to each loose nut.
    5. Determine the farthest nut (maximum distance).
    6. If the man is not carrying any spanners:
       a. Find the nearest usable spanner.
       b. Calculate the distance from the man to the spanner and then to the farthest nut.
       c. The heuristic is the sum of these distances plus the number of loose nuts.
    7. If the man is carrying spanners:
       a. Calculate the number of trips needed (ceil(loose_nuts / carried_spanners)).
       b. The heuristic is the number of trips multiplied by the farthest distance plus the number of loose nuts.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency list for locations based on static links
        self.adjacency = {}
        for fact in static_facts:
            if self.match(fact, "link", "*", "*"):
                loc1, loc2 = self.parse_fact(fact)[1], self.parse_fact(fact)[2]
                if loc1 not in self.adjacency:
                    self.adjacency[loc1] = []
                self.adjacency[loc1].append(loc2)
                if loc2 not in self.adjacency:
                    self.adjacency[loc2] = []
                self.adjacency[loc2].append(loc1)

    def parse_fact(self, fact):
        """Extract components from a PDDL fact string."""
        return fact[1:-1].split()

    def match(self, fact, pattern):
        """Check if a fact matches a given pattern."""
        parts = self.parse_fact(fact)
        return all(fnmatch(part, p) for part, p in zip(parts, pattern))

    def bfs(self, start, target):
        """Find the shortest path distance using BFS."""
        if start == target:
            return 0
        visited = set()
        queue = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.adjacency.get(current, []):
                if neighbor == target:
                    return dist + 1
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        return float('inf')  # No path found

    def __call__(self, node):
        state = node.state
        man_location = None
        loose_nuts = []
        carried_spanners = 0
        nut_locations = set()

        # Extract man's location
        for fact in state:
            parts = self.parse_fact(fact)
            if parts[0] == 'at' and parts[1] == 'bob':
                man_location = parts[2]

        # Extract loose nuts and their locations
        for fact in state:
            parts = self.parse_fact(fact)
            if parts[0] == 'loose':
                nut = parts[1]
                loose_nuts.append(nut)
                # Find the location of the nut
                for fact_loc in state:
                    if self.match(fact_loc, f'(at {nut} * )'):
                        loc = self.parse_fact(fact_loc)[2]
                        nut_locations.add(loc)
                        break

        # Extract carried spanners count
        for fact in state:
            parts = self.parse_fact(fact)
            if parts[0] == 'carrying' and parts[1] == 'bob':
                carried_spanners += 1

        if not loose_nuts:
            return 0

        # Calculate distances from man's location to each nut's location
        max_distance = 0
        farthest_nut_location = None
        for nut_loc in nut_locations:
            distance = self.bfs(man_location, nut_loc)
            if distance > max_distance:
                max_distance = distance
                farthest_nut_location = nut_loc

        if carried_spanners == 0:
            # Find all usable spanners' locations
            spanner_locations = []
            for fact in state:
                if self.match(fact, '(at spanner* *)') and self.match(fact, '(usable spanner*)'):
                    loc = self.parse_fact(fact)[2]
                    spanner_locations.append(loc)
            if not spanner_locations:
                return float('inf')  # No spanner available

            min_total_distance = float('inf')
            for spanner_loc in spanner_locations:
                # Distance from man to spanner
                d1 = self.bfs(man_location, spanner_loc)
                if d1 == float('inf'):
                    continue  # Skip if no path
                # Distance from spanner to farthest nut
                d2 = self.bfs(spanner_loc, farthest_nut_location)
                if d2 == float('inf'):
                    continue  # Skip if no path
                total = d1 + d2
                if total < min_total_distance:
                    min_total_distance = total

            if min_total_distance == float('inf'):
                return float('inf')
            return min_total_distance + len(loose_nuts)
        else:
            # Calculate number of trips needed
            trips = (len(loose_nuts) + carried_spanners - 1) // carried_spanners
            return trips * max_distance + len(loose_nuts)
