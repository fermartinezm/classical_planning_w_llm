from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

class Spanner21Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts by calculating the minimal steps needed for each nut. For each loose nut, the man must either use a carried usable spanner or fetch one from its location. The heuristic sums the minimal steps for all loose nuts.

    # Assumptions
    - The man can carry multiple spanners, but each spanner can be used only once.
    - The man's movement between locations follows the shortest path based on static links.
    - Each nut requires a separate usable spanner.
    - The problem instances ensure there are enough usable spanners to tighten all nuts.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using static link information.
    - Identify the man's name from the initial state by checking 'carrying' and 'at' predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify the man's current location.
    3. For each loose nut:
        a. If the man is carrying a usable spanner:
            i. Add the distance from man's location to nut's location plus one tighten action.
        b. Else:
            i. Find all usable spanners on the ground.
            ii. For each such spanner, compute the total steps: walk to spanner, pickup, walk to nut, tighten.
            iii. Add the minimal steps from the above.
    4. Sum the steps for all loose nuts to get the heuristic value.
    """

    def __init__(self, task):
        self.goals = task.goals
        static = task.static

        # Build graph from static links
        links = set()
        for fact in static:
            parts = fact[1:-1].split()
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                links.add((l1, l2))
                links.add((l2, l1))

        self.graph = {}
        for l1, l2 in links:
            self.graph.setdefault(l1, set()).add(l2)
            self.graph.setdefault(l2, set()).add(l1)

        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        locations = set(self.graph.keys())
        for loc in locations:
            distances = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            self.shortest_paths[loc] = distances

    def __call__(self, node):
        state = node.state

        # Check if all goals are met
        if all(goal in state for goal in self.goals):
            return 0

        # Identify the man's name
        man = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'carrying':
                man = parts[1]
                break
        if not man:
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == 'at' and not (parts[1].startswith('spanner') or parts[1].startswith('nut')):
                    man = parts[1]
                    break
        if not man:
            return float('inf')

        # Find man's current location
        man_loc = None
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] == man:
                man_loc = parts[2]
                break
        if not man_loc:
            return float('inf')

        # Collect loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'loose':
                loose_nuts.append(parts[1])
            elif parts[0] == 'at' and parts[1].startswith('nut'):
                nut_locations[parts[1]] = parts[2]

        total_cost = 0

        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if not nut_loc:
                return float('inf')

            # Check if man is carrying any usable spanner
            carrying_usable = False
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == 'carrying' and parts[1] == man and parts[2].startswith('spanner'):
                    spanner = parts[2]
                    if f'(usable {spanner})' in state:
                        carrying_usable = True
                        break

            if carrying_usable:
                # Distance from man's location to nut's location
                distance = self.shortest_paths.get(man_loc, {}).get(nut_loc, float('inf'))
                total_cost += distance + 1  # +1 for tighten
            else:
                # Find all usable spanners on the ground
                spanner_locs = {}
                for fact in state:
                    parts = fact[1:-1].split()
                    if parts[0] == 'at' and parts[1].startswith('spanner'):
                        spanner_locs[parts[1]] = parts[2]

                usable_spanners = []
                for fact in state:
                    parts = fact[1:-1].split()
                    if parts[0] == 'usable' and parts[1] in spanner_locs:
                        usable_spanners.append((parts[1], spanner_locs[parts[1]]))

                if not usable_spanners:
                    return float('inf')

                min_steps = float('inf')
                for s, s_loc in usable_spanners:
                    d1 = self.shortest_paths.get(man_loc, {}).get(s_loc, float('inf'))
                    d2 = self.shortest_paths.get(s_loc, {}).get(nut_loc, float('inf'))
                    steps = d1 + d2 + 2  # pickup and tighten
                    if steps < min_steps:
                        min_steps = steps
                total_cost += min_steps

        return total_cost
