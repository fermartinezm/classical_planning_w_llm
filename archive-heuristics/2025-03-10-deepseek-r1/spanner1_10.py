from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spanner1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    Estimates the number of actions needed to tighten all loose nuts by calculating the minimal path to collect spanners and reach each nut.

    # Assumptions
    - There is a single man named 'bob' (as per problem examples).
    - Each spanner can be used only once.
    - The man can carry multiple spanners but must pick up each one individually.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using BFS based on static 'link' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Current State Analysis**:
        - Identify the man's current location.
        - Identify all loose nuts and their locations.
        - Identify all usable spanners and their locations (carried or on the ground).
    2. **Cost Calculation for Each Nut**:
        - For each loose nut, compute the minimal cost to use the closest usable spanner.
        - If the spanner is carried, cost is movement to the nut + tighten.
        - If the spanner is on the ground, cost includes movement to the spanner, pickup, movement to the nut, and tighten.
    3. **Greedy Assignment**:
        - Assign the closest spanner to each nut, marking spanners as used to prevent reuse.
    4. **Sum Costs**:
        - Sum the minimal costs for all nuts to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize heuristic with precomputed shortest paths between locations."""
        self.static = task.static
        self.distances = defaultdict(dict)
        self.precompute_distances()

    def precompute_distances(self):
        """Precompute shortest paths between all locations using BFS."""
        graph = defaultdict(list)
        # Build the directed graph from link facts
        for fact in self.static:
            if match(fact, 'link', '*', '*'):
                parts = get_parts(fact)
                from_loc, to_loc = parts[1], parts[2]
                graph[from_loc].append(to_loc)
        # Collect all unique locations
        locations = set()
        for from_loc in graph:
            locations.add(from_loc)
            for to_loc in graph[from_loc]:
                locations.add(to_loc)
        # BFS from each location to compute shortest paths
        for start in locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            # Update distances for all reachable locations
            for loc in locations:
                self.distances[start][loc] = visited.get(loc, float('inf'))

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        # Check if all nuts are tightened (goal state)
        if all(not match(fact, 'loose', '*') for fact in state):
            return 0

        # Find man's current location (assumed to be 'bob')
        current_man_loc = None
        for fact in state:
            if match(fact, 'at', 'bob', '*'):
                current_man_loc = get_parts(fact)[2]
                break
        if not current_man_loc:
            return float('inf')

        # Collect loose nuts and their locations
        loose_nuts = []
        nut_locations = {}
        for fact in state:
            if match(fact, 'loose', '*'):
                nut = get_parts(fact)[1]
                loose_nuts.append(nut)
        for nut in loose_nuts:
            for fact in state:
                if match(fact, 'at', nut, '*'):
                    nut_locations[nut] = get_parts(fact)[2]
                    break

        # Collect usable spanners and their locations (carried or on ground)
        usable_spanners = []
        carried_spanners = []
        for fact in state:
            if match(fact, 'usable', '*'):
                spanner = get_parts(fact)[1]
                usable_spanners.append(spanner)
            if match(fact, 'carrying', 'bob', '*'):
                spanner = get_parts(fact)[2]
                carried_spanners.append(spanner)
        # Determine locations of usable spanners
        spanner_locs = {}
        for spanner in usable_spanners:
            if spanner in carried_spanners:
                spanner_locs[spanner] = current_man_loc
            else:
                for fact in state:
                    if match(fact, 'at', spanner, '*'):
                        spanner_locs[spanner] = get_parts(fact)[2]
                        break

        # Greedily assign the best spanner to each nut
        total_cost = 0
        used_spanners = set()
        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if not nut_loc:
                continue
            best_cost = float('inf')
            best_spanner = None
            for spanner in usable_spanners:
                if spanner in used_spanners or spanner not in spanner_locs:
                    continue
                spanner_loc = spanner_locs[spanner]
                if spanner in carried_spanners:
                    # Cost: move to nut + tighten
                    dist = self.distances[current_man_loc].get(nut_loc, float('inf'))
                    cost = dist + 1
                else:
                    # Cost: move to spanner + pickup + move to nut + tighten
                    dist_to_spanner = self.distances[current_man_loc].get(spanner_loc, float('inf'))
                    dist_to_nut = self.distances[spanner_loc].get(nut_loc, float('inf'))
                    cost = dist_to_spanner + 1 + dist_to_nut + 1
                if cost < best_cost:
                    best_cost = cost
                    best_spanner = spanner
            if best_spanner is not None:
                total_cost += best_cost
                used_spanners.add(best_spanner)
            else:
                return float('inf')  # No spanner available

        return total_cost
