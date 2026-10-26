from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, pattern):
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    for p, pat in zip(parts, pattern_parts):
        if not fnmatch(p, pat):
            return False
    return True

class spanner14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts. It considers the man's current location, the locations of loose nuts, and the available usable spanners (either carried or on the ground). The heuristic calculates the minimal steps needed to collect spanners and reach each nut's location.

    # Assumptions
    - The man can carry multiple spanners but each spanner can be used only once.
    - The shortest path between locations is precomputed using static link information.
    - Each loose nut requires a separate spanner, and the heuristic accounts for the travel and action costs to acquire and use each spanner.

    # Heuristic Initialization
    - Extracts static links to build a location graph and precompute shortest paths between all locations.
    - Identifies all locations from the static link information.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Precompute Shortest Paths**: Use BFS on the location graph from static links to determine the shortest path between all pairs of locations.
    2. **Identify Current State**: Determine the man's current location, loose nuts and their locations, and usable spanners (carried or on the ground).
    3. **Calculate Cost per Nut**: For each loose nut, compute the minimal cost considering:
        - Spanners already carried (cost: walk to nut + tighten).
        - Spanners on the ground (cost: walk to spanner + pickup + walk to nut + tighten).
    4. **Sum Costs**: Sum the minimal costs for all loose nuts to get the heuristic value.
    """

    def __init__(self, task):
        self.static = task.static
        self.locations = set()
        self.location_graph = defaultdict(list)

        # Build location graph from static links
        for fact in self.static:
            if match(fact, "link * *"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.location_graph[l1].append(l2)

        # Precompute shortest paths between all locations
        self.shortest_paths = {}
        for loc in self.locations:
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
        man = None
        carrying_facts = [fact for fact in state if match(fact, "carrying * *")]
        if carrying_facts:
            man = get_parts(carrying_facts[0])[1]
        else:
            # Find man by checking 'at' facts for an object not a spanner, nut, or location
            usable_spanners = set()
            nuts = set()
            for fact in state:
                if match(fact, "usable *"):
                    usable_spanners.add(get_parts(fact)[1])
                if match(fact, "loose *") or match(fact, "tightened *"):
                    nuts.add(get_parts(fact)[1])
            for fact in state:
                if match(fact, "at * *"):
                    parts = get_parts(fact)
                    obj, loc = parts[1], parts[2]
                    if obj not in usable_spanners and obj not in nuts and obj in self.locations:
                        continue
                    if obj not in usable_spanners and obj not in nuts and obj not in self.locations:
                        man = obj
                        break
            if man is None:
                return 0  # Fallback if man not found

        # Get man's current location
        current_man_loc = None
        for fact in state:
            if match(fact, f"at {man} *"):
                current_man_loc = get_parts(fact)[2]
                break
        if current_man_loc is None:
            return 0  # Fallback

        # Collect loose nuts and their locations
        nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "loose *"):
                nuts.add(get_parts(fact)[1])
            if match(fact, "at * *"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in nuts:
                    nut_locations[obj] = loc
        loose_nuts = [(nut, nut_locations[nut]) for nut in nuts if f"(loose {nut})" in state]

        # Collect usable spanners and their locations
        usable_spanners = set()
        spanner_locations = {}
        carried_spanners = set()
        for fact in state:
            if match(fact, "usable *"):
                spanner = get_parts(fact)[1]
                usable_spanners.add(spanner)
            if match(fact, f"carrying {man} *"):
                spanner = get_parts(fact)[2]
                if spanner in usable_spanners:
                    carried_spanners.add(spanner)
            if match(fact, "at * *"):
                parts = get_parts(fact)
                obj, loc = parts[1], parts[2]
                if obj in usable_spanners:
                    spanner_locations[obj] = loc

        total_cost = 0
        for nut, nut_loc in loose_nuts:
            min_cost = float('inf')
            for spanner in usable_spanners:
                if spanner in carried_spanners:
                    # Spanner is carried and usable
                    distance = self.shortest_paths[current_man_loc].get(nut_loc, float('inf'))
                    cost = distance + 1  # walk steps + tighten
                else:
                    # Spanner is on the ground
                    if spanner not in spanner_locations:
                        continue
                    spanner_loc = spanner_locations[spanner]
                    distance_to_spanner = self.shortest_paths[current_man_loc].get(spanner_loc, float('inf'))
                    distance_to_nut = self.shortest_paths[spanner_loc].get(nut_loc, float('inf'))
                    cost = distance_to_spanner + 1 + distance_to_nut + 1  # pickup and tighten
                if cost < min_cost:
                    min_cost = cost
            if min_cost != float('inf'):
                total_cost += min_cost
            else:
                # No usable spanner available for this nut (problem unsolvable)
                total_cost += 0  # Assume solvable as per problem statement

        return total_cost
