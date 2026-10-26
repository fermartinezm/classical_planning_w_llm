from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class Spanner2Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts. It considers the man's current location, the locations of loose nuts, and the availability of usable spanners (either carried or on the ground). The heuristic calculates the minimal path to collect spanners and reach each nut, summing the required walk, pickup, and tighten actions.

    # Assumptions
    - The man can carry multiple spanners but each spanner can be used only once.
    - The problem is solvable (enough usable spanners for the loose nuts).
    - The links between locations form an undirected graph.

    # Heuristic Initialization
    - Precompute shortest paths between all locations using BFS based on static 'link' facts.
    - Extract the man's name from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Identify Current State Components**:
        - Extract the man's current location.
        - Identify all loose nuts and their locations.
        - Identify all usable spanners (carried or on the ground).
    2. **Calculate Movement Costs**:
        - For each loose nut, compute the minimal cost to use an available spanner.
        - If the spanner is carried, cost is walking to the nut's location + tighten.
        - If the spanner is on the ground, cost includes walking to the spanner, picking it up, then walking to the nut's location.
    3. **Greedy Assignment**:
        - For each nut, select the spanner with the minimal cost, mark it as used, and accumulate the total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic with static data and precompute shortest paths."""
        self.goals = task.goals
        self.static = task.static

        # Build location graph from static 'link' facts
        self.graph = defaultdict(list)
        self.locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)
                self.locations.update({loc1, loc2})

        # Precompute shortest paths between all locations using BFS
        self.distance = defaultdict(dict)
        for start in self.graph:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for node, dist in visited.items():
                self.distance[start][node] = dist

        # Extract man's name from initial state (assuming one man)
        self.man_name = None
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'carrying':
                self.man_name = parts[1]
                break
        if self.man_name is None:  # If not carrying, find 'at' fact for man
            for fact in task.initial_state:
                parts = get_parts(fact)
                if parts[0] == 'at' and parts[2] in self.locations:
                    possible_man = parts[1]
                    # Check if not a location, spanner, or nut (heuristic)
                    self.man_name = possible_man
                    break

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check if all goals are satisfied
        if self.goals.issubset(state):
            return 0

        # Extract man's current location
        man_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man_name:
                man_loc = parts[2]
                break
        if man_loc is None:
            return float('inf')  # Invalid state

        # Collect loose nuts and their locations
        loose_nuts = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'loose':
                nut = parts[1]
                # Find nut's location
                for loc_fact in state:
                    loc_parts = get_parts(loc_fact)
                    if loc_parts[0] == 'at' and loc_parts[1] == nut:
                        loose_nuts[nut] = loc_parts[2]
                        break

        # Collect usable spanners (carried or on ground)
        usable_spanners = []
        carried_spanners = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and parts[1] == self.man_name:
                carried_spanners.add(parts[2])
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'usable':
                spanner = parts[1]
                if spanner in carried_spanners:
                    usable_spanners.append((spanner, man_loc, True))
                else:
                    # Find spanner's location
                    for loc_fact in state:
                        loc_parts = get_parts(loc_fact)
                        if loc_parts[0] == 'at' and loc_parts[1] == spanner:
                            usable_spanners.append((spanner, loc_parts[2], False))
                            break

        # Calculate minimal cost for each loose nut
        total_cost = 0
        used_spanners = set()
        for nut, nut_loc in loose_nuts.items():
            min_cost = float('inf')
            best_spanner = None
            for spanner_info in usable_spanners:
                spanner, s_loc, is_carried = spanner_info
                if spanner in used_spanners:
                    continue
                if is_carried:
                    # Cost: walk to nut_loc (if needed) + tighten
                    dist = self.distance.get(man_loc, {}).get(nut_loc, float('inf'))
                    cost = dist + 1
                else:
                    # Cost: walk to spanner, pickup, walk to nut, tighten
                    dist_to_s = self.distance.get(man_loc, {}).get(s_loc, float('inf'))
                    dist_to_nut = self.distance.get(s_loc, {}).get(nut_loc, float('inf'))
                    cost = dist_to_s + 1 + dist_to_nut + 1
                if cost < min_cost:
                    min_cost = cost
                    best_spanner = spanner_info
            if best_spanner is None:
                return float('inf')  # No spanner available, but heuristic assumes solvable
            total_cost += min_cost
            used_spanners.add(best_spanner[0])

        return total_cost
