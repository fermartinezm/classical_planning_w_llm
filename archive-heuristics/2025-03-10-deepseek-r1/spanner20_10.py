import math
from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spanner20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the number of actions required to tighten all loose nuts by calculating the minimal path to collect usable spanners and reach each nut's location.

    Assumptions:
    - Each nut requires a unique usable spanner.
    - The man can carry multiple spanners but can use each only once.
    - Links between locations are directed, and shortest paths are precomputed.

    Heuristic Initialization:
    - Precomputes shortest paths between all locations using static link information.
    - Extracts the man's name from the initial state (assumes the man is 'bob').

    Step-By-Step Thinking for Computing Heuristic:
    1. Determine the man's current location.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners (carried or at a location).
    4. For each loose nut, greedily assign the closest usable spanner:
        a. If the spanner is carried, cost is distance from man's location to nut.
        b. If not, cost includes moving to the spanner, picking it up, then moving to the nut.
    5. Sum the minimal costs for all nuts, ensuring each spanner is used once.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Build the location graph from static links
        self.graph = defaultdict(list)
        for fact in self.static:
            if match(fact, 'link', '*', '*'):
                parts = get_parts(fact)
                from_loc, to_loc = parts[1], parts[2]
                self.graph[from_loc].append(to_loc)

        # Precompute shortest paths between all locations
        self.distances = {}
        for start in self.graph:
            queue = deque([(start, 0)])
            visited = {start: 0}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited or dist + 1 < visited[neighbor]:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            for node, d in visited.items():
                self.distances[(start, node)] = d

        # Extract man's name from initial state (assumes first 'at' fact for man)
        self.man = 'bob'  # Default assumption
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == 'bob':
                self.man = 'bob'
                break

    def get_distance(self, from_loc, to_loc):
        return self.distances.get((from_loc, to_loc), math.inf)

    def __call__(self, node):
        state = node.state
        man_location = None
        loose_nuts = []
        nut_locations = {}
        usable_spanners = []

        # Find man's current location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man:
                man_location = parts[2]
                break
        if not man_location:
            return math.inf

        # Collect loose nuts and their locations
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'loose':
                loose_nuts.append(parts[1])
            elif parts[0] == 'at' and parts[1].startswith('nut'):
                nut_locations[parts[1]] = parts[2]

        # Collect usable spanners (carried or at a location)
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'usable':
                spanner = parts[1]
                carried = False
                spanner_loc = None
                # Check if spanner is carried
                for f in state:
                    f_parts = get_parts(f)
                    if f_parts[0] == 'carrying' and f_parts[1] == self.man and f_parts[2] == spanner:
                        carried = True
                        break
                # If not carried, check location
                if not carried:
                    for f in state:
                        f_parts = get_parts(f)
                        if f_parts[0] == 'at' and f_parts[1] == spanner:
                            spanner_loc = f_parts[2]
                            break
                if carried or spanner_loc is not None:
                    usable_spanners.append( (spanner, carried, spanner_loc) )

        # Check if there are enough spanners
        if len(usable_spanners) < len(loose_nuts):
            return math.inf

        # Greedily assign spanners to nuts
        available_spanners = usable_spanners.copy()
        total_cost = 0
        current_man_loc = man_location

        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut)
            if not nut_loc:
                continue  # Nut location unknown, invalid state

            min_cost = math.inf
            selected_idx = None

            for idx, (spanner, carried, spanner_loc) in enumerate(available_spanners):
                if carried:
                    distance = self.get_distance(current_man_loc, nut_loc)
                    cost = distance + 1  # tighten
                else:
                    distance1 = self.get_distance(current_man_loc, spanner_loc)
                    distance2 = self.get_distance(spanner_loc, nut_loc)
                    if distance1 == math.inf or distance2 == math.inf:
                        continue  # No path
                    cost = distance1 + 1 + distance2 + 1  # pickup + tighten

                if cost < min_cost:
                    min_cost = cost
                    selected_idx = idx

            if selected_idx is None:
                return math.inf  # No available spanner can reach the nut

            total_cost += min_cost
            # Update man's location to the nut's location after tightening
            current_man_loc = nut_loc
            # Remove the used spanner
            available_spanners.pop(selected_idx)

        return total_cost
