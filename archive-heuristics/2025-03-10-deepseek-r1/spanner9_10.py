from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class spanner9Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by considering the minimal path to collect usable spanners and reach each nut's location.

    # Assumptions
    - Each spanner can be used only once.
    - The man can carry multiple spanners, but each tighten action requires one usable spanner.
    - The links between locations form a directed graph, and movement is allowed only in the direction of the links.

    # Heuristic Initialization
    - Precompute the shortest paths between all pairs of locations using BFS based on the static link information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the man's current location from the state.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners (carried or in the world).
    4. For each loose nut, compute the minimal cost to tighten it using the closest available spanner.
    5. Assign spanners to nuts greedily, summing the minimal costs while ensuring each spanner is used at most once.
    6. Return the total estimated cost as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static links and precomputing shortest paths."""
        self.static_links = defaultdict(list)
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.static_links[l1].append(l2)

        # Precompute shortest paths between all pairs of locations
        self.shortest_paths = {}
        locations = set(self.static_links.keys())
        for link_list in self.static_links.values():
            locations.update(link_list)
        locations = list(locations)

        for start in locations:
            # BFS to find shortest paths from start
            queue = deque([(start, 0)])
            visited = {start: 0}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.static_links.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            for end in visited:
                self.shortest_paths[(start, end)] = visited[end]
            # Set unreachable locations to infinity
            for end in locations:
                if (start, end) not in self.shortest_paths:
                    self.shortest_paths[(start, end)] = float('inf')

    def __call__(self, node):
        """Estimate the minimal number of actions to reach the goal from the given state."""
        state = node.state
        man_location = None
        nuts = set()
        nut_locations = {}
        loose_nuts = {}
        usable_carried = set()
        usable_in_world = {}

        # Parse the state to extract relevant information
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj = parts[1]
                loc = parts[2]
                if obj == 'bob':  # Assuming the man is named 'bob'
                    man_location = loc
                else:
                    # Track all objects' locations for nuts and spanners
                    nut_locations[obj] = loc
            elif parts[0] in ['loose', 'tightened']:
                nut = parts[1]
                nuts.add(nut)
                if parts[0] == 'loose':
                    loose_nuts[nut] = None  # Placeholder, will be filled later
            elif parts[0] == 'carrying':
                man, spanner = parts[1], parts[2]
                # Check if the spanner is usable
                if f'(usable {spanner})' in state:
                    usable_carried.add(spanner)
            elif parts[0] == 'usable':
                spanner = parts[1]
                # Check if the spanner is in the world
                for f in state:
                    if f.startswith(f'(at {spanner} '):
                        parts_at = get_parts(f)
                        usable_in_world[spanner] = parts_at[2]
                        break

        # Fill nut locations for loose nuts
        for nut in loose_nuts:
            if nut in nut_locations:
                loose_nuts[nut] = nut_locations[nut]

        # Remove nuts that are not loose or have no location
        loose_nuts = {nut: loc for nut, loc in loose_nuts.items() if loc is not None}

        # Check if there are no loose nuts (goal achieved)
        if not loose_nuts:
            return 0

        # Collect all usable spanners
        usable_spanners = []
        # Carried spanners
        for s in usable_carried:
            usable_spanners.append((s, True, man_location))  # (spanner, is_carried, location)
        # Spanners in the world
        for s, loc in usable_in_world.items():
            usable_spanners.append((s, False, loc))

        # Check if there are enough spanners
        if len(usable_spanners) < len(loose_nuts):
            return float('inf')  # Not solvable, but per problem statement, assume solvable

        # Assign spanners to nuts greedily
        total_cost = 0
        remaining_spanners = usable_spanners.copy()
        loose_nut_list = list(loose_nuts.items())

        for nut, nut_loc in loose_nut_list:
            min_cost = float('inf')
            best_spanner_idx = None

            for idx, (s, is_carried, s_loc) in enumerate(remaining_spanners):
                if is_carried:
                    # Cost is distance from man's current location to nut_loc + 1 (tighten)
                    distance = self.shortest_paths.get((man_location, nut_loc), float('inf'))
                    cost = distance + 1
                else:
                    # Cost is distance to spanner + 1 (pickup) + distance to nut + 1 (tighten)
                    distance_man_to_s = self.shortest_paths.get((man_location, s_loc), float('inf'))
                    distance_s_to_nut = self.shortest_paths.get((s_loc, nut_loc), float('inf'))
                    cost = distance_man_to_s + 1 + distance_s_to_nut + 1

                if cost < min_cost:
                    min_cost = cost
                    best_spanner_idx = idx

            if best_spanner_idx is None:
                return float('inf')  # No spanner available

            total_cost += min_cost
            # Remove the selected spanner from remaining_spanners
            del remaining_spanners[best_spanner_idx]

        return total_cost
