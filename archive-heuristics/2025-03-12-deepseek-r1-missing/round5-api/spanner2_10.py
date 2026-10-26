from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts. It considers the steps to collect required spanners and move to each nut's location, using precomputed shortest paths between locations.

    # Assumptions
    - Each spanner can be used only once.
    - The man can carry multiple spanners but must pick up each one individually.
    - Links between locations are directed as per the static facts.

    # Heuristic Initialization
    - Extract static link facts to build a directed graph of locations.
    - Precompute shortest paths between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the man's current location from the state.
    2. Determine all loose nuts and their respective locations.
    3. Identify usable spanners (either carried or in the world) and their locations.
    4. For each loose nut, calculate the minimal cost to tighten it using the closest available spanner:
       a. If the spanner is carried, cost is the distance from the man's current location to the nut's location plus one tighten action.
       b. If the spanner is in the world, cost includes walking to the spanner, picking it up, walking to the nut's location, and tightening.
    5. Sum the minimal costs for all nuts, ensuring each spanner is used exactly once.
    """

    def __init__(self, task):
        """Initialize the heuristic with static link information and precompute shortest paths."""
        self.static_links = {}
        self.distances = {}

        # Extract link facts from static information
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'link':
                from_loc = parts[1]
                to_loc = parts[2]
                if from_loc not in self.static_links:
                    self.static_links[from_loc] = []
                self.static_links[from_loc].append(to_loc)

        # Precompute shortest paths between all locations
        self.precompute_distances()

    def precompute_distances(self):
        """Precompute shortest paths using BFS for each location."""
        all_locations = set()
        for from_loc in self.static_links:
            all_locations.add(from_loc)
            for to_loc in self.static_links[from_loc]:
                all_locations.add(to_loc)
        all_locations = list(all_locations)

        self.distances = {}
        for loc in all_locations:
            self.distances[loc] = {}
            queue = deque([(loc, 0)])
            visited = set()
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                self.distances[loc][current] = dist
                if current in self.static_links:
                    for neighbor in self.static_links[current]:
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal."""
        state = node.state
        man_location = None
        loose_nuts = []
        available_spanners = []

        # Extract man's current location
        for fact in state:
            if fact.startswith('(at bob '):
                man_location = fact[6:-1].split()[1]
                break

        # Extract loose nuts and their locations
        nut_locations = {}
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'loose':
                nut = parts[1]
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'at' and parts[1] in nut_locations:
                nut_locations[parts[1]] = parts[2]
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'loose':
                nut = parts[1]
                loose_nuts.append((nut, nut_locations.get(nut, None)))

        # Remove nuts without location (shouldn't happen in valid states)
        loose_nuts = [(n, loc) for n, loc in loose_nuts if loc is not None]

        # Extract usable spanners and their locations
        carried_spanners = []
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'carrying' and parts[1] == 'bob':
                carried_spanners.append(parts[2])
        for s in carried_spanners:
            available_spanners.append((s, man_location))

        # Add spanners not carried but usable and in the world
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'usable':
                spanner = parts[1]
                if spanner not in [s for s, _ in available_spanners]:
                    # Find spanner's location
                    for loc_fact in state:
                        loc_parts = loc_fact[1:-1].split()
                        if loc_parts[0] == 'at' and loc_parts[1] == spanner:
                            available_spanners.append((spanner, loc_parts[2]))
                            break

        # Convert to list for processing
        remaining_spanners = available_spanners.copy()
        total_cost = 0

        # Process each loose nut to find minimal cost
        for nut, nut_loc in loose_nuts:
            min_cost = float('inf')
            best_idx = -1

            for idx, (s, s_loc) in enumerate(remaining_spanners):
                # Calculate cost for this spanner
                if s_loc == man_location:
                    # Carried spanner
                    dist = self.distances.get(man_location, {}).get(nut_loc, float('inf'))
                    cost = dist + 1  # walk (if needed) + tighten
                else:
                    # Spanner in the world
                    dist_to_spanner = self.distances.get(man_location, {}).get(s_loc, float('inf'))
                    dist_to_nut = self.distances.get(s_loc, {}).get(nut_loc, float('inf'))
                    cost = dist_to_spanner + 1 + dist_to_nut + 1  # walk to spanner, pickup, walk to nut, tighten

                if cost < min_cost:
                    min_cost = cost
                    best_idx = idx

            if best_idx != -1 and min_cost != float('inf'):
                total_cost += min_cost
                del remaining_spanners[best_idx]

        return total_cost if total_cost != float('inf') else 0
