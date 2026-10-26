from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class SpannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts by considering the minimal path to collect required spanners and reach each nut's location.

    # Assumptions:
    - The man can carry multiple spanners, but each spanner can be used only once.
    - The man starts at his current position and can move between connected locations.
    - Links between locations are directed as per the static facts.

    # Heuristic Initialization
    - Precompute shortest paths between all pairs of locations using BFS based on static link information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the man's current location from the state.
    2. Identify all loose nuts and their locations.
    3. Identify all usable spanners (carried and in the world).
    4. For each loose nut:
        a. If there are carried usable spanners, compute the cost as the distance from man's current location to the nut's location plus one tighten action.
        b. For each usable spanner in the world, compute the cost as the sum of distances from man's location to the spanner's location (pickup) and then to the nut's location (tighten).
        c. Take the minimal cost from the available options.
    5. Sum the minimal costs for all loose nuts.
    """

    def __init__(self, task):
        """Initialize the heuristic with static link information and precompute shortest paths."""
        self.static_links = defaultdict(list)
        self.locations = set()

        # Build the directed graph from static links
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'link':
                start, end = parts[1], parts[2]
                self.static_links[start].append(end)
                self.locations.update([start, end])

        # Precompute shortest paths using BFS for each location
        self.shortest_paths = defaultdict(dict)
        for start in self.locations:
            visited = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                current_dist = visited[current]
                for neighbor in self.static_links.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = current_dist + 1
                        queue.append(neighbor)
            # Update shortest paths for this start location
            for loc in self.locations:
                self.shortest_paths[start][loc] = visited.get(loc, float('inf'))

    def __call__(self, node):
        """Compute the heuristic estimate for the given state."""
        state = node.state
        total_cost = 0

        # Extract man's current location
        man_loc = None
        for fact in state:
            if fact.startswith('(at bob '):
                man_loc = fact.split()[2][:-1]  # Remove closing parenthesis
                break
        if not man_loc:
            return 0  # Should not happen in valid states

        # Extract loose nuts and their locations
        loose_nuts = []
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'loose':
                nut = parts[1]
                # Find the nut's location
                for loc_fact in state:
                    if loc_fact.startswith('(at {} '.format(nut)):
                        nut_loc = loc_fact.split()[2][:-1]
                        loose_nuts.append((nut, nut_loc))
                        break

        # Extract usable spanners (carried and in the world)
        carried_spanners = []
        world_spanners = []
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'carrying' and parts[1] == 'bob':
                spanner = parts[2]
                # Check if the spanner is usable
                if '(usable {})'.format(spanner) in state:
                    carried_spanners.append(spanner)
            elif parts[0] == 'at' and parts[1].startswith('spanner'):
                spanner = parts[1]
                # Check if the spanner is usable
                if '(usable {})'.format(spanner) in state:
                    spanner_loc = parts[2]
                    world_spanners.append((spanner, spanner_loc))

        # Calculate minimal cost for each loose nut
        for nut, nut_loc in loose_nuts:
            min_cost = float('inf')

            # Option 1: Use a carried usable spanner
            if carried_spanners:
                steps = self.shortest_paths[man_loc].get(nut_loc, float('inf'))
                if steps != float('inf'):
                    cost = steps + 1  # Walk steps + tighten
                    min_cost = min(min_cost, cost)

            # Option 2: Use a usable spanner from the world
            for spanner, spanner_loc in world_spanners:
                steps_to_spanner = self.shortest_paths[man_loc].get(spanner_loc, float('inf'))
                steps_to_nut = self.shortest_paths[spanner_loc].get(nut_loc, float('inf'))
                if steps_to_spanner != float('inf') and steps_to_nut != float('inf'):
                    cost = steps_to_spanner + 1 + steps_to_nut + 1  # pickup, tighten
                    min_cost = min(min_cost, cost)

            # Assume solvable, add minimal cost (if no path found, cost remains 0)
            if min_cost != float('inf'):
                total_cost += min_cost

        return total_cost
