from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the minimal elevator movement needed to visit all required origins and destinations, plus the necessary board and depart actions.

    # Assumptions
    - The elevator moves optimally in a single continuous path covering all required stops.
    - Floors are arranged in a linear hierarchy based on the 'above' relations.
    - Each board and depart action takes 1 step, and each elevator movement between adjacent floors takes 1 step.

    # Heuristic Initialization
    - Extract passengers' destinations from static 'destin' facts.
    - Build a floor hierarchy from 'above' facts to determine distances between floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Current State Analysis**:
        - Identify the current floor of the elevator.
        - Determine which passengers are not yet served (not in 'served' state).
        - Separate unserved passengers into those boarded and not boarded.
    2. **Required Stops Collection**:
        - For unboarded passengers, collect their origin and destination floors.
        - For boarded passengers, collect their destination floors.
    3. **Movement Calculation**:
        - Determine the minimal elevator path covering all required stops and the current floor.
        - Calculate movement steps using the formula: (max_floor - min_floor) + max(current_floor's distance to max or min).
    4. **Action Count**:
        - Count board (1 per unboarded passenger) and depart (1 per passenger) actions.
    5. **Total Estimate**:
        - Sum movement steps and action counts for the final heuristic value.
    """

    def __init__(self, task):
        # Extract destin for each passenger from static facts
        self.destin = {}
        # Build floor hierarchy from 'above' facts
        self.floor_indices = {}  # Maps floor to its index in the hierarchy (highest first)
        floors = set()
        graph = {}
        reverse_graph = {}

        # Parse static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            if parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor
            elif parts[0] == 'above':
                higher = parts[1]
                lower = parts[2]
                floors.add(higher)
                floors.add(lower)
                if higher not in graph:
                    graph[higher] = []
                graph[higher].append(lower)
                if lower not in reverse_graph:
                    reverse_graph[lower] = []
                reverse_graph[lower].append(higher)

        # Determine floor order using topological sort (Kahn's algorithm)
        in_degree = {f: 0 for f in floors}
        for f in floors:
            in_degree[f] = len(reverse_graph.get(f, []))

        queue = [f for f in in_degree if in_degree[f] == 0]
        sorted_floors = []

        while queue:
            node = queue.pop(0)
            sorted_floors.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        self.floor_indices = {f: idx for idx, f in enumerate(sorted_floors)}

    def __call__(self, node):
        state = node.state

        # Get current elevator floor
        current_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_floor = fact.split()[1][:-1]  # Extract 'fX' from '(lift-at fX)'
                break
        if not current_floor:
            return 0  # Should not happen if state is valid

        # Collect served passengers
        served = set()
        boarded = set()
        origins = {}  # passenger to origin floor (if not boarded)
        for fact in state:
            parts = fact.strip('()').split()
            if parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'boarded':
                boarded.add(parts[1])
            elif parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                origins[passenger] = floor

        # Determine unserved passengers
        unserved = []
        for p in self.destin:
            if p not in served:
                unserved.append(p)

        if not unserved:
            return 0

        # Collect required stops
        required_floors = []
        for p in unserved:
            if p in origins:  # Not boarded
                required_floors.append(origins[p])
                required_floors.append(self.destin[p])
            else:  # Boarded
                required_floors.append(self.destin[p])

        # Include current floor to compute min and max
        all_floors = required_floors + [current_floor]
        try:
            indices = [self.floor_indices[f] for f in all_floors]
        except KeyError as e:
            # Handle case where a floor is not in the indices (should not happen)
            return float('inf')

        min_idx = min(indices)
        max_idx = max(indices)
        current_idx = self.floor_indices[current_floor]

        distance_to_min = current_idx - min_idx
        distance_to_max = max_idx - current_idx
        movement_steps = (max_idx - min_idx) + max(distance_to_min, distance_to_max)

        # Calculate actions
        unserved_not_boarded = sum(1 for p in unserved if p in origins)
        boarded_not_served = len(unserved) - unserved_not_boarded
        total_actions = unserved_not_boarded * 2 + boarded_not_served * 1

        return movement_steps + total_actions
