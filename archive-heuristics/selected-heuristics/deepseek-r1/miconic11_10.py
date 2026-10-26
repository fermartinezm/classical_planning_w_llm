from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


class Miconic11Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering:
    - The current lift position.
    - The state of each passenger (served, boarded, or waiting).
    - The shortest path between floors based on the 'above' relationships.

    # Assumptions
    - The 'above' relationships form a connected graph allowing movement between any two floors.
    - Each passenger's destination is static and defined in the 'destin' predicate.
    - The elevator can only move one floor per action (up or down) based on the 'above' hierarchy.

    # Heuristic Initialization
    - Extract 'destin' predicates to map each passenger to their destination floor.
    - Build an undirected graph from 'above' static facts to compute shortest paths between floors.
    - Precompute all-pairs shortest paths using BFS for efficient distance lookup.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the current lift position from the state.
    2. For each passenger:
        a. If served: no actions needed.
        b. If boarded: calculate distance from current lift position to destination, add 1 for depart.
        c. If waiting: find origin floor, calculate distance from lift to origin, then to destination, add 2 for board and depart.
    3. Sum all individual passenger costs for the total heuristic value.
    """

    def __init__(self, task):
        """Initialize heuristic with static information from the task."""
        self.destin = {}  # Maps each passenger to their destination floor
        self.graph = defaultdict(list)  # Undirected graph for floor movements
        self.distances = {}  # Shortest path distances between floors

        # Extract 'destin' predicates from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor

        # Build undirected graph from 'above' facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                self.graph[f1].append(f2)
                self.graph[f2].append(f1)  # Allow bidirectional movement

        # Precompute all-pairs shortest paths using BFS
        all_floors = set(self.graph.keys())
        for start in all_floors:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.graph[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for end, dist in visited.items():
                self.distances[(start, end)] = dist

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        current_lift = None
        current_origins = {}
        boarded = set()
        served = set()

        # Extract current lift position, origins, boarded, and served passengers
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift = parts[1]
            elif parts[0] == 'origin':
                passenger, floor = parts[1], parts[2]
                current_origins[passenger] = floor
            elif parts[0] == 'boarded':
                boarded.add(parts[1])
            elif parts[0] == 'served':
                served.add(parts[1])

        if not current_lift:
            return 0  # Invalid state, assume goal reached

        total = 0
        for passenger in self.destin:
            if passenger in served:
                continue
            dest_floor = self.destin[passenger]
            if passenger in boarded:
                # Passenger is boarded: need to move to destination and depart
                dist = self.distances.get((current_lift, dest_floor), 0)
                total += dist + 1  # depart action
            else:
                # Passenger is waiting: need to board and then depart
                origin_floor = current_origins.get(passenger)
                if not origin_floor:
                    continue  # Should not happen in valid states
                dist_to_origin = self.distances.get((current_lift, origin_floor), 0)
                dist_to_dest = self.distances.get((origin_floor, dest_floor), 0)
                total += dist_to_origin + 1 + dist_to_dest + 1  # board and depart actions

        return total
