from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconic25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    Estimates the number of actions needed to serve all passengers by summing the steps required for each passenger based on their current state (boarded or not) and the lift's position. The heuristic precomputes shortest paths between all floors using the 'above' relations for efficient lookup.

    # Assumptions
    - Each passenger's origin is known from the initial state, and their destination is static.
    - The 'above' relations form a bidirectional graph allowing movement between floors via up/down actions.
    - Passengers cannot be boarded or departed multiple times.

    # Heuristic Initialization
    1. Extract passenger origins from the initial state.
    2. Extract destinations and 'above' relations from static facts.
    3. Build a graph of floors and precompute shortest paths between all pairs using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the lift's current position from the state.
    2. For each passenger:
        a. Skip if already served.
        b. If boarded, add steps from current lift position to destination plus depart action.
        c. If not boarded, add steps to origin, board, steps to destination, and depart.
    3. Sum all steps to get the heuristic value.
    """

    def __init__(self, task):
        self.origin = {}
        self.destin = {}
        self.above_graph = defaultdict(list)
        self.floors = set()

        # Extract passenger origins from initial state
        for fact in task.initial_state:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.origin[passenger] = floor
                self.floors.add(floor)

        # Extract destinations and 'above' relations from static facts
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor
                self.floors.add(floor)
            elif parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                self.above_graph[f1].append(f2)
                self.above_graph[f2].append(f1)  # Bidirectional for BFS
                self.floors.update([f1, f2])

        # Precompute shortest paths between all floors
        self.distances = {}
        floors = list(self.floors)
        for start in floors:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.above_graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for floor in floors:
                self.distances[(start, floor)] = visited.get(floor, float('inf'))

    def __call__(self, node):
        state = node.state
        current_lift = next((fact[1:-1].split()[1] for fact in state if fact.startswith('(lift-at')), None)

        if not current_lift:
            return float('inf')

        total = 0
        for passenger in self.origin:
            if f'(served {passenger})' in state:
                continue

            boarded = f'(boarded {passenger})' in state
            dest = self.destin.get(passenger)
            origin = self.origin.get(passenger)

            if not dest or not origin:
                continue

            if boarded:
                dist = self.distances.get((current_lift, dest), float('inf'))
                total += dist + 1  # depart
            else:
                dist_to_origin = self.distances.get((current_lift, origin), float('inf'))
                dist_to_dest = self.distances.get((origin, dest), float('inf'))
                total += dist_to_origin + 1 + dist_to_dest + 1  # board and depart

        return total
