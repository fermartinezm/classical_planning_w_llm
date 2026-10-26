from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconic5Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the minimal elevator movements needed to board and drop off each passenger.

    # Assumptions:
    - The elevator can move between any two floors directly if there's an 'above' relation in either direction.
    - Each boarded passenger requires a depart action.
    - Each unboarded passenger requires a board action and a depart action, plus movements to their origin and destination.

    # Heuristic Initialization
    - Extract passenger origins and destinations from static facts.
    - Build a graph of floor movements based on 'above' predicates and precompute minimal distances between all floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
        a. If boarded, add distance from current elevator floor to their destination plus one depart action.
        b. If not boarded, add distance from current elevator floor to their origin, one board action, distance from origin to destination, and one depart action.
    2. Sum all these values for an estimated total number of actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information and precomputing floor distances."""
        self.static = task.static
        self.goals = task.goals

        # Extract passenger origins and destinations from static facts
        self.origins = {}
        self.destinations = {}
        for fact in self.static:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.origins[passenger] = floor
            elif parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destinations[passenger] = floor

        # Build floor graph from 'above' facts
        self.above_graph = defaultdict(list)
        for fact in self.static:
            if fact.startswith('(above'):
                parts = fact[1:-1].split()
                if parts[0] == 'above':
                    f1, f2 = parts[1], parts[2]
                    self.above_graph[f1].append(f2)
                    self.above_graph[f2].append(f1)  # Bidirectional for up/down movements

        # Precompute minimal distances between all pairs of floors using BFS
        self.distance = defaultdict(dict)
        all_floors = set(self.above_graph.keys())
        for floor in all_floors:
            visited = {floor: 0}
            queue = deque([floor])
            while queue:
                current = queue.popleft()
                for neighbor in self.above_graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            for f, dist in visited.items():
                self.distance[floor][f] = dist

    def __call__(self, node):
        state = node.state
        current_floor = None
        boarded_passengers = set()
        served_passengers = set()

        # Extract current elevator floor and passenger statuses
        for fact in state:
            parts = fact[1:-1].split()
            if parts[0] == 'lift-at':
                current_floor = parts[1]
            elif parts[0] == 'boarded':
                boarded_passengers.add(parts[1])
            elif parts[0] == 'served':
                served_passengers.add(parts[1])

        if current_floor is None:
            return float('inf')

        total_actions = 0

        # Check each passenger's status
        for passenger in self.origins:
            if passenger in served_passengers:
                continue

            if passenger in boarded_passengers:
                dest = self.destinations.get(passenger)
                if dest:
                    dist = self.distance[current_floor].get(dest, float('inf'))
                    total_actions += dist + 1  # depart action
            else:
                origin = self.origins.get(passenger)
                dest = self.destinations.get(passenger)
                if origin and dest:
                    dist1 = self.distance[current_floor].get(origin, float('inf'))
                    dist2 = self.distance[origin].get(dest, float('inf'))
                    total_actions += dist1 + 1 + dist2 + 1  # board and depart actions

        return total_actions if total_actions != 0 else 0
