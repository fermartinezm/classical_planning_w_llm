from heuristics.heuristic_base import Heuristic
from collections import deque

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the minimal movements required by the lift and the necessary boarding and departing actions.

    # Assumptions:
    - The lift can move between floors as defined by the 'above' relationships.
    - Each passenger must board and depart, requiring two actions per passenger.
    - The heuristic assumes that the lift can serve passengers in any order, and the minimal path is the sum of individual distances.

    # Heuristic Initialization
    - Extracts static facts to build a graph of floor connections.
    - Precomputes the shortest path distances between all pairs of floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current position of the lift.
    2. Identify all passengers who are not yet served.
    3. For each passenger, determine their origin and destination floors.
    4. Calculate the distance from the lift's current position to the passenger's origin.
    5. Calculate the distance from the origin to the destination.
    6. Sum these distances for all passengers.
    7. Add two actions for each passenger (boarding and departing).
    8. The total is the estimated number of actions needed.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and precomputing floor distances."""
        self.static_facts = task.static
        self.above_graph = self.build_above_graph()
        self.floor_distances = self.precompute_distances()

    def build_above_graph(self):
        """Construct a graph representing the 'above' relationships between floors."""
        graph = {}
        for fact in self.static_facts:
            if fact.startswith('(above '):
                parts = fact[6:-1].split()
                f1, f2 = parts[0], parts[1]
                if f1 not in graph:
                    graph[f1] = {}
                graph[f1][f2] = 1
                if f2 not in graph:
                    graph[f2] = {}
                graph[f2][f1] = 1
        return graph

    def precompute_distances(self):
        """Precompute the shortest path distances between all pairs of floors using BFS."""
        distances = {}
        for floor in self.above_graph:
            distances[floor] = {}
            queue = deque()
            queue.append((floor, 0))
            visited = {floor}
            while queue:
                current, dist = queue.popleft()
                for neighbor, _ in self.above_graph[current].items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[floor][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        return distances

    def __call__(self, node):
        """Compute the estimated number of actions needed to serve all passengers."""
        state = node.state

        # Extract current lift position
        lift_pos = None
        for fact in state:
            if fact.startswith('(lift-at '):
                lift_pos = fact[8:-1]
                break
        if lift_pos is None:
            return 0  # No lift position, heuristic is 0 (though state is likely unsolvable)

        # Identify served passengers
        served_passengers = set()
        for fact in state:
            if fact.startswith('(served '):
                p = fact[7:-1]
                served_passengers.add(p)

        # Collect passengers who are not served
        passengers = {}
        for fact in state:
            if fact.startswith('(origin '):
                p, f = fact[7:-1].split()
                if p not in served_passengers:
                    passengers[p] = {'origin': f}
            if fact.startswith('(destin '):
                p, f = fact[7:-1].split()
                if p not in served_passengers:
                    passengers[p]['destin'] = f

        total_distance = 0
        num_passengers = len(passengers)
        for p, data in passengers.items():
            origin = data['origin']
            destin = data['destin']

            # Distance from lift to origin
            d1 = self.floor_distances[lift_pos].get(origin, 0)
            # Distance from origin to destination
            d2 = self.floor_distances[origin].get(destin, 0)
            total_distance += d1 + d2

        # Each passenger requires boarding and departing
        total_actions = total_distance + 2 * num_passengers
        return total_actions
