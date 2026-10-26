from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

class Miconic22Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the minimal movement required for each passenger based on their current state (boarded or not) and the elevator's current position. Distances between floors are precomputed using BFS on the 'above' relations.

    # Assumptions:
    - The 'above' relations form a directed graph allowing up and down movements.
    - Each passenger's destination is static and known from the problem's static facts.
    - The elevator can only move between floors connected via 'above' relations.

    # Heuristic Initialization
    - Precompute minimal distances between all floor pairs using BFS based on 'above' relations.
    - Extract each passenger's destination from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Precompute Distances**: Use BFS to calculate the shortest path between all pairs of floors based on 'above' relations.
    2. **Current State Analysis**: Determine the elevator's current position, boarded passengers, served passengers, and origins of unboarded passengers.
    3. **Calculate Steps per Passenger**:
       - **Boarded Passengers**: Distance from current elevator position to destination + 1 depart action.
       - **Unboarded Passengers**: Sum of distances from elevator to origin, origin to destination, + 2 actions (board and depart).
    4. **Sum Steps**: Total heuristic value is the sum of steps for all unserved passengers.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract 'destin' predicates from static facts
        self.destin = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor

        # Build 'above' graph and precompute distances
        self.above = defaultdict(set)
        self.floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'above' and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                self.above[f1].add(f2)
                self.floors.update({f1, f2})

        # Build adjacency list for BFS (directed graph for up/down)
        self.graph = defaultdict(set)
        for f1 in self.above:
            for f2 in self.above[f1]:
                self.graph[f1].add(f2)  # Up action from f1 to f2
                self.graph[f2].add(f1)  # Down action from f2 to f1

        # Precompute minimal distances between all floor pairs
        self.distances = defaultdict(dict)
        for start in self.floors:
            queue = deque([(start, 0)])
            visited = {start: 0}
            while queue:
                current, dist = queue.popleft()
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited or dist + 1 < visited.get(neighbor, float('inf')):
                        visited[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            for floor in self.floors:
                self.distances[start][floor] = visited.get(floor, float('inf'))

    def __call__(self, node):
        state = node.state

        # Find current elevator position
        current_lift_pos = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at' and len(parts) == 2:
                current_lift_pos = parts[1]
                break
        if not current_lift_pos:
            return float('inf')

        # Determine served and boarded passengers
        served = set()
        boarded = set()
        origins = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served' and len(parts) == 2:
                served.add(parts[1])
            elif parts[0] == 'boarded' and len(parts) == 2:
                boarded.add(parts[1])
            elif parts[0] == 'origin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                origins[passenger] = floor

        total = 0
        for passenger in self.destin:
            if passenger in served:
                continue
            if passenger in boarded:
                # Passenger is boarded, need to depart
                dest = self.destin[passenger]
                dist = self.distances[current_lift_pos].get(dest, float('inf'))
                if dist == float('inf'):
                    return float('inf')
                total += dist + 1  # depart action
            else:
                # Passenger is not boarded
                if passenger not in origins:
                    return float('inf')
                origin = origins[passenger]
                dest = self.destin[passenger]
                dist1 = self.distances[current_lift_pos].get(origin, float('inf'))
                dist2 = self.distances[origin].get(dest, float('inf'))
                if dist1 == float('inf') or dist2 == float('inf'):
                    return float('inf')
                total += dist1 + dist2 + 2  # board and depart

        return total
