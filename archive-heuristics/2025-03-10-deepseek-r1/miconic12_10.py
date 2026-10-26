from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by splitting into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """Check if a fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconic12Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    Estimates the number of actions needed to serve all passengers by considering:
    - Current elevator position
    - Distance between floors (precomputed using BFS on 'above' relations)
    - Boarding and departing actions for each passenger

    # Assumptions
    - 'above' relations form a connected graph allowing pathfinding between any two floors relevant to passengers.
    - Each passenger's origin is known from the initial state, and destination is static.
    - Elevator can move between any two connected floors, with each move as one action.

    # Heuristic Initialization
    - Extracts passenger origins from the initial state and destinations from static facts.
    - Builds a floor movement graph from 'above' facts.
    - Precomputes minimal distances between all pairs of floors using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine current elevator position from the state.
    2. Identify served and boarded passengers.
    3. For each unserved passenger:
       a. If boarded: add distance from current floor to destination plus depart action.
       b. If not boarded: add distance to origin, board action, distance to destination, and depart action.
    4. Sum all costs for a total heuristic estimate.
    """

    def __init__(self, task):
        # Extract passenger origins from initial state
        self.passenger_origins = {}
        for fact in task.initial_state:
            if match(fact, 'origin', '*', '*'):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_origins[passenger] = floor

        # Extract passenger destinations from static facts
        self.passenger_destins = {}
        for fact in task.static:
            if match(fact, 'destin', '*', '*'):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destins[passenger] = floor

        # Build floor graph from 'above' facts (bidirectional for up/down)
        self.graph = {}
        floors = set()
        for fact in task.static:
            if match(fact, 'above', '*', '*'):
                parts = get_parts(fact)
                f1, f2 = parts[1], parts[2]
                if f1 not in self.graph:
                    self.graph[f1] = []
                self.graph[f1].append(f2)
                if f2 not in self.graph:
                    self.graph[f2] = []
                self.graph[f2].append(f1)  # Down move is reverse of 'above'
                floors.update([f1, f2])

        # Precompute minimal distances between all pairs of floors using BFS
        self.distances = {}
        for start in floors:
            self.distances[start] = {}
            queue = deque([(start, 0)])
            visited = set()
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                self.distances[start][current] = dist
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state
        current_elevator_floor = None
        for fact in state:
            if match(fact, 'lift-at', '*'):
                current_elevator_floor = get_parts(fact)[1]
                break
        if not current_elevator_floor:
            return float('inf')

        served = set()
        boarded = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'served':
                served.add(parts[1])
            elif parts[0] == 'boarded':
                boarded.add(parts[1])

        total = 0
        for passenger in self.passenger_destins:
            if passenger in served:
                continue
            destin = self.passenger_destins[passenger]
            if passenger in boarded:
                # Need to move to destination and depart
                distance = self.distances[current_elevator_floor].get(destin, float('inf'))
                if distance == float('inf'):
                    return float('inf')
                total += distance + 1  # depart action
            else:
                # Need to board and then depart
                origin = self.passenger_origins.get(passenger)
                if not origin:
                    return float('inf')
                # Distance to origin
                dist_to_origin = self.distances[current_elevator_floor].get(origin, float('inf'))
                # Distance from origin to destination
                dist_origin_destin = self.distances[origin].get(destin, float('inf'))
                if dist_to_origin == float('inf') or dist_origin_destin == float('inf'):
                    return float('inf')
                total += dist_to_origin + dist_origin_destin + 2  # board and depart

        return total
