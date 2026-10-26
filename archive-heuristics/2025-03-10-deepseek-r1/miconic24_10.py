from fnmatch import fnmatch
from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *pattern):
    parts = get_parts(fact)
    return len(parts) == len(pattern) and all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

class Miconic24Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by calculating the minimal elevator movements needed to board and depart each passenger based on their current state (boarded or not) and the elevator's current position.

    # Assumptions
    - Each passenger must be boarded from their origin floor and departed at their destination floor.
    - The elevator can move between any two floors directly if there's an 'above' relation, requiring one action per move.
    - Boarding and departing each take one action each.
    - The heuristic does not account for possible optimizations from serving multiple passengers in a single trip.

    # Heuristic Initialization
    - Extract passenger origins and destinations from static facts.
    - Build a floor graph based on 'above' relations to precompute minimal distances between any two floors using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current Elevator Position**: Determine the current floor of the elevator from the state.
    2. **Passenger State Check**:
       - For each passenger, check if they are already served (no cost).
       - If boarded, calculate the distance from the elevator's current position to their destination and add one action for departing.
       - If not boarded, calculate the distance from the elevator's current position to their origin (boarding), then from origin to destination (moving), adding one action each for boarding and departing.
    3. **Sum Costs**: Sum the calculated costs for all passengers to get the total heuristic estimate.
    """

    def __init__(self, task):
        self.origin = {}
        self.destin = {}
        self.static_floors = set()
        self.edges = defaultdict(set)
        self.distances = {}

        # Extract origin and destination for each passenger from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if match(fact, 'origin', '*', '*'):
                p, f = parts[1], parts[2]
                self.origin[p] = f
            elif match(fact, 'destin', '*', '*'):
                p, f = parts[1], parts[2]
                self.destin[p] = f
            elif match(fact, 'above', '*', '*'):
                f1, f2 = parts[1], parts[2]
                self.static_floors.update({f1, f2})
                self.edges[f1].add(f2)
                self.edges[f2].add(f1)  # Bidirectional for BFS

        # Precompute minimal distances between all floors using BFS
        self.floors = list(self.static_floors)
        for floor in self.floors:
            self.distances[floor] = {}
            visited = {floor: 0}
            queue = deque([floor])
            while queue:
                current = queue.popleft()
                for neighbor in self.edges[current]:
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            self.distances[floor] = visited

    def __call__(self, node):
        state = node.state
        current_lift = None

        # Find current elevator position
        for fact in state:
            if match(fact, 'lift-at', '*'):
                current_lift = get_parts(fact)[1]
                break
        if not current_lift:
            return 0  # Should not happen

        total_cost = 0

        for p in self.origin:
            if f'(served {p})' in state:
                continue  # Already served

            if f'(boarded {p})' in state:
                # Passenger is boarded, need to depart
                dest = self.destin[p]
                distance = self.distances[current_lift].get(dest, 0)
                total_cost += distance + 1  # move + depart
            else:
                # Passenger needs to be boarded
                origin = self.origin[p]
                dest = self.destin[p]
                distance_to_origin = self.distances[current_lift].get(origin, 0)
                distance_to_dest = self.distances[origin].get(dest, 0)
                total_cost += distance_to_origin + 1  # move + board
                total_cost += distance_to_dest + 1  # move + depart

        return total_cost
