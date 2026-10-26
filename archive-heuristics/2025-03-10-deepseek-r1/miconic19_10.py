from collections import deque, defaultdict
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by calculating the minimal elevator movements needed to board and depart each passenger, considering the current state.

    # Assumptions
    - The elevator can move between any two floors directly if there's an 'above' relation in either direction.
    - Passengers are boarded and departed in an optimal order that minimizes backtracking.
    - Boarded passengers are dropped off before picking up new ones to minimize elevator movement.

    # Heuristic Initialization
    1. Extract 'destin' predicates to map each passenger to their destination floor.
    2. Build a graph of floor connections based on 'above' predicates, allowing both up and down movements.
    3. Precompute the minimal steps between all pairs of floors using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the current elevator floor from the state.
    2. Separate passengers into boarded and unboarded.
    3. Process boarded passengers first:
        a. For each boarded passenger, calculate the steps from the current elevator position to their destination.
        b. Add the steps and update the elevator's current position to the destination.
    4. Process unboarded passengers:
        a. For each unboarded passenger, calculate steps from current position to their origin, then to their destination.
        b. Add boarding and departing actions, updating the elevator's position after each step.
    5. Sum all steps to get the heuristic value.
    """

    def __init__(self, task):
        self.destin = {}
        self.graph = defaultdict(list)
        self.distances = {}

        # Extract 'destin' predicates from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor

        # Build graph from 'above' predicates
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                self.graph[f1].append(f2)
                self.graph[f2].append(f1)

        # Precompute minimal distances between all floors
        all_floors = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_floors.update(neighbors)
        all_floors = list(all_floors)
        
        for floor in all_floors:
            self.distances[floor] = {}
            queue = deque([(floor, 0)])
            visited = set()
            while queue:
                current, dist = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                self.distances[floor][current] = dist
                for neighbor in self.graph.get(current, []):
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state
        current_elevator_floor = next((get_parts(fact)[1] for fact in state if fact.startswith('(lift-at ')), None)
        if not current_elevator_floor:
            return 0

        total = 0
        current_floor = current_elevator_floor

        boarded = []
        unboarded = []

        for passenger in self.destin:
            if f'(served {passenger})' in state:
                continue
            if f'(boarded {passenger})' in state:
                boarded.append(passenger)
            else:
                unboarded.append(passenger)

        # Process boarded passengers first
        for p in boarded:
            destin = self.destin[p]
            dist = self.distances.get(current_floor, {}).get(destin, 0)
            total += dist + 1  # depart action
            current_floor = destin

        # Process unboarded passengers
        for p in unboarded:
            origin = next((get_parts(fact)[2] for fact in state if get_parts(fact)[:2] == ['origin', p]), None)
            if not origin:
                continue

            # Move to origin
            dist = self.distances.get(current_floor, {}).get(origin, 0)
            total += dist + 1  # board action
            current_floor = origin

            # Move to destination
            destin = self.destin[p]
            dist = self.distances.get(current_floor, {}).get(destin, 0)
            total += dist + 1  # depart action
            current_floor = destin

        return total
