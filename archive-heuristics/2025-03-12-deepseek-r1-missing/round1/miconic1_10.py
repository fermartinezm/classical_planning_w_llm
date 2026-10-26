from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

class MiconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by calculating the minimal steps needed to board each unserved passenger and transport them to their destination, considering the current elevator position and floor hierarchy.

    # Assumptions
    - The elevator can move between floors connected via 'above' predicates.
    - Passengers need to be boarded from their origin and departed at their destination.
    - The floor hierarchy is derived from 'above' predicates using topological sorting.

    # Heuristic Initialization
    - Extract passenger destinations from static 'destin' facts.
    - Build a floor hierarchy using topological sort on 'above' predicates to determine distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger:
        a. If served, skip.
        b. If boarded, calculate distance from current elevator position to destination.
        c. If not boarded, calculate distance from current elevator position to origin, then to destination.
    2. Sum all required actions (movement, board, depart) for unserved passengers.
    """

    def __init__(self, task):
        self.destin = {}
        above_relations = set()
        floors = set()
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor
            elif parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                above_relations.add((f1, f2))
                floors.update({f1, f2})
        # Build adjacency and in-degree for topological sort
        adjacency = defaultdict(list)
        in_degree = defaultdict(int)
        for f1, f2 in above_relations:
            adjacency[f1].append(f2)
            in_degree[f2] += 1
        for f in floors:
            if f not in in_degree:
                in_degree[f] = 0
        # Kahn's algorithm for topological sort
        queue = deque([f for f in floors if in_degree[f] == 0])
        self.ordered_floors = []
        while queue:
            current = queue.popleft()
            self.ordered_floors.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        self.floor_index = {f: idx for idx, f in enumerate(self.ordered_floors)}
        self.passengers = list(self.destin.keys())

    def __call__(self, node):
        state = node.state
        current_lift = next((fact[1:-1].split()[1] for fact in state if fact.startswith('(lift-at ')), None)
        if not current_lift:
            return 0

        def distance(f1, f2):
            return abs(self.floor_index.get(f1, 0) - self.floor_index.get(f2, 0))

        total = 0
        for passenger in self.passengers:
            if f'(served {passenger})' in state:
                continue
            destin = self.destin[passenger]
            if f'(boarded {passenger})' in state:
                total += distance(current_lift, destin) + 1  # depart
            else:
                origin = next((fact[1:-1].split()[2] for fact in state if fnmatch(fact, f'(origin {passenger} *)')), None)
                if not origin:
                    continue  # invalid state, passenger not boarded and origin missing
                dist_to_origin = distance(current_lift, origin)
                dist_to_destin = distance(origin, destin)
                total += dist_to_origin + 1 + dist_to_destin + 1  # board + depart
        return total
