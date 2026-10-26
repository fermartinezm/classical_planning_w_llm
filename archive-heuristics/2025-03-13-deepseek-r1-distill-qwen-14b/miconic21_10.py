from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the minimal steps required for the lift to reach each passenger's origin and destination floors.

    # Assumptions:
    - The lift can move up or down between floors.
    - Each passenger must be boarded and then departed after reaching their destination.
    - The minimal distance between floors is based on their levels in the hierarchy defined by the 'above' predicates.

    # Heuristic Initialization
    - Extract the hierarchy of floors from the static 'above' facts.
    - Compute the minimal distance between every pair of floors using BFS starting from the top floor (f1).
    - Extract the origin and destination floors for each passenger from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger, check if they are already served. If not, proceed.
    2. For each unserved passenger:
       a. If the passenger is already boarded, calculate the distance from their origin to destination and add 1 action for departing.
       b. If the passenger is not boarded, calculate the distance from the current lift floor to their origin, add 1 action for boarding, then calculate the distance from origin to destination, and add 1 action for departing.
    3. Sum all the calculated actions for an estimate of the total required actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor hierarchy and passenger information."""
        # Extract static facts to build floor hierarchy
        self.static_facts = task.static
        self.floors = set()
        self.above_graph = {}
        self.passenger_origin = {}
        self.passenger_destin = {}

        # Build the graph of floors based on 'above' facts
        for fact in self.static_facts:
            if fact.startswith('(above '):
                f1, f2 = self._parse_fact(fact, 2)
                if f1 not in self.above_graph:
                    self.above_graph[f1] = []
                self.above_graph[f1].append(f2)
                if f2 not in self.above_graph:
                    self.above_graph[f2] = []
                self.above_graph[f2].append(f1)
                self.floors.add(f1)
                self.floors.add(f2)

        # Find the top floor (f1) which is the root of the hierarchy
        self.top_floor = None
        for fact in self.static_facts:
            if fact.startswith('(above f1 '):
                self.top_floor = 'f1'
                break

        # If top_floor is not found, assume it's the one that is not a destination in any 'above' fact
        if self.top_floor is None:
            for floor in self.floors:
                is_above = any(fact.startswith(f'above {floor} ') for fact in self.static_facts)
                if not is_above:
                    self.top_floor = floor
                    break

        # Compute levels using BFS from top_floor
        self.level = {self.top_floor: 0}
        queue = [self.top_floor]
        while queue:
            current = queue.pop(0)
            if current in self.above_graph:
                for neighbor in self.above_graph[current]:
                    if neighbor not in self.level:
                        self.level[neighbor] = self.level[current] + 1
                        queue.append(neighbor)

        # Extract passenger origin and destination from initial state
        for fact in task.initial_state:
            if fact.startswith('(origin '):
                p, f = self._parse_fact(fact, 2)
                self.passenger_origin[p] = f
            elif fact.startswith('(destin '):
                p, f = self._parse_fact(fact, 2)
                self.passenger_destin[p] = f

    def _parse_fact(self, fact, num_parts):
        """Extract components from a PDDL fact."""
        parts = fact[1:-1].split()
        return parts[1:num_parts+1]

    def __call__(self, node):
        """Compute the heuristic value for the given node."""
        state = node.state
        lift_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                lift_floor = fact[8:-1]  # Extract the floor name.

        if lift_floor is None:
            # If lift is not at any floor, which shouldn't happen in miconic
            return 0

        total_cost = 0

        # For each passenger, check if served
        for p in self.passenger_origin:
            served_fact = f'(served {p})'
            if served_fact in state:
                continue  # Already served

            origin = self.passenger_origin[p]
            dest = self.passenger_destin[p]

            boarded_fact = f'(boarded {p})'
            if boarded_fact in state:
                # Already boarded, calculate distance from origin to dest
                distance = abs(self.level[origin] - self.level[dest])
                total_cost += distance + 1  # depart action
            else:
                # Need to move to origin, board, then to dest, then depart
                distance_lift_to_origin = abs(self.level[lift_floor] - self.level[origin])
                distance_origin_to_dest = abs(self.level[origin] - self.level[dest])
                total_cost += distance_lift_to_origin + 1 + distance_origin_to_dest + 1

        return total_cost
