from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers by calculating the required movements of the lift and the boarding/departing actions.

    # Assumptions:
    - Each passenger must be boarded and then served at their destination floor.
    - The lift can move between floors, and the cost is the number of moves required.
    - The heuristic assumes the lift can move optimally to minimize the total number of actions.

    # Heuristic Initialization
    - Extract the origin and destination floors for each passenger from the initial state.
    - Build a floor hierarchy from the static 'above' facts to determine the level of each floor.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not yet served:
       a. If not boarded, calculate the distance from the current lift floor to their origin.
       b. Calculate the distance from their origin to their destination.
       c. Add these distances and account for the boarding and serving actions (2 actions).
    2. Sum the total actions for all passengers to get the heuristic value.
    """

    def __init__(self, task):
        # Extract origin and destination for each passenger from the initial state
        self.passenger_origin = {}
        self.passenger_destin = {}
        for fact in task.initial_state:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                p, f = parts[1], parts[2]
                self.passenger_origin[p] = f
            elif parts[0] == 'destin':
                p, f = parts[1], parts[2]
                self.passenger_destin[p] = f

        # Build floor hierarchy and assign levels based on 'above' static facts
        self.floor_level = {}
        above_facts = [fact for fact in task.static if fact.startswith('(above')]
        floors = set()
        for fact in above_facts:
            f1, f2 = fact[1:-1].split()[1], fact[1:-1].split()[2]
            floors.add(f1)
            floors.add(f2)
        
        # Find the bottom floor (not the second element in any above fact)
        bottom_floor = None
        for f in floors:
            is_above = any(f == parts[2] for fact in above_facts for parts in [fact[1:-1].split()])
            if not is_above:
                bottom_floor = f
                break
        
        # Assign levels using BFS starting from the bottom floor
        if bottom_floor is not None:
            queue = deque()
            queue.append(bottom_floor)
            self.floor_level[bottom_floor] = 0
            while queue:
                current = queue.popleft()
                for fact in above_facts:
                    parts = fact[1:-1].split()
                    if parts[0] == current:
                        next_floor = parts[1]
                        if next_floor not in self.floor_level:
                            self.floor_level[next_floor] = self.floor_level[current] + 1
                            queue.append(next_floor)

    def __call__(self, node):
        state = node.state
        # Extract current lift floor
        current_lift = None
        for fact in state:
            if fact.startswith('(lift-at'):
                current_lift = fact[1:-1].split()[1]
                break
        
        total_cost = 0

        # Process each passenger
        for p in self.passenger_origin:
            served_fact = f'(served {p})'
            boarded_fact = f'(boarded {p})'
            if served_fact in state:
                continue  # Skip if already served
            if boarded_fact in state:
                # Already boarded, need to move to destination and serve
                origin = self.passenger_origin[p]
                destin = self.passenger_destin[p]
                level_origin = self.floor_level[origin]
                level_destin = self.floor_level[destin]
                distance = abs(level_destin - level_origin)
                total_cost += distance + 1  # 1 action for depart
            else:
                # Need to move to origin, board, then to destination, and serve
                origin = self.passenger_origin[p]
                destin = self.passenger_destin[p]
                if current_lift is None:
                    continue  # Lift not present, should not happen in valid state
                level_current = self.floor_level[current_lift]
                level_origin = self.floor_level[origin]
                level_destin = self.floor_level[destin]
                distance_to_origin = abs(level_origin - level_current)
                distance_to_destin = abs(level_destin - level_origin)
                # Actions: move to origin, board, move to destin, depart
                total_cost += distance_to_origin + 1 + distance_to_destin + 1

        return total_cost
