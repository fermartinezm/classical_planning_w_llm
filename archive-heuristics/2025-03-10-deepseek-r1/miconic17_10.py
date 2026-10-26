from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconic17Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by summing the steps needed for each passenger to be boarded (if not already) and then depart at their destination. Movement steps between floors are calculated based on the 'above' hierarchy.

    # Assumptions
    - The 'above' predicates form a linear hierarchy of floors.
    - Each passenger's origin and destination are static (from initial state and static facts).
    - The elevator must visit each passenger's origin (if not boarded) and destination, requiring movement steps and actions.

    # Heuristic Initialization
    - Extract passenger origins from the initial state and destinations from static facts.
    - Build a floor hierarchy from 'above' static facts to compute distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each passenger not served:
        a. If not boarded, calculate steps to move from current lift position to their origin, then to their destination, plus board and depart actions.
        b. If boarded, calculate steps to move from current lift position to their destination plus depart action.
    2. Sum all steps and actions for all unserved passengers.
    """

    def __init__(self, task):
        self.destin = {}
        self.origin = {}
        # Extract destin from static facts
        for fact in task.static:
            if match(fact, 'destin', '*', '*'):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.destin[passenger] = floor
        # Extract origin from initial state
        for fact in task.initial_state:
            if match(fact, 'origin', '*', '*'):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.origin[passenger] = floor
        # Build floor hierarchy
        below_map = {}
        for fact in task.static:
            if match(fact, 'above', '*', '*'):
                parts = get_parts(fact)
                f1, f2 = parts[1], parts[2]
                below_map[f1] = f2
        # Find top floor
        top_floors = [f for f in below_map if f not in below_map.values()]
        top = top_floors[0] if top_floors else None
        ordered_floors = []
        current = top
        while current in below_map:
            ordered_floors.append(current)
            current = below_map[current]
        if current:
            ordered_floors.append(current)
        self.floor_indices = {f: idx for idx, f in enumerate(ordered_floors)} if ordered_floors else {}

    def __call__(self, node):
        state = node.state
        # Current lift floor
        current_lift_floor = next((get_parts(fact)[1] for fact in state if match(fact, 'lift-at', '*')), None)
        if not current_lift_floor or not self.floor_indices:
            return 0
        total = 0
        for passenger in self.destin:
            if f'(served {passenger})' in state:
                continue
            boarded = f'(boarded {passenger})' in state
            origin_floor = self.origin.get(passenger)
            dest_floor = self.destin[passenger]
            if boarded:
                current_idx = self.floor_indices[current_lift_floor]
                dest_idx = self.floor_indices[dest_floor]
                distance = abs(current_idx - dest_idx)
                total += distance + 1
            else:
                if not origin_floor:
                    continue
                current_idx = self.floor_indices[current_lift_floor]
                origin_idx = self.floor_indices[origin_floor]
                dest_idx = self.floor_indices[dest_floor]
                distance_origin = abs(current_idx - origin_idx)
                distance_dest = abs(origin_idx - dest_idx)
                total += distance_origin + distance_dest + 2
        return total
