from collections import defaultdict
from heuristics.heuristic_base import Heuristic

class miconic6Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by considering the necessary movements and actions for each unserved passenger. It groups passengers by their origin and destination floors to minimize redundant lift movements.

    # Assumptions
    - Each passenger must be boarded at their origin and departed at their destination.
    - The lift can move directly between any two floors with an 'above' relation in one action.
    - Boarding and departing each passenger requires a separate action.
    - Passengers can be grouped by origin and destination to optimize lift movements.

    # Heuristic Initialization
    - Extracts static information about each passenger's origin and destination floors from the problem's static facts.
    - Preprocesses the 'origin' and 'destin' predicates to map each passenger to their respective floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State Information**:
       - Determine the current lift position.
       - Identify served, boarded, and unserved passengers.

    2. **Process Unboarded Passengers**:
       - Group unboarded passengers by their origin floor.
       - For each origin group:
         - Add a movement action if the lift is not already at the origin.
         - Add a boarding action for each passenger in the group.
         - Group passengers by destination and add movement and depart actions for each destination.

    3. **Process Boarded Passengers**:
       - Group boarded passengers by their destination floor.
       - For each destination group:
         - Add a movement action if the lift is not already at the destination.
         - Add a depart action for each passenger in the group.

    4. **Sum All Actions**:
       - Total heuristic value is the sum of all movement, board, and depart actions required.
    """

    def __init__(self, task):
        self.origins = {}
        self.destins = {}
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'origin':
                passenger = parts[1]
                floor = parts[2]
                self.origins[passenger] = floor
            elif parts[0] == 'destin':
                passenger = parts[1]
                floor = parts[2]
                self.destins[passenger] = floor

    def __call__(self, node):
        state = node.state
        current_lift_pos = None
        served = set()
        boarded = set()

        for fact in state:
            if fact.startswith('(lift-at '):
                current_lift_pos = fact[1:-1].split()[1]
            elif fact.startswith('(served '):
                served.add(fact[1:-1].split()[1])
            elif fact.startswith('(boarded '):
                boarded.add(fact[1:-1].split()[1])

        all_passengers = set(self.origins.keys())
        unserved = all_passengers - served
        unboarded = [p for p in unserved if p not in boarded]
        boarded_passengers = [p for p in unserved if p in boarded]

        total_cost = 0

        origin_groups = defaultdict(list)
        for p in unboarded:
            origin_groups[self.origins[p]].append(p)

        for origin, passengers in origin_groups.items():
            if current_lift_pos != origin:
                total_cost += 1
            total_cost += len(passengers)
            dest_groups = defaultdict(list)
            for p in passengers:
                dest_groups[self.destins[p]].append(p)
            for dest, dest_passengers in dest_groups.items():
                if dest != origin:
                    total_cost += 1
                total_cost += len(dest_passengers)

        dest_groups = defaultdict(list)
        for p in boarded_passengers:
            dest_groups[self.destins[p]].append(p)

        for dest, passengers in dest_groups.items():
            if current_lift_pos != dest:
                total_cost += 1
            total_cost += len(passengers)

        return total_cost
