from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class Miconic20Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers by calculating the minimal elevator movement and required board/depart actions.

    # Assumptions:
    - The elevator can move optimally between required floors in a single sweep.
    - Each unboarded passenger requires boarding and departing (2 actions).
    - Each boarded passenger requires departing (1 action).
    - Elevator movement between two floors is the difference in their levels.

    # Heuristic Initialization
    - Extract 'above' static facts to determine the level of each floor.
    - Extract 'destin' and 'origin' static facts to know each passenger's destination and origin.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the current elevator position.
    2. For each unserved passenger:
       a. If not boarded, add origin and destination to required floors.
       b. If boarded, add destination to required floors.
    3. Compute minimal elevator movement steps:
       a. Find max and min levels among required floors.
       b. Calculate initial distance from current position to the closest (max or min).
       c. Total movement is initial distance + (max - min levels).
    4. Add 2 actions per unboarded passenger and 1 per boarded.
    """

    def __init__(self, task):
        self.static_above = {}  # Maps each floor to the set of floors above it
        self.destin = {}        # Maps each passenger to their destination floor
        self.origin = {}        # Maps each passenger to their origin floor

        # Process static facts to build floor levels and passenger info
        for fact in task.static:
            parts = fact[1:-1].split()
            if parts[0] == 'above':
                f1, f2 = parts[1], parts[2]
                self.static_above.setdefault(f2, set()).add(f1)
            elif parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
            elif parts[0] == 'origin':
                passenger, floor = parts[1], parts[2]
                self.origin[passenger] = floor

        # Compute floor levels
        self.floor_levels = {}
        all_floors = set(self.static_above.keys())
        for floors in self.static_above.values():
            all_floors.update(floors)
        for floor in all_floors:
            # Number of floors above this floor
            count = len(self.static_above.get(floor, set()))
            self.floor_levels[floor] = count + 1

    def __call__(self, node):
        state = node.state
        current_lift_floor = None
        required_floors = set()
        unboarded = 0
        boarded = 0

        # Extract current lift position
        for fact in state:
            if fact.startswith('(lift-at'):
                current_lift_floor = fact.split()[1][:-1]  # Extract floor from '(lift-at f2)'
                break

        if not current_lift_floor:
            return float('inf')

        # Collect required floors and count passengers
        passengers = {p for p in self.destin.keys()}  # All passengers with destin
        for passenger in passengers:
            served_fact = f'(served {passenger})'
            if served_fact in state:
                continue  # Passenger already served

            boarded_fact = f'(boarded {passenger})'
            if boarded_fact in state:
                # Passenger is boarded, add destination
                required_floors.add(self.destin[passenger])
                boarded += 1
            else:
                # Passenger is unboarded, add origin and destination
                origin = self.origin.get(passenger, None)
                if origin:
                    required_floors.add(origin)
                    required_floors.add(self.destin[passenger])
                    unboarded += 1

        # Calculate movement steps
        if not required_floors:
            return 0  # All passengers served

        required_levels = [self.floor_levels[f] for f in required_floors if f in self.floor_levels]
        if not required_levels:
            return 0

        max_level = max(required_levels)
        min_level = min(required_levels)
        current_level = self.floor_levels.get(current_lift_floor, 0)

        distance_to_max = abs(current_level - max_level)
        distance_to_min = abs(current_level - min_level)
        initial_distance = min(distance_to_max, distance_to_min)
        movement_steps = initial_distance + (max_level - min_level)

        total_actions = movement_steps + (unboarded * 2) + boarded

        return total_actions
