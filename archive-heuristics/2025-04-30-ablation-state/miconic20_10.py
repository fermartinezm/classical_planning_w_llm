import re
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the costs for each unserved passenger independently (board/depart actions)
    plus an estimated lift movement cost to reach the necessary floors.

    # Assumptions
    - Floors are linearly ordered and can be sorted numerically based on their names (e.g., f1, f2, ...).
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic sums individual passenger action costs and a global movement cost estimate.

    # Heuristic Initialization
    - Extracts all floor names from the task definition (initial state and static facts).
    - Sorts the floor names numerically to create a mapping from floor name to its level (integer).
    - Extracts the destination floor for each passenger from static facts.
    - Extracts all passenger names from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, return 0.
    2. Identify the current floor of the lift from the state.
    3. Identify all passengers who have not yet been served (goal condition).
    4. For each unserved passenger:
       - Determine if they are currently boarded in the lift or waiting at their origin floor.
       - Store their origin floor (if unboarded). Destination floor is known from initialization.
    5. Calculate the "action cost" (board/depart actions):
       - Add 1 for each unserved passenger who is currently waiting at their origin (for the `board` action).
       - Add 1 for each unserved passenger (for the `depart` action).
    6. Calculate the "movement cost" (up/down actions):
       - Identify the set of floors the lift *must* visit to serve the remaining passengers. This set includes the origin floors of all unboarded passengers and the destination floors of all boarded passengers.
       - If this set of required floors is empty, the movement cost is 0.
       - Otherwise, find the minimum and maximum floor levels among these required floors.
       - Estimate the minimum number of move actions required to visit all these floors, starting from the current lift floor. A lower bound is the distance from the current floor to the furthest required floor, plus the total vertical range spanned by the required floors. This is calculated as `max(abs(current_level - min_level_needed), abs(current_level - max_level_needed)) + (max_level_needed - min_level_needed)`.
    7. The total heuristic value for the state is the sum of the calculated action cost and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        self.goals = task.goals # Goal conditions are needed for the goal check
        self.static_facts = task.static

        # 1. Extract floor names and create floor_to_level map
        floor_names = set()
        # Look for floors in initial state and static facts
        for fact in task.initial_state | self.static_facts:
            parts = get_parts(fact)
            # Floors appear in (lift-at f), (origin p f), (destin p f), (above f1 f2)
            if parts[0] == 'lift-at' and len(parts) == 2:
                floor_names.add(parts[1])
            elif parts[0] in ['origin', 'destin'] and len(parts) == 3:
                 floor_names.add(parts[2])
            elif parts[0] == 'above' and len(parts) == 3:
                 floor_names.add(parts[1])
                 floor_names.add(parts[2])

        # Sort floors numerically based on the number part (e.g., f1, f10, f2)
        # This assumes floor names follow the pattern 'f' followed by digits.
        def floor_sort_key(floor_name):
            match = re.match(r'f(\d+)', floor_name)
            if match:
                return int(match.group(1))
            # Handle unexpected floor names gracefully, though problem assumes fN format
            return float('inf') # Put non-standard names at the end

        sorted_floors = sorted(list(floor_names), key=floor_sort_key)
        self.floor_to_level = {floor: i + 1 for i, floor in enumerate(sorted_floors)}
        # self.level_to_floor = {i + 1: floor for i, floor in enumerate(sorted_floors)} # Not used in this heuristic

        # 2. Extract passenger destinations and all passenger names from static facts
        self.passenger_destin = {}
        self.all_passengers = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.passenger_destin[passenger] = floor
                self.all_passengers.add(passenger)
            # Also check origin facts in static to ensure we get all passenger names,
            # even if a passenger's destin fact wasn't listed first.
            elif parts[0] == 'origin' and len(parts) == 3:
                 passenger = parts[1]
                 self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state as a frozenset of strings

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify current lift floor
        current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at' and len(parts) == 2:
                current_floor = parts[1]
                break
        # If lift location is not found, something is wrong with the state representation.
        # This should not happen in valid reachable states from a valid initial state.
        # Returning a high value indicates an invalid state for the heuristic.
        if current_floor is None:
             # print("Error: Lift location not found in state.") # For debugging
             return float('inf') # Indicate invalid state

        current_level = self.floor_to_level[current_floor]

        # 3. Identify unserved passengers and their state (unboarded/boarded)
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served' and len(get_parts(fact)) == 2}
        unserved_passengers = self.all_passengers - served_passengers_in_state

        unboarded_passengers_at_origin = set()
        boarded_passengers = set()
        passenger_origin = {} # Store origin floor for unboarded passengers

        for passenger in unserved_passengers:
            if f'(boarded {passenger})' in state:
                boarded_passengers.add(passenger)
            else:
                # If not served and not boarded, they must be at their origin.
                # Find their origin floor in the current state.
                found_origin = False
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == 'origin' and len(parts) == 3 and parts[1] == passenger:
                        origin_floor = parts[2]
                        passenger_origin[passenger] = origin_floor
                        unboarded_passengers_at_origin.add(passenger)
                        found_origin = True
                        break
                # If found_origin is False here, the state is inconsistent with PDDL effects.
                # This heuristic assumes valid states reachable from the initial state.
                # In a real planner, this might indicate a problem or require more robust state handling.
                # For this problem, we assume valid states.

        # 5. Calculate action cost (board + depart)
        # Each unboarded passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action.
        action_cost = len(unboarded_passengers_at_origin) + len(unserved_passengers)

        # 6. Calculate movement cost
        floors_needed_for_pickup = {passenger_origin[p] for p in unboarded_passengers_at_origin}
        floors_needed_for_dropoff = {self.passenger_destin[p] for p in boarded_passengers}
        floors_to_visit = floors_needed_for_pickup | floors_needed_for_dropoff

        movement_cost = 0
        if floors_to_visit:
            levels_to_visit = [self.floor_to_level[f] for f in floors_to_visit]
            min_level_needed = min(levels_to_visit)
            max_level_needed = max(levels_to_visit)

            # Estimate movement as distance to furthest needed floor + range size
            dist_to_min = abs(current_level - min_level_needed)
            dist_to_max = abs(current_level - max_level_needed)
            range_size = max_level_needed - min_level_needed

            movement_cost = max(dist_to_min, dist_to_max) + range_size

        # 7. Total heuristic
        total_heuristic = action_cost + movement_cost

        # Ensure heuristic is non-negative (should be, based on calculation)
        return max(0, total_heuristic)
