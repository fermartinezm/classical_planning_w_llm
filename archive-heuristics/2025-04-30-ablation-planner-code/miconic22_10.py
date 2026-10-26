from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper to extract number from floor name like 'f10'
def get_floor_number(floor_name):
    """Extract the numerical part from a floor name like 'f10'."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    # Handle cases that don't match the expected pattern, though unlikely in miconic
    return float('inf')

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the minimum non-movement actions (board/depart) needed for each unserved
    passenger and adds an estimate of the minimum movement actions needed for the lift
    to visit all required floors (origins of unboarded passengers and destinations
    of boarded passengers).

    Assumptions
    - Floors are ordered numerically based on their names (e.g., f1 < f2 < f10).
    - Movement between adjacent floors costs 1 action.
    - Movement between non-adjacent floors is a sequence of adjacent moves.
    - The 'above' predicate implies a strict ordering of floors, and the numerical
      suffix in floor names determines this order.
    - The goal is to have all passengers served.
    - A passenger is either at their origin, boarded, or served.

    Heuristic Initialization
    - Parses the 'above' facts to determine the floor order and create a mapping
      from floor names to numerical indices. This assumes floor names follow the
      pattern 'f' followed by a number.
    - Parses the 'destin' facts to store the destination floor for each passenger.
    - Identifies all relevant passengers from the goals or static facts.

    Step-By-Step Thinking for Computing Heuristic
    1. Identify the lift's current floor and its corresponding numerical index.
    2. Identify all unserved passengers by checking which passengers do not have
       a '(served ?p)' fact in the current state.
    3. Initialize the total estimated action cost to 0.
    4. Determine the set of floors the lift *must* visit:
       - Iterate through all unserved passengers.
       - If a passenger has an '(origin ?p ?f)' fact in the state (is unboarded):
         - Add 2 to the total action cost (for the 'board' and 'depart' actions needed).
         - Add their origin floor 'f' to the set of required pickup floors.
         - Add their destination floor (looked up from initialization data) to the set of required dropoff floors.
       - If a passenger has a '(boarded ?p)' fact in the state (is boarded):
         - Add 1 to the total action cost (for the 'depart' action needed).
         - Add their destination floor (looked up from initialization data) to the set of required dropoff floors.
    5. Collect the numerical indices for all unique floors in the combined set of
       required pickup and dropoff floors.
    6. Calculate the estimated movement cost:
       - If there are no required floors, the movement cost is 0.
       - If there are required floors, find the minimum and maximum floor indices
         among them.
       - The estimated movement cost is the minimum of two scenarios:
         a) Distance from the current floor index to the minimum required index,
            plus the distance from the minimum required index to the maximum
            required index (a sweep up).
         b) Distance from the current floor index to the maximum required index,
            plus the distance from the maximum required index to the minimum
            required index (a sweep down).
         This estimates the minimum moves to reach one extreme of the required
         floors and then visit all floors up to the other extreme.
    7. The total heuristic value is the sum of the total estimated action cost
       (from step 4) and the estimated movement cost (from step 6).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        # task.goals are needed to identify all passengers that need serving
        # task.static facts contain 'above' and 'destin'

        # 1. Determine floor order and create floor_to_idx map
        # Collect all floors mentioned in 'above' facts
        all_floors = set()
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'above':
                all_floors.add(parts[1])
                all_floors.add(parts[2])

        # Sort floors numerically based on their suffix (e.g., f1, f2, ..., f10)
        # This assumes floor names are consistently 'f' followed by a number.
        sorted_floors = sorted(list(all_floors), key=get_floor_number)

        self.floor_to_idx = {floor: i for i, floor in enumerate(sorted_floors)}
        self.idx_to_floor = {i: floor for i, floor in enumerate(sorted_floors)}

        # 2. Store passenger destinations
        self.passenger_to_destin = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                passenger, floor = parts[1], parts[2]
                self.passenger_to_destin[passenger] = floor

        # Identify all relevant passengers from goals or static facts
        # Passengers in goals must be served. Passengers in static facts (destin) are also relevant.
        self.all_passengers = set(self.passenger_to_destin.keys())
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == 'served':
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Find current lift location and index
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        # Should always find lift location in a valid state
        if current_lift_floor is None:
             # This indicates an unexpected state structure
             return float('inf') # Or a large value

        current_floor_idx = self.floor_to_idx[current_lift_floor]

        # 2. Identify unserved passengers and their state (unboarded or boarded)
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        boarded_passengers_in_state = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'boarded'}
        origin_facts_in_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if get_parts(fact)[0] == 'origin'}

        unserved_passengers = set()
        unboarded_passengers = set() # Unserved and at origin
        currently_boarded_passengers = set() # Unserved and boarded

        for passenger in self.all_passengers:
            if passenger not in served_passengers_in_state:
                unserved_passengers.add(passenger)
                if passenger in boarded_passengers_in_state:
                    currently_boarded_passengers.add(passenger)
                elif passenger in origin_facts_in_state:
                    unboarded_passengers.add(passenger)
                # Passengers not in served, boarded, or origin facts are in an invalid state
                # according to the domain model, so we don't explicitly handle them,
                # assuming the planner won't reach such states.

        # Check for goal state
        if not unserved_passengers:
            return 0

        # 3. Calculate minimum board/depart actions and identify required floors
        action_cost = 0
        required_floor_indices = set()

        # For unboarded passengers: need board (1) + depart (1) = 2 actions
        for passenger in unboarded_passengers:
             action_cost += 2
             origin_floor = origin_facts_in_state[passenger]
             destin_floor = self.passenger_to_destin[passenger]
             required_floor_indices.add(self.floor_to_idx[origin_floor])
             required_floor_indices.add(self.floor_to_idx[destin_floor])

        # For currently boarded passengers: need depart (1) = 1 action
        for passenger in currently_boarded_passengers:
             action_cost += 1
             destin_floor = self.passenger_to_destin[passenger]
             required_floor_indices.add(self.floor_to_idx[destin_floor])

        # 6. Calculate movement cost
        movement_cost = 0
        if required_floor_indices:
            min_req_idx = min(required_floor_indices)
            max_req_idx = max(required_floor_indices)

            # Estimate movement cost as distance to one extreme + sweep distance
            # Cost to go to min required floor, then sweep up to max
            cost_go_min_then_sweep = abs(current_floor_idx - min_req_idx) + (max_req_idx - min_req_idx)
            # Cost to go to max required floor, then sweep down to min
            cost_go_max_then_sweep = abs(current_floor_idx - max_req_idx) + (max_req_idx - min_req_idx)

            movement_cost = min(cost_go_min_then_sweep, cost_go_max_then_sweep)

        # 7. Total heuristic
        total_heuristic = action_cost + movement_cost

        return total_heuristic
