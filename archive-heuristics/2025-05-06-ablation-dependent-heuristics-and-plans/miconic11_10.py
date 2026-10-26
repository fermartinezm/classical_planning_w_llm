from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or invalid formats defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It considers the number of board and depart actions needed for unserved passengers
    and adds an estimate of the lift travel cost to reach the necessary floors.

    # Assumptions
    - The floors are linearly ordered, defined by the 'above' predicate.
    - The cost of each action (board, depart, up, down) is 1.
    - The heuristic is non-admissible and designed to guide a greedy best-first search.

    # Heuristic Initialization
    - Identify all passengers that need to be served from the goal state.
    - Store the origin and destination floor for each relevant passenger.
    - Determine the linear ordering of floors based on the 'above' facts and create
      a mapping from floor names to numerical floor numbers.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the lift.
    2. Identify which passengers are currently served, waiting at their origin,
       or boarded in the lift. Only consider passengers that are part of the goal.
    3. Calculate the "action cost" component:
       - Each unserved passenger who is waiting needs a 'board' action and a 'depart' action.
       - Each unserved passenger who is boarded needs a 'depart' action.
       - A simple estimate is 2 actions for each waiting passenger and 1 action for each boarded passenger.
       - Total action cost = (2 * number of waiting passengers) + (number of boarded passengers).
    4. Calculate the "travel cost" component:
       - Identify the set of floors the lift *must* visit: the origin floors of waiting passengers
         and the destination floors of boarded passengers.
       - If there are no required floors (meaning all relevant passengers are served), the travel cost is 0.
       - Otherwise, find the minimum and maximum floor numbers among the required floors.
       - Estimate the travel cost as the minimum number of moves required for the lift, starting
         from its current floor, to traverse the range between the minimum and maximum required floors.
         This is calculated as the distance from the current floor to the closest end of the required
         floor range, plus the total span of the required floor range.
    5. The total heuristic value is the sum of the action cost and the travel cost.
    6. If all goal passengers are served, the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting passenger information and floor mapping.
        """
        # The base class Heuristic is expected to be available and provide task attributes
        # super().__init__(task) # Assuming Heuristic base class has __init__(self, task)

        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state for origin/destin

        self.all_passengers = set()
        self.passenger_origin = {}
        self.passenger_destin = {}
        self.floor_to_num_map = {}

        # 1. Identify all passengers that need to be served from the goal state.
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served':
                self.all_passengers.add(parts[1])

        # 2. Store the origin and destination floor for each relevant passenger.
        # Origin and destin facts are typically in the initial state.
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'origin' and parts[1] in self.all_passengers:
                    self.passenger_origin[parts[1]] = parts[2]
                elif parts[0] == 'destin' and parts[1] in self.all_passengers:
                    self.passenger_destin[parts[1]] = parts[2]

        # 3. Determine the linear ordering of floors and create floor number mapping.
        floor_above_to_floor_below_map = {}
        all_floors = set()

        # Collect all floors and build the 'above' relationship map
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                f_above, f_below = parts[1], parts[2]
                floor_above_to_floor_below_map[f_above] = f_below
                all_floors.add(f_above)
                all_floors.add(f_below)

        # Add floors from initial state facts if not already included
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts:
                 if parts[0] == 'lift-at':
                     all_floors.add(parts[1])
                 elif parts[0] in ['origin', 'destin'] and len(parts) > 2:
                     all_floors.add(parts[2])

        if not all_floors:
             # Handle case with no floors (shouldn't happen in valid miconic)
             return

        # Find the lowest floor (a floor that is not a value in the above map)
        all_floors_below_in_map = set(floor_above_to_floor_below_map.values())
        lowest_floor = None
        for floor in all_floors:
            if floor not in all_floors_below_in_map:
                lowest_floor = floor
                break

        # Build the floor_to_num_map by traversing upwards from the lowest floor
        if lowest_floor:
            current_floor = lowest_floor
            current_num = 1
            # Need the inverse map to traverse upwards: floor_below_to_floor_above_map
            floor_below_to_floor_above_map = {v: k for k, v in floor_above_to_floor_below_map.items()}

            while current_floor:
                self.floor_to_num_map[current_floor] = current_num
                current_num += 1
                current_floor = floor_below_to_floor_above_map.get(current_floor)
        # else: # If lowest_floor is None, the 'above' facts didn't form a simple chain starting from a unique lowest floor.
               # The floor_to_num_map will remain empty, and the heuristic will likely return inf or behave unexpectedly.
               # This assumes valid miconic problems have a linear floor structure.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify the current location of the lift.
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at':
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None or current_lift_floor not in self.floor_to_num_map:
             # This state is likely invalid or lift is at an unmapped floor
             return float('inf') # Should not happen in valid miconic states

        current_lift_floor_num = self.floor_to_num_map[current_lift_floor]

        # 2. Identify unserved passengers and their state (waiting or boarded).
        served_passengers = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == 'served'}
        unserved_passengers = self.all_passengers - served_passengers

        # 6. If all goal passengers are served, return 0.
        if not unserved_passengers:
            return 0

        waiting_passengers = set()
        boarded_passengers = set()

        for p in unserved_passengers:
            # Check if passenger is waiting at their origin
            # Need to check if origin is known for this passenger
            if p in self.passenger_origin:
                origin_fact = ['origin', p, self.passenger_origin[p]]
                if any(get_parts(fact) == origin_fact for fact in state):
                     waiting_passengers.add(p)
            # Check if passenger is boarded
            boarded_fact = ['boarded', p]
            if any(get_parts(fact) == boarded_fact for fact in state):
                 boarded_passengers.add(p)

        # 3. Calculate the "action cost" component.
        # Each waiting needs board (1) + depart (1) = 2 actions
        # Each boarded needs depart (1) = 1 action
        board_depart_cost = 2 * len(waiting_passengers) + len(boarded_passengers)

        # 4. Calculate the "travel cost" component.
        pickup_floors = {self.passenger_origin[p] for p in waiting_passengers if p in self.passenger_origin}
        dropoff_floors = {self.passenger_destin[p] for p in boarded_passengers if p in self.passenger_destin}
        required_floors = pickup_floors | dropoff_floors

        travel_cost = 0
        if required_floors:
            # Filter out any required floors that weren't mapped (shouldn't happen in valid problems)
            required_floor_nums = {self.floor_to_num_map[f] for f in required_floors if f in self.floor_to_num_map}

            if required_floor_nums: # Ensure we have valid floor numbers to consider
                min_req_floor_num = min(required_floor_nums)
                max_req_floor_num = max(required_floor_nums)

                # Estimate travel cost to cover the range [min_req_floor_num, max_req_floor_num]
                # starting from current_lift_floor_num
                range_travel = max_req_floor_num - min_req_floor_num

                if current_lift_floor_num < min_req_floor_num:
                    # Must go up to min_req, then traverse range
                    travel_cost = (min_req_floor_num - current_lift_floor_num) + range_travel
                elif current_lift_floor_num > max_req_floor_num:
                    # Must go down to max_req, then traverse range
                    travel_cost = (current_lift_floor_num - max_req_floor_num) + range_travel
                else: # current_lift_floor_num is within the range [min_req, max_req]
                    # Option 1: Go down to min_req, then up to max_req
                    cost1 = (current_lift_floor_num - min_req_floor_num) + range_travel
                    # Option 2: Go up to max_req, then down to min_req
                    cost2 = (max_req_floor_num - current_lift_floor_num) + range_travel
                    travel_cost = min(cost1, cost2)

        # 5. Total heuristic value
        return board_depart_cost + travel_cost

