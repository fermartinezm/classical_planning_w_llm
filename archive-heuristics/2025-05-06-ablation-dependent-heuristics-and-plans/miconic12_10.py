from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It considers the estimated vertical movement cost for the elevator to visit necessary floors
    (passenger origins and destinations) and the actions for boarding and departing.

    # Assumptions
    - Floors are totally ordered, defined by the 'above' predicate where `(above f_higher f_lower)`
      means `f_higher` is directly above `f_lower`.
    - Each passenger needs to be picked up at their initial origin and dropped off at their destination.
    - Unserved passengers are always either at their initial origin (indicated by an `(origin p f)` fact)
      or are boarded (indicated by a `(boarded p)` fact).
    - The elevator can carry multiple passengers.

    # Heuristic Initialization
    - Parses the 'above' static facts to establish the floor order and create a mapping
      from floor names to integer indices. It assumes `(above f_higher f_lower)` means
      `f_higher` is directly above `f_lower`.
    - Extracts the destination floor for each passenger from the initial state facts.
    - Extracts the initial origin floor for each passenger from the initial state facts.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check for Goal State: If the current state satisfies all goal conditions (all passengers served), the heuristic value is 0.

    2. Identify Current Elevator Location: Find the floor where the elevator is currently located and get its corresponding integer index using the pre-calculated floor mapping.

    3. Identify Unserved Passengers: Determine which passengers are not yet marked as `served`.

    4. Determine State of Unserved Passengers: For each unserved passenger, check if they are currently at their initial origin floor (indicated by an `(origin p f)` fact in the state) or if they are boarded (indicated by a `(boarded p)` fact in the state).

    5. Collect Required Stops: Create a set of unique floor indices that the elevator *must* visit to make progress towards serving unserved passengers:
       - For each unserved passenger currently at their origin, add the index of their origin floor to the set of required stops.
       - For each unserved passenger who is boarded, add the index of their destination floor to the set of required stops.

    6. Estimate Movement Cost:
       - If there are no required stops but there are still unserved passengers, this indicates an unexpected or potentially invalid state according to the domain rules. Return a large heuristic value to penalize this state.
       - If there are required stops, calculate the minimum and maximum floor indices among these stops.
       - The estimated movement cost is the total vertical span the elevator must cover to visit its current floor and all required stops. This is calculated as `max(current_floor_idx, max_required_idx) - min(current_floor_idx, min_required_idx)`. This represents the distance from the lowest floor visited (either current or lowest required) to the highest floor visited (either current or highest required).

    7. Estimate Boarding/Departing Cost:
       - Each unserved passenger currently at an origin floor requires one `board` action.
       - Each unserved passenger who is currently boarded requires one `depart` action.
       - The estimated boarding/departing cost is the sum of the number of unserved passengers at origins and the number of unserved boarded passengers.

    8. Total Heuristic Value: The total heuristic is the sum of the estimated movement cost and the estimated boarding/departing cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Floor order and index mapping from 'above' static facts.
        - Goal destination for each passenger from initial state facts.
        - Initial origin for each passenger from initial state facts.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state_facts = task.initial_state # Destin and initial origin facts are here

        # 1. Build floor order and index mapping
        all_floors = set()
        above_relations = set() # Store (higher_floor, lower_floor) tuples
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "above":
                higher_floor, lower_floor = parts[1], parts[2]
                all_floors.add(higher_floor)
                all_floors.add(lower_floor)
                above_relations.add((higher_floor, lower_floor))

        self.floor_to_index = {}
        self.index_to_floor = {}

        if not all_floors:
             # No floors defined, likely an invalid problem instance for this domain.
             # Heuristic will return 0 if goals are met, otherwise potentially high.
             pass # Dictionaries remain empty
        elif len(all_floors) == 1:
             # Only one floor, index is 0
             floor = list(all_floors)[0]
             self.floor_to_index = {floor: 0}
             self.index_to_floor = {0: floor}
        else:
            # Find the lowest floor: the one never appearing as the second argument in (above ?x f)
            is_second_arg = {f: False for f in all_floors}
            for higher, lower in above_relations:
                 is_second_arg[lower] = True

            lowest_floor = None
            for f in all_floors:
                if not is_second_arg[f]:
                    lowest_floor = f
                    break

            # Build the direct_above_map: floor_below -> floor_directly_above
            # f_a is directly above f_b if (above f_a f_b) and there is no f_c s.t. (above f_c f_b) and (above f_a f_c)
            direct_above_map = {}
            floors_list = list(all_floors)

            for f_b in floors_list:
                floors_above_b = {f_a for f_a in floors_list if (f_a, f_b) in above_relations}
                if floors_above_b:
                    # Find the minimum floor in floors_above_b according to the 'above' relation
                    floor_directly_above_b = None
                    for f_x in floors_above_b:
                        is_min_above = True
                        for f_y in floors_above_b:
                            if f_x != f_y and (f_y, f_x) in above_relations: # If f_y is above f_x
                                is_min_above = False
                                break
                        if is_min_above:
                            floor_directly_above_b = f_x
                            break
                    if floor_directly_above_b:
                         direct_above_map[f_b] = floor_directly_above_b

            # Traverse from lowest floor using direct_above_map to build index map
            current = lowest_floor
            index = 0
            while current is not None:
                self.floor_to_index[current] = index
                self.index_to_floor[index] = current
                index += 1
                current = direct_above_map.get(current)


        # 2. Extract goal destinations and initial origins for each passenger
        self.passenger_goals = {}
        self.passenger_origins = {}

        for fact in initial_state_facts:
             parts = get_parts(fact)
             if parts[0] == "destin":
                 passenger, destin_floor = parts[1], parts[2]
                 self.passenger_goals[passenger] = destin_floor
             elif parts[0] == "origin":
                 passenger, origin_floor = parts[1], parts[2]
                 self.passenger_origins[passenger] = origin_floor

        # Get the set of all passengers from the initial state
        self.all_passengers = set(self.passenger_origins.keys()).union(set(self.passenger_goals.keys()))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify current lift location
        lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "lift-at":
                lift_floor = parts[1]
                break

        # If lift_floor is not found, the state is likely invalid.
        if lift_floor is None:
             # This shouldn't happen in valid miconic states. Return a high value.
             return 1000

        lift_floor_idx = self.floor_to_index.get(lift_floor)
        if lift_floor_idx is None:
             # Lift is at a floor not in our mapping, likely invalid state.
             return 1000


        # 3. Identify unserved passengers
        served_passengers = set()
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "served":
                served_passengers.add(parts[1])

        unserved_passengers = self.all_passengers - served_passengers

        # 4. Determine State of Unserved Passengers
        unserved_at_origin_floors = {} # floor -> set of unserved passengers at this floor
        unserved_boarded_passengers = set() # set of unserved boarded passengers

        # Find current location/state for unserved passengers from the state facts
        state_origins = {} # floor -> set of passengers at this floor
        state_boarded = set() # set of boarded passengers

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "origin":
                p, f = parts[1], parts[2]
                if f not in state_origins:
                    state_origins[f] = set()
                state_origins[f].add(p)
            elif parts[0] == "boarded":
                p = parts[1]
                state_boarded.add(p)

        for p in unserved_passengers:
            # Check if passenger is at their initial origin floor in the current state
            initial_origin = self.passenger_origins.get(p)
            if initial_origin is not None and initial_origin in state_origins and p in state_origins[initial_origin]:
                 unserved_at_origin_floors[initial_origin] = unserved_at_origin_floors.get(initial_origin, set())
                 unserved_at_origin_floors[initial_origin].add(p)
            elif p in state_boarded:
                 unserved_boarded_passengers.add(p)
            # else: passenger is unserved but not at origin and not boarded.
            # This state should not be reachable in a valid execution.
            # We handle this case below by checking if required_stops_idx is empty
            # while unserved_passengers is not.


        # 5. Collect Required Stops (floors)
        required_stops_idx = set()

        # Floors where unserved passengers are waiting at origin
        for origin_floor in unserved_at_origin_floors:
            # Ensure the floor exists in our mapping (should always be the case for valid origins)
            if origin_floor in self.floor_to_index:
                required_stops_idx.add(self.floor_to_index[origin_floor])

        # Floors where unserved boarded passengers need to depart
        for p in unserved_boarded_passengers:
            destin_floor = self.passenger_goals.get(p) # Get destin from initial state info
            # Ensure the destination floor exists in our mapping
            if destin_floor is not None and destin_floor in self.floor_to_index:
                required_stops_idx.add(self.floor_to_index[destin_floor])
            # else: Passenger has no destination or destination floor is unknown. Invalid state.


        # 6. Calculate estimated cost
        total_cost = 0

        # Handle invalid state where unserved passengers exist but no required stops identified
        if not required_stops_idx and unserved_passengers:
             # This state is likely unreachable or invalid according to the domain rules.
             # Return a high heuristic value to discourage exploring this path.
             return 1000

        # Movement cost
        if required_stops_idx:
            min_required_idx = min(required_stops_idx)
            max_required_idx = max(required_stops_idx)

            # Estimated movement cost: total vertical span from current lift floor
            # to cover the range of required stops.
            total_cost += max(lift_floor_idx, max_required_idx) - min(lift_floor_idx, min_required_idx)

        # 7. Boarding/Departing cost
        # Each unserved passenger at an origin floor needs 1 board action.
        # Each unserved boarded passenger needs 1 depart action.
        total_cost += sum(len(passengers) for passengers in unserved_at_origin_floors.values())
        total_cost += len(unserved_boarded_passengers)

        return total_cost

