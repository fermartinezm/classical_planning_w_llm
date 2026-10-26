from heuristics.heuristic_base import Heuristic

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the number of actions required to reach the goal state
    (all passengers served). The heuristic sums three components:
    1. Estimated move actions for the lift to visit all necessary floors.
    2. Number of board actions needed for waiting passengers.
    3. Number of depart actions needed for boarded passengers.

    Assumptions:
    - The PDDL domain is 'miconic' with standard predicates (origin, destin,
      above, boarded, served, lift-at) and actions (board, depart, up, down).
    - The 'above' predicates define a linear, total order of floors.
    - Passenger destinations are static and provided in the initial state facts
      using the 'destin' predicate.
    - The initial state is valid (e.g., lift is at exactly one floor, passengers
      are either at an origin or served).

    Heuristic Initialization:
    In the constructor (__init__), the heuristic precomputes static information:
    - Parses the 'above' facts to build a mapping between floor names and
      integer indices, representing the floor order. This allows efficient
      calculation of distances between floors.
    - Parses the 'destin' facts to store the destination floor for each
      passenger.
    - Stores the total number of passengers to easily check for the goal state.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state (node):
    1. Identify the lift's current floor by finding the fact '(lift-at ?f)'
       in the state and determine its corresponding floor index using the
       precomputed mapping.
    2. Iterate through the state facts to identify passengers who are
       '(origin ?p ?f)' (waiting) and those who are '(boarded ?p)'. Count
       the number of waiting and boarded passengers. Also, identify passengers
       who are '(served ?p)' and count them.
    3. If the number of served passengers equals the total number of passengers
       in the problem, the state is a goal state, and the heuristic value is 0.
    4. Determine the set of *required floors* that the lift must visit to make
       progress towards the goal. This set includes the origin floor for every
       waiting passenger and the destination floor for every boarded passenger
       (using the precomputed destination map).
    5. Calculate the *move cost*:
       - If the set of required floors is empty, the move cost is 0. This case
         only occurs if all non-served passengers are waiting/boarded at the
         current floor, or if all passengers are served (handled in step 3).
       - If the set of required floors is not empty, find the minimum and
         maximum floor indices among these required floors. The estimated
         minimum number of move actions required for the lift to travel from
         its current floor to visit all required floors is calculated as:
         `min(abs(current_floor_index - min_required_index), abs(current_floor_index - max_required_index)) + (max_required_index - min_required_index)`.
         This formula estimates the cost of traveling to the closest end of the
         required floor range and then sweeping across the entire range.
    6. Calculate the *board cost*: This is simply the number of waiting
       passengers, as each requires one 'board' action.
    7. Calculate the *depart cost*: This is simply the number of boarded
       passengers, as each requires one 'depart' action.
    8. The total heuristic value for the state is the sum of the move cost,
       the board cost, and the depart cost. This value estimates the remaining
       actions needed to get all current waiting passengers boarded and all
       current boarded passengers departed, plus the necessary travel.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task to access static info

        # --- Heuristic Initialization ---
        # Parse floor order and create index mapping
        self.floor_to_index = {}
        self.index_to_floor = {}
        above_map = {}
        all_floors = set()
        floors_with_something_above_them = set()

        for fact in self.task.static:
            if fact.startswith('(above '):
                parts = fact[1:-1].split()
                f_above = parts[1]
                f_below = parts[2]
                above_map[f_below] = f_above
                all_floors.add(f_below)
                all_floors.add(f_above)
                floors_with_something_above_them.add(f_above)

        # Find the lowest floor (the one with nothing immediately below it defined by 'above')
        # Assumes a single lowest floor exists in a valid miconic problem.
        lowest_floors = all_floors - floors_with_something_above_them
        # In a valid miconic problem, there should be exactly one lowest floor.
        lowest_floor = lowest_floors.pop()

        # Build the index mapping by following the 'above' chain
        current_floor = lowest_floor
        index = 0
        while current_floor is not None:
            self.floor_to_index[current_floor] = index
            self.index_to_floor[index] = current_floor
            index += 1
            current_floor = above_map.get(current_floor)
        self.num_floors = index # Total number of floors

        # Parse passenger destinations from static facts
        self.passenger_destin = {}
        for fact in self.task.static:
            if fact.startswith('(destin '):
                parts = fact[1:-1].split()
                p = parts[1]
                f_destin = parts[2]
                self.passenger_destin[p] = f_destin

        # Store total number of passengers
        self.all_passengers_count = len(self.passenger_destin)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Step 1: Identify lift's current floor and its index.
        current_floor = None
        for fact in state:
            if fact.startswith('(lift-at '):
                current_floor = fact[9:-1] # Extract floor name, e.g., 'f2' from '(lift-at f2)'
                break
        # In a valid state, the lift must be at some floor.
        current_floor_idx = self.floor_to_index[current_floor]

        # Step 2 & 3: Identify passenger states (waiting, boarded, served)
        # and their associated required floors (origins for waiting, destinations for boarded).
        waiting_passengers_count = 0
        boarded_passengers_count = 0
        served_passengers_count = 0
        pickup_stops = set()
        dropoff_stops = set()

        for fact in state:
            if fact.startswith('(origin '):
                waiting_passengers_count += 1
                parts = fact[8:-1].split() # e.g., 'p1 f6' from '(origin p1 f6)'
                # p = parts[0] # passenger name not needed for count or pickup_stops set
                f_origin = parts[1]
                pickup_stops.add(f_origin)
            elif fact.startswith('(boarded '):
                boarded_passengers_count += 1
                p = fact[9:-1] # e.g., 'p4' from '(boarded p4)'
                # Need destination for boarded passenger from precomputed map
                f_destin = self.passenger_destin[p]
                dropoff_stops.add(f_destin)
            elif fact.startswith('(served '):
                served_passengers_count += 1
                # p = fact[8:-1] # passenger name not needed for count

        # Step 4: Check if goal is reached.
        if served_passengers_count == self.all_passengers_count:
            return 0 # Goal state reached

        # Step 5: Determine the set of required floors to visit.
        required_floors = pickup_stops | dropoff_stops

        # Step 6: Calculate the move cost.
        move_cost = 0
        if required_floors:
            required_floor_indices = {self.floor_to_index[f] for f in required_floors}
            min_req_idx = min(required_floor_indices)
            max_req_idx = max(required_floor_indices)
            # Minimum moves to visit all floors in the range [min_req_idx, max_req_idx]
            # starting from current_floor_idx.
            move_cost = min(abs(current_floor_idx - min_req_idx), abs(current_floor_idx - max_req_idx)) + (max_req_idx - min_req_idx)
        # Else: If required_floors is empty, it implies all non-served passengers
        # are waiting/boarded at the current floor. Move cost is 0.
        # This case is implicitly handled because if required_floors is empty,
        # pickup_stops and dropoff_stops are empty, which implies
        # waiting_passengers_count and boarded_passengers_count are 0,
        # which implies served_passengers_count == all_passengers_count,
        # and the heuristic returns 0 in Step 4.

        # Step 7 & 8: Calculate board and depart costs (number of actions).
        board_cost = waiting_passengers_count
        depart_cost = boarded_passengers_count

        # Step 9: The heuristic value is the sum of costs.
        h_value = move_cost + board_cost + depart_cost

        return h_value
