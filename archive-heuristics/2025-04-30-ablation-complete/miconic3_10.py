from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    passengers. It sums the minimum number of board and depart actions
    needed for unserved passengers and adds an estimate of the lift movement
    cost to visit all necessary floors (origins for waiting passengers,
    destinations for boarded passengers).

    # Assumptions
    - Floors are named f1, f2, ..., fN and (above fi fj) means fi is higher than fj.
      This implies a linear ordering of floors.
    - The cost of each action (move, board, depart) is 1.
    - The lift can carry multiple passengers.

    # Heuristic Initialization
    - Parses the `above` facts from static information to determine the floor
      order (from highest to lowest) and create a mapping from floor names
      to integer indices (0 for highest, N-1 for lowest).
    - Parses `origin` and `destin` facts from static information to store
      the required routes (origin and destination floors) for each passenger.
      Also identifies all passengers involved in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Parse State:** Identify the current floor of the lift by finding the
        `(lift-at ?f)` fact in the current state.
    2.  **Identify Unserved Passengers:** Iterate through all passengers known
        from the static facts. For each passenger, check if they are marked
        as `served` in the current state using the `(served ?p)` fact. If not,
        they are unserved.
    3.  **Categorize Unserved Passengers:** For each unserved passenger,
        determine their current status:
        -   If the state contains `(origin ?p ?f)`, the passenger is `waiting`
            at floor `?f`.
        -   If the state contains `(boarded ?p)`, the passenger is `boarded`
            in the lift.
        (A passenger is either waiting, boarded, or served if unserved).
    4.  **Count Passenger Actions:**
        -   Each waiting passenger requires a `board` action (cost 1) at their
            origin floor.
        -   Each unserved passenger (waiting or boarded) requires a `depart`
            action (cost 1) at their destination floor.
        -   The total cost for these passenger-specific actions is the sum of
            the number of waiting passengers and the total number of unserved
            passengers.
    5.  **Identify Required Stops:** Determine the set of floors the lift
        *must* visit to pick up and drop off unserved passengers:
        -   Include the origin floor for every waiting passenger.
        -   Include the destination floor for every boarded passenger.
    6.  **Calculate Movement Cost:**
        -   If the set of required stops is empty (meaning all passengers are
            served or no stops are needed), the movement cost is 0.
        -   Otherwise, find the minimum and maximum floor indices among the
            required stops using the pre-calculated floor-to-integer mapping.
        -   Calculate the minimum distance needed to travel from the current
            lift floor to visit all floors within this range of required stops.
            This is estimated as the distance from the current floor to the
            closest end of the required floor range, plus the total width of
            that range. Formula: `min(abs(current_int - min_stop_int), abs(current_int - max_stop_int)) + (max_int - min_int)`.
    7.  **Sum Costs:** The total heuristic value is the sum of the passenger
        action costs (from step 4) and the estimated movement cost (from step 6).
    8.  **Goal State:** If all passengers are served (checked in step 2), the
        heuristic correctly returns 0 as there are no unserved passengers and
        no required stops.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger routes.
        """
        # Assuming task object has 'goals' and 'static' attributes
        self.goals = task.goals
        self.static = task.static

        # 1. Determine floor order and create floor-to-int mapping
        # Extract all floors mentioned in 'above' facts
        all_floors = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                all_floors.add(parts[1])
                all_floors.add(parts[2])

        # Sort floors numerically based on the number part (e.g., f1, f2, ..., f10)
        # Assume f1 is highest, fN is lowest based on typical miconic structure
        # and (above fi fj) means fi is higher than fj.
        # Assign integer indices: 0 for highest, N-1 for lowest.
        # Use a custom key to handle f10, f11 correctly compared to f2, f3 etc.
        floor_names_sorted_by_number = sorted(list(all_floors), key=lambda f: int(f[1:]))

        self.floor_list = floor_names_sorted_by_number # e.g., ['f1', 'f2', ..., 'fN']
        self.floor_to_int = {floor: i for i, floor in enumerate(self.floor_list)}
        # self.int_to_floor = {i: floor for i, floor in enumerate(self.floor_list)} # Not strictly needed for this heuristic

        # 2. Store passenger routes (origin and destination) and identify all passengers
        self.passenger_routes = {}
        self.all_passengers = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "origin":
                p, f = parts[1], parts[2]
                self.all_passengers.add(p)
                if p not in self.passenger_routes:
                    self.passenger_routes[p] = {}
                self.passenger_routes[p]['origin'] = f
            elif parts and parts[0] == "destin":
                p, f = parts[1], parts[2]
                self.all_passengers.add(p)
                if p not in self.passenger_routes:
                    self.passenger_routes[p] = {}
                self.passenger_routes[p]['destin'] = f

        # Ensure passengers mentioned only in goals are included (though origin/destin are static)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served":
                 self.all_passengers.add(parts[1])

        self.all_passengers = list(self.all_passengers) # Convert to list for consistent iteration


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached (all goal predicates are in the state)
        if self.goals <= state:
             return 0

        # 1. Parse State: Get current lift floor
        current_f = None
        # Convert state to set for faster lookups
        state_facts_set = set(state)

        for fact in state_facts_set:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_f = parts[1]
                break

        # This should always be found in a valid state
        if current_f is None:
             # Problem state is malformed or not reachable
             return float('inf') # Indicate unsolvable or error state

        current_f_int = self.floor_to_int[current_f]

        # 2 & 3. Identify Unserved Passengers and Categorize
        num_waiting = 0
        num_boarded = 0
        required_stops = set()

        served_passengers_in_state = {get_parts(fact)[1] for fact in state_facts_set if match(fact, "served", "*")}

        unserved_passengers = [p for p in self.all_passengers if p not in served_passengers_in_state]

        for p in unserved_passengers:
            # Get passenger route info from pre-calculated data
            route = self.passenger_routes.get(p)
            if not route:
                 # Should not happen if all passengers are in static origin/destin
                 continue # Skip passenger if route is unknown

            origin_f = route.get('origin')
            destin_f = route.get('destin')

            # Determine passenger status based on state facts
            is_waiting = False
            # Check specifically if the passenger is at their defined origin floor
            if origin_f is not None: # Ensure origin is defined
                is_waiting = f'(origin {p} {origin_f})' in state_facts_set

            is_boarded = f'(boarded {p})' in state_facts_set

            if is_waiting:
                num_waiting += 1
                required_stops.add(origin_f)
                required_stops.add(destin_f) # Need to visit destin to drop off later
            elif is_boarded:
                num_boarded += 1
                required_stops.add(destin_f) # Need to visit destin to drop off
            # else: # Passenger is unserved but neither waiting nor boarded? Invalid state?
            #    pass # Should not happen in valid states

        # 4. Count Passenger Actions
        # Each waiting passenger needs 1 board action.
        # Each unserved passenger needs 1 depart action.
        num_unserved = num_waiting + num_boarded
        passenger_action_cost = num_waiting + num_unserved

        # 6. Calculate Movement Cost
        movement_cost = 0
        if required_stops:
            required_stop_ints = {self.floor_to_int[f] for f in required_stops}
            min_int = min(required_stop_ints)
            max_int = max(required_stop_ints)

            # Minimum distance to traverse the range [min_int, max_int] starting from current_f_int
            # This is the distance to the closest end of the range plus the width of the range.
            movement_cost = min(abs(current_f_int - min_int), abs(current_f_int - max_int)) + (max_int - min_int)

        # 7. Sum Costs
        total_cost = passenger_action_cost + movement_cost

        return total_cost
