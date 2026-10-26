from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
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
    This heuristic estimates the cost to reach the goal by summing:
    1. The number of passengers waiting at their origin floor (requiring a 'board' action).
    2. The number of passengers currently boarded (requiring a 'depart' action).
    3. An estimate of the floor travel distance required to visit all necessary floors
       (origin floors for waiting passengers, destination floors for boarded passengers).

    # Assumptions
    - The floors are linearly ordered, and the `above` predicates define this order as a single chain.
    - Floor names are consistently formatted (e.g., f1, f2, ...), although the heuristic relies on `above` facts for ordering, not names.
    - The goal is to serve a specific set of passengers.
    - A passenger is either served, boarded, or waiting at their origin.

    # Heuristic Initialization
    - Extracts the linear ordering of floors from the `above` static facts and creates
      a mapping from floor names to numerical floor numbers (0-indexed from the lowest floor).
    - Stores the destination floor for each passenger from the `destin` static facts.
    - Identifies the set of passengers that need to be served based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check for Goal State: If the current state satisfies all goal conditions (all target
       passengers are served), the heuristic value is 0.

    2. Identify Current Lift Location: Find the floor where the lift is currently located
       from the state facts `(lift-at ?f)`. Convert this floor name to its numerical representation using the mapping created during initialization.

    3. Identify Unserved Passengers: Determine which passengers specified in the goal
       are not yet in the `(served ?p)` state.

    4. Categorize Unserved Passengers: For each unserved passenger:
       - If `(boarded ?p)` is true in the state, the passenger is boarded.
       - Otherwise, `(origin ?p ?f)` must be true, and the passenger is waiting at `?f`.
       Count the number of waiting passengers (`passengers_waiting_count`) and boarded passengers (`passengers_boarded_count`).

    5. Identify Required Floors: Determine the set of floors the lift *must* visit to
       serve the remaining passengers:
       - For each waiting passenger, their origin floor is a required pickup floor.
       - For each boarded passenger, their destination floor is a required dropoff floor.
       Collect the numerical representations of these required floor names into a set (`required_floor_numbers`).

    6. Estimate Floor Travel Cost:
       - If the set of `required_floor_numbers` is empty, the estimated travel cost is 0.
       - Otherwise, find the minimum (`min_req_num`) and maximum (`max_req_num`) floor numbers within the `required_floor_numbers` set.
       - The estimated travel cost is the total span of floors from the minimum of (current lift floor number, `min_req_num`) up to the maximum of (current lift floor number, `max_req_num`). This is calculated as `max(lift_floor_number, max_req_num) - min(lift_floor_number, min_req_num)`. This represents a lower bound on the vertical distance the lift must cover to visit all required floors, starting from its current position.

    7. Calculate Total Heuristic: Sum the counts from step 4 (`passengers_waiting_count` + `passengers_boarded_count`) and the estimated travel cost from step 6. This sum represents the estimated remaining effort in terms of board/depart actions and necessary lift movement.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        self.goals = task.goals
        self.static = task.static

        # Build floor mapping: name -> number
        all_floor_names = set()
        above_pairs = [] # Stores (f_above, f_below) tuples from static facts
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1], get_parts(fact)[2]
                above_pairs.append((f_above, f_below))
                all_floor_names.add(f_above)
                all_floor_names.add(f_below)

        # Find the lowest floor: A floor `f` such that there is no `(above x f)` fact.
        floors_that_are_below_something = {f_below for f_above, f_below in above_pairs}
        lowest_floor = None
        for floor in all_floor_names:
             if floor not in floors_that_are_below_something:
                 lowest_floor = floor
                 break

        # Handle edge case: single floor or no above facts but multiple floors
        if lowest_floor is None and all_floor_names:
             if len(all_floor_names) == 1:
                 lowest_floor = list(all_floor_names)[0]
             else:
                 # This implies the 'above' facts don't form a single linear chain.
                 # For valid miconic problems, this should not happen.
                 # If it does, the floor ordering is ambiguous for this heuristic.
                 # We proceed assuming a linear order was intended and lowest_floor was found.
                 pass # lowest_floor should be found if all_floor_names is not empty and forms a chain

        # Build the map: {floor_below: floor_immediately_above}
        immediately_above_map = {}
        for f_higher, f_lower in above_pairs:
            is_immediate = True
            for f_intermediate_higher, f_intermediate_lower in above_pairs:
                # Check if there's an intermediate floor k such that f_higher > k > f_lower
                # i.e., (above f_higher k) and (above k f_lower)
                if (f_intermediate_higher == f_higher and f_intermediate_lower != f_lower):
                    # f_intermediate_lower is below f_higher. Is it above f_lower?
                    # Check if (above f_intermediate_lower f_lower) is true (directly or indirectly)
                    # Perform a simple reachability check downwards from f_intermediate_lower
                    q = [f_intermediate_lower]
                    visited = set()
                    while q:
                        f_check = q.pop(0)
                        if f_check == f_lower:
                            is_immediate = False
                            break
                        if f_check in visited:
                            continue
                        visited.add(f_check)
                        # Find floors immediately below f_check
                        for fa, fb in above_pairs:
                            if fa == f_check:
                                q.append(fb)
                    if not is_immediate:
                        break # Found an intermediate floor

            if is_immediate:
                immediately_above_map[f_lower] = f_higher # Map: {floor_below: floor_immediately_above}


        # Build the ordered list of floors starting from the lowest.
        ordered_floors = []
        current = lowest_floor
        while current is not None:
            ordered_floors.append(current)
            current = immediately_above_map.get(current)

        # Store the floor mapping
        self.floor_to_number = {floor: i for i, floor in enumerate(ordered_floors)}
        self.number_to_floor = {i: floor for i, floor in enumerate(ordered_floors)}


        # Store destination floors for passengers from static facts
        self.passenger_destin = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1], get_parts(fact)[2]
                self.passenger_destin[p] = f

        # Store goal passengers
        self.goal_passengers = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Check for Goal State
        if self.goals <= state:
            return 0

        # 2. Identify Current Lift Location
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        # Assuming lift_floor is always found in a valid state
        lift_floor_number = self.floor_to_number[lift_floor]

        # 3. Identify Unserved Passengers
        passengers_not_served = {p for p in self.goal_passengers if "(served " + p + ")" not in state}

        # 4. Categorize Unserved Passengers and Identify Required Floors
        passengers_waiting_count = 0
        passengers_boarded_count = 0
        required_floor_numbers = set()

        for p in passengers_not_served:
            if "(boarded " + p + ")" in state:
                passengers_boarded_count += 1
                # Add destination floor to required floors
                destin_floor = self.passenger_destin.get(p)
                if destin_floor and destin_floor in self.floor_to_number: # Check if destin_floor is a known floor
                    required_floor_numbers.add(self.floor_to_number[destin_floor])
            else:
                 # Assume waiting at origin if not served and not boarded
                 passengers_waiting_count += 1
                 # Add origin floor to required floors
                 origin_floor = None
                 for fact in state:
                     if match(fact, "origin", p, "*"):
                         origin_floor = get_parts(fact)[2]
                         break # Found origin floor
                 if origin_floor and origin_floor in self.floor_to_number: # Check if origin_floor is a known floor
                     required_floor_numbers.add(self.floor_to_number[origin_floor])

        # 6. Estimate Floor Travel Cost
        estimated_moves = 0
        if required_floor_numbers:
            min_req_num = min(required_floor_numbers)
            max_req_num = max(required_floor_numbers)
            # Estimated moves is the span from the lowest required floor to the highest,
            # including the current lift floor.
            estimated_moves = max(lift_floor_number, max_req_num) - min(lift_floor_number, min_req_num)

        # 7. Calculate Total Heuristic
        # Heuristic = (number of board actions needed) + (number of depart actions needed) + estimated_moves
        # Number of board actions needed = passengers_waiting_count
        # Number of depart actions needed = passengers_not_served count = passengers_waiting_count + passengers_boarded_count
        # A simpler heuristic for greedy:
        # h = passengers_waiting_count (cost for board) + passengers_boarded_count (cost for depart) + estimated_moves (cost for travel)
        h = passengers_waiting_count + passengers_boarded_count + estimated_moves

        return h
