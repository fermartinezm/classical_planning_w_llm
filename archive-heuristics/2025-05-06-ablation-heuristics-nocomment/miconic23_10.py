from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Splits a PDDL fact string into its components."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of arguments if args are provided
    if args and len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Estimates the cost to reach the goal state (all passengers served)
    by summing the number of board/depart actions needed and the estimated
    lift travel cost.

    Summary:
    The heuristic counts the number of passengers waiting at their origin
    (who need a 'board' action) and the number of passengers boarded
    (who need a 'depart' action). It adds an estimate of the minimum
    vertical travel the lift must perform to visit all floors where
    passengers need to be picked up or dropped off.

    Assumptions:
    - The floor structure forms a single linear tower defined by 'above' predicates.
    - 'above f1 f2' means f1 is immediately above f2.
    - Passenger origins and destinations are defined in the static facts.
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    - Parses 'above' facts from static information to determine the level of each floor.
      This allows calculating the vertical distance between any two floors.
    - Parses 'origin' and 'destin' facts from static information to store
      the required pickup and dropoff floors for each passenger.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current floor of the lift from the state.
    2. Identify all passengers who are currently 'served' from the state.
    3. Determine the set of unserved passengers by subtracting served passengers
       from the total set of passengers known from static facts.
    4. Categorize unserved passengers based on the state: 'waiting' (have an
       '(origin p O)' fact) or 'boarded' (have a '(boarded p)' fact).
    5. Count the number of waiting passengers (N_waiting). Each requires a 'board' action.
    6. Count the number of boarded passengers (N_boarded). Each requires a 'depart' action.
    7. Determine the set of 'required stops': this includes the origin floor for
       each waiting passenger and the destination floor for each boarded passenger.
       These are the floors the lift must visit to make progress for unserved passengers.
    8. If the set of required stops is empty, it means all unserved passengers
       are boarded and are already at their destination floor. In this specific
       case, only 'depart' actions are needed. The heuristic is simply N_boarded.
    9. If there are required stops, calculate the estimated travel cost:
       - Find the minimum and maximum floor levels among the required stops.
       - Find the current floor level of the lift.
       - The estimated travel cost is the minimum vertical distance required to reach
         one extreme of the required floor range (min or max level) from the current
         level, plus the vertical distance spanning the required floor range
         (max level - min level). This estimates the travel needed to reach the
         relevant floors and sweep through them.
       - Travel cost = min(abs(current_level - min_level) + (max_level - min_level),
                           abs(current_level - max_level) + (max_level - min_level)).
    10. The total heuristic value is the sum of N_waiting, N_boarded, and the estimated travel cost.
        Heuristic = N_waiting + N_boarded + Travel cost.
    """
    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse 'above' facts to determine floor levels
        # Build map: floor -> floor_immediately_below_it
        floor_below = {}
        all_floors = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f1, f2 = get_parts(fact)[1:]
                floor_below[f1] = f2
                all_floors.add(f1)
                all_floors.add(f2)

        # Find the lowest floor (a floor that is in all_floors but not a key in floor_below)
        floors_with_floor_below = set(floor_below.keys())
        lowest_floor = None
        for floor in all_floors:
             if floor not in floors_with_floor_below:
                 lowest_floor = floor
                 break # Assuming a single tower

        # Assign levels starting from the lowest floor
        self.floor_levels = {}
        if lowest_floor:
            current_floor = lowest_floor
            level = 1
            # Build reverse map: floor -> floor_immediately_above_it
            floor_above = {v: k for k, v in floor_below.items()}

            while current_floor in all_floors: # Loop until we go above the highest floor
                self.floor_levels[current_floor] = level
                if current_floor in floor_above:
                    current_floor = floor_above[current_floor]
                    level += 1
                else:
                    break # Reached the highest floor
        # else: Handle case where all_floors is empty or structure is not a simple tower (e.g., cyclic)
        # For valid miconic, lowest_floor should be found if all_floors is not empty.


        # 2. Parse 'origin' and 'destin' facts for passenger info
        self.passenger_info = {}
        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                p, f = get_parts(fact)[1:]
                if p not in self.passenger_info:
                    self.passenger_info[p] = {}
                self.passenger_info[p]['origin'] = f
            elif match(fact, "destin", "*", "*"):
                p, f = get_parts(fact)[1:]
                if p not in self.passenger_info:
                    self.passenger_info[p] = {}
                self.passenger_info[p]['destin'] = f

        # Get the set of all passengers involved in the problem
        self.all_passengers = set(self.passenger_info.keys())


    def __call__(self, node):
        state = node.state

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or floor levels not initialized, return infinity or a large value
        if current_lift_floor is None or not self.floor_levels:
             # Cannot compute meaningful heuristic without lift location or floor structure
             # Returning a large value indicates this state is likely undesirable or invalid
             # In a well-formed problem, this case should not be reached for reachable states.
             # As a fallback, return the number of unserved passengers * a large constant
             served_passengers_count = sum(1 for fact in state if match(fact, "served", "*"))
             return (len(self.all_passengers) - served_passengers_count) * 1000 # Arbitrary large cost


        current_level = self.floor_levels.get(current_lift_floor, 0) # Default to 0 if floor not found (shouldn't happen)


        # 2. Identify served passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # 3. Determine unserved passengers
        unserved_passengers = self.all_passengers - served_passengers

        # If all passengers are served, heuristic is 0
        if not unserved_passengers:
             return 0

        # 4. Categorize unserved passengers and 5. Count N_waiting and N_boarded
        waiting_passengers = set()
        boarded_passengers = set()

        # Build sets of waiting and boarded passengers from state facts
        state_waiting_passengers = {get_parts(fact)[1] for fact in state if match(fact, "origin", "*", "*")}
        state_boarded_passengers = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}

        # Filter by unserved status
        waiting_passengers = unserved_passengers.intersection(state_waiting_passengers)
        boarded_passengers = unserved_passengers.intersection(state_boarded_passengers)

        n_waiting = len(waiting_passengers)
        n_boarded = len(boarded_passengers)


        # 7. Determine required stops
        required_stops = set()
        for p in waiting_passengers:
            # Add origin floor for waiting passengers
            origin_floor = self.passenger_info.get(p, {}).get('origin')
            if origin_floor and origin_floor in self.floor_levels: # Ensure origin is known and has a level
                required_stops.add(origin_floor)

        for p in boarded_passengers:
            # Add destination floor for boarded passengers
            destin_floor = self.passenger_info.get(p, {}).get('destin')
            if destin_floor and destin_floor in self.floor_levels: # Ensure destination is known and has a level
                 required_stops.add(destin_floor)

        # 8. Calculate estimated travel cost
        travel_cost = 0
        if required_stops:
            required_levels = [self.floor_levels[f] for f in required_stops]
            min_level = min(required_levels)
            max_level = max(required_levels)

            # Estimated travel cost: min distance to reach one end of the range
            # plus the distance to traverse the range.
            dist_to_min = abs(current_level - min_level)
            dist_to_max = abs(current_level - max_level)
            range_dist = max_level - min_level

            travel_cost = min(dist_to_min + range_dist, dist_to_max + range_dist)
        # else: required_stops is empty. This means all unserved passengers are boarded
        # and are already at their destination floor. Travel cost is 0.
        # The heuristic will be n_waiting + n_boarded + 0 = 0 + n_boarded = n_boarded,
        # which is the number of depart actions needed. This is handled correctly.


        # 10. Calculate total heuristic value
        # Heuristic = (board actions needed) + (depart actions needed) + (estimated travel actions)
        # Each waiting passenger needs 1 board action.
        # Each boarded passenger needs 1 depart action.
        heuristic_value = n_waiting + n_boarded + travel_cost

        return heuristic_value

