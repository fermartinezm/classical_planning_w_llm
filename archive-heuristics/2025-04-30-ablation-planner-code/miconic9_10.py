# from heuristics.heuristic_base import Heuristic # Assuming this is provided by the framework

# Define a dummy Heuristic base class if running standalone for testing
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        state = node.state
        raise NotImplementedError

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    # Check if the number of parts matches the number of args, considering wildcards.
    # If the number of parts is less than the number of non-wildcard args, it can't match.
    non_wildcard_args = [arg for arg in args if arg != '*']
    if len(parts) < len(non_wildcard_args):
        return False
    # Check if each part matches the corresponding argument pattern up to the length of parts or args
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the remaining effort to serve all passengers.
    It considers the movement cost for the elevator to visit all necessary floors
    (origin floors for unpicked passengers and destination floors for boarded passengers)
    plus the cost of the required board and depart actions.

    # Assumptions
    - Floors are ordered linearly (f1 < f2 < ... < fn). The 'above' facts define this order.
      If no 'above' facts exist but floors are mentioned in 'destin' facts, all floors
      are assumed to be on the same level (movement cost 0).
    - The elevator has infinite capacity.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - All passengers mentioned in the problem have a static destination defined by a 'destin' fact.
    - A passenger is either at their origin, boarded, or served (not a mix in a valid state).
    - The elevator's location is always defined by a `(lift-at ?f)` fact in a valid state.
    - PDDL facts are well-formed strings.

    # Heuristic Initialization
    - Extracts the floor ordering from the static 'above' facts to create a mapping
      from floor names (e.g., 'f1') to integer floor numbers (e.g., 1). This is done
      by counting how many floors are strictly below each floor based on the 'above'
      relations. If no 'above' facts define an order, but floors are mentioned in
      'destin' facts, all such floors are mapped to integer 1.
    - Extracts the destination floor for each passenger from the static 'destin' facts
      and stores them in a dictionary. Also collects the set of all passengers known
      from these static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the elevator's current floor from the `(lift-at ?f)` fact.
    2. Identify all passengers who are currently served from the `(served ?p)` facts.
    3. Initialize counts for unpicked and boarded passengers and sets for their required floors.
    4. Iterate through all passengers known in the problem (from static 'destin' facts).
    5. For each passenger who is *not* served:
       - Check the current state to see if the passenger is at their origin floor (`(origin ?p ?f)`). If so, increment the count of unpicked passengers and add their origin floor to the set of 'pickup floors'.
       - Check the current state to see if the passenger is boarded (`(boarded ?p)`). If so, increment the count of boarded passengers and add their destination floor (looked up from static data) to the set of 'dropoff floors'.
       - (A passenger should be either at origin, boarded, or served in a valid state).
    6. Combine 'pickup floors' and 'dropoff floors' into a single set of 'required floors' that the elevator must visit.
    7. If there are no required floors (meaning all unserved passengers are neither at origin nor boarded, which implies they must be served), the heuristic is 0.
    8. If there are required floors:
       - Get the integer floor numbers for the current elevator floor and all required floors using the mapping created during initialization.
       - Calculate the movement cost: This is the minimum number of floor movements required to visit all required floors starting from the current floor. A lower bound is calculated as the distance from the current floor to the closest extreme required floor (min or max) plus the distance between the min and max required floors.
         Movement Cost = min(|current_floor_int - min_req_floor_int|, |current_floor_int - max_req_floor_int|) + (max_req_floor_int - min_req_floor_int).
         If the floor mapping is based on all floors being level 1 (single-floor case), this calculation correctly results in 0 movement cost.
       - Calculate the action cost: This is the estimated number of 'board' and 'depart' actions needed for the unserved passengers. Each unpicked passenger requires one 'board' and one 'depart' action (total 2). Each boarded passenger requires one 'depart' action (total 1).
         Action Cost = (Number of unpicked passengers * 2) + (Number of boarded passengers * 1).
    9. The total heuristic value is the sum of the movement cost and the action cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        # self.goals = task.goals # Not directly used in this heuristic calculation
        static_facts = task.static

        # 1. Build floor_to_int mapping from 'above' facts
        floors_below_count = {}
        all_floors_from_above = set()
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) > 2:
                    f1, f2 = parts[1:]
                    all_floors_from_above.add(f1)
                    all_floors_from_above.add(f2)
                    floors_below_count[f2] = floors_below_count.get(f2, 0) + 1

        self.floor_to_int = {}
        if all_floors_from_above:
            # Assign integer floor numbers based on the count of floors below
            for floor in all_floors_from_above:
                 self.floor_to_int[floor] = floors_below_count.get(floor, 0) + 1
        # else: If no 'above' facts, self.floor_to_int remains empty initially.

        # 2. Store destination floor for each passenger from 'destin' facts
        self.passenger_to_destin = {}
        self.all_passengers = set()
        all_floors_from_destin = set() # Also collect floors from destin
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) > 2:
                    p, d = parts[1:]
                    if p and d: # Ensure parts are not empty strings
                        self.passenger_to_destin[p] = d
                        self.all_passengers.add(p)
                        all_floors_from_destin.add(d)

        # If floor_to_int is still empty after processing 'above' facts,
        # but there are floors mentioned in 'destin' facts, assume they are all on level 1.
        # This handles single-floor problems or problems missing 'above' facts where floors exist.
        if not self.floor_to_int and all_floors_from_destin:
             for floor in all_floors_from_destin:
                 self.floor_to_int[floor] = 1

        # Note: Floors mentioned *only* in 'origin' facts in the initial state
        # might not be included in self.floor_to_int if they are not in 'above' or 'destin'.
        # This could lead to a KeyError if such a floor becomes a required stop.
        # Assuming valid miconic problems define all relevant floors in static facts.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        current_floor = None
        served_passengers = set()
        passengers_at_origin_in_state = {} # {passenger: floor}
        boarded_passengers_in_state = set() # {passenger}

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]

            if predicate == "lift-at" and len(parts) > 1:
                current_floor = parts[1]
            elif predicate == "origin" and len(parts) > 2:
                p, f = parts[1:]
                passengers_at_origin_in_state[p] = f
            elif predicate == "boarded" and len(parts) > 1:
                p = parts[1]
                boarded_passengers_in_state.add(p)
            elif predicate == "served" and len(parts) > 1:
                p = parts[1]
                served_passengers.add(p)

        # Identify unserved passengers and their status (at origin or boarded)
        unserved_passengers_at_origin_count = 0
        unserved_boarded_passengers_count = 0
        pickup_floors_for_unserved = set()
        dropoff_floors_for_unserved = set()

        # Iterate through all passengers we know about from static destin facts
        # Only consider those not yet served.
        unserved_passengers = self.all_passengers - served_passengers

        for p in unserved_passengers:
            # Check if this unserved passenger is at their origin in the current state
            if p in passengers_at_origin_in_state:
                unserved_passengers_at_origin_count += 1
                pickup_floors_for_unserved.add(passengers_at_origin_in_state[p])
            # Check if this unserved passenger is boarded in the current state
            elif p in boarded_passengers_in_state:
                unserved_boarded_passengers_count += 1
                # Add destination floor to dropoff floors
                if p in self.passenger_to_destin:
                    dropoff_floors_for_unserved.add(self.passenger_to_destin[p])
                # else: Passenger is boarded but has no static destination? Malformed problem.
                #      Ignoring this passenger for heuristic calculation.


        # Required floors are the union of origin floors for unpicked and destin floors for boarded
        required_floors = pickup_floors_for_unserved.union(dropoff_floors_for_unserved)

        # If no passengers need service or are in transit requiring elevator stops, heuristic is 0.
        if not required_floors:
            return 0

        # Calculate movement cost
        # Ensure current_floor is known and in the floor mapping.
        # Ensure all required floors are in the floor mapping.
        # If not, it's a malformed problem instance relative to the static definition or state.
        # Fallback to action cost only if floor mapping is incomplete or current_floor is missing.
        if current_floor is None or current_floor not in self.floor_to_int or \
           not all(f in self.floor_to_int for f in required_floors):
             # Fallback to action cost only
             return unserved_passengers_at_origin_count * 2 + unserved_boarded_passengers_count * 1


        min_req_floor_int = min(self.floor_to_int[f] for f in required_floors)
        max_req_floor_int = max(self.floor_to_int[f] for f in required_floors)
        current_floor_int = self.floor_to_int[current_floor]

        # Movement cost: distance to closest extreme + distance between extremes
        movement_cost = min(abs(current_floor_int - min_req_floor_int), abs(current_floor_int - max_req_floor_int)) + (max_req_floor_int - min_req_floor_int)

        # Calculate action cost
        action_cost = unserved_passengers_at_origin_count * 2 + unserved_boarded_passengers_count * 1

        # Total heuristic is the sum
        total_cost = movement_cost + action_cost

        return total_cost
