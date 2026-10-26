from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Or raise an error, depending on expected input robustness
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
    passengers. It counts the number of necessary board and depart actions
    and adds an estimate of the minimum travel cost required to visit all
    relevant floors (origins for waiting passengers, destinations for boarded passengers).

    # Assumptions
    - Floors are linearly ordered as defined by the `above` predicate.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The lift can carry multiple passengers.

    # Heuristic Initialization
    - Extracts passenger destinations from static facts.
    - Builds a mapping from floor names to numerical floor levels based on the `above` predicate.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all passengers and their goal destinations (from initialization).
    2. Determine which passengers are not yet served by checking the current state.
    3. For unserved passengers, determine if they are waiting at their origin or are already boarded by checking the current state.
    4. Count the number of waiting passengers (each needs a 'board' action).
    5. Count the total number of unserved passengers (each needs a 'depart' action).
    6. Identify the set of 'required floors':
       - For each waiting passenger, their origin floor is a required stop.
       - For each boarded passenger, their destination floor is a required stop.
    7. If there are no unserved passengers, the heuristic is 0.
    8. If there are unserved passengers, find the lift's current floor from the current state.
    9. Map the current lift floor and all required floors to their numerical levels using the pre-calculated floor mapping.
    10. Find the minimum and maximum numerical levels among the required floors.
    11. Estimate the travel cost: This is the minimum distance the lift must travel to visit the two extreme required floors. This distance is calculated as `min(abs(current_floor_num - min_req_floor_num), abs(current_floor_num - max_req_floor_num)) + abs(max_req_floor_num - min_req_floor_num)`.
    12. The total heuristic value is the sum of:
        - Number of waiting passengers (for board actions).
        - Number of unserved passengers (for depart actions).
        - Estimated travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.static = task.static

        # 1. Extract passenger destinations from static facts
        self.destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destinations[passenger] = floor

        # 2. Build floor mapping from above facts
        # below_map: {floor_above: floor_below} derived from (above floor_above floor_below)
        below_map = {}
        all_floors = set()

        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, floor_above, floor_below = get_parts(fact)
                below_map[floor_above] = floor_below
                all_floors.add(floor_above)
                all_floors.add(floor_below)

        self.floor_to_num = {}
        if not all_floors:
            pass # floor_to_num remains empty
        elif len(all_floors) == 1 and not below_map:
             # Single floor case
             self.floor_to_num = {list(all_floors)[0]: 1}
        else:
            # Find the lowest floor: the one that is not a key in below_map
            lowest_floor = None
            floors_that_are_above_others = set(below_map.keys())
            for floor in all_floors:
                 if floor not in floors_that_are_above_others:
                     lowest_floor = floor
                     break

            if lowest_floor is None:
                 # This case should ideally not happen in a valid multi-floor linear domain
                 # with multiple floors, implies a cycle or single floor not handled
                 # by the 'not in below_map.keys()' check if it's the only floor.
                 # If all_floors has one element and it's not in below_map.keys()
                 # (because below_map is empty), it's the lowest.
                 if len(all_floors) == 1:
                      lowest_floor = list(all_floors)[0]
                 else:
                      # Handle potential error or return empty map
                      print("Warning: Could not determine lowest floor from 'above' facts.")
                      return # floor_to_num remains empty


            # Build map starting from the lowest floor
            current_f = lowest_floor
            current_num = 1
            # Need the map from floor_below to floor_above to traverse upwards
            above_map = {v: k for k, v in below_map.items()}

            while current_f:
                self.floor_to_num[current_f] = current_num
                # Find the floor that is immediately above current_f
                next_f = above_map.get(current_f)
                current_f = next_f
                current_num += 1


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        all_passengers = set(self.destinations.keys())
        unserved_passengers = {p for p in all_passengers if p not in served_passengers}

        # If all passengers are served, the heuristic is 0
        if not unserved_passengers:
            return 0

        # 2. Identify waiting and boarded passengers among the unserved
        waiting_passengers = set()
        boarded_passengers = set()
        passenger_origins = {} # Store origins for waiting passengers

        for fact in state:
            if match(fact, "origin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                if passenger in unserved_passengers:
                    waiting_passengers.add(passenger)
                    passenger_origins[passenger] = floor
            elif match(fact, "boarded", "*"):
                _, passenger = get_parts(fact)
                if passenger in unserved_passengers:
                    boarded_passengers.add(passenger)

        # 3. Count board and depart actions needed
        num_board_actions = len(waiting_passengers)
        num_depart_actions = len(unserved_passengers) # Each unserved passenger needs one depart action

        # 4. Identify required floors
        required_floors = set()
        for p in waiting_passengers:
            # Ensure origin is known (it should be in state if waiting)
            if p in passenger_origins:
                required_floors.add(passenger_origins[p])
        for p in boarded_passengers:
            # Ensure destination is known (it should be in self.destinations)
            if p in self.destinations:
                 required_floors.add(self.destinations[p])

        # If there are unserved passengers but no required floors found, something is wrong
        # (e.g., passenger origin/destin not in floor list from 'above' facts)
        if not required_floors:
             # This implies unserved_passengers > 0 but no required floors were identified.
             # This state might be unreachable or indicates a problem parsing the state/goals.
             # Return a high value.
             return float('inf')


        # 5. Find current lift floor
        current_lift_f = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_f = get_parts(fact)
                break

        # Should always find the lift location in a valid state
        if current_lift_f is None:
             # This indicates an invalid state representation, return infinity
             return float('inf')

        # 6. Map floors to numbers and calculate travel cost
        current_lift_f_num = self.floor_to_num.get(current_lift_f)

        # If current lift floor is not in our mapping, something is wrong
        if current_lift_f_num is None:
             return float('inf')

        required_floor_nums = {self.floor_to_num.get(f) for f in required_floors if f in self.floor_to_num}

        # If required floors could not be mapped (e.g., floor names not in static facts)
        if len(required_floor_nums) != len(required_floors):
             # Some required floors are not in our mapping. Problem parsing or malformed input.
             return float('inf') # Cannot estimate travel

        min_req_floor_num = min(required_floor_nums)
        max_req_floor_num = max(required_floor_nums)

        # Calculate travel cost: min distance to reach the range of required floors
        # and traverse the range.
        # This is the distance from the current floor to the nearest extreme required floor,
        # plus the distance between the two extreme required floors.
        travel_cost = min(abs(current_lift_f_num - min_req_floor_num),
                          abs(current_lift_f_num - max_req_floor_num)) + \
                      abs(max_req_floor_num - min_req_floor_num)

        # 7. Total heuristic cost
        total_cost = num_board_actions + num_depart_actions + travel_cost

        return total_cost
