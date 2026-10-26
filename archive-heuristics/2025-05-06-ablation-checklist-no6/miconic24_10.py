import sys
from fnmatch import fnmatch
# Assuming Heuristic base class is available from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the Heuristic base class if it's not provided in the execution environment
# This is a placeholder based on the examples provided.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError("Heuristic must implement __call__")


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost as the sum of:
    1. Minimum movement cost for the lift to cover the range of floors
       with pending pickup or dropoff tasks, starting from the current floor.
    2. Number of board actions needed (one for each waiting passenger).
    3. Number of depart actions needed (one for each unserved passenger).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor ordering to create a floor name -> number mapping.
        - Passenger destinations.
        - List of all passengers relevant to the goal.
        """
        super().__init__(task)

        # Build floor mapping from static 'above' facts
        self.floor_to_num = {}
        self.num_to_floor = {}
        above_facts_parts = [get_parts(fact) for fact in self.static if match(fact, "above", "*", "*")]

        all_floors = set()
        floors_that_are_above_something = set() # These are the second argument in (above f_higher f_lower)
        floors_that_are_below_something = set() # These are the first argument in (above f_higher f_lower)

        for parts in above_facts_parts:
            f_higher, f_lower = parts[1], parts[2]
            all_floors.add(f_higher)
            all_floors.add(f_lower)
            floors_that_are_below_something.add(f_higher)
            floors_that_are_above_something.add(f_lower)

        # The lowest floor is one that is not the second argument of any 'above' fact.
        # i.e., no floor is above it.
        possible_lowest_floors = all_floors - floors_that_are_above_something

        if not all_floors:
             # No floors defined
             pass # No floors to map
        elif len(all_floors) == 1: # Single floor
            floor = list(all_floors)[0]
            self.floor_to_num[floor] = 1
            self.num_to_floor[1] = floor
        elif not possible_lowest_floors:
             # No ordering information or cannot determine lowest floor reliably.
             # Fallback: Use alphabetical order as floor numbers.
             # print("Warning: No 'above' facts or cannot determine lowest floor. Using alphabetical order.")
             sorted_floors = sorted(list(all_floors))
             for i, floor in enumerate(sorted_floors):
                 self.floor_to_num[floor] = i + 1
                 self.num_to_floor[i + 1] = floor

        else:
            # Assuming there is exactly one lowest floor in a valid problem
            # If multiple, pick the alphabetically first one.
            lowest_floor = sorted(list(possible_lowest_floors))[0]

            current_floor = lowest_floor
            current_num = 1
            self.floor_to_num[current_floor] = current_num
            self.num_to_floor[current_num] = current_floor

            # Build the rest of the mapping by following the 'above' chain upwards
            while True:
                # Find the floor directly above the current_floor
                next_floor = None
                for parts in above_facts_parts:
                    f_higher, f_lower = parts[1], parts[2]
                    if f_lower == current_floor:
                        next_floor = f_higher
                        break # Assuming only one floor is directly above another

                if next_floor and next_floor not in self.floor_to_num:
                    current_num += 1
                    self.floor_to_num[next_floor] = current_num
                    self.num_to_floor[current_num] = next_floor
                    current_floor = next_floor
                else:
                    break # Reached the top floor or a floor already mapped

        # Check if all floors were mapped. If not, there's an issue with 'above' facts (disconnected/cycle).
        # Fallback to alphabetical order if mapping is incomplete.
        if len(self.floor_to_num) != len(all_floors):
             # print("Warning: Floor ordering graph is disconnected or cyclic. Using alphabetical order.")
             self.floor_to_num = {}
             self.num_to_floor = {}
             sorted_floors = sorted(list(all_floors))
             for i, floor in enumerate(sorted_floors):
                 self.floor_to_num[floor] = i + 1
                 self.num_to_floor[i + 1] = floor


        # Extract passenger destinations and list of all passengers relevant to the goal
        self.passenger_destins = {}
        self.all_passengers = set()

        # Passengers relevant to the goal are those mentioned in the goal (served)
        # and those with a destination defined in static facts.
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.all_passengers.add(passenger)

        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_destins[passenger] = floor
                self.all_passengers.add(passenger) # Add passenger if they have a destination


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check if goal is reached
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        if served_passengers == self.all_passengers:
            return 0 # Goal state

        # 2. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid miconic problem state
             return sys.maxsize # Indicate a bad state


        current_lift_num = self.floor_to_num.get(current_lift_floor)
        if current_lift_num is None:
             # Should not happen if floor mapping is correct and state is valid
             return sys.maxsize


        # 3. Identify unserved passengers and pending tasks
        unserved_passengers = self.all_passengers - served_passengers
        pickup_floors = set()
        dropoff_floors = set()
        n_waiting = 0
        n_boarded = 0

        for passenger in unserved_passengers:
            is_waiting = False
            is_boarded = False
            origin_floor = None

            # Check state for origin or boarded status
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    is_waiting = True
                    origin_floor = get_parts(fact)[2]
                    break
                if match(fact, "boarded", passenger):
                    is_boarded = True
                    break

            if is_waiting:
                if origin_floor:
                    pickup_floors.add(origin_floor)
                    n_waiting += 1
                else:
                    # Should not happen in a valid state
                    return sys.maxsize
            elif is_boarded:
                # Passenger is boarded, needs to go to destination
                destin_floor = self.passenger_destins.get(passenger)
                if destin_floor:
                    dropoff_floors.add(destin_floor)
                    n_boarded += 1
                else:
                    # Should not happen in a valid problem (passenger boarded but no destination?)
                    return sys.maxsize
            # else: passenger is unserved but neither waiting nor boarded?
            # This shouldn't happen in a valid state progression.
            # A passenger is either (origin ?p ?f), (boarded ?p), or (served ?p).
            # If unserved, they must be either waiting or boarded.
            # If this case occurs, it suggests an invalid state.
            # We could ignore them or return inf. Let's assume valid states.


        # 4. Calculate movement cost
        required_floors = pickup_floors | dropoff_floors

        if not required_floors:
            # If no required floors, all unserved passengers must be boarded
            # and already at their destination, just needing to depart.
            # Movement cost is 0.
            movement_cost = 0
        else:
            required_floor_nums = set()
            for f in required_floors:
                num = self.floor_to_num.get(f)
                if num is None:
                    # Should not happen if floor mapping is complete
                    return sys.maxsize
                required_floor_nums.add(num)

            min_req_num = min(required_floor_nums)
            max_req_num = max(required_floor_nums)

            # Movement cost is the distance from the lowest relevant floor
            # to the highest relevant floor, including the current lift floor.
            # This is abs(max(curr, max_req) - min(curr, min_req))
            movement_cost = abs(max(current_lift_num, max_req_num) - min(current_lift_num, min_req_num))


        # 5. Calculate action cost
        # Each waiting passenger needs 1 board action.
        # Each unserved passenger (waiting or boarded) needs 1 depart action.
        # Total actions = N_waiting (board) + N_unserved (depart)
        # N_unserved = n_waiting + n_boarded
        action_cost = n_waiting + (n_waiting + n_boarded) # Board + Depart
        # Simplified: action_cost = 2 * n_waiting + n_boarded

        # Note: If required_floors is empty, n_waiting must be 0.
        # In that case, action_cost = 2 * 0 + n_boarded = n_boarded.
        # This correctly reflects that only depart actions are needed for boarded passengers
        # who are already at their destination.
        # So the formula `action_cost = 2 * n_waiting + n_boarded` is generally correct.


        # 6. Total heuristic cost
        total_cost = movement_cost + action_cost

        return total_cost
