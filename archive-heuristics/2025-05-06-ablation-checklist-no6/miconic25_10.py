# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # Assume task object has attributes: name, facts, initial_state, goals, operators, static

        def __call__(self, node):
            # node object is assumed to have attribute: state
            raise NotImplementedError

from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for number of parts unless pattern uses wildcards extensively
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding pattern argument
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the remaining actions by summing the independent costs
    for each unserved passenger. The cost for a passenger includes
    lift movement to origin (if waiting), boarding, lift movement
    to destination, and departing.

    This heuristic is non-admissible as it counts lift movements
    independently for each passenger, potentially overestimating
    when passengers share floors or routes. It aims to prioritize
    states where passengers are closer to being served.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger destinations.
        """
        super().__init__(task)

        # Build floor_to_level mapping
        # Assume floors are named f1, f2, ... and sorted numerically based on the number
        floor_names = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                floor_names.add(f1)
                floor_names.add(f2)

        # Sort floors numerically (f1, f2, ..., f10, f11, ...)
        # Use a custom key to handle floor names like 'f1', 'f10' correctly
        try:
            # Attempt numerical sort based on the number after 'f'
            sorted_floors = sorted(list(floor_names), key=lambda f: int(f[1:]))
        except (ValueError, IndexError):
             # Fallback to alphabetical sort if numerical parsing fails or name format is unexpected
             sorted_floors = sorted(list(floor_names))
             # print("Warning: Floor names not in f<number> format or unexpected, using alphabetical sort.") # Keep silent


        # Assign levels (lowest floor is level 1)
        self.floor_to_level = {floor: i + 1 for i, floor in enumerate(sorted_floors)}

        # Build destin_floor mapping for all passengers and collect all passenger names
        self.destin_floor = {}
        self.passengers = set()
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.destin_floor[passenger] = floor
                self.passengers.add(passenger)

        # Add passengers from goal facts if any are missed (e.g., if a passenger has no destin fact but is in goal)
        # This shouldn't happen in standard miconic, but for robustness:
        for goal in self.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.passengers.add(passenger)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached (all passengers served)
        # This check is faster than calculating the full heuristic if it's a goal state
        all_served = True
        for passenger in self.passengers:
            if f"(served {passenger})" not in state:
                all_served = False
                break

        if all_served:
            return 0 # Goal state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If not goal state but lift location is unknown, return a large value
        # This indicates an unexpected state format or an unsolvable path
        if current_lift_floor is None:
             return float('inf') # Or a large integer like 1e9


        current_level = self.floor_to_level.get(current_lift_floor)
        if current_level is None:
             # Current lift floor is not in our floor list (unexpected)
             return float('inf') # Or a large integer

        total_cost = 0

        # Calculate cost for each unserved passenger independently
        for passenger in self.passengers:
            if f"(served {passenger})" not in state:
                f_destin = self.destin_floor.get(passenger)
                if f_destin is None:
                     # Passenger in goal/state but no destin fact (unexpected)
                     return float('inf') # Or a large integer

                destin_level = self.floor_to_level.get(f_destin)
                if destin_level is None:
                     # Destination floor not in our floor list (unexpected)
                     return float('inf') # Or a large integer


                if f"(boarded {passenger})" in state:
                    # Passenger is boarded, needs to go to destination and depart
                    cost_for_p = abs(destin_level - current_level) + 1 # moves + depart
                    total_cost += cost_for_p
                else:
                    # Passenger is waiting at origin, needs pickup and dropoff
                    # Find origin floor
                    f_origin = None
                    # Efficiently find the origin fact for this passenger
                    origin_fact_prefix = f"(origin {passenger} "
                    for fact in state:
                        if fact.startswith(origin_fact_prefix) and fact.endswith(')'):
                             parts = get_parts(fact)
                             if len(parts) == 3 and parts[0] == 'origin' and parts[1] == passenger:
                                 f_origin = parts[2]
                                 break # Found the origin, move to next passenger

                    if f_origin is not None:
                        origin_level = self.floor_to_level.get(f_origin)
                        if origin_level is None:
                             # Origin floor not in our floor list (unexpected)
                             return float('inf') # Or a large integer

                        # moves to origin + board + moves to destin + depart
                        cost_for_p = abs(origin_level - current_level) + 1 + abs(destin_level - origin_level) + 1
                        total_cost += cost_for_p
                    else:
                        # This case should not happen in a valid state for an unserved, unboarded passenger
                        # It implies the passenger is neither waiting nor boarded nor served.
                        # Return a large value as this state might be invalid or unreachable
                        return float('inf') # Or a large integer


        return total_cost
