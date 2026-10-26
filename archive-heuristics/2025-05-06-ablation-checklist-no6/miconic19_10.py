from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


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


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the cost as the sum of:
    1. Minimum moves required to visit all floors where pickups or dropoffs are needed.
    2. Number of board actions needed (one for each waiting passenger).
    3. Number of depart actions needed (one for each unserved passenger).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build floor order and mapping
        above_facts = [get_parts(fact) for fact in static_facts if match(fact, "above", "*", "*")]
        floors = set()
        floors_below = {} # map floor -> floor below it
        floors_above = {} # map floor -> floor above it

        for _, f_lower, f_higher in above_facts:
            floors.add(f_lower)
            floors.add(f_higher)
            floors_above[f_lower] = f_higher
            floors_below[f_higher] = f_lower

        self.floor_order = []
        self.floor_to_index = {}

        if floors:
            # Find the lowest floor (a floor that is not a 'higher' floor in any 'above' fact)
            all_higher_floors = set(floors_below.keys())
            lowest_floor = None
            # Iterate through all floors found in 'above' facts
            for floor in floors:
                 # If a floor is never the 'higher' floor in an 'above' fact, it must be the lowest
                 if floor not in all_higher_floors:
                     lowest_floor = floor
                     break

            # Build the ordered list of floors starting from the lowest
            if lowest_floor is not None:
                current_floor = lowest_floor
                while current_floor is not None:
                    self.floor_order.append(current_floor)
                    current_floor = floors_above.get(current_floor)

                # Map floor name to index
                self.floor_to_index = {floor: i for i, floor in enumerate(self.floor_order)}
            # Note: If floors is not empty but lowest_floor is None, it implies an invalid
            # 'above' structure (e.g., cycle, disconnected). Assuming valid miconic PDDL.
            # If floors is empty, floor_order and floor_to_index remain empty,
            # which is correct for problems with no floors or only one floor (no above facts).


        # Store destination floors for all passengers
        self.passenger_destinations = {}
        # Also collect all passenger names mentioned in static facts or goals
        self.all_passengers = set()
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.passenger_destinations[passenger] = floor
                 self.all_passengers.add(passenger)

        # Collect passengers from goals as well
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Check if goal is reached
        # The goal is a frozenset of facts. Check if the state contains all goal facts.
        if self.goals <= state:
            return 0

        # Get current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # If lift location is unknown, something is wrong with the state or domain.
        # Assuming lift-at is always present in a valid state.
        # If there are no floors (e.g., trivial problem), index lookup might fail.
        # Handle case with no floors gracefully (heuristic is 0 if no unserved passengers).
        if not self.floor_to_index:
             # This case should only happen if there are no floors or only one floor
             # and no 'above' facts. If there are unserved passengers, they must
             # all be on the single floor. Moves = 0.
             current_lift_index = 0 # Dummy index if no floors defined by 'above'
        else:
             current_lift_index = self.floor_to_index[current_lift_floor]


        # Identify unserved passengers and count waiting/unserved
        num_waiting = 0
        num_unserved = 0
        floors_needing_stop_indices = set()

        for passenger in self.all_passengers:
            # Check if passenger is served
            if f"(served {passenger})" not in state:
                num_unserved += 1

                is_waiting = False
                f_origin = None
                # Check if passenger is waiting and get origin floor
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        _, p, f_origin = get_parts(fact)
                        num_waiting += 1
                        is_waiting = True
                        # Add origin floor to floors needing stop
                        if f_origin in self.floor_to_index:
                            floors_needing_stop_indices.add(self.floor_to_index[f_origin])
                        # else: Origin floor not in defined floors? Malformed problem.
                        break # Found origin, passenger is waiting

                # If passenger is waiting or boarded, their destination floor needs a stop
                # Check if passenger is boarded
                is_boarded = f"(boarded {passenger})" in state

                if is_waiting or is_boarded:
                     # Get destination floor (should be in self.passenger_destinations)
                     f_destin = self.passenger_destinations.get(passenger)
                     if f_destin and f_destin in self.floor_to_index:
                         floors_needing_stop_indices.add(self.floor_to_index[f_destin])
                     # else: Destination floor not in defined floors or destination unknown? Malformed problem.


        # Estimate moves required to visit all necessary floors
        estimated_moves = 0
        if floors_needing_stop_indices:
            min_stop_index = min(floors_needing_stop_indices)
            max_stop_index = max(floors_needing_stop_indices)

            # Minimum moves to visit all floors in the range [min_stop_index, max_stop_index]
            # starting from current_lift_index.
            # This is the distance to the closer end of the range plus the range size.
            range_size = max_stop_index - min_stop_index
            dist_to_min = abs(current_lift_index - min_stop_index)
            dist_to_max = abs(current_lift_index - max_stop_index)

            # The lift must travel from current to either min or max, then sweep the range.
            # Option 1: current -> min -> max. Moves = dist_to_min + range_size
            # Option 2: current -> max -> min. Moves = dist_to_max + range_size
            estimated_moves = min(dist_to_min, dist_to_max) + range_size


        # Total heuristic = estimated moves + board actions + depart actions
        # Board actions needed = number of waiting passengers
        # Depart actions needed = number of unserved passengers
        h = estimated_moves + num_waiting + num_unserved

        return h
