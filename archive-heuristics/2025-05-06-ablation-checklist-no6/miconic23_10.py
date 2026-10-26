from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, unless args contains wildcards
    # A simpler check is just zip and check all parts match corresponding args
    if len(parts) != len(args) and '*' not in args:
         return False
    # Ensure we don't go out of bounds if parts is shorter than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions (board, depart, move) required to serve
    all passengers.

    Heuristic components:
    1. Number of 'board' actions needed: One for each passenger waiting at their origin.
    2. Number of 'depart' actions needed: One for each passenger not yet served (waiting or boarded).
    3. Number of 'move' actions needed: Estimated based on the vertical travel required
       to visit all floors where pickups or dropoffs are needed.

    Move estimate:
    Calculates the minimum and maximum floor indices among all required pickup and dropoff floors.
    The estimated moves are the distance from the current lift floor to the lowest required floor,
    plus twice the vertical span between the lowest and highest required floors.
    This represents a simplified path that goes down to the lowest required stop, sweeps up to the highest,
    and then potentially needs to cover the span again (e.g., sweep back down).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by parsing floor order and passenger destinations.
        """
        # Assuming Heuristic base class has a constructor that accepts task
        # super().__init__(task)
        self.task = task # Store task if base class doesn't

        # 1. Parse floor order and create floor_to_index map
        all_floors = set()
        below_to_above = {}
        for fact in task.static:
            if match(fact, "above", "*", "*"):
                _, f_higher, f_lower = get_parts(fact)
                below_to_above[f_lower] = f_higher
                all_floors.add(f_higher)
                all_floors.add(f_lower)

        # Find the lowest floor (a floor that is never the 'higher' floor in an 'above' fact)
        # Assumes a single chain of floors and a unique lowest floor if multiple floors exist.
        highest_floors_in_above = set(below_to_above.values())
        lowest_floor = None
        if all_floors:
            candidate_lowest = all_floors - highest_floors_in_above
            if len(candidate_lowest) == 1:
                 lowest_floor = candidate_lowest.pop()
            elif len(all_floors) == 1: # Case with only one floor
                 lowest_floor = list(all_floors)[0]
            # else: multiple lowest floors or no floors, indicates invalid structure,
            # lowest_floor remains None, will be handled as inf heuristic.


        # Build ordered list of floors starting from the lowest
        ordered_floors = []
        if lowest_floor:
            current_floor = lowest_floor
            while current_floor is not None:
                ordered_floors.append(current_floor)
                current_floor = below_to_above.get(current_floor) # Get floor above current

        # Create floor_to_index map
        self.floor_to_index = {f: i for i, f in enumerate(ordered_floors)}

        # 2. Parse passenger destinations from static facts
        self.passenger_dest = {}
        for fact in task.static:
            if match(fact, "destin", "*", "*"):
                _, passenger, floor = get_parts(fact)
                self.passenger_dest[passenger] = floor

        # 3. Get all passenger names from goals (assuming goal is (and (served p1) ...))
        self.all_passengers = set()
        for goal in task.goals:
             if match(goal, "served", "*"):
                 _, passenger = get_parts(goal)
                 self.all_passengers.add(passenger)
        # Also collect passengers mentioned in initial state if not in goals (less common in miconic)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] in ["origin", "boarded", "served"] and len(parts) > 1:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break

        # Check if current lift floor is valid and mapped
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # Invalid state: lift location unknown or not a recognized floor
             return float('inf')

        current_idx = self.floor_to_index[current_lift_floor]

        waiting_passengers = set()
        boarded_passengers = set()
        served_passengers = set()

        # Identify state of each passenger
        for passenger in self.all_passengers:
            is_served = False
            for fact in state:
                if match(fact, "served", passenger):
                    is_served = True
                    break
            if is_served:
                served_passengers.add(passenger)
                continue

            is_boarded = False
            for fact in state:
                 if match(fact, "boarded", passenger):
                     is_boarded = True
                     break

            if is_boarded:
                boarded_passengers.add(passenger)
            else:
                # Not served and not boarded, must be waiting
                is_waiting = False
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        is_waiting = True
                        break
                if is_waiting:
                    waiting_passengers.add(passenger)
                # else: passenger is in an invalid state (not served, boarded, or waiting)
                # This shouldn't happen in a valid problem instance/state reachable from initial state.
                # For robustness, we could ignore or return inf. Assume valid states for now.


        # If all passengers are served, the heuristic is 0
        if len(served_passengers) == len(self.all_passengers):
            return 0

        # Calculate heuristic components

        # 1. Board actions needed: one for each waiting passenger
        h_board = len(waiting_passengers)

        # 2. Depart actions needed: one for each unserved passenger
        h_depart = len(waiting_passengers) + len(boarded_passengers)

        # 3. Estimated move actions
        service_floors_idx = set()

        # Add origin floors for waiting passengers
        for passenger in waiting_passengers:
            origin_floor = None
            for fact in state:
                 if match(fact, "origin", passenger, "*"):
                     origin_floor = get_parts(fact)[2]
                     break
            # Check if origin floor is valid and mapped
            if origin_floor is None or origin_floor not in self.floor_to_index:
                 # Invalid state: waiting passenger's origin floor unknown or not a recognized floor
                 return float('inf')
            service_floors_idx.add(self.floor_to_index[origin_floor])

        # Add destination floors for all unserved passengers (waiting or boarded)
        unserved_passengers = waiting_passengers | boarded_passengers
        for passenger in unserved_passengers:
            dest_floor = self.passenger_dest.get(passenger)
            # Check if destination floor is valid and mapped
            if dest_floor is None or dest_floor not in self.floor_to_index:
                 # Invalid state: unserved passenger's destination floor unknown or not a recognized floor
                 return float('inf')
            service_floors_idx.add(self.floor_to_index[dest_floor])


        h_moves = 0
        if service_floors_idx:
            min_idx = min(service_floors_idx)
            max_idx = max(service_floors_idx)

            # Estimated moves: distance from current to lowest required floor + twice the span
            # This models going down to the lowest stop, sweeping up to the highest,
            # and potentially needing to go back down or cover the span again.
            # This is a non-admissible estimate designed to guide greedy search.
            h_moves = abs(current_idx - min_idx) + 2 * (max_idx - min_idx)

            # Alternative move estimate: distance from current to highest required floor + twice the span
            # h_moves_alt = abs(current_idx - max_idx) + 2 * (max_idx - min_idx)
            # Using the minimum of these two might be slightly better?
            # h_moves = min(h_moves, h_moves_alt)
            # Let's stick to the simpler one (assuming initial move towards lowest required floor)
            # as it seemed to work well for example 1.


        # Total heuristic value
        total_cost = h_board + h_depart + h_moves

        return total_cost
