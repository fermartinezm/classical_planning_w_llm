# Need to import fnmatch for pattern matching
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic # This import is assumed

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts is at least the number of args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    Estimates the number of actions required to serve all passengers.
    The heuristic considers:
    1. The number of board actions needed (for unboarded passengers).
    2. The number of depart actions needed (for boarded passengers).
    3. The estimated lift travel distance to visit all necessary floors
       (origins of unboarded, destinations of boarded).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Floor ordering and mapping from floor name to index.
        - Origin and destination floors for each passenger.
        """
        super().__init__(task)

        # 1. Parse floor ordering from 'above' facts
        # Assuming floors are named f1, f2, ... and are ordered numerically.
        # We need to extract all floor names and sort them.
        floor_names = set()
        for fact in self.static:
            if match(fact, "above", "*", "*"):
                _, f1, f2 = get_parts(fact)
                floor_names.add(f1)
                floor_names.add(f2)

        # Sort floor names numerically (e.g., f1, f2, f10)
        # This assumes the format "f" followed by a number.
        def floor_sort_key(floor_name):
            try:
                # Extract the number part after 'f'
                return int(floor_name[1:])
            except (ValueError, IndexError):
                # Handle cases that don't fit f<number> pattern if necessary.
                # Returning a large value puts non-standard names at the end.
                # Based on domain examples, this case might not occur.
                return float('inf')

        self.sorted_floors = sorted(list(floor_names), key=floor_sort_key)
        self.floor_to_index = {floor: i for i, floor in enumerate(self.sorted_floors)}

        # 2. Parse origin and destination for each passenger from static facts
        self.passenger_info = {} # passenger -> (origin_floor, destin_floor)
        for fact in self.static:
            if match(fact, "origin", "*", "*"):
                _, passenger, origin_floor = get_parts(fact)
                if passenger not in self.passenger_info:
                    self.passenger_info[passenger] = [None, None]
                self.passenger_info[passenger][0] = origin_floor
            elif match(fact, "destin", "*", "*"):
                _, passenger, destin_floor = get_parts(fact)
                if passenger not in self.passenger_info:
                    self.passenger_info[passenger] = [None, None]
                self.passenger_info[passenger][1] = destin_floor

        # Convert list to tuple for immutability
        for p, info in self.passenger_info.items():
             self.passenger_info[p] = tuple(info)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Find current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # This should always be found in a valid state for this domain
        if current_lift_floor is None:
             # Cannot compute heuristic without lift location.
             # Return infinity or a very large number to prune this path.
             return float('inf')

        current_lift_index = self.floor_to_index[current_lift_floor]

        num_board_actions = 0
        num_depart_actions = 0
        floors_to_visit_indices = set()

        # Iterate through all passengers known from static facts
        for passenger, (origin_floor, destin_floor) in self.passenger_info.items():
            # Check if the passenger is already served
            if f"(served {passenger})" in state:
                continue # This passenger is done

            # Passenger is not served. Calculate their contribution.
            origin_index = self.floor_to_index[origin_floor]
            destin_index = self.floor_to_index[destin_floor]

            # Check if the passenger is boarded
            if f"(boarded {passenger})" in state:
                # Passenger is boarded, needs to depart at destination
                num_depart_actions += 1
                floors_to_visit_indices.add(destin_index)
            else:
                # Passenger is not boarded, needs to board at origin and then depart at destination
                num_board_actions += 1
                num_depart_actions += 1 # They will need to depart eventually
                floors_to_visit_indices.add(origin_index)

        # Calculate estimated lift travel cost
        travel_cost = 0
        if floors_to_visit_indices:
            min_floor_needed_index = min(floors_to_visit_indices)
            max_floor_needed_index = max(floors_to_visit_indices)

            # Estimate travel as the distance to cover the range [min_needed, max_needed]
            # starting from the current floor.
            # This is the size of the minimal segment covering current floor and all needed floors.
            travel_cost = max(current_lift_index, max_floor_needed_index) - min(current_lift_index, min_floor_needed_index)

        # Total heuristic is the sum of necessary actions (board/depart) and estimated travel
        total_heuristic = num_board_actions + num_depart_actions + travel_cost

        # The heuristic is 0 iff all passengers are served (checked this logic earlier).
        # If there are no unserved passengers, num_board_actions=0, num_depart_actions=0,
        # floors_to_visit_indices is empty, travel_cost=0. Total=0.
        # If total_heuristic=0, then num_depart_actions=0, meaning no unserved passengers.

        return total_heuristic
