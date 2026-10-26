from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

# Helper functions (can be outside the class)
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input format
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
    This heuristic estimates the remaining effort by summing, for each unserved passenger,
    the estimated cost to get them to the next required state (either boarded or served).
    The estimated cost includes the lift movement to the relevant floor (origin or destination)
    plus the cost of the required action (board or depart).

    # Assumptions
    - Floors are linearly ordered and named like 'f1', 'f2', etc., allowing numerical sorting.
    - Each unserved passenger requires at least one board action and one depart action.
    - The cost of moving the lift between floors is the absolute difference in their floor indices.
    - The cost of board and depart actions is 1.

    # Heuristic Initialization
    - Extracts the goal conditions (which implicitly define served passengers).
    - Extracts static facts, specifically:
        - The destination floor for each passenger.
        - The 'above' relationships to determine the floor ordering and create a floor-to-index mapping.
        - Collects all passenger names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the lift's current floor from the state.
    2. Initialize the total heuristic cost to 0.
    3. Iterate through all passengers identified during initialization.
    4. For each passenger:
       - Check if the passenger is already served (i.e., the fact '(served <passenger>)' is in the current state). If yes, add 0 cost for this passenger and continue to the next.
       - If not served, check if the passenger is currently boarded in the lift (i.e., the fact '(boarded <passenger>)' is in the current state).
       - If boarded:
         - Find the passenger's destination floor using the pre-calculated map from static facts.
         - Calculate the distance between the lift's current floor and the passenger's destination floor using the floor index map (absolute difference of indices).
         - Add this distance plus 1 (representing the 'depart' action cost) to the total heuristic cost.
       - If not boarded (meaning they are waiting at their origin floor):
         - Find the passenger's origin floor by searching for the fact '(origin <passenger> <floor>)' in the current state.
         - Calculate the distance between the lift's current floor and the passenger's origin floor using the floor index map.
         - Add this distance plus 1 (representing the 'board' action cost) to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and all passenger names from the task's static facts and initial state.
        """
        self.goals = task.goals
        self.static = task.static

        # Build floor index map
        floor_names = set()
        # Collect floors from above facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'above':
                floor_names.add(parts[1])
                floor_names.add(parts[2])
        # Collect floors from initial lift location (in case it's the only floor mentioned initially)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'lift-at':
                 floor_names.add(parts[1])

        # Sort floor names numerically (assuming f1, f2, f10 format)
        # This assumes floor names are consistently formatted as 'f' followed by a number.
        # A more general approach would involve building a graph from 'above' facts,
        # but for typical miconic instances, this numerical sort is sufficient.
        try:
            sorted_floor_names = sorted(list(floor_names), key=lambda x: int(x[1:]))
        except (ValueError, IndexError):
             # Fallback if floor names are not in expected 'fN' format.
             # This might lead to incorrect floor ordering if names are arbitrary.
             print("Warning: Floor names might not be in 'fN' format. Sorting alphabetically.")
             sorted_floor_names = sorted(list(floor_names)) # Simple alphabetical sort

        self.floor_indices = {floor: i for i, floor in enumerate(sorted_floor_names)}

        # Build passenger destination map and collect all passenger names
        self.passenger_destinations = {}
        self.all_passengers = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts:
                if parts[0] == 'destin':
                    passenger, destination = parts[1], parts[2]
                    self.passenger_destinations[passenger] = destination
                    self.all_passengers.add(passenger)
                elif parts[0] == 'origin': # Also collect passengers from origin facts in static
                     passenger, origin = parts[1], parts[2]
                     self.all_passengers.add(passenger)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached (all goals are served predicates)
        # This check is technically redundant if we sum costs for unserved passengers,
        # but it's a quick exit for the goal state.
        if self.goals <= state:
            return 0

        # Find the lift's current floor
        current_f = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'lift-at':
                current_f = parts[1]
                break

        # If lift location is not found, something is wrong with the state representation
        # or domain definition. Return a large value to penalize this state.
        if current_f is None:
             # print("Warning: Lift location not found in state.") # Optional warning
             return 1000000 # Large arbitrary value indicating an invalid or highly undesirable state

        total_heuristic = 0

        # Iterate through all known passengers
        for passenger in self.all_passengers:
            # Check if the passenger is served
            if f'(served {passenger})' in state:
                continue # Passenger is served, no cost for this passenger

            # Passenger is not served. Check if boarded or waiting at origin.
            if f'(boarded {passenger})' in state:
                # Passenger is boarded, needs to go to destination
                destin_f = self.passenger_destinations.get(passenger)
                if destin_f is None:
                     # Destination not found for a boarded passenger - indicates a problem
                     # with the static facts or state consistency. Penalize.
                     # print(f"Warning: Destination not found for boarded passenger {passenger}.") # Optional warning
                     total_heuristic += 1000 # Arbitrary penalty
                     continue

                # Cost for this passenger: move lift from current_f to destin_f + depart action
                current_floor_idx = self.floor_indices.get(current_f)
                destin_floor_idx = self.floor_indices.get(destin_f)

                if current_floor_idx is None or destin_floor_idx is None:
                     # Floor name from state or static not found in our index map. Penalize.
                     # print(f"Warning: Floor index not found for {current_f} or {destin_f}.") # Optional warning
                     total_heuristic += 1000 # Arbitrary penalty
                     continue

                move_cost = abs(current_floor_idx - destin_floor_idx)
                action_cost = 1 # Cost of the 'depart' action
                total_heuristic += move_cost + action_cost

            else:
                # Passenger is not served and not boarded, must be waiting at origin.
                # Find the origin floor for this passenger in the current state.
                origin_f = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == 'origin' and parts[1] == passenger:
                        origin_f = parts[2]
                        break

                if origin_f is None:
                    # Passenger is unserved, not boarded, and not at origin in state?
                    # This state should ideally not be reachable in a valid problem. Penalize.
                    # print(f"Warning: Origin not found in state for unserved, unboarded passenger {passenger}.") # Optional warning
                    total_heuristic += 1000 # Arbitrary penalty
                    continue

                # Cost for this passenger: move lift from current_f to origin_f + board action
                current_floor_idx = self.floor_indices.get(current_f)
                origin_floor_idx = self.floor_indices.get(origin_f)

                if current_floor_idx is None or origin_floor_idx is None:
                     # Floor name from state not found in our index map. Penalize.
                     # print(f"Warning: Floor index not found for {current_f} or {origin_f}.") # Optional warning
                     total_heuristic += 1000 # Arbitrary penalty
                     continue

                move_cost = abs(current_floor_idx - origin_floor_idx)
                action_cost = 1 # Cost of the 'board' action
                total_heuristic += move_cost + action_cost

        return total_heuristic
