import re
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all unserved passengers.
    For each unserved passenger, it estimates the cost based on their current state:
    - If waiting at their origin floor: Estimate includes moving the lift to the origin, boarding, moving to the destination, and departing.
    - If boarded: Estimate includes moving the lift to the destination, and departing.
    The total heuristic is the sum of these estimates for all unserved passengers.

    # Assumptions
    - Floors are named 'fX' where X is an integer, and their order corresponds to the numerical value of X.
    - The cost of moving between adjacent floors is 1.
    - Boarding a passenger costs 1 action.
    - Departing a passenger costs 1 action.
    - The heuristic assumes passengers are served one by one in some optimal order (which is not explicitly planned by the heuristic, but the sum provides an estimate).

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts (`destin` predicate).
    - Builds a mapping from floor names (e.g., 'f1', 'f2') to integer indices based on the numerical suffix to calculate floor distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse static facts to determine the destination floor for each passenger and build the floor-to-index mapping based on floor names (e.g., 'f1' -> 0, 'f2' -> 1).
    2. In the `__call__` method, get the current state.
    3. Identify the current floor of the lift.
    4. Iterate through all passengers whose destinations are known (from initialization).
    5. For each passenger:
       - Check if the passenger is already served (`(served ?p)`). If yes, cost is 0 for this passenger.
       - If not served, check if the passenger is boarded (`(boarded ?p)`).
         - If boarded:
           - Get the passenger's destination floor (`f_destin`) from the pre-calculated goal information.
           - Calculate the distance between the current lift floor and `f_destin` using the floor-to-index mapping: `abs(floor_index(current_lift_floor) - floor_index(f_destin))`.
           - Add the cost of departing (1 action).
           - Total estimated cost for this boarded passenger: `abs(current_lift_index - destin_index) + 1`.
         - If not boarded (meaning they are waiting at their origin):
           - Find the passenger's origin floor (`f_origin`) from the current state (`(origin ?p ?f_origin)`).
           - Get the passenger's destination floor (`f_destin`) from the pre-calculated goal information.
           - Calculate the distance from the current lift floor to `f_origin`: `abs(current_lift_index - origin_index)`.
           - Add the cost of boarding (1 action).
           - Calculate the distance from `f_origin` to `f_destin`: `abs(origin_index - destin_index)`.
           - Add the cost of departing (1 action).
           - Total estimated cost for this waiting passenger: `abs(current_lift_index - origin_index) + 1 + abs(origin_index - destin_index) + 1`.
    6. Sum the estimated costs for all unserved passengers. This sum is the heuristic value.
    7. If the sum is 0, it means all passengers are served, which is the goal state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Include initial state to find all floor names

        # Store destination floor for each passenger from static facts
        self.goal_destinations = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.goal_destinations[passenger] = floor

        # Build floor-to-index mapping based on numerical suffix
        self.floor_to_index = {}
        floor_names = set()
        # Find all floor names mentioned in the initial state and static facts
        # Need to look in initial state as well to get all floors present in the problem
        for fact in initial_state | static_facts:
            parts = get_parts(fact)
            for part in parts:
                # Check if the part looks like a floor name (starts with 'f' followed by digits)
                if part.startswith('f') and re.fullmatch(r'f\d+', part):
                    floor_names.add(part)

        # Sort floor names numerically (e.g., f1, f2, f10)
        # Use the integer value after 'f' for sorting
        sorted_floor_names = sorted(list(floor_names), key=lambda f: int(f[1:]))

        for index, floor_name in enumerate(sorted_floor_names):
            self.floor_to_index[floor_name] = index

        if not self.floor_to_index:
             # This case might occur for trivial problems or parsing errors
             print("Warning: No floors found in the problem definition.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        total_cost = 0  # Initialize action cost counter.

        # Find the current lift location
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown or not a recognized floor, return infinity
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             return float('inf')

        current_lift_index = self.floor_to_index[current_lift_floor]

        # Identify served passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through all passengers whose destinations we know (all passengers in the problem)
        for passenger, destin_floor in self.goal_destinations.items():
            if passenger in served_passengers:
                continue # This passenger is already served

            # Check if the passenger is boarded
            is_boarded = any(match(fact, "boarded", passenger) for fact in state)

            destin_index = self.floor_to_index.get(destin_floor)
            if destin_index is None:
                 # Destination floor not found in floor mapping - indicates problem parsing issue
                 # Treat this passenger as contributing infinity cost, or skip?
                 # Skipping might be better for a heuristic, assuming valid problem
                 continue

            if is_boarded:
                # Passenger is boarded, needs to reach destination and depart
                # Cost = moves to destination + depart action
                total_cost += abs(current_lift_index - destin_index) + 1

            else:
                # Passenger is not served and not boarded, must be waiting at origin
                # Find the origin floor from the current state
                origin_floor = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts[0] == "origin" and len(parts) == 3 and parts[1] == passenger:
                        origin_floor = parts[2]
                        break

                if origin_floor is None or origin_floor not in self.floor_to_index:
                     # Origin floor not found in state or not a recognized floor
                     # Indicates an invalid state for an unserved, unboarded passenger
                     # Treat as infinity or skip? Skipping is safer for heuristic.
                     continue

                origin_index = self.floor_to_index[origin_floor]

                # Cost = moves to origin + board action + moves from origin to destin + depart action
                total_cost += abs(current_lift_index - origin_index) + 1 + abs(origin_index - destin_index) + 1

        return total_cost
