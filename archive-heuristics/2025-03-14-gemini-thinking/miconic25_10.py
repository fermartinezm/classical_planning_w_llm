from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_floor_index(floor_name):
    """
    Extracts the floor index from a floor name like 'f1', 'f2', etc.
    Returns 0 if parsing fails, assuming 'f' followed by digits.
    """
    try:
        return int(floor_name[1:])
    except (ValueError, TypeError):
        return 0

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(lift-at f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers in the Miconic domain.
    It calculates the cost for each unserved passenger based on their origin and destination floors and the current lift position.

    # Assumptions
    - The heuristic assumes that for each unserved passenger, the lift will first go to their origin floor, then to their destination floor.
    - It simplifies the movement cost between floors by considering the absolute difference in floor indices as a proxy for the number of up/down actions.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger that is not yet served:
    1. Determine the passenger's origin floor and destination floor from the static facts.
    2. Check if the passenger is currently boarded in the given state.
    3. Get the current floor where the lift is located.
    4. If the passenger is not boarded:
       - Estimate the cost to move the lift from its current floor to the passenger's origin floor. This is approximated by the absolute difference of their floor indices.
       - Add 1 for the 'board' action.
       - Estimate the cost to move the lift from the origin floor to the destination floor. This is approximated by the absolute difference of their floor indices.
       - Add 1 for the 'depart' action.
    5. If the passenger is boarded:
       - Estimate the cost to move the lift from its current floor to the passenger's destination floor. This is approximated by the absolute difference of their floor indices.
       - Add 1 for the 'depart' action.
    6. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic. Extracts goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static
        self.passenger_origins = {}
        self.passenger_destinations = {}

        for fact in self.static_facts:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_origins[passenger] = floor
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destinations[passenger] = floor

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0
        lift_floor = None
        served_passengers = set()
        boarded_passengers = set()

        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
            elif match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])

        if lift_floor is None:
            return float('inf') # Should not happen in valid miconic problems

        unserved_passengers = set()
        for passenger in self.passenger_origins:
            if passenger not in served_passengers:
                unserved_passengers.add(passenger)

        if not unserved_passengers:
            return 0 # Goal state reached

        for passenger in unserved_passengers:
            origin_floor = self.passenger_origins[passenger]
            destination_floor = self.passenger_destinations[passenger]

            if passenger not in boarded_passengers:
                heuristic_value += 1 # board action
                heuristic_value += abs(get_floor_index(lift_floor) - get_floor_index(origin_floor)) # moves to origin
                heuristic_value += abs(get_floor_index(origin_floor) - get_floor_index(destination_floor)) # moves to destin
                heuristic_value += 1 # depart action
            else:
                heuristic_value += 1 # depart action
                heuristic_value += abs(get_floor_index(lift_floor) - get_floor_index(destination_floor)) # moves to destin

        return heuristic_value
