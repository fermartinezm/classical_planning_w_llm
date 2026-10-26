from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary board and depart actions for each unserved passenger,
    and adds move actions if the lift is not at the required floor for boarding or departing.

    # Assumptions
    - Each unserved passenger requires at least a board and a depart action.
    - Moving the lift to the origin floor before boarding and to the destination floor before departing are necessary steps if the lift is not already at the correct floor.
    - The heuristic does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - No specific initialization is needed for this heuristic besides the standard heuristic base class initialization.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger:
    1. Check if the passenger is already served. If yes, no further actions are needed for this passenger.
    2. If not served, consider the actions needed:
        a. If the passenger is not yet boarded:
           - Count a 'board' action.
           - Check if the lift is at the passenger's origin floor. If not, count a 'move' action to the origin floor.
        b. If the passenger is boarded:
           - Count a 'depart' action.
           - Check if the lift is at the passenger's destination floor. If not, count a 'move' action to the destination floor.
    3. Sum up the counts of 'board', 'depart', and 'move' actions for all unserved passengers to get the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        super().__init__(task)

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0
        lift_floor = None
        origin_floors = {}
        destination_floors = {}

        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
            elif match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                origin_floors[passenger] = floor
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                destination_floors[passenger] = floor

        unserved_passengers = []
        for fact in self.task.goals:
            if match(fact, "served", "*"):
                goal_passenger = get_parts(fact)[1]
                served = False
                for state_fact in state:
                    if match(state_fact, "served", goal_passenger):
                        served = True
                        break
                if not served:
                    unserved_passengers.append(goal_passenger)

        for passenger in unserved_passengers:
            if any(match(fact, "served", passenger) for fact in state):
                continue

            if not any(match(fact, "boarded", passenger) for fact in state):
                heuristic_value += 1  # For board action
                origin_floor = origin_floors.get(passenger)
                if lift_floor != origin_floor:
                    heuristic_value += 1  # For move to origin floor
            else:
                heuristic_value += 1  # For depart action
                destination_floor = destination_floors.get(passenger)
                if lift_floor != destination_floor:
                    heuristic_value += 1  # For move to destination floor

        return heuristic_value
