from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    by considering the necessary steps for each unserved passenger: moving the lift
    to the passenger's origin floor, boarding, moving to the destination floor, and departing.
    It simplifies the cost of moving between any two different floors as 1, and 0 if they are the same.

    # Assumptions:
    - Each move action (up or down between different floors) costs 1.
    - Boarding and departing actions each cost 1.
    - The heuristic is calculated by summing up the estimated costs for each passenger individually.
    - It assumes that serving each passenger is independent of others, which might not be optimal in all cases but provides a reasonable estimate.

    # Heuristic Initialization
    - Extracts the goal conditions (served predicates) and static facts (destin, origin, above predicates) from the task.
    - Stores the destination and origin floor for each passenger for efficient access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger, the heuristic estimates the remaining actions based on their current state:

    1. Check if the passenger is already served. If yes, no further actions are needed for this passenger (cost 0).
    2. If not served, determine if the passenger is boarded or not.
    3. If not boarded:
       a. Determine the passenger's origin floor and the current lift floor.
       b. Estimate the cost to move the lift to the origin floor (1 if lift is not at the origin floor, 0 otherwise).
       c. Add 1 for the 'board' action.
       d. Determine the passenger's destination floor and the origin floor.
       e. Estimate the cost to move the lift from the origin floor to the destination floor (1 if origin and destination floors are different, 0 otherwise).
       f. Add 1 for the 'depart' action.
       g. The total estimated cost for this passenger is the sum of costs from steps 3b, 3c, 3e, and 3f.
    4. If boarded:
       a. Determine the passenger's destination floor and the current lift floor.
       b. Estimate the cost to move the lift to the destination floor (1 if lift is not at the destination floor, 0 otherwise).
       c. Add 1 for the 'depart' action.
       d. The total estimated cost for this passenger is the sum of costs from steps 4b and 4c.
    5. The final heuristic value for the state is the sum of the estimated costs for all unserved passengers.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        for fact in static_facts:
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
        """Estimate the number of actions needed to serve all passengers."""
        state = node.state
        heuristic_value = 0

        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            return float('inf') # Should not happen in valid states, but handle for robustness

        passengers_served = set()
        passengers_boarded = set()
        passengers_origin = {}
        passengers_destin = {}

        for fact in state:
            if match(fact, "served", "*"):
                passengers_served.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                passengers_boarded.add(get_parts(fact)[1])

        all_passengers = set(self.passenger_origins.keys())

        for passenger in all_passengers:
            if passenger in passengers_served:
                continue

            origin_floor = self.passenger_origins[passenger]
            destination_floor = self.passenger_destinations[passenger]

            if passenger in passengers_boarded:
                move_to_destination_cost = 1 if lift_floor != destination_floor else 0
                heuristic_value += move_to_destination_cost + 1 # depart action
            else:
                move_to_origin_cost = 1 if lift_floor != origin_floor else 0
                move_to_destination_cost = 1 if origin_floor != destination_floor else 0

                heuristic_value += move_to_origin_cost + 1 + move_to_destination_cost + 1 # board, depart actions

        return heuristic_value
