from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_floor_number(floor_name):
    """Extracts the floor number from a floor name like 'f1', 'f2', etc."""
    return int(floor_name[1:])

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
    This heuristic estimates the number of actions required to serve all passengers
    by considering the necessary lift movements and board/depart actions for each unserved passenger.

    # Assumptions:
    - The floor names are in the format 'f1', 'f2', 'f3', ... and their order corresponds to their numerical suffix.
    - The cost of moving between adjacent floors (up or down) is 1 action.
    - Boarding and departing actions each cost 1 action.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Extracts the origin floor for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger that is not yet served:
    1. Check if the passenger is already boarded.
    2. If not boarded:
        - Determine the passenger's origin floor and the current lift floor.
        - Estimate the number of 'up' or 'down' actions needed to move the lift to the origin floor by calculating the absolute difference between their floor numbers.
        - Add this number to the heuristic estimate, plus 1 for the 'board' action.
    3. If boarded:
        - Determine the passenger's destination floor and the current lift floor.
        - Estimate the number of 'up' or 'down' actions needed to move the lift to the destination floor by calculating the absolute difference between their floor numbers.
        - Add this number to the heuristic estimate, plus 1 for the 'depart' action.
    4. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - Origin floor for each passenger.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_destinations = {}
        self.passenger_origins = {}

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                destination_floor = parts[2]
                self.passenger_destinations[passenger] = destination_floor
            elif match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                origin_floor = parts[2]
                self.passenger_origins[passenger] = origin_floor


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        lift_location = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_location = get_parts(fact)[1]
                break

        if lift_location is None:
            return float('inf') # Should not happen in valid miconic problems

        served_passengers = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])

        boarded_passengers = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])

        origin_passengers = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                origin_passengers.add(get_parts(fact)[1])


        for passenger in set(self.passenger_destinations.keys()) | set(self.passenger_origins.keys()):
            if passenger not in served_passengers:
                if passenger not in boarded_passengers:
                    origin_floor = self.passenger_origins.get(passenger)
                    if origin_floor is not None:
                        heuristic_value += abs(get_floor_number(origin_floor) - get_floor_number(lift_location)) + 1 # Move to origin and board
                else:
                    destination_floor = self.passenger_destinations.get(passenger)
                    if destination_floor is not None:
                        heuristic_value += abs(get_floor_number(destination_floor) - get_floor_number(lift_location)) + 1 # Move to destination and depart

        # Goal state check: if all passengers are served, heuristic is 0
        all_served = True
        for goal in self.goals:
            if goal not in state:
                all_served = False
                break
        if all_served:
            return 0

        return heuristic_value
