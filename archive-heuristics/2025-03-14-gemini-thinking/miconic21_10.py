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
    This heuristic estimates the number of actions required to serve all passengers
    by considering the necessary board, depart, up, and down actions for each unserved passenger.
    It simplifies the move cost by counting each move action as 1, regardless of the distance.

    # Assumptions
    - The heuristic assumes that for each unserved passenger, the lift needs to:
        1. Move to the passenger's origin floor (if not already there).
        2. Board the passenger.
        3. Move to the passenger's destination floor (if not already there).
        4. Depart the passenger.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:
    1. Initialize the heuristic value to 0.
    2. Identify the current lift floor.
    3. For each passenger:
        a. Check if the passenger is already served. If yes, no further actions are needed for this passenger.
        b. If not served, determine the passenger's origin and destination floors.
        c. If the passenger is not yet boarded:
           i. If the lift is not at the passenger's origin floor, increment the heuristic by 1 (for a move action).
           ii. Increment the heuristic by 1 (for the 'board' action).
        d. If the passenger is boarded:
           i. If the lift is not at the passenger's destination floor, increment the heuristic by 1 (for a move action).
           ii. Increment the heuristic by 1 (for the 'depart' action).
    4. The total heuristic value is the sum of the increments from step 3 for all unserved passengers.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger from static facts.
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
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                origin_floor = parts[2]
                self.passenger_origins[passenger] = origin_floor


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0

        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            return float('inf') # Should not happen in valid states

        for passenger in self.passenger_destinations:
            if f'(served {passenger})' in state:
                continue

            origin_floor = self.passenger_origins.get(passenger)
            destination_floor = self.passenger_destinations.get(passenger)
            if origin_floor is None or destination_floor is None:
                continue # Should not happen in valid problems

            is_boarded = f'(boarded {passenger})' in state

            if not is_boarded:
                if f'(origin {passenger} {origin_floor})' in state: # Only consider passengers who are still at origin
                    if lift_floor != origin_floor:
                        heuristic_value += 1 # Move to origin
                    heuristic_value += 1     # Board action
            else:
                if lift_floor != destination_floor:
                    heuristic_value += 1     # Move to destination
                heuristic_value += 1         # Depart action

        return heuristic_value
