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
    This heuristic estimates the number of actions required to serve all passengers in the Miconic domain.
    It calculates the cost for each unserved passenger based on their current state (waiting for lift or boarded)
    and the lift's current location.

    # Assumptions:
    - For each unserved passenger, we need to perform a sequence of actions: move lift to origin floor (if needed), board,
      move lift to destination floor (if needed), and depart.
    - The heuristic simplifies the movement cost between floors to a constant value (currently 1) for each move action
      required to reach the origin or destination floor. This is a simplification and might not be perfectly accurate
      but aims for efficiency.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger and the origin floor for each passenger from the static facts.
    - Stores these in dictionaries for quick access during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger, the heuristic estimates the remaining actions as follows:
    1. Check if the passenger is already served. If yes, the cost for this passenger is 0.
    2. If not served, check if the passenger is boarded.
    3. If not boarded:
       - Determine the passenger's origin floor.
       - Estimate the cost to move the lift to the origin floor (simplified to 1 move action if lift is not at origin).
       - Add 1 action for the 'board' action.
    4. If boarded:
       - Determine the passenger's destination floor.
       - Estimate the cost to move the lift to the destination floor (simplified to 1 move action if lift is not at destination).
       - Add 1 action for the 'depart' action.
    5. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic.
        Extracts origin and destination floors for each passenger from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                origin_floor = parts[2]
                self.passenger_origins[passenger] = origin_floor
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                destination_floor = parts[2]
                self.passenger_destinations[passenger] = destination_floor

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        Estimates the number of actions needed to reach the goal state from the current state.
        """
        state = node.state
        heuristic_value = 0
        lift_floor = None
        served_passengers = set()
        boarded_passengers = set()
        origin_passengers = {} # passenger -> floor

        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
            elif match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])
            elif match(fact, "origin", "*", "*"):
                passenger = get_parts(fact)[1]
                floor = get_parts(fact)[2]
                origin_passengers[passenger] = floor


        for passenger in self.passenger_destinations: # Iterate through all passengers defined in the problem
            if f'(served {passenger})' in state:
                continue # Passenger already served, no cost

            if f'(boarded {passenger})' not in state:
                # Passenger not boarded yet
                origin_floor = self.passenger_origins[passenger]

                if f'(origin {passenger} {origin_floor})' in state or f'(origin {passenger} {origin_floor})' not in state: # check if still needs to board or already boarded and origin removed
                    if lift_floor != origin_floor:
                        heuristic_value += 1 # Cost to move lift to origin floor (simplified)
                    heuristic_value += 1 # Cost to board

            else:
                # Passenger is boarded
                destination_floor = self.passenger_destinations[passenger]
                if lift_floor != destination_floor:
                    heuristic_value += 1 # Cost to move lift to destination floor (simplified)
                heuristic_value += 1 # Cost to depart

        return heuristic_value
