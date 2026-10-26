from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(origin p1 f1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all passengers
    by considering the necessary steps for each unserved passenger: moving the lift to the
    passenger's origin floor, boarding, moving to the destination floor, and departing.

    # Assumptions
    - The heuristic assumes that each unserved passenger needs to be boarded, transported, and departed.
    - It simplifies the movement cost of the lift by counting each necessary move (up or down) as a single action,
      regardless of the distance between floors.

    # Heuristic Initialization
    - The heuristic initializes by pre-processing the static facts to store the destination floor for each passenger.
    - Origin floors are not pre-processed as they are expected to change as passengers are boarded (origin predicate is removed).

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger that is not yet served:
    1. Check if the passenger is already served. If yes, no further actions are needed for this passenger.
    2. If not served, check if the passenger is boarded.
       a. If not boarded:
          i.  Identify the passenger's origin floor and destination floor from the initial state and static facts.
          ii. Find the current floor of the lift.
          iii.Estimate the cost to move the lift to the origin floor (1 action if lift is not at the origin floor, 0 otherwise).
          iv.Add 1 action for boarding the passenger.
          v. Estimate the cost to move the lift to the destination floor (1 action if lift is not at the destination floor after moving to origin and boarding, 0 otherwise).
          vi.Add 1 action for departing the passenger.
       b. If boarded:
          i.  Identify the passenger's destination floor.
          ii. Find the current floor of the lift.
          iii.Estimate the cost to move the lift to the destination floor (1 action if lift is not at the destination floor, 0 otherwise).
          iv.Add 1 action for departing the passenger.
    3. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger from static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        self.passenger_destinations = {}
        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                destination_floor = parts[2]
                self.passenger_destinations[passenger] = destination_floor

    def __call__(self, node):
        """Compute an estimate of the minimal number of actions required to serve all passengers."""
        state = node.state
        heuristic_value = 0

        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            return float('inf') # Should not happen in valid states

        served_passengers = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])

        boarded_passengers = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])

        origin_floors = {}
        for fact in state:
            if match(fact, "origin", "*", "*"):
                origin_floors[get_parts(fact)[1]] = get_parts(fact)[2]


        passengers_to_serve = set(self.passenger_destinations.keys()) - served_passengers

        for passenger in passengers_to_serve:
            if passenger in boarded_passengers:
                destination_floor = self.passenger_destinations[passenger]
                if lift_floor != destination_floor:
                    heuristic_value += 1 # Move to destination
                heuristic_value += 1 # Depart
            else:
                origin_floor = origin_floors.get(passenger)
                destination_floor = self.passenger_destinations[passenger]
                if origin_floor: # Passenger is still at origin
                    if lift_floor != origin_floor:
                        heuristic_value += 1 # Move to origin
                    heuristic_value += 1 # Board
                    if lift_floor != destination_floor: # Lift is now at origin floor
                        heuristic_value += 1 # Move to destination
                    heuristic_value += 1 # Depart
        return heuristic_value
