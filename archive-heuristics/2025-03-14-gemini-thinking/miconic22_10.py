from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_floor_number(floor_name):
    """Extracts the floor number from a floor name like 'f1', 'f2', etc."""
    return int(floor_name[1:])

def match(fact, *args):
    """
    Utility function to check if a PDDL fact matches a given pattern.
    - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
    - `args`: The pattern to match (e.g., "at", "*", "rooma").
    - Returns `True` if the fact matches the pattern, `False` otherwise.
    """
    parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    by considering the movements of the lift and the boarding/departing actions
    for each unserved passenger. It calculates the estimated cost for each passenger
    individually and sums them up to get the total heuristic value.

    # Assumptions:
    - Floors are ordered and their names (f1, f2, f3...) reflect this order, allowing
      us to use the numerical part of the floor name to estimate distances.
    - The heuristic is not necessarily admissible but aims to be informative and efficient
      for guiding greedy best-first search.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - No other pre-computation is needed.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger, the heuristic estimates the remaining actions needed to serve them:

    1. Identify the current location of the lift.
    2. For each passenger:
       a. Check if the passenger is already served. If yes, the cost for this passenger is 0.
       b. If not served, check if the passenger is boarded.
          i. If not boarded:
             - Find the origin floor of the passenger.
             - Calculate the distance (number of floors to move) from the current lift floor to the origin floor.
             - Add this distance to the heuristic cost.
             - Add 1 to the cost for the 'board' action.
             - Calculate the distance from the origin floor to the destination floor.
             - Add this distance to the heuristic cost.
             - Add 1 to the cost for the 'depart' action.
          ii. If boarded:
              - Find the destination floor of the passenger.
              - Calculate the distance from the current lift floor to the destination floor.
              - Add this distance to the heuristic cost.
              - Add 1 to the cost for the 'depart' action.
    3. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        self.passenger_destinations = {}
        self.passenger_origins = {}

        for fact in self.static_facts:
            if match(fact, "destin", "*", "*"):
                parts = fact[1:-1].split()
                passenger = parts[1]
                destination_floor = parts[2]
                self.passenger_destinations[passenger] = destination_floor
            elif match(fact, "origin", "*", "*"):
                parts = fact[1:-1].split()
                passenger = parts[1]
                origin_floor = parts[2]
                self.passenger_origins[passenger] = origin_floor


    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0

        lift_floor_str = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if lift_floor_str:
            lift_floor = get_floor_number(lift_floor_str.split()[1])
        else:
            return float('inf') # Should not happen in valid states, but handle for robustness

        served_passengers = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(fact.split()[1][1:-1]) # Extract passenger name

        boarded_passengers = set()
        for fact in state:
            if match(fact, "boarded", "*"):
                boarded_passengers.add(fact.split()[1][1:-1]) # Extract passenger name


        for passenger in self.passenger_destinations:
            if passenger in served_passengers:
                continue # Passenger already served, no cost

            if passenger not in boarded_passengers:
                origin_floor_name = self.passenger_origins.get(passenger)
                destination_floor_name = self.passenger_destinations.get(passenger)
                if origin_floor_name and destination_floor_name:
                    origin_floor = get_floor_number(origin_floor_name)
                    destination_floor = get_floor_number(destination_floor_name)
                    heuristic_value += abs(lift_floor - origin_floor) # Move lift to origin
                    heuristic_value += 1 # Board action
                    heuristic_value += abs(origin_floor - destination_floor) # Move lift to destination (from origin) - approximation, should be from current floor but origin is closer to reality
                    heuristic_value += 1 # Depart action
            else: # Passenger is boarded but not served
                destination_floor_name = self.passenger_destinations.get(passenger)
                if destination_floor_name:
                    destination_floor = get_floor_number(destination_floor_name)
                    heuristic_value += abs(lift_floor - destination_floor) # Move lift to destination
                    heuristic_value += 1 # Depart action

        return heuristic_value
