from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_string):
    """
    Extracts objects from a PDDL fact string.
    For example, '(predicate_name object1 object2)' returns ['object1', 'object2'].
    It removes the parentheses and splits the string by spaces, ignoring the predicate name.
    """
    parts = fact_string[1:-1].split()
    return parts[1:]

def get_predicate_name(fact_string):
    """
    Extracts the predicate name from a PDDL fact string.
    For example, '(predicate_name object1 object2)' returns 'predicate_name'.
    """
    parts = fact_string[1:-1].split()
    return parts[0]

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all passengers
    by considering the necessary moves and actions for each unserved passenger. It calculates
    the cost based on moving the lift to each passenger's origin floor, boarding them, moving
    to their destination floor, and departing.

    # Assumptions:
    - Floors are numerically ordered based on their names (e.g., f1, f2, f3...).
    - The 'above' predicates define the floor order, but the heuristic directly uses the numerical
      part of the floor names to calculate distances, assuming a linear floor arrangement.
    - The heuristic assumes that for each unserved passenger, the lift needs to visit their origin
      and destination floors in sequence.

    # Heuristic Initialization
    - Extracts static information about passenger destinations and origin floors.
    - Stores destination floor for each passenger.
    - Stores origin floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger that is not yet served:
    1. Check if the passenger is already served. If yes, no cost is added for this passenger.
    2. If the passenger is not served, check if they are boarded.
    3. If not boarded:
       a. Determine the passenger's origin floor and the current lift floor.
       b. Estimate the cost to move the lift from its current floor to the passenger's origin floor.
          This is approximated by the absolute difference in floor levels (extracted from floor names).
       c. Add 1 to the cost for the 'board' action.
    4. If boarded:
       a. Determine the passenger's destination floor and the current lift floor.
       b. Estimate the cost to move the lift from its current floor to the passenger's destination floor.
          This is approximated by the absolute difference in floor levels.
       c. Add 1 to the cost for the 'depart' action.
    5. Sum up the costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic by pre-processing static facts to extract
        passenger destinations and origins.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_destinations = {}
        self.passenger_origins = {}

        for fact in static_facts:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'destin':
                passenger, floor = objects
                self.passenger_destinations[passenger] = floor
            elif predicate == 'origin': # Although origin might change, we can store initial origin for reference if needed.
                passenger, floor = objects
                self.passenger_origins[passenger] = floor
        for goal_fact in self.goals:
            if get_predicate_name(goal_fact) == 'served':
                passenger = get_objects_from_fact(goal_fact)[0]
                if passenger not in self.passenger_destinations: # In case goal is given before origin in static facts
                    self.passenger_destinations[passenger] = None # Destination might be unknown at init, but will be used in heuristic

    def __call__(self, node):
        """
        Calculate the heuristic value for a given state in the miconic domain.
        """
        state = node.state
        current_lift_floor = None
        boarded_passengers = set()
        served_passengers = set()
        origin_locations = {}

        for fact in state:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'lift-at':
                current_lift_floor = objects[0]
            elif predicate == 'boarded':
                boarded_passengers.add(objects[0])
            elif predicate == 'served':
                served_passengers.add(objects[0])
            elif predicate == 'origin':
                passenger, floor = objects
                origin_locations[passenger] = floor


        heuristic_value = 0
        passengers_to_serve = set()
        for passenger in self.passenger_destinations: # Iterate through all passengers that have destinations defined.
            passengers_to_serve.add(passenger)

        for passenger in passengers_to_serve:
            if '(served {})'.format(passenger) in state:
                continue # Passenger already served, no cost

            destination_floor = self.passenger_destinations[passenger]
            origin_floor = origin_locations.get(passenger, self.passenger_origins.get(passenger)) # Get origin from current state if available, else from static init

            if '(boarded {})'.format(passenger) not in state:
                # Passenger not boarded yet
                if origin_floor is not None and current_lift_floor is not None:
                    origin_floor_level = int(origin_floor[1:])
                    current_lift_floor_level = int(current_lift_floor[1:])
                    heuristic_value += abs(current_lift_floor_level - origin_floor_level) if current_lift_floor != origin_floor else 0
                heuristic_value += 1 # Board action
            else:
                # Passenger is boarded
                if destination_floor is not None and current_lift_floor is not None:
                    destination_floor_level = int(destination_floor[1:])
                    current_lift_floor_level = int(current_lift_floor[1:])
                    heuristic_value += abs(current_lift_floor_level - destination_floor_level) if current_lift_floor != destination_floor else 0
                heuristic_value += 1 # Depart action

        return heuristic_value
