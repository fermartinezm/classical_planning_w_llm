from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(origin p1 f6)' it returns ['p1', 'f6'].
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[1:]  # Return objects, skipping predicate name

def get_predicate_name(fact_str):
    """
    Extracts the predicate name from a PDDL fact string.
    For example, from '(origin p1 f6)' it returns 'origin'.
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[0]

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    in the miconic domain. It considers the necessary steps for each unserved passenger:
    moving the lift to the passenger's origin floor, boarding, moving to the destination floor, and departing.

    # Assumptions:
    - Floors are ordered based on the 'above' predicates provided in the static facts.
    - The heuristic assumes the lift always takes the shortest path (in terms of number of floors)
      to reach a target floor.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.
    - It counts the number of up/down actions as the difference in floor levels.

    # Heuristic Initialization
    - Extracts static information about floor order from 'above' predicates.
    - Identifies passenger origins and destinations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Determine the current location of the lift.
    3. For each passenger:
       a. Check if the passenger is already served. If yes, ignore this passenger.
       b. If not served, check if the passenger is boarded.
       c. If not boarded:
          i.  Find the passenger's origin and destination floors.
          ii. Estimate the number of 'move' actions (up or down) to reach the origin floor from the lift's current floor.
          iii.Add this number to the heuristic value.
          iv. Add 1 to the heuristic value for the 'board' action.
          v. Estimate the number of 'move' actions to reach the destination floor from the origin floor.
          vi.Add this number to the heuristic value.
          vii.Add 1 to the heuristic value for the 'depart' action.
       d. If boarded:
          i.  Find the passenger's destination floor.
          ii. Estimate the number of 'move' actions to reach the destination floor from the lift's current floor.
          iii.Add this number to the heuristic value.
          iv. Add 1 to the heuristic value for the 'depart' action.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by pre-processing static facts to determine floor order and passenger information."""
        self.goals = task.goals
        static_facts = task.static

        self.floor_order = []
        above_relations = []
        for fact in static_facts:
            if get_predicate_name(fact) == 'above':
                above_relations.append(get_objects_from_fact(fact))

        floors_set = set()
        for rel in above_relations:
            floors_set.add(rel[0])
            floors_set.add(rel[1])
        if not floors_set: # Handle case with no floors
            self.floor_order = []
            return

        sorted_floors = sorted(list(floors_set), key=lambda f: int(f[1:])) # Assumes floor names like f1, f2, ...
        self.floor_order = sorted_floors


        self.passenger_origins = {}
        self.passenger_destinations = {}
        for fact in static_facts:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'origin':
                self.passenger_origins[objects[0]] = objects[1]
            elif predicate == 'destin':
                self.passenger_destinations[objects[0]] = objects[1]

    def get_floor_index(self, floor_name):
        """Returns the index of a floor in the pre-calculated floor order."""
        try:
            return self.floor_order.index(floor_name)
        except ValueError: # Handle cases where floor is not in order (should not happen in well-formed problems)
            return -1

    def get_distance(self, floor1, floor2):
        """Calculates the distance (number of floors to move) between two floors."""
        if floor1 == floor2:
            return 0
        index1 = self.get_floor_index(floor1)
        index2 = self.get_floor_index(floor2)
        if index1 == -1 or index2 == -1:
            return 1000 # Return a large value if floor is not found in order.
        return abs(index1 - index2)

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0

        lift_floor = None
        for fact in state:
            if get_predicate_name(fact) == 'lift-at':
                lift_floor = get_objects_from_fact(fact)[0]
                break
        if lift_floor is None:
            return float('inf') # No lift location, should not happen in valid states, but handle for robustness

        served_passengers = set()
        boarded_passengers = set()
        origin_passengers = {} # Current origins, might change after boarding
        for fact in state:
            predicate = get_predicate_name(fact)
            objects = get_objects_from_fact(fact)
            if predicate == 'served':
                served_passengers.add(objects[0])
            elif predicate == 'boarded':
                boarded_passengers.add(objects[0])
            elif predicate == 'origin':
                origin_passengers[objects[0]] = objects[1]


        for passenger in self.passenger_destinations: # Iterate through all passengers defined in destin
            if passenger in served_passengers:
                continue # Passenger already served, no cost

            destination_floor = self.passenger_destinations[passenger]
            origin_floor_current_state = origin_passengers.get(passenger, self.passenger_origins.get(passenger)) # Get current origin or initial origin if not changed

            if passenger not in boarded_passengers:
                origin_floor = origin_floor_current_state
                if origin_floor is None: # Passenger has no origin, should not happen in valid problems, but handle
                    continue

                heuristic_value += self.get_distance(lift_floor, origin_floor) # Move to origin
                heuristic_value += 1 # Board
                heuristic_value += self.get_distance(origin_floor, destination_floor) # Move to destin
                heuristic_value += 1 # Depart
            else: # Passenger is boarded
                heuristic_value += self.get_distance(lift_floor, destination_floor) # Move to destin
                heuristic_value += 1 # Depart

        return heuristic_value
