from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string.
    For example, from '(origin p1 f6)' it returns ['p1', 'f6'].
    """
    fact_content = fact_str[1:-1].split()
    return fact_content[1:]  # Return objects, skip predicate name

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It calculates the minimum moves needed for the lift to reach each passenger's origin and destination floors,
    plus one action for boarding and one for departing each passenger.

    # Assumptions:
    - The heuristic assumes that for each unserved passenger, the lift needs to:
        1. Move to the passenger's origin floor (if not already there).
        2. Board the passenger.
        3. Move to the passenger's destination floor (if not already there).
        4. Depart the passenger.
    - It calculates the moves based on the lexicographical order of floor names, assuming 'f1', 'f2', 'f3', ... represent increasing floor levels.
    - The heuristic does not consider optimizing lift movements for multiple passengers simultaneously. It calculates the cost for each unserved passenger independently.

    # Heuristic Initialization
    - Extracts static information: destination floor for each passenger and the 'above' relationships between floors.
    - Precomputes the order of floors based on lexicographical comparison of floor names.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Determine the current lift floor from the state.
    3. For each passenger:
        a. Check if the passenger is already served. If yes, no cost is added for this passenger.
        b. If not served, check if the passenger is boarded.
        c. If not boarded:
            i.  Find the passenger's origin floor and destination floor from the static facts.
            ii. Calculate the number of moves needed to bring the lift from its current floor to the passenger's origin floor. This is estimated as the absolute difference in the lexicographical order of floor names.
            iii.Add the calculated moves and 1 (for the 'board' action) to the heuristic value.
            iv. Update the 'current lift floor' to be the passenger's origin floor for the next passenger's calculation.
        d. If boarded:
            i.  Find the passenger's destination floor.
            ii. Calculate the number of moves needed to bring the lift from its current floor to the passenger's destination floor.
            iii.Add the calculated moves and 1 (for the 'depart' action) to the heuristic value.
            iv. Update the 'current lift floor' to be the passenger's destination floor for the next passenger's calculation.
    4. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic.
        Extracts destination floors for each passenger and precomputes floor order.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above_relations = set()
        self.floors = set()

        for fact in static_facts:
            fact_str = str(fact)
            if fact_str.startswith('(destin'):
                passenger, floor = get_objects_from_fact(fact_str)
                self.passenger_destinations[passenger] = floor
            elif fact_str.startswith('(origin'):
                passenger, floor = get_objects_from_fact(fact_str)
                self.passenger_origins[passenger] = floor
            elif fact_str.startswith('(above'):
                floor1, floor2 = get_objects_from_fact(fact_str)
                self.above_relations.add((floor1, floor2))
                self.floors.add(floor1)
                self.floors.add(floor2)
        for fact in task.initial_state:
            fact_str = str(fact)
            if fact_str.startswith('(origin'):
                passenger, floor = get_objects_from_fact(fact_str)
                if passenger not in self.passenger_origins: # origin can be defined in initial state too
                    self.passenger_origins[passenger] = floor


        self.floor_list = sorted(list(self.floors)) # Lexicographical sort as floor order proxy


    def get_floor_index(self, floor_name):
        """
        Returns the index of a floor in the precomputed floor list.
        Uses lexicographical order as a proxy for floor level.
        """
        try:
            return self.floor_list.index(floor_name)
        except ValueError: # Handle cases where floor is not in the list (should not happen in well-formed problems)
            return 0 # Default to lowest floor index


    def __call__(self, node):
        """
        Compute the heuristic value for a given state in the miconic domain.
        """
        state = node.state
        heuristic_value = 0
        current_lift_floor = None

        for fact in state:
            fact_str = str(fact)
            if fact_str.startswith('(lift-at'):
                current_lift_floor = get_objects_from_fact(fact_str)[0]
                break # Assuming only one lift

        if current_lift_floor is None:
            return float('inf') # No lift location known, should not happen in valid states

        unserved_passengers = set()
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            fact_str = str(fact)
            if fact_str.startswith('(served'):
                served_passengers.add(get_objects_from_fact(fact_str)[0])
            elif fact_str.startswith('(boarded'):
                boarded_passengers.add(get_objects_from_fact(fact_str)[0])
            elif fact_str.startswith('(origin'):
                passenger = get_objects_from_fact(fact_str)[0]
                if passenger not in served_passengers:
                    unserved_passengers.add(passenger)
            elif fact_str.startswith('(destin'):
                passenger = get_objects_from_fact(fact_str)[0]
                if passenger not in served_passengers:
                    unserved_passengers.add(passenger)


        current_heuristic_lift_floor = current_lift_floor # Lift floor for heuristic calculation, can change during loop

        for passenger in self.passenger_destinations: # Iterate through all passengers with destinations
            if passenger not in served_passengers:
                if passenger not in boarded_passengers:
                    origin_floor = self.passenger_origins.get(passenger)
                    if origin_floor: # Origin might not be in initial state, but should be in domain definition
                        origin_floor_index = self.get_floor_index(origin_floor)
                        current_lift_floor_index = self.get_floor_index(current_heuristic_lift_floor)
                        heuristic_value += abs(origin_floor_index - current_lift_floor_index) + 1 # Moves + board
                        current_heuristic_lift_floor = origin_floor # Update lift floor for next passenger
                else: # Passenger is boarded
                    destination_floor = self.passenger_destinations[passenger]
                    destination_floor_index = self.get_floor_index(destination_floor)
                    current_lift_floor_index = self.get_floor_index(current_heuristic_lift_floor)
                    heuristic_value += abs(destination_floor_index - current_lift_floor_index) + 1 # Moves + depart
                    current_heuristic_lift_floor = destination_floor # Update lift floor


        return heuristic_value
