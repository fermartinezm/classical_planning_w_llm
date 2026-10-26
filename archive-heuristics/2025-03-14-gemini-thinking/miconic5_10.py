from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    It calculates the minimum number of moves (up/down) and board/depart actions
    needed for each passenger to reach their destination floor.

    # Assumptions:
    - The heuristic assumes that for each unserved passenger, the lift will first
      go to their origin floor, then board them, then go to their destination floor,
      and finally depart them.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.
    - It assumes that moving between adjacent floors costs 1 action, and boarding/departing
      each cost 1 action.

    # Heuristic Initialization
    - Extracts static facts, specifically 'origin', 'destin', and 'above' predicates.
    - Creates mappings from passengers to their origin and destination floors.
    - Creates a list of floors, ordered based on the 'above' predicates to estimate distances.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger that is not yet served:
    1. Determine the passenger's origin and destination floors from the static facts.
    2. Find the current floor of the lift from the current state.
    3. If the passenger is not yet boarded:
       - Estimate the number of moves required to bring the lift from its current floor
         to the passenger's origin floor. This is done by counting the number of 'above'
         predicates that need to be traversed.
       - Add 1 action for the 'board' action.
    4. If the passenger is boarded:
       - Estimate the number of moves required to bring the lift from its current floor
         to the passenger's destination floor.
       - Add 1 action for the 'depart' action.
    5. Sum up the estimated actions for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic. Extracts passenger origins, destinations,
        and floor order from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above_relations = []
        self.floors = set()

        for fact in static_facts:
            if match(fact, 'origin', '*', '*'):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_origins[passenger] = floor
                self.floors.add(floor)
            elif match(fact, 'destin', '*', '*'):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destinations[passenger] = floor
                self.floors.add(floor)
            elif match(fact, 'above', '*', '*'):
                parts = get_parts(fact)
                floor1 = parts[1]
                floor2 = parts[2]
                self.above_relations.append((floor1, floor2))
                self.floors.add(floor1)
                self.floors.add(floor2)

        self.floor_list = sorted(list(self.floors)) # Simple alphabetical sorting of floors. More sophisticated ordering based on 'above' could be implemented.
        self.floor_indices = {floor: index for index, floor in enumerate(self.floor_list)}


    def get_floor_distance(self, floor1, floor2):
        """
        Estimates the distance between two floors based on their indices in the sorted floor list.
        This is a simplification and assumes floors are linearly ordered.
        """
        if floor1 not in self.floor_indices or floor2 not in self.floor_indices:
            return 0 # Handle cases where floor is not known.
        return abs(self.floor_indices[floor1] - self.floor_indices[floor2])


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0
        lift_floor = None

        for fact in state:
            if match(fact, 'lift-at', '*'):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            return float('inf') # No lift location known, should not happen in valid states

        unserved_passengers = []
        for passenger in self.passenger_origins: # Iterate through all passengers defined in the problem
            served_predicate = f'(served {passenger})'
            if served_predicate not in state:
                unserved_passengers.append(passenger)

        for passenger in unserved_passengers:
            origin_floor = self.passenger_origins[passenger]
            destination_floor = self.passenger_destinations[passenger]
            boarded_predicate = f'(boarded {passenger})'

            if boarded_predicate not in state:
                # Passenger not boarded yet
                distance_to_origin = self.get_floor_distance(lift_floor, origin_floor)
                heuristic_value += distance_to_origin # Moves to origin floor
                heuristic_value += 1 # Board action
                lift_floor = origin_floor # Assume lift is now at origin floor for next passenger's calculation (simplified heuristic)
            else:
                # Passenger boarded
                distance_to_destination = self.get_floor_distance(lift_floor, destination_floor)
                heuristic_value += distance_to_destination # Moves to destination floor
                heuristic_value += 1 # Depart action
                lift_floor = destination_floor # Assume lift is now at destination floor (simplified heuristic)

        return heuristic_value
