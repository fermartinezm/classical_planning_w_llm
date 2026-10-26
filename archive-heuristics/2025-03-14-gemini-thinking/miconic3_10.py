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
    This heuristic estimates the number of actions required to serve all passengers.
    It calculates the minimum number of moves and actions (board and depart) needed for each unserved passenger.

    # Assumptions:
    - The heuristic assumes that for each unserved passenger, the lift will move to their origin floor, board them, move to their destination floor, and depart them.
    - It estimates the number of lift movements as the difference in floor indices based on the 'above' predicates.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - Extracts the destination floor for each passenger from the static facts.
    - Creates a floor index mapping based on the 'above' predicates to estimate floor distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost to 0.
    2. Determine the current lift floor from the state.
    3. Identify all passengers who are not yet served.
    4. For each unserved passenger:
        a. If the passenger is not yet boarded:
            i. Estimate the number of 'up' or 'down' actions to move the lift from its current floor to the passenger's origin floor.
            ii. Add 1 for the 'board' action.
            iii. Estimate the number of 'up' or 'down' actions to move the lift from the origin floor to the passenger's destination floor.
            iv. Add 1 for the 'depart' action.
        b. If the passenger is already boarded:
            i. Estimate the number of 'up' or 'down' actions to move the lift from its current floor to the passenger's destination floor.
            ii. Add 1 for the 'depart' action.
    5. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    6. The floor distance is calculated based on the pre-computed floor indices derived from 'above' predicates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - Floor order based on 'above' predicates.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_destinations = {}
        self.passenger_origins = {}
        self.above_relations = []
        self.floors = set()
        self.passengers = set()

        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                destination_floor = parts[2]
                self.passenger_destinations[passenger] = destination_floor
                self.passengers.add(passenger)
                self.floors.add(destination_floor)
            elif match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                origin_floor = parts[2]
                self.passenger_origins[passenger] = origin_floor
                self.passengers.add(passenger)
                self.floors.add(origin_floor)
            elif match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                floor1 = parts[1]
                floor2 = parts[2]
                self.above_relations.append((floor1, floor2))
                self.floors.add(floor1)
                self.floors.add(floor2)

        self.floor_list = sorted(list(self.floors), key=lambda f: self._get_floor_index(f))
        self.floor_indices = {floor: index for index, floor in enumerate(self.floor_list)}


    def _get_floor_index(self, floor):
        """Helper function to determine a rough index for each floor based on 'above' relations.
           This is a simplification and might not perfectly represent the floor order in all cases,
           but it serves as a heuristic estimate for floor distance."""
        index = 0
        for f1, f2 in self.above_relations:
            if f2 == floor:
                index += 1
        return index


    def _floor_distance(self, floor1, floor2):
        """Calculates the distance between two floors based on their indices."""
        if floor1 == floor2:
            return 0
        return abs(self.floor_indices.get(floor1, 0) - self.floor_indices.get(floor2, 0))


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0
        current_lift_floor = None
        served_passengers = set()
        boarded_passengers = set()
        origin_passengers = {}

        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
            elif match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])
            elif match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                origin_passengers[parts[1]] = parts[2]


        unserved_passengers = [p for p in self.passengers if p not in served_passengers]

        if not unserved_passengers:
            return 0

        for passenger in unserved_passengers:
            origin_floor = self.passenger_origins.get(passenger)
            destination_floor = self.passenger_destinations.get(passenger)

            if passenger not in boarded_passengers:
                if current_lift_floor:
                    heuristic_value += self._floor_distance(current_lift_floor, origin_floor) # Moves to origin
                heuristic_value += 1 # Board action
                current_lift_floor = origin_floor # Assume lift is now at origin floor
                heuristic_value += self._floor_distance(current_lift_floor, destination_floor) # Moves to destination
                heuristic_value += 1 # Depart action
                current_lift_floor = destination_floor # Assume lift is now at destination floor
            else:
                if current_lift_floor:
                    heuristic_value += self._floor_distance(current_lift_floor, destination_floor) # Moves to destination
                heuristic_value += 1 # Depart action
                current_lift_floor = destination_floor # Assume lift is now at destination floor

        return heuristic_value
