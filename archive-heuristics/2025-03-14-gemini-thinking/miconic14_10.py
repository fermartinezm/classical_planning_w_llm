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

    # Assumptions:
    - The heuristic assumes that each unserved passenger needs to be boarded, transported to their destination floor, and departed.
    - It estimates the number of up and down actions required to move the lift to the origin and destination floors for each passenger.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - Extracts static facts: destination and origin floors for each passenger, and the 'above' relationships between floors to determine floor order.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Initialize the heuristic value to 0.
    2. Determine the current location of the lift.
    3. Identify all passengers who are not yet served.
    4. For each unserved passenger:
        a. If the passenger is not yet boarded:
           i.  Determine the origin floor of the passenger.
           ii. Calculate the number of 'up' or 'down' actions needed to move the lift from its current location to the passenger's origin floor. Add this count to the heuristic value.
           iii.Increment the heuristic value by 1 (for the 'board' action).
           iv. Update the current lift location to the passenger's origin floor.
        b. If the passenger is boarded:
           i.  Determine the destination floor of the passenger.
           ii. Calculate the number of 'up' or 'down' actions needed to move the lift from its current location to the passenger's destination floor. Add this count to the heuristic value.
           iii.Increment the heuristic value by 1 (for the 'depart' action).
           iv. Update the current lift location to the passenger's destination floor.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - Origin floor for each passenger.
        - 'above' relationships between floors.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above_relations = set()
        self.floors = set()

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_origins[passenger] = floor
                self.floors.add(floor)
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destinations[passenger] = floor
                self.floors.add(floor)
            elif match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                floor1 = parts[1]
                floor2 = parts[2]
                self.above_relations.add((floor1, floor2))
                self.floors.add(floor1)
                self.floors.add(floor2)
        self.floor_list = sorted(list(self.floors), key=lambda f: int(f[1:])) # Simple sort based on floor number, assuming f1, f2, ...

    def get_floor_index(self, floor_name):
        """Get the index of a floor in the sorted floor list."""
        try:
            return self.floor_list.index(floor_name)
        except ValueError:
            return -1

    def get_move_cost(self, current_floor, target_floor):
        """Calculate the number of up or down actions needed to move between floors."""
        if current_floor == target_floor:
            return 0
        current_index = self.get_floor_index(current_floor)
        target_index = self.get_floor_index(target_floor)
        if current_index == -1 or target_index == -1:
            return 0 # Should not happen in valid problems, but handle for robustness
        return abs(target_index - current_index)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0
        lift_floor = None
        boarded_passengers = set()
        served_passengers = set()
        unserved_passengers = set()

        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])
            elif match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])
            elif match(fact, "origin", "*", "*"):
                unserved_passengers.add(get_parts(fact)[1])
            elif match(fact, "destin", "*", "*"):
                unserved_passengers.add(get_parts(fact)[1])

        current_lift_floor = lift_floor

        for passenger in unserved_passengers:
            if passenger in served_passengers:
                continue

            if passenger not in boarded_passengers:
                origin_floor = self.passenger_origins[passenger]
                if current_lift_floor != origin_floor:
                    heuristic_value += self.get_move_cost(current_lift_floor, origin_floor)
                    current_lift_floor = origin_floor
                heuristic_value += 1 # board action
            else:
                destination_floor = self.passenger_destinations[passenger]
                if current_lift_floor != destination_floor:
                    heuristic_value += self.get_move_cost(current_lift_floor, destination_floor)
                    current_lift_floor = destination_floor
                heuristic_value += 1 # depart action

        return heuristic_value
