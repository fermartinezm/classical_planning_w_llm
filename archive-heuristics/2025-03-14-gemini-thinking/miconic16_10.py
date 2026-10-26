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
    This heuristic estimates the number of actions required to serve all passengers
    by considering the necessary moves of the lift and the board/depart actions for each passenger.
    It calculates the minimum number of moves for the lift to reach each passenger's origin and destination floors.

    # Assumptions:
    - The heuristic assumes that for each unserved passenger, the lift needs to visit their origin floor to board them,
      and then their destination floor to let them depart.
    - It simplifies the problem by considering each passenger independently and summing up the estimated costs.
    - It assumes that the 'above' predicates define a linear ordering of floors.

    # Heuristic Initialization
    - Extracts the 'above' relationships from the static facts to determine the floor order.
    - Creates a mapping from floor names to their index based on the 'above' ordering.
    - Stores destination floors for each passenger from the static facts.
    - Stores origin floors for each passenger from the static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Extract the current lift location from the state.
    3. For each passenger:
        a. Check if the passenger is already served. If yes, no cost is added for this passenger.
        b. If not served, check if the passenger is boarded.
            i. If not boarded, find the passenger's origin floor and the current lift floor.
               Estimate the number of 'move' actions needed to reach the origin floor from the current lift floor
               (using the pre-calculated floor order). Add this number and 1 (for the 'board' action) to the heuristic value.
            ii. If boarded, find the passenger's destination floor and the current lift floor.
                Estimate the number of 'move' actions needed to reach the destination floor from the current lift floor.
                Add this number and 1 (for the 'depart' action) to the heuristic value.
    4. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic.

        - Extracts floor order from 'above' predicates.
        - Creates a floor index map for efficient distance calculation.
        - Stores origin and destination floors for each passenger.
        """
        self.goals = task.goals
        static_facts = task.static

        self.floor_order = []
        above_relations = []
        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                above_relations.append(get_parts(fact)[1:])

        floors_set = set()
        for rel in above_relations:
            floors_set.add(rel[0])
            floors_set.add(rel[1])
        sorted_floors = sorted(list(floors_set)) # Assuming floors can be sorted alphabetically if no explicit order

        if above_relations: # Try to infer order from 'above' if available, otherwise alphabetical
            floor_map = {floor: set() for floor in floors_set}
            for f1, f2 in above_relations:
                floor_map[f1].add(f2)

            bottom_floor = sorted_floors[0] # Assume first floor alphabetically is the bottom if no other info
            ordered_floors = [bottom_floor]
            current_floor = bottom_floor
            while True:
                next_floor_options = floor_map[current_floor]
                if not next_floor_options:
                    break
                next_floor = min(next_floor_options, key=sorted_floors.index) # Pick the next floor in sorted order
                ordered_floors.append(next_floor)
                current_floor = next_floor
            self.floor_order = ordered_floors
        else:
            self.floor_order = sorted_floors


        self.floor_index = {floor: index for index, floor in enumerate(self.floor_order)}

        self.passenger_origins = {}
        self.passenger_destinations = {}
        for fact in static_facts:
            if match(fact, "destin", "*", "*"):
                passenger, floor = get_parts(fact)[1:]
                self.passenger_destinations[passenger] = floor
            if match(fact, "origin", "*", "*"):
                passenger, floor = get_parts(fact)[1:]
                self.passenger_origins[passenger] = floor


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        - Calculates the estimated cost to serve all unserved passengers.
        """
        state = node.state
        heuristic_value = 0

        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            return float('inf') # Should not happen in valid states but handle for robustness

        unserved_passengers = set()
        for passenger in self.passenger_origins: # Consider all passengers defined in the problem
            served = False
            for fact in state:
                if match(fact, "served", passenger):
                    served = True
                    break
            if not served:
                unserved_passengers.add(passenger)

        for passenger in unserved_passengers:
            boarded = False
            for fact in state:
                if match(fact, "boarded", passenger):
                    boarded = True
                    break

            if not boarded:
                origin_floor = self.passenger_origins[passenger]
                if lift_floor is None or origin_floor not in self.floor_index or lift_floor not in self.floor_index:
                    return float('inf') # Handle cases where floor is not defined or lift location is invalid
                floor_distance = abs(self.floor_index.get(origin_floor, 0) - self.floor_index.get(lift_floor, 0))
                heuristic_value += floor_distance + 1 # Moves to origin + board action
            else:
                destination_floor = self.passenger_destinations[passenger]
                if lift_floor is None or destination_floor not in self.floor_index or lift_floor not in self.floor_index:
                    return float('inf') # Handle cases where floor is not defined or lift location is invalid
                floor_distance = abs(self.floor_index.get(destination_floor, 0) - self.floor_index.get(lift_floor, 0))
                heuristic_value += floor_distance + 1 # Moves to destination + depart action

        return heuristic_value
