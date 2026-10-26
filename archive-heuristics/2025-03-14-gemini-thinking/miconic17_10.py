from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_objects_from_fact(fact_str):
    """
    Extracts objects from a PDDL fact string, ignoring surrounding brackets.
    For example, '(predicate_name object1 object2)' becomes ['predicate_name', 'object1', 'object2'].
    """
    return fact_str[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers in the Miconic domain.
    It calculates the cost for each unserved passenger based on their origin and destination floors and the current lift position.

    # Assumptions:
    - Each move action (up or down) has a cost of 1.
    - Boarding and departing actions each have a cost of 1.
    - The heuristic assumes an optimal sequence of up and down actions to reach a target floor.
    - It does not consider the interaction between serving multiple passengers concurrently,
      potentially underestimating the actual cost in complex scenarios.

    # Heuristic Initialization
    - Extracts static information: destination floor for each passenger and the 'above' relationships between floors.
    - Determines a sorted list of floors based on 'above' relationships to estimate move costs.

    # Step-By-Step Thinking for Computing Heuristic
    For each passenger that is not yet served:
    1. If the passenger is not yet boarded:
        a. Calculate the number of moves required to bring the lift from its current floor to the passenger's origin floor.
        b. Add 1 action for the 'board' action.
        c. Calculate the number of moves required to bring the lift from the origin floor to the passenger's destination floor.
        d. Add 1 action for the 'depart' action.
    2. If the passenger is already boarded:
        a. Calculate the number of moves required to bring the lift from its current floor to the passenger's destination floor.
        b. Add 1 action for the 'depart' action.
    3. Sum up the costs for all unserved passengers to get the total heuristic estimate.

    The number of moves between two floors is estimated by finding their indices in a sorted list of floors derived from the 'above' predicates and taking the absolute difference of these indices.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic.
        Extracts destination floors for each passenger and builds a sorted list of floors based on 'above' predicates.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_destinations = {}
        self.above_relations = []
        self.floors_list = set()

        for fact in static_facts:
            parts = get_objects_from_fact(fact)
            if parts[0] == 'destin':
                self.passenger_destinations[parts[1]] = parts[2]
            elif parts[0] == 'above':
                self.above_relations.append((parts[1], parts[2]))
                self.floors_list.add(parts[1])
                self.floors_list.add(parts[2])

        if not self.floors_list: # Handle case with only one floor
            self.sorted_floors = list()
        else:
            self.sorted_floors = sorted(list(self.floors_list), key=lambda floor: self.get_floor_level(floor)) # Sort floors based on level
        self.floor_indices = {floor: index for index, floor in enumerate(self.sorted_floors)}


    def get_floor_level(self, floor_name):
        """
        Determines a level for each floor based on 'above' relations.
        This is a simplified approach and might not be perfect for all possible 'above' configurations,
        but it serves to provide a relative ordering for heuristic calculation.
        """
        level = 0
        current_floor = floor_name
        processed_floors = {floor_name} # To avoid infinite loops in cyclic above definitions (though unlikely in typical miconic problems)

        while True:
            found_below = False
            for f1, f2 in self.above_relations:
                if f2 == current_floor:
                    current_floor = f1
                    level += 1
                    if current_floor in processed_floors: # Basic cycle detection
                        return level # Break cycle, level might be underestimated
                    processed_floors.add(current_floor)
                    found_below = True
                    break
            if not found_below:
                break
        return level


    def get_moves_cost(self, from_floor, to_floor):
        """
        Estimates the number of moves between two floors based on their indices in the sorted floor list.
        If floors are not in the sorted list (e.g., problem definition error), returns 0 cost.
        """
        if from_floor not in self.floor_indices or to_floor not in self.floor_indices:
            return 0
        return abs(self.floor_indices[to_floor] - self.floor_indices[from_floor])


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0

        lift_floor = None
        served_passengers = set()
        boarded_passengers = set()
        origin_passengers = {}

        for fact in state:
            parts = get_objects_from_fact(fact)
            predicate = parts[0]
            if predicate == 'lift-at':
                lift_floor = parts[1]
            elif predicate == 'served':
                served_passengers.add(parts[1])
            elif predicate == 'boarded':
                boarded_passengers.add(parts[1])
            elif predicate == 'origin':
                origin_passengers[parts[1]] = parts[2]

        unserved_passengers = set(self.passenger_destinations.keys()) - served_passengers

        for passenger in unserved_passengers:
            destination_floor = self.passenger_destinations[passenger]
            if passenger in boarded_passengers:
                # Passenger is boarded, need to move to destination and depart
                heuristic_value += self.get_moves_cost(lift_floor, destination_floor) + 1 # Moves + depart
            else:
                origin_floor = origin_passengers.get(passenger)
                if origin_floor: # Origin should always be defined for unserved and not boarded passenger
                    # Passenger not boarded, need to move to origin, board, move to destination, depart
                    heuristic_value += self.get_moves_cost(lift_floor, origin_floor) + 1 # Moves to origin + board
                    heuristic_value += self.get_moves_cost(origin_floor, destination_floor) + 1 # Moves to destination + depart

        return heuristic_value
