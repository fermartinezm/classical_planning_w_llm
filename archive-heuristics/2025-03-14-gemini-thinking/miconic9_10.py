from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_floor_number(floor_name):
    """Extracts the floor number from a floor name like 'f1', 'f2', etc."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    return -1  # Return -1 if the floor name doesn't match the expected format

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    by considering the necessary board, depart, up, and down actions. It calculates
    the minimum moves needed for the lift to reach each passenger's origin and
    destination floors and sums up the actions for all passengers.

    # Assumptions:
    - The floors are linearly ordered based on the 'above' predicates.
    - The heuristic assumes that for each unserved passenger, the lift will first
      go to their origin floor, then to their destination floor.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - Extracts static information about passenger origins, destinations, and floor order
      from the task definition.
    - Pre-computes an ordered list of floors based on 'above' predicates to efficiently
      estimate lift movement costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost to 0.
    2. Identify all passengers who are not yet served.
    3. For each unserved passenger:
        a. Check if the passenger is boarded.
        b. If not boarded:
            i.  Determine the passenger's origin floor.
            ii. Determine the current lift floor.
            iii.Estimate the number of 'up' or 'down' actions needed to move the lift
                from its current floor to the passenger's origin floor. Add this to the cost.
            iv. Add 1 action for the 'board' action.
        c. If boarded:
            i.  Determine the passenger's destination floor.
            ii. Determine the current lift floor.
            iii.Estimate the number of 'up' or 'down' actions needed to move the lift
                from its current floor to the passenger's destination floor. Add this to the cost.
            iv. Add 1 action for the 'depart' action.
    4. The total accumulated cost is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic. Extracts passenger origins, destinations,
        and floor order from the task's static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.origin_floors = {}
        self.destination_floors = {}
        self.above_relations = set()
        self.all_floors_ordered = []

        floors_set = set()
        passengers_set = set()

        for fact in static_facts:
            fact_parts = self._get_fact_parts(fact)
            if fact_parts[0] == 'origin':
                passenger = fact_parts[1]
                floor = fact_parts[2]
                self.origin_floors[passenger] = floor
                passengers_set.add(passenger)
                floors_set.add(floor)
            elif fact_parts[0] == 'destin':
                passenger = fact_parts[1]
                floor = fact_parts[2]
                self.destination_floors[passenger] = floor
                passengers_set.add(passenger)
                floors_set.add(floor)
            elif fact_parts[0] == 'above':
                floor1 = fact_parts[1]
                floor2 = fact_parts[2]
                self.above_relations.add((floor1, floor2))
                floors_set.add(floor1)
                floors_set.add(floor2)

        self.passengers = list(passengers_set)
        self.floors = list(floors_set)
        self.floors.sort(key=get_floor_number) # Order floors based on number in name

    def _get_fact_parts(self, fact_str):
        """Extracts predicate and objects from a fact string."""
        return fact_str[1:-1].split()

    def __call__(self, node):
        """
        Computes the heuristic value for a given state in the miconic domain.
        Estimates the number of actions to serve all passengers.
        """
        state = node.state
        served_passengers = set()
        boarded_passengers = set()
        current_lift_floor = None
        current_origin_floors = {}

        for fact in state:
            fact_parts = self._get_fact_parts(fact)
            predicate = fact_parts[0]
            if predicate == 'served':
                served_passengers.add(fact_parts[1])
            elif predicate == 'boarded':
                boarded_passengers.add(fact_parts[1])
            elif predicate == 'lift-at':
                current_lift_floor = fact_parts[1]
            elif predicate == 'origin':
                current_origin_floors[fact_parts[1]] = fact_parts[2]


        unserved_passengers = [p for p in self.passengers if p not in served_passengers]
        heuristic_value = 0
        lift_location = current_lift_floor

        for passenger in unserved_passengers:
            if passenger not in boarded_passengers:
                origin_floor = self.origin_floors[passenger]

                if lift_location != origin_floor:
                    origin_floor_index = self.floors.index(origin_floor)
                    lift_floor_index = self.floors.index(lift_location)
                    heuristic_value += abs(origin_floor_index - lift_floor_index)
                    lift_location = origin_floor
                heuristic_value += 1 # board action
            else:
                destination_floor = self.destination_floors[passenger]
                if lift_location != destination_floor:
                    destination_floor_index = self.floors.index(destination_floor)
                    lift_floor_index = self.floors.index(lift_location)
                    heuristic_value += abs(destination_floor_index - lift_floor_index)
                    lift_location = destination_floor
                heuristic_value += 1 # depart action

        return heuristic_value
