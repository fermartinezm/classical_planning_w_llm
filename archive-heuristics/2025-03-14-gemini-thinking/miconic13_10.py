from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import re

def get_floor_number(floor_name):
    """Extract the floor number from a floor name like 'f1', 'f2', etc."""
    match = re.match(r'f(\d+)', floor_name)
    if match:
        return int(match.group(1))
    return None

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
    by considering the necessary lift movements and board/depart actions for each unserved passenger.

    # Assumptions:
    - Floors are named in a sequence like f1, f2, f3, ... and the 'above' predicate defines their order.
    - The heuristic assumes that for each unserved passenger, the lift needs to move to their origin floor,
      they board, the lift moves to their destination floor, and they depart.
    - It calculates the number of up/down actions based on the difference in floor numbers.

    # Heuristic Initialization
    - Extracts static information about passenger origins and destinations, and 'above' relationships
      to understand floor order.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Determine the current location of the lift from the state.
    3. For each passenger:
       a. Check if the passenger is already served. If yes, no further actions are needed for this passenger.
       b. If not served, check if the passenger is boarded.
          i. If not boarded, find the passenger's origin floor.
             - Estimate the number of 'up' or 'down' actions needed to move the lift from its current location to the origin floor based on the difference in floor numbers. Add this to the heuristic value.
             - Add 1 to the heuristic value for the 'board' action.
             - Update the assumed lift location to the origin floor for the next passenger's calculation.
          ii. If boarded, find the passenger's destination floor.
              - Estimate the number of 'up' or 'down' actions needed to move the lift from its current location to the destination floor. Add this to the heuristic value.
              - Add 1 to the heuristic value for the 'depart' action.
              - Update the assumed lift location to the destination floor for the next passenger's calculation.
    4. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting passenger origins, destinations, and floor order."""
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above_relations = set()

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_origins[passenger] = floor
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                passenger = parts[1]
                floor = parts[2]
                self.passenger_destinations[passenger] = floor
            elif match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                floor1 = parts[1]
                floor2 = parts[2]
                self.above_relations.add((floor1, floor2))

    def __call__(self, node):
        """Estimate the number of actions to reach the goal state from the current state."""
        state = node.state
        heuristic_value = 0
        current_lift_floor = None

        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
            return float('inf') # Should not happen in valid problems

        current_lift_floor_num = get_floor_number(current_lift_floor)

        passengers = set()
        for fact in state:
            if match(fact, "origin", "*", "*"):
                passengers.add(get_parts(fact)[1])
            elif match(fact, "destin", "*", "*"):
                passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                passengers.add(get_parts(fact)[1])
            elif match(fact, "served", "*"):
                passengers.add(get_parts(fact)[1])

        for passenger in passengers:
            if f'(served {passenger})' not in state:
                if f'(boarded {passenger})' not in state:
                    origin_floor = self.passenger_origins.get(passenger)
                    if origin_floor:
                        origin_floor_num = get_floor_number(origin_floor)
                        if origin_floor_num is not None and current_lift_floor_num is not None:
                            heuristic_value += abs(origin_floor_num - current_lift_floor_num)
                            current_lift_floor_num = origin_floor_num # Assume lift moves to origin
                        heuristic_value += 1 # board action
                else:
                    destination_floor = self.passenger_destinations.get(passenger)
                    if destination_floor:
                        destination_floor_num = get_floor_number(destination_floor)
                        if destination_floor_num is not None and current_lift_floor_num is not None:
                            heuristic_value += abs(destination_floor_num - current_lift_floor_num)
                            current_lift_floor_num = destination_floor_num # Assume lift moves to destination
                        heuristic_value += 1 # depart action

        return heuristic_value
