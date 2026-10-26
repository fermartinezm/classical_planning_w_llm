from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_floor_level(floor_name):
    """Extract the floor level (integer) from the floor name (e.g., 'f1' -> 1)."""
    return int(floor_name[1:])

def calculate_moves(from_floor_name, to_floor_name):
    """Calculate the number of up/down moves between two floors."""
    from_level = get_floor_level(from_floor_name)
    to_level = get_floor_level(to_floor_name)
    return abs(to_level - from_level)

def get_object_from_fact(fact_str, object_type_index):
    """
    Extracts an object name from a PDDL fact string.

    Args:
        fact_str (str): The PDDL fact string, e.g., '(origin p1 f6)'.
        object_type_index (int): The index of the object in the fact string (starting from 1).

    Returns:
        str: The extracted object name, e.g., 'p1' or 'f6'.
    """
    parts = fact_str[1:-1].split() # Remove parentheses and split
    return parts[object_type_index]

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all passengers
    in the miconic domain. It considers the necessary moves for the lift to reach passenger origins,
    board them, move to their destinations, and depart.

    # Assumptions:
    - The heuristic assumes that each passenger needs to be boarded and departed exactly once.
    - It estimates the number of up/down actions based on the difference in floor levels.
    - It does not consider potential optimizations like serving multiple passengers in a single lift trip.

    # Heuristic Initialization
    - Extracts static information about passenger origins and destinations from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic cost to 0.
    2. Determine the current lift floor from the state.
    3. For each passenger:
       a. Check if the passenger is already served. If yes, no further actions are needed for this passenger.
       b. If not served, check if the passenger is boarded.
          i. If boarded, estimate the cost to move the lift from its current floor to the passenger's destination floor and then depart (1 action).
          ii. If not boarded, estimate the cost to move the lift from its current floor to the passenger's origin floor, board the passenger (1 action), move from the origin floor to the destination floor, and depart (1 action).
    4. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by pre-processing static facts to store origin and destination floors for each passenger."""
        self.goals = task.goals
        static_facts = task.static

        self.origin_floors = {}
        self.destin_floors = {}

        for fact in static_facts:
            if fact.startswith('(origin'):
                passenger = get_object_from_fact(fact, 1)
                floor = get_object_from_fact(fact, 2)
                self.origin_floors[passenger] = floor
            elif fact.startswith('(destin'):
                passenger = get_object_from_fact(fact, 1)
                floor = get_object_from_fact(fact, 2)
                self.destin_floors[passenger] = floor

    def __call__(self, node):
        """Calculate the heuristic value for a given state."""
        state = node.state
        heuristic_value = 0
        lift_floor = None
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            if fact.startswith('(lift-at'):
                lift_floor = get_object_from_fact(fact, 1)
            elif fact.startswith('(boarded'):
                boarded_passengers.add(get_object_from_fact(fact, 1))
            elif fact.startswith('(served'):
                served_passengers.add(get_object_from_fact(fact, 1))

        if lift_floor is None:
            return float('inf') # Should not happen in valid states, but handle for robustness

        passengers = set(self.origin_floors.keys())

        for passenger in passengers:
            if f'(served {passenger})' in state:
                continue # Passenger already served, no cost

            origin_floor = self.origin_floors[passenger]
            destin_floor = self.destin_floors[passenger]

            if f'(boarded {passenger})' in state:
                # Passenger is boarded, need to move to destination and depart
                moves_to_dest = calculate_moves(lift_floor, destin_floor)
                heuristic_value += moves_to_dest + 1 # +1 for depart action
            else:
                # Passenger not boarded, need to move to origin, board, move to destination, depart
                moves_to_origin = calculate_moves(lift_floor, origin_floor)
                moves_origin_to_dest = calculate_moves(origin_floor, destin_floor)
                heuristic_value += moves_to_origin + 1 + moves_origin_to_dest + 1 # +1 for board, +1 for depart

        return heuristic_value
