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
    by considering the necessary moves of the lift and the board/depart actions for each passenger.
    It calculates the cost based on the current state and the remaining tasks for each passenger.

    # Assumptions:
    - The heuristic assumes that for each unserved passenger, the lift needs to move to their origin floor,
      they need to board, the lift needs to move to their destination floor, and they need to depart.
    - The cost of moving between adjacent floors is considered as 1 action.
    - The cost of boarding and departing is also considered as 1 action each.
    - The heuristic is not admissible but aims to provide a good estimate for greedy best-first search.

    # Heuristic Initialization
    - Extracts static information about floor order from the `above` predicates.
    - Stores the origin and destination floors for each passenger.
    - Determines an ordered list of floors based on the `above` relations to calculate move costs.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract the current lift location from the state.
    2. For each passenger:
       - Check if the passenger is already served. If yes, no further actions are needed for this passenger.
       - If not served, check if the passenger is boarded.
         - If boarded, determine the destination floor and calculate the number of moves needed to reach it from the current lift location. Add the move cost and the cost of 'depart' action (1) to the heuristic. Update the lift location to the destination floor.
         - If not boarded, determine the origin floor and calculate the number of moves needed to reach it from the current lift location. Add the move cost and the cost of 'board' action (1) to the heuristic. Update the lift location to the origin floor.
    3. The total accumulated cost is the heuristic value for the given state.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic.
        Extracts passenger origins, destinations, and floor order from the task definition.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.above_relations = set()
        self.floors_list = set()

        for fact in static_facts:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_origins[parts[1]] = parts[2]
                self.floors_list.add(parts[2])
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]
                self.floors_list.add(parts[2])
            elif match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                self.above_relations.add((parts[1], parts[2]))
                self.floors_list.add(parts[1])
                self.floors_list.add(parts[2])

        self.ordered_floors = sorted(list(self.floors_list)) # Simple alphabetical order, can be improved with topological sort if needed for complex instances

    def get_floor_index(self, floor_name):
        """Returns the index of a floor in the ordered floor list."""
        try:
            return self.ordered_floors.index(floor_name)
        except ValueError:
            return 0 # Default to 0 if floor not found, handle error if needed

    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        Estimates the number of actions to reach the goal state from the current state.
        """
        state = node.state
        heuristic_cost = 0
        current_lift_floor = None
        served_passengers = set()
        boarded_passengers = set()
        origin_passengers_current = {}

        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
            elif match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])
            elif match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                origin_passengers_current[parts[1]] = parts[2]


        if current_lift_floor is None:
            return float('inf') # No lift-at predicate, should not happen in valid states but handle for robustness

        passengers_to_serve = set(self.passenger_origins.keys())

        for passenger in passengers_to_serve:
            if f'(served {passenger})' in state:
                continue

            if f'(boarded {passenger})' in state:
                destination_floor = self.passenger_destinations[passenger]
                if current_lift_floor != destination_floor:
                    move_cost = abs(self.get_floor_index(current_lift_floor) - self.get_floor_index(destination_floor))
                    heuristic_cost += move_cost
                    current_lift_floor = destination_floor # Assume lift moves
                heuristic_cost += 1 # depart action
            else:
                origin_floor = self.passenger_origins[passenger]
                if current_lift_floor != origin_floor:
                    move_cost = abs(self.get_floor_index(current_lift_floor) - self.get_floor_index(origin_floor))
                    heuristic_cost += move_cost
                    current_lift_floor = origin_floor # Assume lift moves
                heuristic_cost += 1 # board action

        return heuristic_cost
