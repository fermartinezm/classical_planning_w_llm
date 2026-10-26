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
    This heuristic estimates the minimum number of actions required to serve all passengers
    by considering the necessary steps for each unserved passenger: moving the lift to
    the origin floor, boarding, moving to the destination floor, and departing.

    # Assumptions:
    - The heuristic assumes that for each unserved passenger, the lift needs to visit their
      origin floor to board them and their destination floor to let them depart.
    - It estimates the cost based on the number of floor movements and board/depart actions.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.

    # Heuristic Initialization
    - Extracts static information about passenger origins and destinations, and the floor order
      from the 'above' predicates.
    - Precomputes the origin and destination floor for each passenger and the ordered list of floors.

    # Step-By-Step Thinking for Computing Heuristic
    1. Extract passenger origins and destinations from the static facts.
    2. Determine the order of floors based on the 'above' predicates.
    3. For a given state, iterate through each passenger.
    4. If a passenger is already served, no further actions are needed for them.
    5. If a passenger is not served:
       a. Determine the passenger's origin and destination floors.
       b. If the passenger is not yet boarded:
          i. Estimate the cost to move the lift from its current floor to the passenger's origin floor.
             This is calculated as the number of floors to traverse based on the precomputed floor order.
          ii. Add 1 to the cost for the 'board' action.
       c. Estimate the cost to move the lift from the current (or origin) floor to the passenger's destination floor.
          This is again calculated based on the floor order.
       d. Add 1 to the cost for the 'depart' action.
    6. Sum up the estimated costs for all unserved passengers to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the miconic heuristic by extracting passenger origins, destinations,
        and floor order from the static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.passenger_origins = {}
        self.passenger_destinations = {}
        self.floor_order = []

        above_relations = []

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
                above_relations.append((parts[1], parts[2]))

        # Determine floor order - assuming a simple linear order from 'above' facts.
        floors_set = set()
        for p in self.passenger_origins:
            floors_set.add(self.passenger_origins[p])
            floors_set.add(self.passenger_destinations[p])
        lift_floors_init = set()
        for fact in task.initial_state:
            if match(fact, "lift-at", "*"):
                lift_floors_init.add(get_parts(fact)[1])
        floors_set.update(lift_floors_init)

        if floors_set:
            bottom_floor = None
            top_floor = None
            floor_names = sorted(list(floors_set)) # Sort floors alphabetically to get a consistent order if 'above' is incomplete
            if floor_names:
                bottom_floor = floor_names[0]
                current_floor = bottom_floor
                ordered_floors = [bottom_floor]
                processed_floors = {bottom_floor}

                while True:
                    next_floor = None
                    for f1, f2 in above_relations:
                        if f2 == current_floor and f1 not in processed_floors:
                            next_floor = f1
                            break
                    if next_floor:
                        ordered_floors.append(next_floor)
                        processed_floors.add(next_floor)
                        current_floor = next_floor
                    else:
                        break
                self.floor_order = ordered_floors
        else:
            self.floor_order = []


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.
        """
        state = node.state
        heuristic_value = 0
        lift_floor = None

        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        if lift_floor is None:
            return float('inf') # No lift location, should not happen in valid states, but handle for robustness

        for passenger in self.passenger_origins:
            if f'(served {passenger})' in state:
                continue # Passenger already served, no cost

            origin_floor = self.passenger_origins[passenger]
            destination_floor = self.passenger_destinations[passenger]
            is_boarded = f'(boarded {passenger})' in state

            current_lift_floor = lift_floor

            if not is_boarded:
                if current_lift_floor != origin_floor:
                    if self.floor_order:
                        try:
                            origin_index = self.floor_order.index(origin_floor)
                            current_index = self.floor_order.index(current_lift_floor)
                            heuristic_value += abs(origin_index - current_index)
                        except ValueError:
                            heuristic_value += 1 # If floor order is incomplete, assume 1 move
                    else:
                        heuristic_value += 1 # Default to 1 move if floor order is not available
                heuristic_value += 1 # board action
                current_lift_floor = origin_floor # Lift is now at origin floor (virtually)

            if current_lift_floor != destination_floor:
                if self.floor_order:
                    try:
                        destination_index = self.floor_order.index(destination_floor)
                        current_index = self.floor_order.index(current_lift_floor)
                        heuristic_value += abs(destination_index - current_index)
                    except ValueError:
                        heuristic_value += 1 # If floor order is incomplete, assume 1 move
                else:
                    heuristic_value += 1 # Default to 1 move if floor order is not available
            heuristic_value += 1 # depart action

        return heuristic_value
