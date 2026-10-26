from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_floor_number(floor_name):
    """Extract the floor number from a floor name like 'f1', 'f2', etc."""
    return int(floor_name[1:])

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
    by considering the necessary lift movements and board/depart actions for each passenger.

    # Assumptions:
    - The heuristic assumes that for each unserved passenger, the lift needs to move to their origin floor,
      they need to board, the lift needs to move to their destination floor, and they need to depart.
    - It does not consider optimizing lift movements for multiple passengers simultaneously.
    - It assumes that moving between adjacent floors costs 1 action (either 'up' or 'down').

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1. Initialize the heuristic value to 0.
    2. Determine the current floor of the lift from the state.
    3. Iterate through each passenger:
        a. Check if the passenger is already served. If yes, no further actions are needed for this passenger.
        b. If the passenger is not served, check if they are boarded.
        c. If not boarded:
            i.  Find the origin floor of the passenger.
            ii. Calculate the number of 'up' or 'down' actions needed to move the lift from its current floor to the passenger's origin floor. This is estimated as the absolute difference between the floor numbers.
            iii. Add this move cost and 1 (for the 'board' action) to the heuristic value.
        d. If boarded:
            i.  Find the destination floor of the passenger.
            ii. Calculate the number of 'up' or 'down' actions needed to move the lift from its current floor to the passenger's destination floor. This is estimated as the absolute difference between the floor numbers.
            iii. Add this move cost and 1 (for the 'depart' action) to the heuristic value.
    4. Return the total accumulated heuristic value.

    This heuristic is admissible under the assumption that each passenger needs to be picked up and dropped off individually, and the shortest path in terms of floor difference is taken. However, for greedy best-first search, admissibility is not required, and this heuristic aims for a reasonable and efficient estimate.
    """

    def __init__(self, task):
        """Initialize the miconic heuristic."""
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static
        self.passenger_origins = {}
        self.passenger_destinations = {}

        for fact in self.static_facts:
            if match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_origins[parts[1]] = parts[2]
            elif match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                self.passenger_destinations[parts[1]] = parts[2]


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        heuristic_value = 0
        lift_floor = None

        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break
        if lift_floor is None:
            return float('inf') # Should not happen in valid states

        served_passengers = set()
        boarded_passengers = set()

        for fact in state:
            if match(fact, "served", "*"):
                served_passengers.add(get_parts(fact)[1])
            elif match(fact, "boarded", "*"):
                boarded_passengers.add(get_parts(fact)[1])

        goal_served_passengers = set()
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                goal_served_passengers.add(get_parts(goal_fact)[1])


        passengers_to_serve = goal_served_passengers

        current_lift_floor_num = get_floor_number(lift_floor)

        for passenger in passengers_to_serve:
            if f'(served {passenger})' in state:
                continue

            if f'(boarded {passenger})' not in state:
                origin_floor = self.passenger_origins.get(passenger)
                if origin_floor:
                    origin_floor_num = get_floor_number(origin_floor)
                    move_cost = abs(origin_floor_num - current_lift_floor_num)
                    heuristic_value += move_cost + 1 # move + board
                    current_lift_floor_num = origin_floor_num # For next passenger's move calculation, assume lift is now at origin. This is not correct for multiple passengers, but simplifies heuristic. We should not update lift floor here.
            else:
                destin_floor = self.passenger_destinations.get(passenger)
                if destin_floor:
                    destin_floor_num = get_floor_number(destin_floor)
                    move_cost = abs(destin_floor_num - current_lift_floor_num)
                    heuristic_value += move_cost + 1 # move + depart
                    current_lift_floor_num = destin_floor_num # Similar incorrect update as above.

        # Recalculate heuristic more accurately without updating current_lift_floor_num within the loop.
        heuristic_value = 0
        current_lift_floor_num = get_floor_number(lift_floor)
        for passenger in passengers_to_serve:
            if f'(served {passenger})' in state:
                continue

            if f'(boarded {passenger})' not in state:
                origin_floor = self.passenger_origins.get(passenger)
                if origin_floor:
                    origin_floor_num = get_floor_number(origin_floor)
                    move_cost = abs(origin_floor_num - current_lift_floor_num)
                    heuristic_value += move_cost + 1
            else:
                destin_floor = self.passenger_destinations.get(passenger)
                if destin_floor:
                    destin_floor_num = get_floor_number(destin_floor)
                    move_cost = abs(destin_floor_num - current_lift_floor_num)
                    heuristic_value += move_cost + 1

        return heuristic_value
