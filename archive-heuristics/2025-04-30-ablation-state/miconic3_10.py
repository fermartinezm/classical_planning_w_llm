from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to transport all
    passengers to their destination floors. It sums the estimated cost for
    each unserved passenger. The cost for a passenger depends on whether
    they are waiting at their origin or are already boarded.

    # Assumptions
    - Floors are linearly ordered, defined by `(above f_higher f_lower)` facts.
    - Each passenger has a unique origin and destination floor, specified in the initial state.
    - The goal is to have all specified passengers served.
    - The lift has infinite capacity.

    # Heuristic Initialization
    - Parses static facts (`above` predicates) to determine the floor ordering
      and map floor names to numerical levels. Assumes `(above f_higher f_lower)`
      means `f_higher` is directly above `f_lower`.
    - Extracts the origin and destination floors for each passenger that needs
      to be served according to the goal state, using facts from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:
    1. Identify the current floor of the lift by finding the `(lift-at ?f)` fact.
    2. Identify which passengers have already been served by finding `(served ?p)` facts.
    3. Initialize the total heuristic cost to 0.
    4. For each passenger that is listed in the goal as needing to be served:
       a. Check if the passenger is already in the set of served passengers. If yes, continue to the next passenger (cost for this passenger is 0).
       b. Retrieve the passenger's origin and destination floors (pre-calculated in init).
       c. Check if the passenger is currently boarded in the lift by finding the `(boarded ?p)` fact.
       d. Get the numerical levels for the current lift floor, the passenger's origin floor, and the passenger's destination floor using the floor-to-level map.
       e. If the passenger is boarded:
          - The estimated cost for this passenger is the number of move actions
            required to travel from the current lift floor to the passenger's
            destination floor, plus 1 action for the `depart` action.
          - Cost = `abs(level(current_lift_floor) - level(destination_floor)) + 1`.
       f. If the passenger is not boarded (meaning they are waiting at their origin):
          - The estimated cost for this passenger is the number of move actions
            required to travel from the current lift floor to the passenger's
            origin floor, plus 1 action for the `board` action, plus the number
            of move actions required to travel from the origin floor to the
            destination floor, plus 1 action for the `depart` action.
          - Cost = `abs(level(current_lift_floor) - level(origin_floor)) + 1 + abs(level(origin_floor) - level(destination_floor)) + 1`.
       g. Add the calculated cost for this unserved passenger to the total heuristic cost.
    5. Return the total heuristic value.
    6. If the lift location or a required floor level cannot be determined, return infinity, indicating a potentially unsolvable or malformed state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger info.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state contains origin/destin

        # --- Parse floor ordering and assign levels ---
        # Map floor_above -> floor_below from (above f_above f_below) facts
        floor_above_to_floor_below = {}
        all_floors_in_above = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f_above, f_below = parts[1], parts[2]
                floor_above_to_floor_below[f_above] = f_below
                all_floors_in_above.add(f_above)
                all_floors_in_above.add(f_below)

        self.floor_to_level = {}
        if all_floors_in_above:
            # Find the highest floor (a floor that is never f_below in any (above f_above f_below) fact)
            floors_that_are_below_something = set(floor_above_to_floor_below.values())
            highest_floor = None
            for floor in all_floors_in_above:
                if floor not in floors_that_are_below_something:
                    highest_floor = floor
                    break

            if highest_floor:
                # Build floor_to_level map by traversing downwards from the highest floor
                current_floor = highest_floor
                # Assign levels starting from 0 for the lowest floor.
                # The highest floor will have level = number_of_floors - 1.
                # We can determine the number of floors by traversing the chain.
                num_floors = 0
                temp_floor = highest_floor
                while temp_floor is not None:
                    num_floors += 1
                    temp_floor = floor_above_to_floor_below.get(temp_floor)

                current_floor = highest_floor
                level = num_floors - 1
                while current_floor is not None:
                    self.floor_to_level[current_floor] = level
                    level -= 1
                    current_floor = floor_above_to_floor_below.get(current_floor)
            # else: Handle case where above facts don't form a single chain? Assume valid problems do.

        # --- Extract passenger origin and destination info ---
        # Collect origin and destin for all passengers mentioned in initial state
        all_passenger_info = {}
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "origin" and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 if p not in all_passenger_info:
                     all_passenger_info[p] = [None, None] # [origin, destin]
                 all_passenger_info[p][0] = f
             elif parts and parts[0] == "destin" and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 if p not in all_passenger_info:
                     all_passenger_info[p] = [None, None]
                 all_passenger_info[p][1] = f

        # Identify passengers that need to be served according to the goal
        passengers_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served" and len(parts) == 2:
                passengers_to_serve.add(parts[1])

        # Store info only for passengers that need serving
        self.passenger_info = {
            p: info for p, info in all_passenger_info.items()
            if p in passengers_to_serve
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid miconic state
             return float('inf') # Cannot proceed without lift location

        # Get set of served passengers
        served_passengers = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) == 2:
                served_passengers.add(parts[1])

        total_cost = 0  # Initialize action cost counter.

        # Iterate through passengers that need serving (identified in init)
        for passenger, (origin_floor, destin_floor) in self.passenger_info.items():
            # If passenger is already served, they don't contribute to the heuristic
            if passenger in served_passengers:
                continue

            # Check if the passenger is boarded
            is_boarded = f"(boarded {passenger})" in state

            # Get floor levels (handle potential missing floors defensively, though unlikely in valid problems)
            # If a floor is not in floor_to_level, it means it wasn't part of the 'above' chain.
            # This might happen if there's only one floor, or if the PDDL is malformed.
            # If a floor is missing, we can't calculate distance, so return infinity.
            current_level = self.floor_to_level.get(current_lift_floor)
            origin_level = self.floor_to_level.get(origin_floor)
            destin_level = self.floor_to_level.get(destin_floor)

            if current_level is None or origin_level is None or destin_level is None:
                 # This indicates an issue with floor parsing or a malformed problem
                 # where origin/destin floors are not in the 'above' chain.
                 # In a well-formed miconic problem, all relevant floors should be ordered.
                 # Returning infinity signals an unsolvable state or parsing error.
                 return float('inf')

            if is_boarded:
                # Passenger is in the lift, needs to go to destination and depart
                # Cost = move to destin + depart
                cost = abs(current_level - destin_level) + 1
                total_cost += cost
            else:
                # Passenger is waiting at origin, needs pickup, transport, and dropoff
                # Cost = move to origin + board + move to destin + depart
                cost = abs(current_level - origin_level) + 1 + abs(origin_level - destin_level) + 1
                total_cost += cost

        return total_cost
