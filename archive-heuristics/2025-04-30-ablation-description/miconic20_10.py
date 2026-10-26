from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the total number of actions (moves, board, depart)
    required to serve all passengers who are not yet served. It calculates the
    cost for each unserved passenger independently, assuming the lift can go
    directly to their location (origin or current lift position if boarded)
    and then to their destination.

    # Assumptions
    - The floors are linearly ordered by the `(above f1 f2)` predicates.
    - The cost of moving between adjacent floors is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic calculates the sum of costs for each unserved passenger
      as if they could be served sequentially without interference, ignoring
      lift capacity or optimal routing for multiple passengers.

    # Heuristic Initialization
    - Extracts the linear ordering of floors from the `(above f1 f2)` static facts.
    - Creates a map from floor names to their numerical index (level) in the ordered list.
    - Extracts the destination floor for each passenger from the `(destin p f)`
      facts in the static information or goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the current floor of the lift.
    3. Initialize the total heuristic cost to 0.
    4. Iterate through all passengers whose destination is known (from initialization).
    5. For each passenger:
       a. Check if the passenger is already served (`(served p)` is true). If yes, skip this passenger.
       b. Get the passenger's destination floor.
       c. Check if the passenger is currently boarded (`(boarded p)` is true).
       d. If the passenger is boarded:
          - The estimated cost for this passenger is the distance (absolute difference in floor index) from the current lift floor to their destination floor, plus 1 for the `depart` action.
          - Add this cost to the total heuristic.
       e. If the passenger is not boarded:
          - Find the passenger's origin floor (`(origin p f)` is true).
          - The estimated cost for this passenger is the distance from the current lift floor to their origin floor (to pick them up), plus 1 for the `board` action, plus the distance from their origin floor to their destination floor (to transport them), plus 1 for the `depart` action.
          - Add this cost to the total heuristic.
    6. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor ordering and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions
        static_facts = task.static  # Facts that are not affected by actions.

        # 1. Extract floor ordering and create floor index map
        floor_above_map = {} # Maps floor_below -> floor_above
        all_floors = set()
        floors_that_are_above_others = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                f_above, f_below = get_parts(fact)[1:]
                floor_above_map[f_below] = f_above
                all_floors.add(f_above)
                all_floors.add(f_below)
                floors_that_are_above_others.add(f_above)

        self.ordered_floors = []
        self.floor_index_map = {}

        if all_floors: # Handle case with no floors or no above facts
            # Find the bottom floor: a floor that is in all_floors but is not
            # the first argument of any (above f_above f_below) fact.
            bottom_floor = None
            for floor in all_floors:
                if floor not in floors_that_are_above_others:
                    bottom_floor = floor
                    break # Found the bottom floor

            if bottom_floor:
                # Rebuild floor_above_map to map f_below -> f_above
                # This map is already built correctly in the first loop.
                # floor_above_map contains f_below -> f_above for (above f_above f_below)

                current_floor = bottom_floor
                index = 0
                # Build the ordered list and index map by following the 'above' chain upwards
                while current_floor is not None:
                    self.ordered_floors.append(current_floor)
                    self.floor_index_map[current_floor] = index
                    index += 1
                    # Find the floor directly above the current_floor using the map
                    current_floor = floor_above_map.get(current_floor)

            else:
                 # Fallback if no clear bottom floor is found (e.g., single floor or complex structure)
                 # Just add all floors found. Distance will be 0 if only one.
                 # If multiple floors but no clear order, distance calculation will be wrong.
                 # Sorting alphabetically is a simple fallback if ordering fails.
                 sorted_floors = sorted(list(all_floors))
                 self.ordered_floors = sorted_floors
                 self.floor_index_map = {floor: i for i, floor in enumerate(sorted_floors)}


        # 2. Extract passenger destinations
        self.passenger_destinations = {}
        # Destinations can be in static facts or goals
        facts_to_check = set(static_facts) | set(self.goals)
        for fact in facts_to_check:
            if match(fact, "destin", "*", "*"):
                p, f_destin = get_parts(fact)[1:]
                self.passenger_destinations[p] = f_destin

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state.

        # 1. Check if goal is reached
        # The goal is defined as a set of facts.
        # The Task object has a goal_reached method, but we can also check directly.
        # The heuristic should be 0 if and only if the goal is reached.
        # Checking self.goals <= state is the correct way.
        if self.goals <= state:
             return 0

        # 2. Find current lift floor
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # This case indicates an invalid state for the miconic domain.
        # Returning infinity signals this state is undesirable/unreachable by valid actions.
        if lift_floor is None:
             return float('inf')

        total_cost = 0  # Initialize action cost counter.

        # 4. Iterate through passengers and calculate cost
        for passenger, dest_floor in self.passenger_destinations.items():
            # 5a. Check if served
            if f"(served {passenger})" in state:
                continue # Passenger is already served

            # 5c. Check if boarded
            is_boarded = f"(boarded {passenger})" in state

            if is_boarded:
                # 5d. Passenger is boarded
                # Cost = distance from current lift floor to destination + 1 (depart)
                current_floor = lift_floor
                # Use .get() with a default of 0 in case a floor isn't in the map (shouldn't happen in valid states)
                cost_for_passenger = abs(self.floor_index_map.get(current_floor, 0) - self.floor_index_map.get(dest_floor, 0)) + 1
                total_cost += cost_for_passenger
            else:
                # 5e. Passenger is not boarded, find origin
                origin_floor = None
                # Search for the origin fact in the current state
                for fact in state:
                    if match(fact, "origin", passenger, "*"):
                        origin_floor = get_parts(fact)[2]
                        break

                # If origin_floor is None, the passenger is unserved, not boarded,
                # and not waiting at an origin. This state is likely invalid
                # or indicates a passenger was somehow removed from their origin
                # without being boarded. For a valid state reachable by domain actions,
                # an unserved, unboarded passenger must have an (origin p f) fact.
                # We should only calculate cost for passengers in expected states.
                # If origin is None, skip this passenger as they are not in a state
                # the heuristic can interpret based on the domain definition.
                if origin_floor is None:
                     continue # Skip passengers in unexpected states

                # Passenger is waiting at origin_floor
                # Cost = distance(lift, origin) + 1 (board) + distance(origin, destin) + 1 (depart)
                cost_for_passenger = (
                    abs(self.floor_index_map.get(lift_floor, 0) - self.floor_index_map.get(origin_floor, 0))
                    + 1
                    + abs(self.floor_index_map.get(origin_floor, 0) - self.floor_index_map.get(dest_floor, 0))
                    + 1
                )
                total_cost += cost_for_passenger

        return total_cost
