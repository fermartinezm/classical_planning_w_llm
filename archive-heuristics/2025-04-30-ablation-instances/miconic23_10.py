from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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
    This heuristic estimates the number of actions needed to serve all passengers.
    It counts the required board and depart actions, plus an estimate of the
    minimum number of floor movements (stops at distinct required floors).

    # Assumptions
    - Floors are linearly ordered, defined by `(above f_lower f_higher)` facts
      where `f_higher` is immediately above `f_lower`.
    - All passengers have a defined origin (initially) and destination (static).
    - The goal is to have all passengers `served`.
    - A passenger is either at their origin, boarded, or served.

    # Heuristic Initialization
    - Extracts the linear order of floors and creates a mapping from floor name to number.
      (Note: The current heuristic doesn't strictly *use* the number mapping,
       but building it helps validate the floor structure and could be used
       for distance-based heuristics).
    - Extracts the destination floor for each passenger from static facts.
    - Identifies all passengers involved in the problem by looking at static `destin` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift.
    2. Identify all passengers who are not yet `served`.
    3. Count the number of `board` actions needed: This is the number of unserved
       passengers who are currently at their `origin` floor.
    4. Count the number of `depart` actions needed: This is the total number of
       unserved passengers (each unserved passenger must eventually `depart`
       at their destination).
    5. Identify the set of floors the lift *must* visit:
       - All origin floors of unserved passengers currently waiting there.
       - All destination floors of unserved passengers.
    6. Estimate the number of `move` actions: This is the number of distinct
       floors in the set identified in step 5, excluding the current lift floor.
       This represents the minimum number of *stops* at new floors required.
    7. The total heuristic value is the sum of the counts from steps 3, 4, and 6.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order, passenger destinations,
        and the set of all passengers.
        """
        self.goals = task.goals
        self.static = task.static

        # Build floor mapping: (above f_lower f_higher) means f_higher is one level above f_lower.
        # We want f_lowest -> 1, ..., f_highest -> N.
        floor_above_map = {} # f_lower -> f_higher
        all_floors = set()
        floors_with_floor_below = set() # Floors that appear as f_higher in (above f_lower f_higher)

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'above' and len(parts) == 3:
                f_lower, f_higher = parts[1], parts[2]
                floor_above_map[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)
                floors_with_floor_below.add(f_higher)

        # Find the lowest floor (appears as f_lower but never as f_higher)
        lowest_floor = None
        self.floor_to_number = {}
        self.number_to_floor = {}

        if not all_floors:
             pass # No floors defined
        elif len(all_floors) == 1:
             floor = list(all_floors)[0]
             self.floor_to_number = {floor: 1}
             self.number_to_floor = {1: floor}
        else:
            # Find potential lowest floors (those not appearing as f_higher)
            potential_lowest = all_floors - floors_with_floor_below
            if len(potential_lowest) == 1:
                 lowest_floor = list(potential_lowest)[0]
            elif len(potential_lowest) > 1:
                 # Multiple disconnected chains or malformed. Pick one.
                 lowest_floor = sorted(list(potential_lowest))[0] # Arbitrary but deterministic pick
                 # print(f"Warning: Multiple potential lowest floors found: {potential_lowest}. Picking {lowest_floor}.")
            # else: Cycle or no floors. lowest_floor remains None.

            # Build the mapping upwards from the lowest floor if found
            if lowest_floor:
                current_floor = lowest_floor
                current_number = 1

                while current_floor is not None and current_floor in all_floors:
                    if current_floor in self.floor_to_number: # Cycle detected?
                         # print(f"Warning: Cycle detected in floor relations at {current_floor}.")
                         break # Stop mapping
                    self.floor_to_number[current_floor] = current_number
                    self.number_to_floor[current_number] = current_floor
                    current_number += 1
                    current_floor = floor_above_map.get(current_floor) # Get the floor immediately above

                # Check if all floors were mapped (only relevant for linear chains)
                # if len(self.floor_to_number) != len(all_floors):
                     # print("Warning: Not all floors are in a single linear chain.")


        # Extract passenger destinations and collect all passenger names
        self.passenger_destinations = {}
        self.all_passengers = set()

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'destin' and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.passenger_destinations[passenger] = floor
                self.all_passengers.add(passenger)

        # Add passengers from goals (should also be in destin static, but safety)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'served' and len(parts) == 2:
                 self.all_passengers.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*")):
                current_lift_floor = get_parts(fact)[1]
                break

        if current_lift_floor is None:
             # Should not happen in a valid miconic state
             # print("Warning: Lift location not found in state.")
             return float('inf') # Cannot solve if lift location is unknown

        # Identify unserved passengers
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = {p for p in self.all_passengers if p not in served_passengers}

        # If all passengers are served, the goal is reached.
        if not unserved_passengers:
            return 0

        # Count non-move actions (board and depart)
        board_actions_needed = 0
        depart_actions_needed = len(unserved_passengers) # Each unserved needs 1 depart

        # Collect required floors for moves
        required_floors = set()

        for passenger in unserved_passengers:
            is_origin = False
            origin_floor = None

            # Check if passenger is at origin
            for fact in state:
                if match(fact, "origin", passenger, "*")):
                    is_origin = True
                    origin_floor = get_parts(fact)[2]
                    break

            # Add required floors based on passenger status
            if is_origin:
                board_actions_needed += 1
                required_floors.add(origin_floor)

            # Destination floor is required for all unserved passengers
            dest_floor = self.passenger_destinations.get(passenger)
            if dest_floor:
                required_floors.add(dest_floor)
            # else: Passenger has no destination? Problematic instance.


        # Estimate move actions
        # Count distinct required floors excluding the current one.
        move_actions_needed = len(required_floors - {current_lift_floor})

        # Total heuristic value
        total_cost = board_actions_needed + depart_actions_needed + move_actions_needed

        return total_cost
