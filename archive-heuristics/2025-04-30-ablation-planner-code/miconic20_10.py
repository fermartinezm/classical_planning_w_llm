# Assuming heuristic_base.py provides a Heuristic base class
# from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

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

# Define a dummy Heuristic base class if not provided, just for structure
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

class miconicHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Miconic domain.

    Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It sums the required board and depart actions for unserved passengers and adds
    an estimate of the lift movement cost based on the range of floors that
    need to be visited (origin floors for waiting passengers and destination floors
    for all unserved passengers).

    Assumptions
    - The lift can carry multiple passengers.
    - The 'above' predicates define a linear order of floors.
    - The cost of each action (move, board, depart) is 1.

    Heuristic Initialization
    - Extracts the destination floor for each passenger from static facts.
    - Builds a mapping from floor names to numerical levels based on the 'above' predicates. The lowest floor is assigned level 1.

    Step-By-Step Thinking for Computing Heuristic
    1. Identify the current floor of the lift from the state and get its numerical level using the pre-calculated floor mapping.
    2. Identify all unserved passengers by checking which passengers do not have the '(served ?p)' fact in the state.
    3. Initialize the heuristic value to 0.
    4. Determine the set of floors that the lift *must* visit:
       - All origin floors of waiting passengers.
       - All destination floors of *all* unserved passengers (both waiting and boarded).
    5. Count the number of 'board' actions needed: This is the number of passengers currently waiting at their origin floor. Add this count to the heuristic.
    6. Count the number of 'depart' actions needed: This is the total number of unserved passengers. Add this count to the heuristic.
    7. If there are no required floors to visit (i.e., the set from step 4 is empty), the movement cost is 0. This occurs when all passengers are served.
    8. If there are required floors:
       - Get the numerical levels for all required floors.
       - Find the minimum and maximum floor levels among the required floors.
       - Estimate the movement cost based on the current lift floor level and the range [min_level, max_level]:
         - If the current floor is below the minimum required level, the cost is the distance up to the minimum required level plus the distance from minimum to maximum (`(min_level - current_level) + (max_level - min_level)`).
         - If the current floor is above the maximum required level, the cost is the distance down to the maximum required level plus the distance from maximum to minimum (`(current_level - max_level) + (max_level - min_level)`).
         - If the current floor is within the range [min_level, max_level]:
           - Check if there are required floors both above and below the current floor.
           - If both directions are needed, the estimated moves is the distance to the nearest end of the range plus the full range distance (`(max_level - min_level) + min(abs(current_level - min_level), abs(current_level - max_level))`).
           - If only floors above are needed, the cost is the distance up to the maximum required level (`max_level - current_level`).
           - If only floors below are needed, the cost is the distance down to the minimum required level (`current_level - min_level`).
    9. Add the estimated movement cost to the total heuristic value.
    10. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Destination floor for each passenger.
        - Floor level mapping from 'above' predicates.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract passenger destinations
        self.passenger_destin = {}
        for fact in static_facts:
             if match(fact, "destin", "*", "*"):
                 _, passenger, floor = get_parts(fact)
                 self.passenger_destin[passenger] = floor

        # Build floor level mapping
        # (above f_above f_below) means f_above is immediately above f_below
        # We build a map f_below -> f_above
        floor_above_map = {}
        all_floors = set()

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                _, f_above, f_below = get_parts(fact)
                all_floors.add(f_above)
                all_floors.add(f_below)
                floor_above_map[f_below] = f_above

        # Find the lowest floor: a floor that is not the 'f_above' in any (above f_above f_below)
        # i.e., a floor that is not a VALUE in floor_above_map
        lowest_floor = None
        above_values = set(floor_above_map.values())
        for f in all_floors:
            if f not in above_values:
                 lowest_floor = f
                 break

        # Build the level mapping by following the 'immediately_above' chain from the lowest floor
        self.floor_to_level = {}
        current_floor = lowest_floor
        level = 1
        while current_floor is not None:
            self.floor_to_level[current_floor] = level
            # Find the floor immediately above current_floor
            # This is the floor f_above such that (above f_above current_floor) is true.
            # This means current_floor is the f_below in the (above f_above f_below) fact.
            next_floor = floor_above_map.get(current_floor)

            current_floor = next_floor
            level += 1

        # Need the set of all passengers to identify unserved ones later
        self.all_passengers = set(self.passenger_destin.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Find current lift floor
        current_lift_floor_name = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor_name = get_parts(fact)[1]
                break
        # If lift-at fact is not found, something is wrong with the state representation.
        # Assuming it's always present in valid states.
        current_lift_floor_level = self.floor_to_level[current_lift_floor_name]

        heuristic = 0  # Initialize action cost counter.

        origins_needed_levels = set()
        destinations_needed_levels = set()

        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_passengers = self.all_passengers - served_passengers

        num_waiting = 0

        for passenger in unserved_passengers:
            # Check if passenger is waiting at origin
            is_waiting = False
            origin_floor_name = None
            for fact in state:
                if match(fact, "origin", passenger, "*"):
                    is_waiting = True
                    origin_floor_name = get_parts(fact)[2]
                    break

            if is_waiting:
                num_waiting += 1
                origins_needed_levels.add(self.floor_to_level[origin_floor_name])

            # All unserved passengers need to reach their destination
            destin_floor_name = self.passenger_destin[passenger]
            destinations_needed_levels.add(self.floor_to_level[destin_floor_name])


        # Total board actions needed = number of passengers currently waiting at origin
        heuristic += num_waiting

        # Total depart actions needed = number of unserved passengers (both waiting and boarded)
        heuristic += len(unserved_passengers)

        # Calculate movement cost
        required_floors_levels = origins_needed_levels.union(destinations_needed_levels)

        if not required_floors_levels:
            moves = 0
        else:
            min_level = min(required_floors_levels)
            max_level = max(required_floors_levels)

            moves = 0
            if current_lift_floor_level < min_level:
                # Must go up at least to the minimum required floor, then potentially up to max
                moves = (min_level - current_lift_floor_level) + (max_level - min_level)
            elif current_lift_floor_level > max_level:
                # Must go down at least to the maximum required floor, then potentially down to min
                moves = (current_lift_floor_level - max_level) + (max_level - min_level)
            else: # current_lift_floor_level is within [min_level, max_level]
                needs_up = any(level > current_lift_floor_level for level in required_floors_levels)
                needs_down = any(level < current_lift_floor_level for level in required_floors_levels)

                if needs_up and needs_down:
                    moves = (max_level - min_level) + min(abs(current_lift_floor_level - min_level), abs(current_lift_floor_level - max_level))
                elif needs_up:
                    moves = max_level - current_lift_floor_level
                elif needs_down:
                    moves = current_lift_floor_level - min_level

        heuristic += moves

        return heuristic
