from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to match facts
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
    This heuristic estimates the cost to serve all passengers. It considers
    the number of 'board' and 'depart' actions needed for unserved passengers
    and estimates the minimum lift movement required to visit all necessary
    pickup and dropoff floors.

    # Assumptions
    - Floors are linearly ordered. The 'above' predicates define this order.
      Specifically, we assume floors are named 'f1', 'f2', ..., 'fn' and
      '(above fi fj)' implies fj is lower than fi if i < j.
    - Each 'board' and 'depart' action costs 1.
    - Each 'up' or 'down' action between adjacent floors costs 1.

    # Heuristic Initialization
    - Parses the 'above' facts and object definitions to identify all floor
      objects.
    - Assigns a numerical level to each floor based on the assumption that
      floors sorted numerically by their index (e.g., f1 < f2 < ... < fn)
      correspond to decreasing physical height (f1 is highest, fn is lowest).
      The lowest floor is assigned level 1.
    - Extracts the destination floor for each passenger from the static
      'destin' facts.
    - Identifies all passengers in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Identify the current floor of the lift and its corresponding numerical level.
    3. Identify all passengers who are not yet 'served' in the current state.
    4. For each unserved passenger:
       - Determine their current status: are they waiting at their origin floor
         (predicate '(origin p f)') or are they already boarded
         (predicate '(boarded p)')?
       - Collect the set of all floors that the lift *must* visit to make progress
         towards serving these passengers:
         - Add the origin floor for every unboarded passenger.
         - Add the destination floor for every boarded passenger.
    5. Calculate the base action cost:
       - Each unboarded passenger requires at least a 'board' and a 'depart' action (cost 2).
       - Each boarded passenger requires at least a 'depart' action (cost 1).
       - Sum these costs for all unserved passengers.
    6. Calculate the estimated lift movement cost:
       - If there are no required visit floors (this happens only if there are no unserved passengers, which is covered by step 1), the estimated moves is 0.
       - Otherwise, find the minimum and maximum floor levels among the required visit floors.
       - The estimated number of move actions is a lower bound calculated as the vertical span of the required floors plus the minimum distance from the current lift floor to either the lowest or highest required floor. This is given by the formula:
         `(max_target_level - min_target_level) + min(abs(current_level - min_target_level), abs(current_level - max_target_level))`.
    7. The total heuristic value is the sum of the base action cost and the estimated lift movement cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting floor levels and passenger destinations."""
        # Assuming task object has attributes: goals, static, initial_state
        super().__init__(task) # Call base class constructor

        # 1. Parse floor objects and determine floor levels.
        # Collect all objects that are floors. Assuming floors start with 'f'.
        floor_objects = set()
        # Check initial state, goals, and static facts for objects that look like floors
        for fact in self.initial_state | self.goals | self.static:
             parts = get_parts(fact)
             for part in parts: # Check all parts, including predicate name in case it's a floor (unlikely)
                 if part.startswith('f'): # Simple check for floor objects
                     floor_objects.add(part)

        # Sort floors based on the number part (assuming format f<number>)
        # This relies on the naming convention f1, f2, ...
        def floor_sort_key(floor_name):
            try:
                # Extract number after 'f' and convert to int
                return int(floor_name[1:])
            except (ValueError, IndexError):
                # Handle cases where floor name is not 'f' followed by a number
                # This heuristic relies on a consistent f<number> naming for levels.
                # If other names exist, this heuristic might be inaccurate.
                # Assign a large value to push them to the end, assuming they are highest.
                # This is a potential limitation if floor names are arbitrary.
                # For standard miconic benchmarks, f<number> is typical.
                print(f"Warning: Unexpected floor name format encountered: {floor_name}. Heuristic might be inaccurate.")
                return float('inf') # Assign a large value for sorting

        sorted_floors = sorted(list(floor_objects), key=floor_sort_key)

        # Assign levels: lowest floor gets level 1, next gets level 2, etc.
        # Based on example instances, floors sorted numerically correspond to
        # decreasing physical height (f1 is highest, fn is lowest).
        # So, the last floor in the sorted list is the lowest.
        self.floor_levels = {}
        # Assign levels in reverse order of the sorted list
        for level, floor in enumerate(reversed(sorted_floors), 1):
             self.floor_levels[floor] = level

        # 2. Extract passenger destinations from static facts.
        self.passenger_destinations = {}
        for fact in self.static:
            if match(fact, "destin", "*", "*"):
                passenger, destination_floor = get_parts(fact)[1:]
                self.passenger_destinations[passenger] = destination_floor

        # Identify all passengers from destinations
        self.all_passengers = set(self.passenger_destinations.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Check if goal is reached.
        # The goal is a set of (served ?p) facts.
        # Check if all goal facts are present in the current state.
        if self.goals <= state:
            return 0

        # 2. Find current lift floor and level.
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break
        # If lift location is not in state, something is wrong with the state representation
        # or problem definition. Assume it's always present.
        # Add a check just in case, though it indicates a problem with the state.
        if current_lift_floor is None:
             # Cannot compute heuristic without lift location. Return infinity or a large value.
             # Or return a value indicating an unsolvable state if lift can disappear.
             # Assuming valid states always have lift-at.
             # For robustness, return a large value.
             # print("Error: Lift location not found in state.")
             return float('inf') # Should not happen in valid miconic states

        current_lift_level = self.floor_levels[current_lift_floor]

        # 3. Identify unserved passengers and their state (origin or boarded).
        unserved_passengers = set()
        boarded_passengers = set()
        passengers_at_origin = {} # {passenger: origin_floor}

        # Find passengers who are currently served
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Iterate through all known passengers
        for passenger in self.all_passengers:
            # If a passenger is not in the set of served passengers in the current state, they are unserved.
            if passenger not in served_passengers_in_state:
                unserved_passengers.add(passenger)

                # Check if the unserved passenger is boarded
                if f"(boarded {passenger})" in state:
                    boarded_passengers.add(passenger)
                else:
                    # If not served and not boarded, they must be at their origin floor.
                    # Find their origin floor from the current state.
                    found_origin = False
                    for fact in state:
                        if match(fact, "origin", passenger, "*"):
                            origin_floor = get_parts(fact)[2]
                            passengers_at_origin[passenger] = origin_floor
                            found_origin = True
                            break # Found origin for this passenger
                    # If an unserved, unboarded passenger doesn't have an origin fact,
                    # something is inconsistent with the domain state representation.
                    # Assuming valid states.

        # If no unserved passengers were found, but the goal check failed,
        # it implies the goal definition includes more than just (served ?p).
        # Based on the domain file, the goal is *only* (served ?p) for all passengers.
        # So, if unserved_passengers is empty, self.goals <= state must be True.
        # This check is a safeguard but should not be reached in valid miconic problems
        # where the goal check at the start works correctly.
        if not unserved_passengers:
             return 0 # Goal state detected (or inconsistent state)

        # 4. Collect required visit floors.
        required_visit_floors = set()
        # Add origin floors for passengers waiting to be picked up
        for passenger, origin_floor in passengers_at_origin.items():
            required_visit_floors.add(origin_floor)
        # Add destination floors for passengers who are boarded
        for passenger in boarded_passengers:
            destination_floor = self.passenger_destinations[passenger]
            required_visit_floors.add(destination_floor)

        # 5. Calculate base action cost (board/depart).
        # Each unboarded passenger needs board (1) + depart (1) = 2 actions.
        # Each boarded passenger needs depart (1) = 1 action.
        action_cost = len(passengers_at_origin) * 2 + len(boarded_passengers) * 1

        # 6. Calculate estimated lift movement cost.
        estimated_moves = 0
        if required_visit_floors:
            # Get levels for all floors that need visiting
            target_levels = [self.floor_levels[f] for f in required_visit_floors]
            min_target_level = min(target_levels)
            max_target_level = max(target_levels)

            # Calculate minimum moves to cover the range [min_target_level, max_target_level]
            # starting from current_lift_level.
            current_level = current_lift_level

            # Minimum moves = span of required floors + min distance from current to range ends
            range_span = max_target_level - min_target_level
            estimated_moves = range_span + min(abs(current_level - min_target_level), abs(current_level - max_target_level))

        # 7. Total heuristic value.
        total_cost = action_cost + estimated_moves

        return total_cost
