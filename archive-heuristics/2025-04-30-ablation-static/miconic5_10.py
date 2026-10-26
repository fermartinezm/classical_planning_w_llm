# Import necessary modules
from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided in the execution environment
# This is just for standalone testing or if the base class is not strictly required for the code itself
# but is part of the expected structure. If the environment provides it, this can be removed.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle empty or invalid facts gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return [] # Return empty list for invalid format
    return fact[1:-1].split()

# Helper function to match PDDL facts using fnmatch patterns
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 room1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all passengers.
    It considers the number of passengers needing pickup and dropoff, and the
    estimated vertical movement cost for the lift to visit all necessary floors.

    # Assumptions
    - Floors are ordered linearly based on `above` predicates.
    - The lift can carry multiple passengers.
    - The lift must visit a passenger's current origin floor to board them (if waiting)
      and their destination floor to depart them (if unserved).

    # Heuristic Initialization
    - Extract the floor ordering from the `above` static facts to create a
      mapping from floor names to integer indices.
    - Extract the initial origin and destination floor for each passenger
      from the static facts (initial state). This defines the set of all
      passengers in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the current location of the lift from the state facts.
    2. Identify the state of each passenger (waiting at origin, boarded, or served)
       by checking the state facts. Determine the set of unserved passengers
       (those not marked as 'served' among all passengers in the problem).
    3. Identify the set of necessary pickup floors: these are the origin floors
       of passengers who are currently waiting (indicated by `(origin ?p ?f)`
       facts in the current state).
    4. Identify the set of necessary dropoff floors: these are the destination
       floors (extracted during initialization from static facts) for all
       passengers who are currently unserved.
    5. Combine the pickup and dropoff floors to get the set of all active floors
       the lift must visit.
    6. If there are no active floors, all passengers are served, and the heuristic is 0.
    7. If there are active floors:
       - Map the current lift floor and all active floors to their integer indices
         using the precomputed floor ordering.
       - Find the minimum and maximum integer indices among the active floors.
       - Estimate the minimum number of moves required to visit all active floors
         starting from the current lift floor. A reasonable estimate is the distance
         from the current floor to the closest extreme active floor (min or max),
         plus the distance between the min and max active floors (covering the sweep).
         This is calculated as `(max_active_int - min_active_int) + min(abs(current_lift_int - min_active_int), abs(current_lift_int - max_active_int))`.
    8. Count the number of passengers currently waiting at their origin (from state facts).
       Each waiting passenger needs one 'board' action.
    9. Count the total number of passengers who are not yet served (among all passengers
       in the problem). Each unserved passenger needs one 'depart' action at their
       destination.
    10. The total heuristic value is the sum of the estimated moves, the number of
        board actions needed, and the number of depart actions needed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information from the task.
        """
        self.goals = task.goals
        static_facts = task.static # Static facts include initial origins, destinations, and 'above' relations

        # 1. Build floor ordering and mapping (floor_to_int)
        above_facts_parts = [get_parts(fact) for fact in static_facts if match(fact, "above", "*", "*")]
        all_floors = set()
        floors_above = set()
        # Collect all floors and identify floors that are "above" something else
        for parts in above_facts_parts:
            if len(parts) == 3: # Ensure correct structure (above f1 f2)
                _, f_below, f_above = parts
                all_floors.add(f_below)
                all_floors.add(f_above)
                floors_above.add(f_above)

        self.floor_to_int = {}
        self.int_to_floor = {}

        # Handle case with floors defined
        if all_floors:
            # The lowest floor is one in all_floors but not in floors_above
            # Assumes a single lowest floor exists and floors form a linear chain
            potential_lowest = all_floors - floors_above
            if not potential_lowest:
                 # This might happen if floors form a cycle or there are no 'above' facts for multiple floors
                 # In a standard miconic domain, there should be a unique lowest floor.
                 # As a fallback, just sort floors alphabetically, but this is not PDDL-correct.
                 # Let's assume valid miconic structure with a unique lowest floor.
                 # If it's truly empty, maybe there's only one floor?
                 if len(all_floors) == 1:
                     lowest_floor = list(all_floors)[0]
                 else:
                     # This indicates a problem with the 'above' facts structure
                     # print("Warning: Could not determine unique lowest floor from 'above' facts.") # Debugging
                     # Fallback: Pick an arbitrary floor as start, or handle as error
                     lowest_floor = sorted(list(all_floors))[0] # Arbitrary start based on sorting
            else:
                 lowest_floor = potential_lowest.pop()


            # Build the ordered list of floors starting from the lowest
            floor_order = [lowest_floor]
            current_floor = lowest_floor
            # Map floor -> floor_above
            above_map = {f_below: f_above for _, f_below, f_above in above_facts_parts if len(parts) == 3}

            # Build the order following the 'above' chain
            while current_floor in above_map:
                next_floor = above_map[current_floor]
                # Prevent infinite loops in case of cycles or malformed 'above' facts
                if next_floor in floor_order:
                     # print("Warning: Cycle detected or malformed 'above' relations.") # Debugging
                     break
                floor_order.append(next_floor)
                current_floor = next_floor

            self.floor_to_int = {floor: i for i, floor in enumerate(floor_order)}
            self.int_to_floor = {i: floor for floor, i in self.floor_to_int.items()}

        # 2. Extract initial origin and destination floors for each passenger from static facts
        self.passenger_origins_init = {} # Store initial origins
        self.passenger_destinations = {}
        self.all_passengers = set()

        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if len(args) >= 2: # Ensure predicate has enough arguments for passenger and floor
                if predicate == "origin":
                    person, floor = args
                    self.passenger_origins_init[person] = floor
                    self.all_passengers.add(person)
                elif predicate == "destin":
                     person, floor = args
                     self.passenger_destinations[person] = floor
                     self.all_passengers.add(person)
            # Also add passengers mentioned in goals but not in origin/destin facts (unlikely in miconic)
            # For robustness, could iterate goals here too, but standard miconic has goals like (served p)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                current_lift_floor = get_parts(fact)[1]
                break

        # If lift location is unknown, something is wrong with the state representation
        # or domain definition. Return infinity as this state is likely unreachable/invalid.
        if current_lift_floor is None:
             # print("Warning: Lift location not found in state.") # Debugging
             return float('inf')

        # Handle case with no floors defined or current lift floor not in mapping
        if not self.floor_to_int or current_lift_floor not in self.floor_to_int:
             # print("Warning: Floors not defined or current lift floor not in mapping.") # Debugging
             # If there are unserved passengers but no valid floor mapping, it's impossible.
             # If no unserved passengers, h=0.
             passengers_served_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
             passengers_unserved = self.all_passengers - passengers_served_in_state
             return 0 if not passengers_unserved else float('inf')


        current_lift_int = self.floor_to_int[current_lift_floor]

        # 2. Identify state of each passenger from current state facts
        passengers_waiting_in_state = set()
        passengers_boarded_in_state = set()
        passengers_served_in_state = set()

        for fact in state:
            parts = get_parts(fact)
            if len(parts) > 1: # Ensure there's at least a passenger argument
                if parts[0] == "origin":
                    passengers_waiting_in_state.add(parts[1])
                elif parts[0] == "boarded":
                    passengers_boarded_in_state.add(parts[1])
                elif parts[0] == "served":
                    passengers_served_in_state.add(parts[1])

        # Passengers who are not served (among all passengers in the problem instance)
        passengers_unserved = self.all_passengers - passengers_served_in_state

        # 3. Identify necessary pickup and dropoff floors
        # Pickup floors are the origins of passengers currently waiting (from state facts)
        pickup_floors = {get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        # Dropoff floors are the destinations of all unserved passengers (from static facts)
        # Ensure passenger exists in self.passenger_destinations (should be true if in self.all_passengers)
        dropoff_floors = {self.passenger_destinations[p] for p in passengers_unserved if p in self.passenger_destinations}

        # 4. Combine active floors
        active_floors = pickup_floors | dropoff_floors

        # 5. If no active floors, goal is reached
        if not active_floors:
            return 0

        # 6. Estimate moves
        # Map active floors to integer indices. Handle cases where a floor might not be in floor_to_int
        # (e.g., if the problem instance is malformed, though unlikely).
        active_floors_int = {self.floor_to_int[f] for f in active_floors if f in self.floor_to_int}

        # If somehow active_floors_int becomes empty (e.g., active floors not in floor_to_int mapping),
        # this indicates an issue, but we should handle it gracefully. If there are unserved passengers
        # but no valid active floors, it's likely unsolvable or malformed.
        if not active_floors_int:
             # print("Warning: Active floors not found in floor mapping.") # Debugging
             return float('inf') # Cannot reach goal if needed floors are not mapped

        min_active_int = min(active_floors_int)
        max_active_int = max(active_floors_int)

        # Moves estimate: distance to closest extreme + span of active floors
        # This is equivalent to max(current, max_active) - min(current, min_active) + (max_active - min_active)
        moves = (max_active_int - min_active_int) + min(
            abs(current_lift_int - min_active_int),
            abs(current_lift_int - max_active_int)
        )

        # 7. Count board actions needed
        # Only passengers currently waiting need a board action
        board_actions_needed = len(passengers_waiting_in_state)

        # 8. Count depart actions needed
        # All unserved passengers need a depart action
        depart_actions_needed = len(passengers_unserved)

        # 9. Total heuristic
        total_cost = moves + board_actions_needed + depart_actions_needed

        return total_cost
