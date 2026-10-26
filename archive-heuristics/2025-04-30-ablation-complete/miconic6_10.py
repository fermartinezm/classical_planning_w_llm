# Need to import the base class if it's in a separate file
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return [] # Return empty list for malformed input

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
         return False # Mismatch in arity

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions (moves, boards, departs)
    required to serve all passengers. It calculates the minimum number of
    lift movements needed to visit all floors where passengers are waiting
    or need to be dropped off, and adds a fixed cost (1 for depart, 2 for board+depart)
    for each passenger not yet served.

    # Assumptions
    - Floors are linearly ordered (e.g., f1 < f2 < ... < fn).
    - The 'above' predicate defines this linear order: (above f_lower f_higher)
      means f_lower is immediately below f_higher.
    - Passenger destinations are static and available from the initial state
      facts (`destin` predicate).
    - The cost of each action (move, board, depart) is 1.

    # Heuristic Initialization
    - Parses the 'above' facts from static information to build a mapping
      from floor names to numerical indices, representing their order.
    - Extracts the destination floor for each passenger from the initial state
      facts (`destin` predicate).
    - Collects the set of all relevant passengers from initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unserved Passengers: Determine which passengers are not yet
       in the goal state (i.e., `(served ?p)` is false).

    2. Determine Required Floors: For each unserved passenger:
       - Get the passenger's origin floor `?f_origin` and destination floor `?f_destin`.
       - If the passenger is waiting at their origin `(origin ?p ?f_origin)`, the lift must visit both `?f_origin` and `?f_destin`. Add both to the set of required floors.
       - If the passenger is boarded `(boarded ?p)`, the lift must visit `?f_destin`. Add `?f_destin` to the set of required floors.

    3. Get Current Lift Floor: Find the floor where the lift is currently located
       using the `(lift-at ?f)` fact.

    4. Map Floors to Indices: Convert the current lift floor and all required
       floors from names (e.g., 'f1', 'f2') to their numerical indices using
       the floor map built during initialization. Handle cases where a floor
       might not be in the map (indicates problem parsing or state issue).

    5. Calculate Minimum Moves: If there are required floors:
       - Find the minimum and maximum floor indices among the required floors.
       - Calculate the minimum number of move actions needed for the lift to
         travel from its current floor index to visit all floors within the
         range defined by the minimum and maximum required floor indices.
         This is calculated as `min(abs(current_idx - min_required_idx), abs(current_idx - max_required_idx)) + (max_required_idx - min_required_idx)`.
       - If there are no required floors, the minimum moves is 0.

    6. Calculate Passenger Action Cost: For each passenger not yet served:
       - If the passenger is boarded, add 1 (for the `depart` action).
       - If the passenger is waiting at an origin, add 2 (for the `board` and `depart` actions).

    7. Sum Costs: The total heuristic value is the sum of the minimum move
       actions and the total passenger action cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the floor map and storing
        passenger destinations.
        """
        super().__init__(task) # Call the base class constructor

        # Build the floor index map from static 'above' facts.
        # (above f_lower f_higher) means f_lower is immediately below f_higher.
        # We want f_lowest -> index 0, f_next_lowest -> index 1, etc.
        above_relations = {} # f_lower -> f_higher
        all_floors = set()
        floors_that_are_higher = set() # Floors that appear as f_higher

        # 'above' facts are typically static
        facts_to_check_for_above = set(self.static)

        for fact in facts_to_check_for_above:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, f_lower, f_higher = parts # Corrected order based on action 'up'
                    above_relations[f_lower] = f_higher
                    all_floors.add(f_lower)
                    all_floors.add(f_higher)
                    floors_that_are_higher.add(f_higher)
                # else: malformed fact warning is optional

        # Find the lowest floor (the one not appearing as f_higher in any above fact)
        lowest_floor = None
        potential_lowest_floors = all_floors - floors_that_are_higher
        if len(potential_lowest_floors) == 1:
             lowest_floor = potential_lowest_floors.pop()
        elif len(all_floors) > 0:
             # Fallback: Simple sorting if naming convention is consistent (f1, f2, ...)
             try:
                 sorted_floors = sorted(list(all_floors), key=lambda f: int(f[1:]) if f.startswith('f') and f[1:].isdigit() else f)
                 if sorted_floors:
                     lowest_floor = sorted_floors[0]
                 # else: No floors found, handled below
             except (ValueError, IndexError):
                 lowest_floor = None # Cannot determine lowest floor

        self.floor_to_index = {}
        self.index_to_floor = {}
        if lowest_floor:
            # Build map by traversing up from the lowest floor
            current_floor = lowest_floor
            current_index = 0
            while current_floor is not None:
                if current_floor in self.floor_to_index:
                     break # Avoid infinite loop on cyclic 'above' relations
                self.floor_to_index[current_floor] = current_index
                self.index_to_floor[current_index] = current_floor
                current_index += 1
                current_floor = above_relations.get(current_floor)

            if len(self.floor_to_index) != len(all_floors):
                 # Warning: Built incomplete floor map.
                 pass # Continue with potentially incomplete map, errors will occur in __call__ if needed floors are missing


        # Store passenger destinations from initial state
        self.passenger_destinations = {}
        # Also collect all passengers mentioned in initial state or goals
        self.all_passengers = set()
        # self.passenger_origins = {} # Initial origin is not needed, current origin is in state

        for fact in self.task.initial_state:
             if match(fact, "destin", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     _, passenger, floor = parts
                     self.passenger_destinations[passenger] = floor
                     self.all_passengers.add(passenger)
             elif match(fact, "origin", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3:
                     _, passenger, floor = parts
                     # self.passenger_origins[passenger] = floor # Not needed
                     self.all_passengers.add(passenger)
             elif match(fact, "boarded", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2:
                     _, passenger = parts
                     self.all_passengers.add(passenger)
             # lift-at is also in initial state, but handled in __call__

        # Add passengers from goals who might not be in initial state facts (e.g. already served)
        for goal in self.goals:
             if match(goal, "served", "*"):
                 parts = get_parts(goal)
                 if len(parts) == 2:
                     _, passenger = parts
                     self.all_passengers.add(passenger)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        # Find current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, current_lift_floor = parts
                    break
                # else: malformed fact warning is optional

        if current_lift_floor is None:
             # This indicates an invalid state where lift location is unknown
             # Should not happen in a valid problem trace
             return float('inf')

        current_floor_idx = self.floor_to_index.get(current_lift_floor)
        if current_floor_idx is None:
             # This indicates a floor in the state was not in the initial 'above' facts
             return float('inf')


        # Identify required floors and track passenger states
        required_floors = set()
        served_passengers = set()
        boarded_passengers = set()
        current_origin_passengers = {} # passenger -> origin_floor (from current state)

        # Populate sets/dicts from current state
        for fact in state:
            if match(fact, "served", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, p = parts
                    served_passengers.add(p)
                # else: malformed fact warning is optional
            elif match(fact, "boarded", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    _, p = parts
                    boarded_passengers.add(p)
                # else: malformed fact warning is optional
            elif match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    _, p, f = parts
                    current_origin_passengers[p] = f
                # else: malformed fact warning is optional

        # Determine required floors and calculate passenger action cost for unserved passengers
        passenger_action_cost = 0
        unserved_passengers_exist = False

        for p in self.all_passengers:
            if p not in served_passengers:
                unserved_passengers_exist = True

                dest_floor = self.passenger_destinations.get(p)
                if dest_floor is None:
                     # This means a passenger in goals/initial state didn't have a destination
                     return float('inf')

                if p in boarded_passengers:
                    # Passenger is boarded, needs to go to destination and depart
                    required_floors.add(dest_floor)
                    passenger_action_cost += 1 # 1 depart action

                elif p in current_origin_passengers:
                    # Passenger is waiting at origin, needs to go to origin, board, go to dest, depart
                    origin_floor = current_origin_passengers[p]
                    required_floors.add(origin_floor) # Must visit origin
                    required_floors.add(dest_floor) # Must visit destination
                    passenger_action_cost += 2 # 1 board + 1 depart action

                # else: passenger is not served, not boarded, not at origin. Invalid state.
                # This case should not add to required_floors or passenger_action_cost
                # as it represents an impossible state transition.

        # If no passengers need service, heuristic is 0
        if not unserved_passengers_exist:
             return 0

        # Calculate minimum moves to visit required floors
        min_moves = 0
        if required_floors:
            required_indices = []
            for f in required_floors:
                idx = self.floor_to_index.get(f)
                if idx is None:
                    return float('inf') # Should not be reachable if floor map is complete
                required_indices.append(idx)

            if not required_indices: # Should not happen if required_floors was not empty, but safety check
                 min_moves = 0
            else:
                min_required_idx = min(required_indices)
                max_required_idx = max(required_indices)

                # Minimum moves to cover the range [min_required_idx, max_required_idx]
                # starting from current_floor_idx.
                min_moves = min(abs(current_floor_idx - min_required_idx), abs(current_floor_idx - max_required_idx)) + (max_required_idx - min_required_idx)

        total_heuristic = min_moves + passenger_action_cost

        return total_heuristic
