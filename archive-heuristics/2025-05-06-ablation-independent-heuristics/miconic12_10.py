# Need to import Heuristic base class
from heuristics.heuristic_base import Heuristic
# Need Task class for type hinting in __init__
from task import Task

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string like '(predicate arg1 arg2)' into (predicate, [arg1, arg2])."""
    # Remove leading/trailing brackets and split by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class miconicHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the minimum travel cost
    for the lift to visit all necessary floors (origins of unboarded passengers
    and destinations of all unserved passengers) and the number of boarding
    and departing actions required for unserved passengers.

    Assumptions:
    - Floor names are of the format 'fN' where N is an integer, and the order
      of floors corresponds to the numerical order of N. The 'above' predicates
      in the static information confirm this ordering.
    - The goal is to serve all passengers mentioned in the initial state's
      origin/destin facts.
    - Valid states: Unserved passengers are either at their origin or boarded.

    Heuristic Initialization:
    - Parses facts from the task (static, initial state, goal) to identify
      all floor objects.
    - Assumes floor names are 'f' followed by a number and sorts them numerically
      to create a mapping from floor names (e.g., 'f1', 'f2') to numerical
      floor indices (e.g., 1, 2).
    - Parses the 'destin' predicates from the static facts to create a mapping
      from passenger names to their destination floor names.
    - Identifies all passengers present in the problem instance based on initial
      'origin' or static 'destin' facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the goal is already reached (all passengers served). If yes, return 0.
    2. Identify the current floor of the lift from the state. Get its numerical index.
       If the lift floor is not found or not indexed, return infinity.
    3. Identify all passengers that have not yet been served.
    4. Separate the unserved passengers into those still at their origin (unboarded)
       and those who are boarded. While doing this, record the current origin floor
       for each unboarded passenger. Passengers unserved, not boarded, and not
       at origin in the current state are considered "lost" for the purpose of
       identifying pickup stops, but still require a dropoff.
    5. Determine the set of floor indices the lift *must* visit:
       - The index of the origin floor for each unboarded passenger (if origin is known and indexed).
       - The index of the destination floor for each unserved passenger (if destination is known and indexed).
    6. Calculate the minimum travel distance required for the lift to start at its
       current floor index and visit all required floor indices. This is calculated
       as the distance from the current floor index to the nearest extreme required
       floor index (minimum or maximum) plus the distance between the minimum and
       maximum required floor indices. If there are no required floors, travel is 0.
    7. Count the number of boarding actions needed (one for each unboarded passenger).
    8. Count the number of departing actions needed (one for each unserved passenger).
    9. The heuristic value is the sum of the minimum travel distance, the number
       of boarding actions, and the number of departing actions.
    """
    def __init__(self, task: Task):
        super().__init__()
        self.task = task

        # 1. Build floor name -> index map
        self.floor_to_index = {}
        self.index_to_floor = {}
        floor_names = set()

        # Collect all potential floor names from relevant facts
        facts_to_check = list(task.static) + list(task.initial_state) + list(task.goals)
        for fact_str in facts_to_check:
            pred, args = parse_fact(fact_str)
            # Check predicates known to involve floors
            if pred in ['above', 'lift-at', 'origin', 'destin']:
                 # Floors are typically arguments in these predicates
                 # A simple heuristic: check if argument starts with 'f' and contains a digit
                 for arg in args:
                     if isinstance(arg, str) and arg.startswith('f') and any(char.isdigit() for char in arg):
                         floor_names.add(arg)

        # Assuming floor names are fN, extract N and sort numerically
        try:
            sorted_floors = sorted(list(floor_names), key=lambda f: int(f[1:]))
            for i, floor_name in enumerate(sorted_floors):
                self.floor_to_index[floor_name] = i + 1 # Use 1-based indexing
                self.index_to_floor[i + 1] = floor_name
        except ValueError:
             # If floor names are not in the expected fN format, we cannot build the index map.
             # The heuristic will likely return infinity later if floors are needed.
             print(f"Error: Unexpected floor name format in {floor_names}. Cannot build floor index map.")
             self.floor_to_index = {}
             self.index_to_floor = {}


        # 2. Build passenger -> destin_floor map from static facts
        self.passenger_to_destin = {}
        for fact_str in task.static:
            pred, args = parse_fact(fact_str)
            if pred == 'destin':
                passenger, destin_floor = args
                self.passenger_to_destin[passenger] = destin_floor

        # 3. Identify all passengers in the problem from initial state and static facts
        self.all_passengers = set(self.passenger_to_destin.keys())
        for fact_str in task.initial_state:
             pred, args = parse_fact(fact_str)
             if pred == 'origin':
                 self.all_passengers.add(args[0])


    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        # The goal is (served p) for all passengers identified in init/static.
        all_served = True
        for p in self.all_passengers:
            if f'(served {p})' not in state:
                all_served = False
                break
        if all_served:
            return 0 # Goal reached

        # 2. Find current lift floor
        current_lift_floor = None
        for fact_str in state:
            pred, args = parse_fact(fact_str)
            if pred == 'lift-at':
                current_lift_floor = args[0]
                break

        # If lift-at is not found or floor is not indexed, problem state is invalid or unexpected
        if current_lift_floor is None or current_lift_floor not in self.floor_to_index:
             # print(f"Warning: Current lift floor {current_lift_floor} not found or not indexed.")
             return float('inf') # Cannot compute heuristic, likely unreachable state

        current_lift_index = self.floor_to_index[current_lift_floor]

        # 3. Identify unserved passengers
        unserved_passengers = {p for p in self.all_passengers if f'(served {p})' not in state}

        # 4. Separate unserved passengers by state (origin or boarded)
        unboarded_passengers = set()
        boarded_passengers = set()
        passenger_origin_floor = {} # Store origin floor for unboarded passengers

        # Build sets for quick lookup in state
        state_predicates = {}
        for fact_str in state:
             pred, args = parse_fact(fact_str)
             if pred not in state_predicates:
                 state_predicates[pred] = set()
             state_predicates[pred].add(tuple(args)) # Use tuple for hashable args

        for p in unserved_passengers:
            is_boarded = False
            # Check if boarded
            if ('boarded', (p,)) in state_predicates.get('boarded', set()):
                 boarded_passengers.add(p)
                 is_boarded = True

            if not is_boarded:
                 # If not boarded and unserved, assume they are at their origin in the current state
                 # Find origin floor in the current state
                 found_origin = False
                 for args_tuple in state_predicates.get('origin', set()):
                     if args_tuple[0] == p:
                         unboarded_passengers.add(p)
                         passenger_origin_floor[p] = args_tuple[1]
                         found_origin = True
                         break # Found origin

                 # If not found at origin either, this passenger is in an unexpected state.
                 # They remain in unserved_passengers but won't contribute to unboarded counts/origin stops.
                 # They still contribute to destination stops and depart actions.
                 if not found_origin:
                      # print(f"Warning: Unserved passenger {p} is neither at origin nor boarded in state.")
                      pass # Continue, passenger is just in unserved_passengers set


        # 5. Determine required floors (indices)
        required_indices = set()
        for p in unboarded_passengers:
            origin_floor = passenger_origin_floor.get(p)
            if origin_floor and origin_floor in self.floor_to_index: # Ensure origin was found and is a known floor
                required_indices.add(self.floor_to_index[origin_floor])
            # else: print(f"Warning: Origin floor for unboarded passenger {p} not found or not indexed.") # Debugging

        for p in unserved_passengers: # Both unboarded and boarded
            destin_floor = self.passenger_to_destin.get(p)
            if destin_floor and destin_floor in self.floor_to_index: # Ensure destination is known and is a known floor
                 required_indices.add(self.floor_to_index[destin_floor])
            # else: print(f"Warning: Destination floor for unserved passenger {p} not found or not indexed.") # Debugging


        # 6. Calculate minimum travel cost
        min_travel = 0
        if required_indices: # Only calculate travel if there are floors to visit
            min_idx = min(required_indices)
            max_idx = max(required_indices)
            # Minimum travel is distance from current to nearest extreme + distance between extremes
            min_travel = min(abs(current_lift_index - min_idx), abs(current_lift_index - max_idx)) + (max_idx - min_idx)


        # 7. Calculate non-travel actions
        num_board = len(unboarded_passengers)
        num_depart = len(unserved_passengers) # Each unserved passenger needs a depart action

        # 8. Total heuristic
        h_value = min_travel + num_board + num_depart

        return h_value
