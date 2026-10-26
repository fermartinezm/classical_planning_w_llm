import re
from heuristics.heuristic_base import Heuristic
# Operator and Task classes are part of the planner's domain model,
# needed to understand the context of the Task object passed to the heuristic.
# While not directly used in the __call__ method's logic, they are relevant
# to the __init__ method which processes the Task object.
from task import Operator, Task


class miconicHeuristic(Heuristic):
    """
    Summary:
    Domain-dependent heuristic for the Miconic domain. Estimates the cost
    as the sum of minimum required board/depart actions and an estimate
    of the lift travel distance.

    Assumptions:
    - Floor names are strings of the form 'f<number>', where <number> is an integer,
      and these numbers define the floor order (e.g., f1 is below f2).
    - The 'above' predicate correctly defines the total order of floors (f_i f_j means f_j is above f_i).
    - The 'destin' predicate facts are static and available in task.static.
    - All passengers in the problem instance are mentioned in 'destin' facts in task.static.

    Heuristic Initialization:
    - Parses all floor names from the task facts and static facts.
    - Creates a mapping from floor name string to an integer index based on numerical order (f1 -> 1, f2 -> 2, ...).
    - Parses the static 'destin' facts to store the destination floor for each passenger.
    - Identifies all passengers in the problem instance based on the 'destin' facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the goal is reached by counting served passengers. If all passengers are served, return 0.
    2. Identify the current floor of the lift from the state.
    3. Iterate through the state facts to identify passengers who are waiting at their origin
       or are currently boarded.
    4. Count the number of passengers waiting at origin (N_origin) and the number
       of passengers boarded (N_boarded).
    5. Calculate the base cost: This is the minimum number of board and depart actions needed.
       Each waiting passenger needs one board and one depart action (2 actions).
       Each boarded passenger needs one depart action (1 action).
       Base cost = (2 * N_origin) + N_boarded.
    6. Identify the set of floors the lift *must* visit to pick up waiting passengers
       or drop off boarded passengers. This set includes the origin floors of waiting
       passengers and the destination floors of boarded passengers.
    7. If this set of needed floors is empty, it means all remaining passengers are
       either waiting at the current floor or are boarded and need to depart at the
       current floor. In this specific case, the travel cost is 0.
    8. If the set of needed floors is not empty:
       - Get the integer index for the current lift floor.
       - Find the minimum and maximum integer floor indices among the needed floors.
       - Estimate travel cost as the total span of floors from the lowest point
         (minimum of current floor index and minimum needed floor index) to the
         highest point (maximum of current floor index and maximum needed floor index).
         Travel cost = max(current_floor_idx, max_needed_idx) - min(current_floor_idx, min_needed_idx).
    9. The total heuristic value is the base cost plus the estimated travel cost.
    """

    def __init__(self, task):
        super().__init__()
        # Store goals for goal check (alternative to counting served passengers)
        # self.goals = task.goals # Not strictly needed if we count served vs all passengers

        self.destinations = {}
        floor_names = set()
        all_passengers_set = set() # Use a set to collect all passengers

        # Helper function to parse fact strings
        def parse_fact(fact_str):
             # Use regex to find predicate and arguments
             match = re.match(r'\(([\w-]+)(.*)\)', fact_str)
             if not match: return None, []
             predicate = match.group(1)
             args_str = match.group(2).strip()
             # Split arguments by spaces, handling potential multiple spaces
             args = args_str.split() if args_str else []
             return predicate, args

        # Collect information from all facts (initial state, static, and general facts)
        # task.facts contains all possible ground facts in the domain, including those
        # that might be true in the initial state or static.
        # task.static contains facts explicitly marked as static.
        # Combining them ensures we get all relevant info.
        all_relevant_facts = set(task.facts) | set(task.static)

        for fact_str in all_relevant_facts:
            predicate, args = parse_fact(fact_str)
            if not predicate: continue

            if predicate in ['lift-at', 'origin', 'destin']:
                if len(args) > 0:
                    # Last argument is the floor for these predicates
                    floor_names.add(args[-1])
                    if predicate in ['origin', 'destin'] and len(args) > 1:
                         # First argument is the passenger
                         all_passengers_set.add(args[0])
                         if predicate == 'destin':
                             # Store destination mapping
                             self.destinations[args[0]] = args[1]

            elif predicate == 'above':
                 if len(args) > 1:
                    # Both arguments are floors
                    floor_names.add(args[0])
                    floor_names.add(args[1])

        # Identify all passengers based on who has a destination
        self.all_passengers = frozenset(self.destinations.keys())

        # 1) Extract objects from facts (specifically floors) and create mapping
        # Assumes floor names are f<number>
        try:
            # Filter out any non-floor strings that might have been added
            valid_floor_names = [f for f in floor_names if re.match(r'f\d+$', f)]
            # Sort numerically based on the number part
            sorted_floor_names = sorted(list(valid_floor_names), key=lambda f: int(f[1:]))
            # Create mapping from floor name to 1-based integer index
            self.floor_to_int = {floor_name: i + 1 for i, floor_name in enumerate(sorted_floor_names)}
        except (ValueError, IndexError) as e:
             # If floor parsing fails (e.g., names not f<number>), print warning
             # and create an empty mapping. This makes floor_name_to_int return 0,
             # effectively setting travel_cost to 0.
             print(f"Warning: Error parsing floor names ({e}). Floor names found: {floor_names}. Travel cost will be ignored.")
             self.floor_to_int = {}


    def floor_name_to_int(self, floor_name):
        """Converts a floor name string (e.g., 'f5') to its integer index."""
        # Return 0 if floor name is not found in the mapping.
        # This handles cases where floor_to_int is empty due to parsing errors,
        # or if a floor somehow exists in the state but wasn't in the initial
        # fact collection (shouldn't happen in valid PDDL).
        return self.floor_to_int.get(floor_name, 0)

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        served_passengers = set()
        lift_at_floor = None
        passengers_at_origin = {} # {passenger: floor}
        passengers_boarded = set()

        # Helper function to parse fact strings (repeated for clarity within __call__)
        def parse_fact(fact_str):
             match = re.match(r'\(([\w-]+)(.*)\)', fact_str)
             if not match: return None, []
             predicate = match.group(1)
             args_str = match.group(2).strip()
             args = args_str.split() if args_str else []
             return predicate, args

        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if not predicate: continue

            if predicate == 'served' and len(args) > 0:
                served_passengers.add(args[0])
            elif predicate == 'lift-at' and len(args) > 0:
                lift_at_floor = args[0]
            elif predicate == 'origin' and len(args) > 1:
                passengers_at_origin[args[0]] = args[1]
            elif predicate == 'boarded' and len(args) > 0:
                passengers_boarded.add(args[0])

        # Check if goal is reached (all passengers identified in init are served)
        if served_passengers == self.all_passengers:
             return 0

        # Identify passengers who still need service
        passengers_needing_service = self.all_passengers - served_passengers

        # 3. Identify passengers waiting or boarded among those needing service
        passengers_waiting = {p for p in passengers_needing_service if p in passengers_at_origin}
        passengers_currently_boarded = {p for p in passengers_needing_service if p in passengers_boarded}

        N_origin = len(passengers_waiting)
        N_boarded = len(passengers_currently_boarded)

        # If no passengers need service but goal check failed, something is wrong.
        # If floor mapping failed, we can't calculate travel cost.
        # In these cases, return just the base cost.
        if not self.floor_to_int or (N_origin == 0 and N_boarded == 0):
             # If N_origin + N_boarded == 0, goal should have been reached.
             # If not, self.all_passengers might be wrong, or state is invalid.
             # Returning 0 might cause infinite loops if goal is truly unreachable but h=0.
             # Returning base_cost is safer than 0 if goal check failed unexpectedly.
             # If floor_to_int is empty, travel_cost calculation would fail, so return base_cost.
             return (2 * N_origin) + N_boarded


        # 5. Calculate base cost (board/depart actions)
        base_cost = (2 * N_origin) + N_boarded

        # 6. Identify floors the lift must visit
        floors_needed = set()
        for p in passengers_waiting:
             # Get origin floor from state
             origin_floor = passengers_at_origin.get(p)
             if origin_floor: # Ensure origin floor exists in state
                 floors_needed.add(origin_floor)
        for p in passengers_currently_boarded:
             # Get destination floor from static destinations
             destin_floor = self.destinations.get(p)
             if destin_floor: # Ensure destination exists (should always for valid passengers)
                 floors_needed.add(destin_floor)

        # 7. Calculate travel cost
        travel_cost = 0
        # Only calculate travel if there are floors to visit AND we have a valid lift location
        if floors_needed and lift_at_floor:
            try:
                current_floor_idx = self.floor_name_to_int(lift_at_floor)
                # Get indices only for needed floors that are in our mapping
                needed_floor_indices = [self.floor_name_to_int(f) for f in floors_needed if f in self.floor_to_int]

                if needed_floor_indices: # Ensure we have valid indices after filtering
                    min_needed_idx = min(needed_floor_indices)
                    max_needed_idx = max(needed_floor_indices)

                    # 8. Estimate travel cost
                    # Distance from the lowest point (min of current and min needed)
                    # to the highest point (max of current and max needed).
                    travel_cost = max(current_floor_idx, max_needed_idx) - min(current_floor_idx, min_needed_idx)
                # else: floors_needed had names not in mapping, travel_cost remains 0

            except Exception as e:
                 # This catch should ideally not be hit if floor_name_to_int returns 0
                 # and needed_floor_indices is checked. It's a final safeguard.
                 print(f"Unexpected error during travel cost calculation: {e}")
                 travel_cost = 0 # Fallback

        # 9. Total heuristic value
        h_value = base_cost + travel_cost

        # The logic ensures h_value > 0 if not goal (unless base_cost + travel_cost happens to be 0,
        # which only occurs when N_origin=0, N_boarded=0, and floors_needed is empty,
        # implying the goal state).

        return h_value
