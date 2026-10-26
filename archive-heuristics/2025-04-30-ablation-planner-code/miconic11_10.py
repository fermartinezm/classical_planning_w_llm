from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # Assuming task object also provides access to objects
            self.objects = task.objects if hasattr(task, 'objects') else {}

        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions required to serve all passengers.
    It sums the number of boarding actions needed, the number of departing actions needed,
    and an estimate of the lift movement cost.

    # Assumptions
    - Each passenger needs one board action and one depart action.
    - The lift can carry multiple passengers.
    - The movement cost is estimated by the distance required to traverse the range
      of floors where pickups or dropoffs are needed, plus the distance from the
      current floor to this range.
    - Floors are ordered linearly by the `above` predicates.

    # Heuristic Initialization
    - Extracts the list of all passengers and floors from the task objects.
    - Determines the destination floor for each passenger from the static facts.
    - Determines the sorted order of floors and their indices based on the `above` predicates.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the goal is already met (all passengers served). The goal is met if `(served p)` is true for every passenger `p`. If so, the heuristic is 0.
    2. Identify the current floor of the lift by finding the fact `(lift-at ?f)`.
    3. Identify unserved passengers. For each unserved passenger:
       - If `(origin p ?f)` is true in the current state, the passenger is waiting at `?f` and needs a `board` action. Add 1 to base action count and add `?f` to required floors.
       - If `(boarded p)` is true in the current state, the passenger is boarded and needs a `depart` action at their destination floor. Add 1 to base action count and add their destination floor (looked up from static facts) to required floors.
       - If an unserved passenger is neither at origin nor boarded in the current state, this indicates a potentially malformed state. The heuristic cannot reliably estimate cost for such a passenger and returns infinity.
    4. The base action cost is the total count of `board` and `depart` actions needed for all unserved passengers found in steps above.
    5. Calculate movement cost:
       - If there are no required floors identified in step 3, the movement cost is 0.
       - If there are required floors, find the minimum and maximum floor indices among them using the pre-calculated floor ordering.
       - Calculate the movement cost estimate: `abs(current_floor_index - min_required_floor_index) + (max_required_floor_index - min_required_floor_index)`.
    6. The total heuristic value is the sum of the base action cost (step 4) and the movement cost (step 5).
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        super().__init__(task) # Call base class constructor

        # Extract passengers and floors from task objects
        # Assuming task.objects is a dict like {'obj_name': 'obj_type'}
        self.passengers = [obj for obj, obj_type in self.objects.items() if obj_type == 'passenger']
        self.floors = [obj for obj, obj_type in self.objects.items() if obj_type == 'floor']

        # Determine destination for each passenger from static facts
        self.destinations = {}
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "destin":
                _, passenger, floor = parts
                self.destinations[passenger] = floor

        # Determine floor ordering and indices from static facts
        self.floor_to_index = self._get_floor_ordering()
        # Optional: self.index_to_floor = {v: k for k, v in self.floor_to_index.items()}
        self.num_floors = len(self.floors)

        # Pre-calculate goal set for efficiency
        self._goal_set = {f"(served {p})" for p in self.passengers}


    def _get_floor_ordering(self):
        """
        Determines the sorted order of floors based on 'above' predicates
        and returns a mapping from floor name to its index (0-based).
        Assumes 'above f_lower f_higher' means f_higher is immediately above f_lower.
        Assumes floors form a single linear sequence.
        Handles cases with no 'above' facts or inconsistent facts by falling back
        to alphabetical sorting of all known floors.
        """
        above_map = {} # maps lower floor to higher floor
        below_map = {} # maps higher floor to lower floor
        all_floors_in_above = set()

        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "above":
                f_lower, f_higher = parts[1], parts[2]
                above_map[f_lower] = f_higher
                below_map[f_higher] = f_lower
                all_floors_in_above.add(f_lower)
                all_floors_in_above.add(f_higher)

        # Find the lowest floor: a floor that is the first argument of an 'above'
        # but is NOT the second argument of ANY 'above' predicate.
        potential_lowest = set(above_map.keys()) - set(below_map.keys())

        if len(potential_lowest) == 1:
            lowest_floor = potential_lowest.pop()
            # Build the sorted list of floors by following the chain
            sorted_floors = []
            current = lowest_floor
            while current is not None:
                sorted_floors.append(current)
                current = above_map.get(current)

            # Check if all floors involved in 'above' are in the chain
            if len(sorted_floors) == len(all_floors_in_above):
                 # Check if all *defined* floors are in the chain. If not, fallback.
                 # This handles cases where some floors exist but are not in 'above' chain.
                 if all(f in sorted_floors for f in self.floors):
                      # Success: Found a consistent ordering covering all floors
                      return {floor: i for i, floor in enumerate(sorted_floors)}
                 # else: print("Warning: Floor ordering from 'above' predicates does not include all defined floors. Falling back to alphabetical order.")


        # Fallback: If finding the unique lowest floor failed, or the chain is incomplete,
        # or no 'above' predicates exist, sort all known floors alphabetically.
        if self.floors:
            # print("Warning: Could not determine unique floor ordering from 'above' predicates. Using alphabetical order.")
            return {floor: i for i, floor in enumerate(sorted(self.floors))}

        # No floors defined
        return {}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # State is a frozenset of fact strings

        # 1. Check if goal is met
        # Goal is (served p) for all passengers p
        # Check if the current state contains all facts in the goal set
        if self._goal_set.issubset(state):
             return 0 # Goal state

        # If no floors are defined or ordering failed critically, cannot compute heuristic
        if not self.floor_to_index and self.floors:
             # If floors exist but ordering failed, we can't proceed reliably
             # print("Error: Floor ordering not available for existing floors. Cannot compute heuristic.")
             return float('inf')
        elif not self.floors:
             # No floors defined in the problem. If not goal, something is wrong.
             # But if there are no passengers either, goal is trivially met (h=0).
             # If passengers exist but no floors, they can never be served.
             # This case should be caught by the goal check if there are passengers.
             # If we reach here and there are passengers but no floors, it's unsolvable.
             if self.passengers:
                  # print("Error: Passengers exist but no floors defined. Problem likely unsolvable.")
                  return float('inf')
             else:
                  # No passengers, no floors. Goal is met (h=0).
                  return 0 # Should be covered by goal check


        # 2. Find current lift floor
        current_floor = None
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "lift-at":
                current_floor = parts[1]
                break
        if current_floor is None:
             # This should not happen in a valid miconic state with a lift
             # print("Error: Lift location not found in state.")
             return float('inf') # Should not happen in valid states

        current_floor_index = self.floor_to_index.get(current_floor)
        if current_floor_index is None:
             # This might happen if the lift is at a floor not in the floor list/ordering
             # print(f"Error: Lift at floor {current_floor} not found in floor ordering map.")
             return float('inf') # Should not happen in valid states


        # 3. Identify unserved passengers and required actions/floors
        base_actions = 0
        required_floors = set()
        served_passengers = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Track state of all passengers for efficiency
        passenger_states = {} # {p: 'origin_f' or 'boarded' or 'served'}
        for fact in state:
             parts = get_parts(fact)
             if len(parts) == 2 and parts[0] == "served":
                  passenger_states[parts[1]] = 'served'
             elif len(parts) == 2 and parts[0] == "boarded":
                  passenger_states[parts[1]] = 'boarded'
             elif len(parts) == 3 and parts[0] == "origin":
                  passenger_states[parts[1]] = parts[2] # Store origin floor


        for passenger in self.passengers:
            state_info = passenger_states.get(passenger)

            if state_info == 'served':
                 continue # Passenger is served, no cost

            elif isinstance(state_info, str) and state_info.startswith('f'): # Passenger is at origin floor
                origin_floor = state_info
                # Passenger is waiting at origin, needs boarding
                base_actions += 1 # Cost for board action
                required_floors.add(origin_floor)

            elif state_info == 'boarded':
                # Passenger is boarded, needs departing
                base_actions += 1 # Cost for depart action
                # Find destination floor from static facts
                destin_floor = self.destinations.get(passenger)
                if destin_floor:
                     required_floors.add(destin_floor)
                else:
                     # This should not happen if problem is well-formed and destinations are in static
                     # print(f"Error: Destination not found in static facts for boarded passenger {passenger}.")
                     return float('inf') # Invalid problem definition?

            else:
                # Unserved passenger is neither at origin nor boarded.
                # This indicates a potentially malformed state.
                # print(f"Warning: Unserved passenger {passenger} is neither at origin nor boarded in state.")
                return float('inf') # Cannot reliably estimate cost for this passenger.


        # 4. Calculate movement cost
        movement_cost = 0
        if required_floors:
            required_floor_indices = []
            for f in required_floors:
                 idx = self.floor_to_index.get(f)
                 if idx is not None:
                      required_floor_indices.append(idx)
                 else:
                      # A required floor is not in our ordering map. Problematic.
                      # print(f"Error: Required floor {f} not found in floor ordering map.")
                      return float('inf') # Should not happen in valid states

            if required_floor_indices: # Ensure list is not empty after checks
                min_req_idx = min(required_floor_indices)
                max_req_idx = max(required_floor_indices)

                # Cost to reach the range + cost to traverse the range
                movement_cost = abs(current_floor_index - min_req_idx) + (max_req_idx - min_req_idx)
            # else: movement_cost remains 0


        # 5. Total heuristic
        total_heuristic = base_actions + movement_cost

        return total_heuristic
