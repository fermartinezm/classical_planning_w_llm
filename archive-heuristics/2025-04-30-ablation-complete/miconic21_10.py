from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import sys # Import sys for returning a large number in case of errors

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    whose status (waiting at origin, boarded, or served) is explicitly represented
    in the current state. It sums the number of boarding actions needed, the number
    of departing actions needed, and an estimate of the minimum lift movement
    required to visit all necessary floors for these passengers.

    # Assumptions
    - Each unboarded, unserved passenger explicitly listed at an origin in the state
      requires one 'board' action.
    - Each unserved passenger (boarded or unboarded) explicitly listed in the state
      requires one 'depart' action.
    - The lift movement cost is estimated based on the range of floors that must be visited
      to pick up unboarded passengers and drop off boarded passengers whose status
      is represented in the state.
    - The floor numbering is consistent (e.g., f1 < f2 < ...). The heuristic infers
      the floor order from the 'above' predicates.
    - Passengers whose status is not explicitly mentioned in the state (not in origin,
      boarded, or served facts) are not considered for the remaining plan from this state.
    - The state representation is consistent with domain rules (e.g., unserved, unboarded
      passengers are at their origin). Invalid states (e.g., lift at unknown floor,
      passenger destination unknown) are penalized with a large heuristic value.

    # Heuristic Initialization
    - Parses static facts to build a mapping from floor names to numerical indices
      based on the 'above' predicates. Handles the single-floor case.
    - Stores the destination floor for each passenger from the 'destin' predicates
      found in static facts or initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state (all required 'served' facts from the task goal are true). If yes, return 0.
    2. Identify the current floor of the lift from the state. If the lift location is missing or unknown, return a large penalty.
    3. Initialize counters for needed board actions, needed depart actions, and a set for required floor indices.
    4. Identify passengers whose status (origin, boarded, served) is explicitly mentioned in the current state facts.
    5. For each such passenger:
       - Check if the passenger is 'served' in the current state. If yes, continue to the next passenger.
       - If not served:
         - Increment the count of needed depart actions (each unserved passenger needs one final depart).
         - Check if the passenger is 'boarded' in the current state.
           - If 'boarded': Get their destination floor. If destination is unknown or floor is not in map, return a large penalty. Add the destination floor index to the set of required floor indices.
           - If not 'boarded' (and must therefore be at an origin according to domain rules and state representation assumption): Increment the count of needed board actions. Find their origin floor from the state facts and their destination floor from the stored destinations. If origin or destination is unknown or floors are not in map, return a large penalty. Add both the origin floor index and the destination floor index to the set of required floor indices.
    6. Estimate the lift movement cost:
       - If the set of required floor indices is empty (meaning all relevant passengers in the state are served), the movement cost is 0.
       - Otherwise, find the minimum and maximum floor indices among the required floors.
       - Get the index of the current lift floor.
       - The estimated movement cost is the distance from the current floor index to the nearest
         required floor index, plus the total span of the required floor indices:
         `min(|current_idx - min_req_idx|, |current_idx - max_req_idx|) + (max_req_idx - min_req_idx)`.
    7. The total heuristic value is the sum of the estimated board actions, depart actions,
       and lift movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals # Goal conditions, used to check for goal state.
        static_facts = task.static # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial facts, might contain destin/origin

        # 1. Build floor mapping (name -> index)
        above_facts_parts = [get_parts(fact) for fact in static_facts if match(fact, "above", "*", "*")]

        if not above_facts_parts:
            # Handle single floor case: find the single floor name
            all_floors_found = set()
            # Look for floors in initial state and static facts
            for fact in initial_state | static_facts:
                parts = get_parts(fact)
                for part in parts:
                    # Simple check: starts with 'f' and potentially followed by digits
                    if part.startswith('f') and (len(part) > 1 and part[1:].isdigit() or len(part) == 1):
                         all_floors_found.add(part)
            # Assuming there's exactly one floor in this case
            if len(all_floors_found) == 1:
                 single_floor = list(all_floors_found)[0]
                 self.floor_to_index = {single_floor: 0}
            else:
                 # Fallback if no floors found or multiple found unexpectedly
                 # print(f"Warning: Could not determine single floor in domain without 'above' facts. Found {all_floors_found}. Using empty map.")
                 self.floor_to_index = {} # Empty map will likely cause errors later, but indicates problem setup issue.

        else:
            # Multiple floors case: build map from 'above' facts
            immediately_above = {}
            all_floors = set()
            for _, f_lower, f_higher in above_facts_parts:
                immediately_above[f_lower] = f_higher
                all_floors.add(f_lower)
                all_floors.add(f_higher)

            # Find the lowest floor
            lowest_floor = None
            above_second_args = set(immediately_above.values())
            for floor in all_floors:
                 if floor not in above_second_args:
                     lowest_floor = floor
                     break
            # Fallback if lowest floor not found (e.g., cyclic 'above' or disconnected floors)
            if lowest_floor is None and all_floors:
                 # This shouldn't happen in valid PDDL, but sort and pick first as a fallback
                 # print("Warning: Could not find a unique lowest floor. Using alphabetically first.")
                 lowest_floor = sorted(list(all_floors))[0]

            # Build the ordered list of floors and the mapping
            self.floor_to_index = {}
            current_floor = lowest_floor
            index = 0
            # Handle case where lowest_floor might still be None if all_floors was empty
            while current_floor is not None and current_floor in all_floors:
                if current_floor in self.floor_to_index:
                     # Prevent infinite loops in case of cyclic 'above' (shouldn't happen in valid PDDL)
                     # print(f"Warning: Cyclic 'above' detected involving {current_floor}. Stopping floor mapping.")
                     break
                self.floor_to_index[current_floor] = index
                index += 1
                current_floor = immediately_above.get(current_floor)


        # 2. Store passenger destinations
        self.passenger_destinations = {}
        # Destinations are typically static, but might be in initial state too.
        for fact in static_facts | initial_state:
            if match(fact, "destin", "*", "*"):
                _, passenger, destination_floor = get_parts(fact)
                self.passenger_destinations[passenger] = destination_floor


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Current world state.

        # 1. Check if the goal is reached
        if self.goals <= state:
            return 0

        # 2. Identify current lift floor
        current_lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                _, current_lift_floor = get_parts(fact)
                break
        # Handle case where lift-at might be missing (shouldn't happen in valid states)
        if current_lift_floor is None:
             # Cannot compute heuristic without lift location
             # Return a large number as a penalty for an invalid state
             # print("Warning: lift-at predicate missing in state.")
             return sys.maxsize # Large penalty

        current_lift_floor_index = self.floor_to_index.get(current_lift_floor)
        if current_lift_floor_index is None:
             # Handle case where lift is at an unknown floor (shouldn't happen)
             # print(f"Warning: Lift at unknown floor '{current_lift_floor}'.")
             return sys.maxsize # Large penalty


        # 3. Initialize counters and required floors set
        board_actions_needed = 0
        depart_actions_needed = 0
        required_floor_indices = set()

        # Get current status of passengers from the state for quick lookup
        served_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        boarded_passengers_in_state = {get_parts(fact)[1] for fact in state if match(fact, "boarded", "*")}
        origin_passengers_in_state = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "origin", "*", "*")}

        # Identify all passengers whose status is mentioned in the state
        passengers_in_state = set(served_passengers_in_state) | set(boarded_passengers_in_state) | set(origin_passengers_in_state.keys())

        # 4 & 5. Iterate through passengers mentioned in the state and determine needs
        for passenger in passengers_in_state:
            if passenger in served_passengers_in_state:
                continue # Passenger is already served

            # Passenger is unserved (and mentioned in state)
            depart_actions_needed += 1 # Needs one depart action eventually

            if passenger in boarded_passengers_in_state:
                # Passenger is unserved and boarded
                destination_floor = self.passenger_destinations.get(passenger)
                if destination_floor:
                    dest_idx = self.floor_to_index.get(destination_floor)
                    if dest_idx is not None:
                        required_floor_indices.add(dest_idx)
                    else:
                         # print(f"Warning: Destination floor '{destination_floor}' for boarded passenger '{passenger}' not found in floor map.")
                         return sys.maxsize # Penalty for invalid state/data
                else:
                     # print(f"Warning: Destination for boarded passenger '{passenger}' not found.")
                     return sys.maxsize # Penalty for invalid state/data

            elif passenger in origin_passengers_in_state:
                # Passenger is unserved and waiting at origin (explicitly in state)
                board_actions_needed += 1 # Needs one board action
                origin_floor = origin_passengers_in_state[passenger]
                destination_floor = self.passenger_destinations.get(passenger)

                origin_idx = self.floor_to_index.get(origin_floor)
                dest_idx = self.floor_to_index.get(destination_floor)

                if origin_idx is not None:
                    required_floor_indices.add(origin_idx)
                else:
                     # print(f"Warning: Origin floor '{origin_floor}' for passenger '{passenger}' not found in floor map.")
                     return sys.maxsize # Penalty for invalid state/data

                if dest_idx is not None:
                    required_floor_indices.add(dest_idx)
                else:
                     # print(f"Warning: Destination floor '{destination_floor}' for passenger '{passenger}' not found in floor map.")
                     return sys.maxsize # Penalty for invalid state/data

            # Note: Passengers unserved, not boarded, and not at origin in state are ignored
            # by iterating only over passengers explicitly mentioned in the state facts.


        # 6. Estimate lift movement cost
        movement_cost = 0
        if required_floor_indices:
            min_req_idx = min(required_floor_indices)
            max_req_idx = max(required_floor_indices)

            # Cost to reach the range + cost to traverse the range
            # This assumes the lift goes to one end of the required range first,
            # then traverses the range.
            movement_cost = min(abs(current_lift_floor_index - min_req_idx),
                                abs(current_lift_floor_index - max_req_idx)) + \
                            (max_req_idx - min_req_idx)

        # 7. Total heuristic estimate
        total_cost = board_actions_needed + depart_actions_needed + movement_cost

        return total_cost
