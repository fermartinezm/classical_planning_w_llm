# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
import math # For float('inf')

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    This heuristic estimates the cost to serve all passengers. It sums the estimated cost for each unserved passenger, considering whether they are waiting at their origin or are already boarded. The cost for a passenger is estimated by the number of floor movements required for the lift to reach their origin (if waiting) or destination (if boarded), plus the cost of the board and depart actions.

    # Assumptions
    - The floors are linearly ordered by the 'above' predicate, defining direct adjacency via facts like (above f_i f_{i+1}). The set of 'above' facts must allow for the reconstruction of this unique linear order.
    - The cost of moving one floor up or down is 1.
    - The cost of boarding a passenger is 1.
    - The cost of departing a passenger is 1.
    - The heuristic calculates costs for each unserved passenger independently and sums them, ignoring potential synergies (like picking up/dropping off multiple passengers at the same floor) or lift capacity. This makes it non-admissible but potentially effective for greedy best-first search.
    - Assumes valid states conform to the domain: a passenger is either at their origin, boarded, or served. The lift is always at some floor. Origin and destination for goal passengers are defined in the initial state.

    # Heuristic Initialization
    - Parses the 'above' facts from the static information to create a mapping from floor names to numerical levels. If a unique linear order cannot be determined from 'above' facts, a fallback heuristic is used.
    - Parses 'origin' and 'destin' facts from the initial state to store the origin and destination floor for each passenger.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Check if the current state is a goal state (all passengers served). If yes, the heuristic is 0.
    2. If the floor level mapping failed during initialization (e.g., due to missing or ambiguous 'above' facts), return a fallback heuristic (twice the number of unserved passengers).
    3. Find the current floor of the lift from the state. If the lift location is not found or its floor level is unknown, return infinity as the state is likely invalid or unreachable.
    4. Initialize the total heuristic cost to 0.
    5. Iterate through all passengers whose origin and destination were identified during initialization.
    6. For each passenger:
       - Check if the passenger is already 'served' in the current state. If yes, add 0 to the total cost for this passenger and continue.
       - If not 'served', check if the passenger is 'boarded'.
         - If 'boarded':
           - Get the passenger's destination floor and its corresponding level. If the destination floor level is unknown, return infinity (invalid state).
           - Calculate the number of floor moves required for the lift to go from its current floor to the passenger's destination floor (absolute difference in floor levels).
           - Add this number of moves + 1 (for the 'depart' action) to the total cost.
         - If not 'boarded' (meaning they are waiting at their origin):
           - Check if the passenger is at their origin floor in the current state. If not, the state is invalid for this passenger; return infinity.
           - Get the passenger's origin and destination floors and their corresponding levels. If either level is unknown, return infinity (invalid state).
           - Calculate the number of floor moves required for the lift to go from its current floor to the passenger's origin floor.
           - Add this number of moves + 1 (for the 'board' action) to the cost.
           - Calculate the number of floor moves required for the lift to go from the origin floor to the destination floor.
           - Add this number of moves + 1 (for the 'depart' action) to the cost.
    7. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor levels and passenger info.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # 1. Build floor level mapping from 'above' facts
        above_map = {} # Maps floor f_i to f_{i+1} if (above f_i f_{i+1}) is present
        all_floors_mentioned = set()

        # First pass: Identify direct 'above' relations and all mentioned floors
        potential_predecessors = set() # Floors that are the first arg of some 'above'
        potential_successors = set()  # Floors that are the second arg of some 'above'

        # Flag to indicate if 'above' facts suggest a non-linear structure
        non_linear_above = False

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above" and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                # Assume (above f_i f_{i+1}) facts define the direct chain
                # If f1 is already a key, it means we found a cycle or multiple successors,
                # which indicates a non-linear structure or invalid 'above' facts for this heuristic.
                if f1 in above_map:
                     # print(f"Warning: Floor '{f1}' appears as first argument in multiple 'above' facts. Cannot build linear map.")
                     non_linear_above = True
                     break # Stop processing above facts

                above_map[f1] = f2
                potential_predecessors.add(f1)
                potential_successors.add(f2)
                all_floors_mentioned.add(f1)
                all_floors_mentioned.add(f2)

        self.floor_level = {}
        if above_map and not non_linear_above:
            # Find the lowest floor: a floor that is a predecessor but not a successor
            potential_lowest_floors = potential_predecessors - potential_successors

            if len(potential_lowest_floors) == 1:
                lowest_floor = potential_lowest_floors.pop()
                current_floor = lowest_floor
                level = 1
                # Traverse the direct 'above' chain to build levels
                while current_floor in above_map:
                    self.floor_level[current_floor] = level
                    current_floor = above_map[current_floor]
                    level += 1
                # Add the highest floor (the last one in the chain)
                self.floor_level[current_floor] = level

                # Basic check: Ensure all floors mentioned in 'above' facts got a level
                if len(self.floor_level) != len(all_floors_mentioned):
                     # This indicates the 'above' facts didn't form a single linear chain
                     # (e.g., disconnected floors mentioned in 'above' facts)
                     self.floor_level = None # Indicate failure
                     # print("Warning: 'above' facts found, but could not build a complete linear floor map covering all mentioned floors.")

            else:
                 # Could not determine a unique lowest floor from 'above' facts
                 self.floor_level = None # Indicate failure
                 # print("Warning: Could not determine unique lowest floor from 'above' facts.")
        else: # above_map is empty or non_linear_above is True
             self.floor_level = None
             # if non_linear_above:
             #     print("Warning: 'above' facts suggest non-linear structure. Cannot build linear floor map.")
             # elif all_floors_mentioned:
             #      print("Warning: 'above' facts found, but could not build linear floor map (possibly disconnected).")


        # If floor_level is still None, check if there are multiple floors in initial state
        # to determine if the fallback heuristic is appropriate (multiple floors without order)
        self._multiple_floors_in_init = False
        if self.floor_level is None:
             all_floors_in_init = set()
             for fact in initial_state:
                 parts = get_parts(fact)
                 if parts:
                     if parts[0] == "lift-at" and len(parts) == 2:
                         all_floors_in_init.add(parts[1])
                     elif parts[0] in ["origin", "destin"] and len(parts) == 3:
                         all_floors_in_init.add(parts[2])
             self._multiple_floors_in_init = len(all_floors_in_init) > 1
             # if self._multiple_floors_in_init:
                 # print("Warning: Multiple floors detected in initial state, but no floor order could be established from 'above' facts. Using fallback heuristic.")
             # If not multiple floors in init, and floor_level is None, it implies 0 or 1 floor,
             # or invalid init state. The fallback will still work (unserved_count * 2).


        # 2. Store passenger origin and destination floors
        self.passenger_info = {} # {passenger_name: {'origin': floor, 'destin': floor}}
        # Extract from initial state
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == "origin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                if p not in self.passenger_info:
                    self.passenger_info[p] = {}
                self.passenger_info[p]['origin'] = f
            elif parts and parts[0] == "destin" and len(parts) == 3:
                p, f = parts[1], parts[2]
                if p not in self.passenger_info:
                    self.passenger_info[p] = {}
                self.passenger_info[p]['destin'] = f

        # Note: Passengers mentioned in goals but not in initial origin/destin facts
        # will not be included in self.passenger_info and thus won't contribute
        # to the heuristic sum. This assumes valid PDDL where all goal passengers
        # have origin/destin defined in the initial state.


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Handle case where floor mapping failed during initialization
        if self.floor_level is None:
             # Fallback heuristic: Count unserved passengers * 2 (board + depart)
             # This fallback is used if floor order is unknown (multiple floors without 'above' facts
             # or 'above' facts don't form a linear chain).
             unserved_count = 0
             for p in self.passenger_info:
                 if f"(served {p})" not in state:
                     unserved_count += 1
             # If there's only one floor, movement cost is always 0, so this is accurate.
             # If there are multiple floors but no order, this is a weak but finite heuristic.
             return unserved_count * 2


        # Find current lift location
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at" and len(parts) == 2:
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This state is likely invalid or unreachable in miconic
             # (lift must always be somewhere). Return infinity.
             return math.inf

        current_lift_level = self.floor_level.get(current_lift_floor)
        if current_lift_level is None:
             # Current lift floor not found in floor map - problem with parsing or state.
             # This shouldn't happen if floor_level was built correctly and state is valid.
             return math.inf # Indicate severe problem


        total_cost = 0

        # Iterate through all passengers we know about from initial state
        for p, info in self.passenger_info.items():
            origin_f = info.get('origin')
            destin_f = info.get('destin')

            # Should have origin/destin info for all passengers in self.passenger_info
            # but add check for robustness.
            if not origin_f or not destin_f:
                 # This passenger info is incomplete, cannot calculate cost.
                 # This case indicates a problem with the initial state definition
                 # for this passenger. Return infinity as state is likely invalid.
                 return math.inf


            # Check if passenger is served
            if f"(served {p})" in state:
                continue # Passenger is served, cost is 0 for this one

            # Check if passenger is boarded
            boarded_fact = f"(boarded {p})"
            if boarded_fact in state:
                # Passenger is boarded, needs to reach destination and depart
                destin_level = self.floor_level.get(destin_f)
                if destin_level is None:
                     # Destination floor not in map - problem. Return infinity.
                     return math.inf

                # Cost = moves to destination + depart action
                moves_to_destin = abs(current_lift_level - destin_level)
                total_cost += moves_to_destin + 1 # 1 for depart action

            # Passenger is not served and not boarded, must be at origin
            else: # Check if they are at origin explicitly
                 origin_fact = f"(origin {p} {origin_f})"
                 if origin_fact in state:
                     # Passenger is waiting at origin, needs board, move to destin, depart
                     origin_level = self.floor_level.get(origin_f)
                     destin_level = self.floor_level.get(destin_f)

                     if origin_level is None or destin_level is None:
                         # Origin or Destination floor not in map - problem. Return infinity.
                         return math.inf

                     # Cost = moves to origin + board + moves from origin to destination + depart
                     moves_to_origin = abs(current_lift_level - origin_level)
                     moves_origin_to_destin = abs(origin_level - destin_level)

                     total_cost += moves_to_origin + 1 + moves_origin_to_destin + 1 # 1 for board, 1 for depart
                 else:
                     # Passenger is not served, not boarded, and not at origin.
                     # This indicates an invalid state for this passenger according to domain rules.
                     # Return infinity.
                     return math.inf


        return total_cost
