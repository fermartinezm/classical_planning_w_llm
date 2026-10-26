from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists and provides task.goals and task.static

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This might happen if the state representation is different than expected.
        # Based on the example state, facts are strings like '(predicate arg1 arg2)'
        # If the input is already a tuple/list, adjust accordingly.
        # Assuming string format based on example.
        # print(f"Warning: Unexpected fact format: {fact}") # Optional warning
        return [] # Return empty list for malformed facts

    return fact[1:-1].split()

# Helper function to match PDDL facts
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
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the required board and depart actions and adds an estimate
    of the necessary lift movement actions. The movement estimate is based on
    the span of floors where actions (pick-up or drop-off) are needed and the
    distance from the current lift floor to the closest floor requiring an action.

    # Assumptions
    - Floors are ordered linearly, defined by `(above f_higher f_lower)` facts in static.
    - All passengers mentioned in the goal must be served.
    - The `destin` predicate for a passenger is static (present in the initial state or static facts).
    - The PDDL problem is well-formed (e.g., unique lowest floor, no cycles in above facts).

    # Heuristic Initialization
    - Parses the `(above f_higher f_lower)` facts from the static information
      to build a mapping from floor names to integer indices representing their order.
      It identifies the lowest floor as the one that is a 'lower_floor' in an `above`
      fact but never a 'higher_floor'. It then follows the 'immediately above' chain.
      If no `above` facts exist but floors are present, it assumes a single floor.
      If floor ordering cannot be determined (e.g., multiple floors but no `above` facts,
      or invalid `above` structure), floor lookups will fail later, resulting in infinity.
    - Extracts the destination floor for each passenger from the initial state or static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current floor of the lift by finding the fact `(lift-at ?f)`.
    2. Identify all unserved passengers. These are passengers mentioned in the goal
       `(served ?p)` that do not have `(served ?p)` true in the current state.
    3. From the state, identify which unserved passengers are waiting (`(origin p f)`)
       and which are boarded (`(boarded p)`).
    4. Count the number of waiting passengers. Each requires a `board` action (cost 1).
    5. Count the number of unserved passengers (waiting + boarded). Each requires a `depart` action (cost 1).
    6. Determine the set of "action floors" where the lift must stop:
       - The origin floor for every waiting passenger.
       - The destination floor for every unserved passenger (both waiting and boarded).
    7. If there are no action floors, all necessary pick-ups and drop-offs are done.
       If there are also no unserved passengers, the goal is reached, and the heuristic is 0.
       If there are unserved passengers but no action floors, this indicates an issue
       (e.g., passenger needs dropping off but is not boarded or at origin, or destination is unknown).
       Assuming valid states, if no action floors are needed, all unserved passengers must be boarded
       and at their destination floor, or already served. The check for `unserved_passengers` handles the goal state.
       If `unserved_passengers > 0` but `action_floor_indices` is empty, something is wrong with the state/problem definition.
       We return infinity in case of errors like unknown floors. If action_floor_indices is empty but unserved > 0, it implies an unsolvable state from here, so infinity is appropriate.
    8. If there are action floors:
       - Map the action floors to their corresponding integer indices using the
         floor order established during initialization. If any floor is unknown, return infinity.
       - Find the minimum (`min_idx`) and maximum (`max_idx`) index among the action floor indices.
       - Find the index (`closest_idx`) of the action floor that is closest
         in terms of floor index to the current lift floor index (`current_lift_idx`).
       - Estimate the movement cost as the distance from the current lift floor
         to the closest action floor (`abs(current_lift_idx - closest_idx)`) plus
         the span of the action floors (`max_idx - min_idx`). This estimates the cost
         to reach the "active" range of floors and then traverse that range.
    9. The total heuristic value is the sum of the number of waiting passengers
       (for board actions), the number of unserved passengers (for depart actions),
       and the estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting floor order and passenger destinations.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # 1. Build floor order map (floor name -> index)
        self.floor_to_index = {}
        self.index_to_floor = {}
        floor_above_map = {} # maps lower_floor -> higher_floor
        higher_floors = set()
        lower_floors = set()
        all_floors_from_above = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "above":
                f_higher, f_lower = parts[1], parts[2]
                all_floors_from_above.add(f_higher)
                all_floors_from_above.add(f_lower)
                floor_above_map[f_lower] = f_higher
                higher_floors.add(f_higher)
                lower_floors.add(f_lower)

        if not all_floors_from_above:
             # Case with no above facts. Try to find floors mentioned in any fact.
             floors_in_facts = set()
             for fact in initial_state | self.goals | static_facts:
                  parts = get_parts(fact)
                  for part in parts:
                       # Simple heuristic: assume floor objects start with 'f'
                       if part.startswith('f'):
                            floors_in_facts.add(part)

             if len(floors_in_facts) == 1:
                  # Assume it's the only floor, index 0
                  floor_name = list(floors_in_facts)[0]
                  self.floor_to_index = {floor_name: 0}
                  self.index_to_floor = {0: floor_name}
             elif len(floors_in_facts) > 1:
                  # Multiple floors but no 'above' facts to order them.
                  # Cannot determine order, floor lookups will fail later.
                  print("Warning: Multiple floors found but no 'above' facts to define order.")
                  # Maps remain empty, lookups will return None.
             # If no floors found at all, maps are empty.

        else: # We have above facts, build the chain
            # Find the lowest floor: it's in LowerFloors but not in HigherFloors
            potential_lowest = lower_floors - higher_floors

            if len(potential_lowest) == 1:
                lowest_floor = list(potential_lowest)[0]
                current_floor = lowest_floor
                index = 0
                # Follow the chain upwards using the floor_above_map
                while current_floor is not None:
                    if current_floor in self.floor_to_index:
                         # Cycle or duplicate floor detected - invalid structure
                         print(f"Error: Cycle or duplicate floor '{current_floor}' detected in floor order.")
                         self.floor_to_index = {} # Clear maps
                         self.index_to_floor = {}
                         break # Stop building map

                    self.floor_to_index[current_floor] = index
                    self.index_to_floor[index] = current_floor
                    current_floor = floor_above_map.get(current_floor) # Get floor immediately above
                    index += 1
            elif len(potential_lowest) == 0:
                 # No floor is only in LowerFloors. Could be one floor (handled above)
                 # or invalid structure (cycle, or disconnected components).
                 print("Error: Could not find a unique lowest floor from 'above' facts.")
                 # Maps remain empty, lookups will fail.
            else:
                 # Multiple potential lowest floors - invalid structure
                 print("Error: Found multiple potential lowest floors.")
                 # Maps remain empty, lookups will fail.


        # 2. Extract passenger destinations
        self.passenger_destinations = {}
        # Destinations are typically in the initial state or static facts
        for fact in initial_state | static_facts:
             parts = get_parts(fact)
             if parts and parts[0] == "destin":
                 passenger, floor = parts[1], parts[2]
                 self.passenger_destinations[passenger] = floor

        # As a fallback, check goals if destin wasn't in initial/static (less common)
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served":
                 passenger = parts[1]
                 # If passenger needs serving but destin isn't found yet, try goals
                 if passenger not in self.passenger_destinations:
                      # This is less likely for destin facts, but included for robustness
                      for goal_fact in self.goals:
                           goal_parts = get_parts(goal_fact)
                           if goal_parts and goal_parts[0] == "destin" and goal_parts[1] == passenger:
                                self.passenger_destinations[passenger] = goal_parts[2]
                                break


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Find current lift floor
        current_lift_floor = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "lift-at":
                current_lift_floor = parts[1]
                break

        if current_lift_floor is None:
             # This should not happen in a valid miconic state
             # print("Error: Lift location not found in state.") # Debugging
             return float('inf') # Cannot proceed without lift location

        current_lift_idx = self.floor_to_index.get(current_lift_floor)
        if current_lift_idx is None:
             # Lift is at a floor not found during initialization (invalid problem/state)
             # print(f"Error: Unknown floor '{current_lift_floor}' found in state.") # Debugging
             return float('inf') # Unknown floor

        # 2. Identify unserved passengers and their status
        waiting_passengers = set()
        boarded_passengers = set()
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if parts:
                if parts[0] == "origin":
                    waiting_passengers.add(parts[1])
                elif parts[0] == "boarded":
                    boarded_passengers.add(parts[1])
                elif parts[0] == "served":
                    served_passengers.add(parts[1])

        # Identify all passengers that need to be served according to the goal
        all_passengers_in_goals = {get_parts(goal)[1] for goal in self.goals if get_parts(goal) and get_parts(goal)[0] == "served"}
        unserved_passengers = all_passengers_in_goals - served_passengers

        # Filter waiting/boarded sets to include only those who are also unserved
        waiting_passengers = waiting_passengers.intersection(unserved_passengers)
        boarded_passengers = boarded_passengers.intersection(unserved_passengers)

        # Check if goal is reached (all passengers in goals are served)
        if not unserved_passengers:
            return 0

        # 3. Count board actions needed (one for each waiting passenger)
        num_board_actions = len(waiting_passengers)

        # 4. Count depart actions needed (one for each unserved passenger)
        num_depart_actions = len(unserved_passengers)

        # 5. Determine action floors (origins for waiting, destins for unserved)
        action_floor_indices = set()

        # Origins of waiting passengers
        for p in waiting_passengers:
            # Find origin floor for p in the current state
            origin_floor = None
            for fact in state:
                 parts = get_parts(fact)
                 if parts and parts[0] == "origin" and parts[1] == p:
                      origin_floor = parts[2]
                      break
            if origin_floor:
                 origin_idx = self.floor_to_index.get(origin_floor)
                 if origin_idx is not None:
                      action_floor_indices.add(origin_idx)
                 else:
                      # Origin floor not found in floor map (invalid problem/state)
                      # print(f"Error: Unknown origin floor '{origin_floor}' for passenger {p}") # Debugging
                      return float('inf')
            else:
                 # Passenger is in waiting_passengers but has no origin fact? Invalid state.
                 # print(f"Error: Waiting passenger {p} has no origin fact in state.") # Debugging
                 return float('inf')


        # Destinations of unserved passengers (waiting + boarded)
        for p in unserved_passengers:
             destin_floor = self.passenger_destinations.get(p)
             if destin_floor:
                  destin_idx = self.floor_to_index.get(destin_floor)
                  if destin_idx is not None:
                       action_floor_indices.add(destin_idx)
                  else:
                       # Destination floor not found in floor map (invalid problem/state)
                       # print(f"Error: Unknown destination floor '{destin_floor}' for passenger {p}") # Debugging
                       return float('inf')
             else:
                  # Unserved passenger has no destination recorded (invalid problem)
                  # print(f"Error: Destination not found for unserved passenger {p}") # Debugging
                  return float('inf')


        # If no action floors are identified but there are unserved passengers,
        # this implies an issue (e.g., passengers are stuck). Return infinity.
        # This check also covers the case where floor ordering failed and action_floor_indices is empty.
        if not action_floor_indices:
             # This case should ideally not be reached if unserved_passengers > 0
             # and floor mapping is correct, as unserved passengers need pickup/dropoff.
             # If reached, it suggests an unsolvable state or parsing issue.
             # print("Warning: No action floors found for unserved passengers.") # Debugging
             return float('inf')


        # 6. Estimate movement cost
        min_action_idx = min(action_floor_indices)
        max_action_idx = max(action_floor_indices)

        # Find the closest action floor index to the current lift index
        closest_action_idx = min(action_floor_indices, key=lambda idx: abs(idx - current_lift_idx))

        # Movement cost: distance from current floor to closest action floor
        # plus the span of floors that need visiting.
        movement_cost = abs(current_lift_idx - closest_action_idx) + (max_action_idx - min_action_idx)

        # 7. Total heuristic
        # Sum of board actions, depart actions, and estimated movement actions.
        total_cost = num_board_actions + num_depart_actions + movement_cost

        return total_cost

