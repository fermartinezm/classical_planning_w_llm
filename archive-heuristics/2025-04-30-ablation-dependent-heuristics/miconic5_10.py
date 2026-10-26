# Helper function to parse fact strings
def parse_fact(fact_str):
    """Removes parentheses and splits a PDDL fact string into parts."""
    # Handle potential empty string or non-string input defensively
    if not isinstance(fact_str, str) or len(fact_str) < 2:
        return []
    # Remove leading/trailing parentheses and split by space
    return fact_str[1:-1].split()


class miconicHeuristic:
    """
    Domain-dependent heuristic for the miconic domain.

    Summary:
    Estimates the cost to reach the goal by summing the number of unserved
    passengers (weighted by whether they are waiting or boarded) and the
    estimated movement cost for the lift to visit all necessary floors.
    Necessary floors are the origin floors of waiting passengers and the
    destination floors of boarded passengers.

    Assumptions:
    - Floors are linearly ordered (f1 < f2 < ... < fN). The (above f_i f_j)
      facts define this order, where (above f_i f_j) means f_j is immediately
      above f_i.
    - All floors mentioned in the problem (initial state, goal, static)
      that are relevant for passenger origins/destinations are part of this
      linear ordering defined by the static (above) facts.
    - Passenger destinations are provided in the static facts.

    Heuristic Initialization:
    - Parses the static facts to build a mapping from floor names (e.g., 'f1')
      to integer indices (e.g., 0). This is done by finding the lowest floor
      (one not appearing as the second argument in any (above) fact) and
      following the (above) chain upwards.
    - Parses the static facts to build a mapping from passenger names to their
      destination floor names.
    - Identifies all passengers present in the problem from initial state
      and static facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the state is a goal state (all passengers served). If yes,
       the heuristic value is 0.
    2. Identify the current floor of the lift from the state.
    3. Initialize counts for passengers waiting at their origin and passengers
       boarded in the lift, and sets for floors that need to be visited.
    4. Iterate through all known passengers:
       - If the passenger is already served (check state), ignore them.
       - If the passenger is not served:
         - If the passenger is at their origin floor (check state), increment
           the waiting count and add their origin floor to the set of floors
           to visit.
         - If the passenger is boarded (check state), increment the boarded
           count and add their destination floor (looked up from initialization data)
           to the set of floors to visit.
    5. If the set of floors to visit is empty (which should only happen if all
       passengers are served, handled in step 1), the movement cost is 0.
    6. If there are floors to visit:
       - Get the integer indices for the current floor and all floors to visit
         that have a corresponding index.
       - If there are valid target floors with indices:
         - Find the minimum and maximum indices among the valid target floors.
         - Calculate the estimated movement cost: This is the minimum distance
           from the current floor index to either the minimum or maximum target
           floor index, plus the distance between the minimum and maximum target
           floor indices. This estimates the travel needed to reach the "action
           zone" and traverse it.
           Movement cost = min(abs(current_index - min_target_index),
                               abs(current_index - max_target_index)) +
                           (max_target_index - min_target_index)
       - If there are no valid target floors with indices (e.g., all relevant
         floors are outside the defined 'above' chain), movement cost is 0.
    7. Calculate the total heuristic value:
       Heuristic = 2 * (waiting_count) + 1 * (boarded_count) + movement_cost.
       Each waiting passenger needs a board and a depart action (2 actions).
       Each boarded passenger needs depart (1 action). Movement cost
       is added as an estimate of travel actions.
    """

    def __init__(self, task):
        self.task = task
        self.floor_to_index = {}
        self.passenger_to_destin = {}
        self.all_passengers = set()

        # 1. Build floor_to_index mapping
        above_map = {} # {floor_below: floor_above}
        all_floors = set()

        for fact_str in task.static:
            parts = parse_fact(fact_str)
            if not parts: continue # Skip invalid facts
            if parts[0] == 'above' and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                above_map[f1] = f2
                all_floors.add(f1)
                all_floors.add(f2)
            elif parts[0] == 'destin' and len(parts) == 3:
                 p, f = parts[1], parts[2]
                 self.passenger_to_destin[p] = f
                 self.all_passengers.add(p)
            # Collect passengers from other static predicates if any (miconic only has destin)

        # Collect passengers from initial state and goal
        for fact_str in task.initial_state:
             parts = parse_fact(fact_str)
             if not parts: continue # Skip invalid facts
             if parts[0] in ('origin', 'boarded', 'served') and len(parts) > 1:
                 self.all_passengers.add(parts[1])

        for goal_fact_str in task.goals:
             parts = parse_fact(goal_fact_str)
             if not parts: continue # Skip invalid facts
             if parts[0] == 'served' and len(parts) > 1:
                 self.all_passengers.add(parts[1])


        # Find the lowest floor (a floor in all_floors that is not a value in above_map)
        floors_that_are_above_of_something = set(above_map.values())
        lowest_floor = None
        if all_floors:
            potential_lowest = all_floors - floors_that_are_above_of_something
            if len(potential_lowest) == 1:
                lowest_floor = list(potential_lowest)[0]
            elif len(all_floors) == 1: # Case with only one floor
                 lowest_floor = list(all_floors)[0]
            # else: Multiple potential lowest floors or none found. Assume linear chain structure holds.
            # If lowest_floor remains None, floor_to_index will be empty or incomplete.

        if lowest_floor:
            current_floor = lowest_floor
            index = 0
            while current_floor is not None:
                self.floor_to_index[current_floor] = index
                index += 1
                current_floor = above_map.get(current_floor) # Get the floor directly above


    def __call__(self, state):
        # 1. Check for goal state
        # Assuming task.goal_reached(state) is available and correct
        if self.task.goal_reached(state):
            return 0

        # 2. Find current lift floor
        current_floor = None
        # Pre-parse relevant facts from the state for faster lookup
        state_facts_by_pred = {}
        for fact_str in state:
             parts = parse_fact(fact_str)
             if not parts: continue # Skip invalid facts
             pred = parts[0]
             if pred not in state_facts_by_pred:
                 state_facts_by_pred[pred] = []
             state_facts_by_pred[pred].append(parts)

        lift_at_facts = state_facts_by_pred.get('lift-at', [])
        if lift_at_facts and len(lift_at_facts[0]) > 1:
             current_floor = lift_at_facts[0][1]

        # Should always find lift-at in a valid state, but handle defensively
        if current_floor is None:
             # This state is likely invalid or represents an unrecoverable error
             # If there are unserved passengers but no lift location, we can't estimate movement.
             # Return a value based only on passenger counts?
             # Let's count unserved passengers as a fallback.
             served_passengers = {f[1] for f in state_facts_by_pred.get('served', []) if len(f) > 1}
             waiting_count = 0
             boarded_count = 0
             boarded_passengers_in_state = {f[1] for f in state_facts_by_pred.get('boarded', []) if len(f) > 1}
             origin_passengers_in_state = {f[1]: f[2] for f in state_facts_by_pred.get('origin', []) if len(f) > 2}

             for p in self.all_passengers:
                 if p in served_passengers:
                     continue
                 if p in origin_passengers_in_state:
                     waiting_count += 1
                 elif p in boarded_passengers_in_state:
                     boarded_count += 1
             return 2 * waiting_count + 1 * boarded_count # Fallback heuristic

        # 3. Identify unserved passengers and required floors
        waiting_count = 0
        boarded_count = 0
        floors_with_waiting = set()
        floors_with_destinations = set()

        served_passengers = {f[1] for f in state_facts_by_pred.get('served', []) if len(f) > 1}
        boarded_passengers_in_state = {f[1] for f in state_facts_by_pred.get('boarded', []) if len(f) > 1}
        origin_passengers_in_state = {f[1]: f[2] for f in state_facts_by_pred.get('origin', []) if len(f) > 2}


        for p in self.all_passengers:
            if p in served_passengers:
                continue # Passenger is served

            # Passenger is not served
            if p in origin_passengers_in_state:
                 # Passenger is waiting at origin
                 waiting_count += 1
                 origin_f = origin_passengers_in_state[p]
                 floors_with_waiting.add(origin_f)
            elif p in boarded_passengers_in_state:
                 # Passenger is boarded
                 boarded_count += 1
                 # Destination floor is needed for boarded passengers
                 destin_f = self.passenger_to_destin.get(p)
                 if destin_f: # Ensure destination is known
                     floors_with_destinations.add(destin_f)
                 # else: Passenger destination unknown, cannot add to floors_with_destinations

        # 4. Combine target floors
        all_target_floors = floors_with_waiting | floors_with_destinations

        # 5. Calculate movement cost
        movement_cost = 0
        # Ensure current floor has a known index
        current_index = self.floor_to_index.get(current_floor)

        if all_target_floors and current_index is not None:
            # Ensure all target floors have a known index
            valid_target_floors = {f for f in all_target_floors if f in self.floor_to_index}

            if valid_target_floors:
                target_indices = {self.floor_to_index[f] for f in valid_target_floors}
                min_target_index = min(target_indices)
                max_target_index = max(target_indices)

                # Estimate movement cost to cover the range of target floors
                # Distance to nearest end of the range + the span of the range
                dist_to_min = abs(current_index - min_target_index)
                dist_to_max = abs(current_index - max_target_index)
                span = max_target_index - min_target_index

                movement_cost = min(dist_to_min, dist_to_max) + span
            # else: No valid target floors with known indices. Movement cost remains 0.

        # 6. Calculate total heuristic
        # Each waiting passenger needs board (1) + depart (1) = 2 actions
        # Each boarded passenger needs depart (1) = 1 action
        # Add estimated movement actions
        heuristic_value = 2 * waiting_count + 1 * boarded_count + movement_cost

        return heuristic_value
