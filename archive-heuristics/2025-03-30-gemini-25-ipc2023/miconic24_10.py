import math # Used for float('inf')

# Try to import the base class, provide a dummy if not found (e.g., for standalone testing)
try:
    # This assumes the heuristic is used within a framework where heuristic_base is available.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Using dummy base class.")
    # Define a dummy base class if the import fails
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): pass

# Helper function to parse PDDL facts: "(predicate obj1 obj2 ...)" -> ["predicate", "obj1", "obj2", ...]
def get_parts(fact_str):
    """Removes parentheses and splits a PDDL fact string into parts."""
    # Handles potential leading/trailing whitespace and removes parentheses
    return fact_str.strip()[1:-1].split()

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all passengers
    by summing the estimated costs for each unserved passenger individually. The cost
    for a passenger includes the 'board' and 'depart' actions, plus the estimated number
    of 'up'/'down' actions (lift movements) needed to reach their origin (if waiting)
    and then their destination.

    # Assumptions
    - The '(above f1 f2)' predicate means floor f1 is directly above floor f2, defining
      a single, linear elevator shaft.
    - Each 'up' or 'down' action moves the lift exactly one floor between adjacent floors.
    - The heuristic sums costs per passenger, ignoring potential optimizations from
      serving multiple passengers in one trip. This makes it non-admissible but aims
      for informative guidance in a greedy search.
    - The PDDL problem is well-formed (connected floors, defined origins/destinations for
      relevant passengers).

    # Heuristic Initialization
    - Parses static facts ('above', 'destin') and initial state facts ('origin', 'lift-at')
      to understand the problem setup.
    - Computes the 'level' (integer height) for each floor based on 'above' facts,
      starting with level 0 for the bottom-most floor. Stores this in `self.floor_levels`.
      Handles cases with single floors or potentially disconnected floors (assigning level -1).
    - Stores passenger destinations in `self.destin`.
    - Identifies all unique passengers mentioned in the problem in `self.all_passengers`.
    - Identifies passengers required to be served in the goal state in `self.goal_passengers`.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Goal Check:** If all passengers listed in the goal conjuncts (typically `(served p)`)
       are currently in the 'served' state, return 0.
    2. **State Parsing:** Extract the lift's current floor (`lift_floor`), the set of
       boarded passengers (`boarded`), a dictionary of waiting passengers and their
       origins (`waiting`), and the set of served passengers (`served`) from the input state.
    3. **Sanity Checks:**
       - Verify the lift's location is known. If not, return infinity (error state).
       - Check if floor levels were successfully computed. If not (and passengers exist),
         return a basic fallback estimate (e.g., 2 * number of unserved goal passengers).
       - Get the current level of the lift (`lift_level`). If the lift's floor or its level
         is unknown/invalid, return infinity.
    4. **Cost Calculation:** Initialize total cost `h = 0`. Iterate through all passengers
       (`self.all_passengers`) potentially needing service:
       a. **Skip Served:** If the passenger `p` is in the `served` set, continue.
       b. **Check Destination:** Get `p`'s destination floor (`dest_floor`) and level (`dest_level`).
          If the destination or its level is unknown or invalid (e.g., level -1), print an
          error, add a large penalty (1000) to `h`, and skip to the next passenger.
       c. **Boarded Passenger:** If `p` is in the `boarded` set:
          - Add 1 (for the future 'depart' action).
          - Add lift movement cost: `abs(lift_level - dest_level)`.
       d. **Waiting Passenger:** If `p` is in the `waiting` dictionary:
          - Add 1 (for 'board') + 1 (for 'depart').
          - Get the origin floor (`origin_floor`) and level (`origin_level`).
          - If the origin level is unknown or invalid, print error, add penalty, and skip.
          - Add lift movement L -> O: `abs(lift_level - origin_level)`.
          - Add lift movement O -> D: `abs(origin_level - dest_level)`.
       e. **Unexpected State:** If an unserved passenger `p` is neither boarded nor waiting
          (this indicates an unexpected state, possibly during plan execution if actions
          are modeled differently), log a warning and add a fallback cost estimate
          (e.g., 2 actions + move L -> D).
    5. **Return Value:** Return the total calculated cost `h`, ensuring it is non-negative.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # --- 1. Precompute floor levels ---
        self.floor_levels = {}
        above_pairs = []
        all_floors = set()

        # Extract floors and 'above' relationships from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0] if parts else None
            if predicate == 'above' and len(parts) == 3:
                f1, f2 = parts[1], parts[2]
                above_pairs.append((f1, f2))
                all_floors.update([f1, f2])
            elif predicate == 'destin' and len(parts) == 3:
                 all_floors.add(parts[2]) # Ensure destination floors are known

        # Extract floors from initial state facts
        for fact in initial_state:
             parts = get_parts(fact)
             predicate = parts[0] if parts else None
             if predicate == 'lift-at' and len(parts) == 2:
                 all_floors.add(parts[1])
             elif predicate == 'origin' and len(parts) == 3:
                 all_floors.add(parts[2]) # Ensure origin floors are known

        if not all_floors:
             print("Info: No floors found in the problem definition.")
        elif not above_pairs:
             # No 'above' facts: Assume single floor or disconnected floors
             if len(all_floors) == 1:
                 floor_name = list(all_floors)[0]
                 self.floor_levels[floor_name] = 0
                 print(f"Info: Only one floor found ('{floor_name}'). Setting level to 0.")
             elif len(all_floors) > 1:
                 print("Warning: Multiple floors exist but no 'above' facts found. Assigning level 0 to all.")
                 for f in all_floors:
                     self.floor_levels[f] = 0 # Ambiguous, heuristic may be inaccurate
        else:
            # Build adjacency maps assuming (above f1 f2) means f1 is directly above f2
            adj = {f2: f1 for f1, f2 in above_pairs} # floor -> floor directly above
            rev_adj = {f1: f2 for f1, f2 in above_pairs} # floor -> floor directly below

            # Find the bottom floor robustly
            floors_with_floor_below = set(rev_adj.keys())
            floors_with_floor_above = set(adj.keys())
            # Bottom floor should have a floor above it but no floor below it
            possible_bottoms = floors_with_floor_above - floors_with_floor_below

            bottom_floor = None
            if len(possible_bottoms) == 1:
                bottom_floor = list(possible_bottoms)[0]
            else:
                 # Fallback: Find floor not above any other floor (works if top floor isn't listed as 'above' anything)
                 floors_that_are_above_others = set(adj.values()) # Corrected: values are floors above
                 possible_bottoms_alt = all_floors - floors_that_are_above_others
                 if len(possible_bottoms_alt) == 1:
                     bottom_floor = list(possible_bottoms_alt)[0]
                 # Handle single floor case if it wasn't caught earlier
                 elif not possible_bottoms and not possible_bottoms_alt and len(all_floors) == 1:
                     bottom_floor = list(all_floors)[0]
                 else:
                     # If still ambiguous, raise error
                     raise ValueError(f"Cannot uniquely determine bottom floor. Check 'above' facts are consistent and form a line. "
                                      f"Candidates (method 1): {possible_bottoms}, "
                                      f"Candidates (method 2): {possible_bottoms_alt}.")

            # BFS from bottom floor to assign levels
            q = [(bottom_floor, 0)]
            visited = {bottom_floor}
            self.floor_levels[bottom_floor] = 0
            head = 0
            while head < len(q):
                curr_f, curr_level = q[head]; head += 1
                f_above = adj.get(curr_f) # Find floor directly above current
                if f_above:
                    if f_above not in all_floors:
                         print(f"Warning: Floor '{f_above}' (above '{curr_f}') seems to exist but wasn't initially listed.")
                         all_floors.add(f_above) # Track it anyway
                    if f_above not in visited:
                         visited.add(f_above)
                         self.floor_levels[f_above] = curr_level + 1
                         q.append((f_above, curr_level + 1))

            # Check for unreached floors (indicates disconnected graph or errors in 'above')
            unreached = all_floors - visited
            if unreached:
                 print(f"Warning: Some floors were not reached during level assignment: {unreached}. Assigning level -1.")
                 for f_unreached in unreached:
                     self.floor_levels[f_unreached] = -1 # Mark level as unknown

        # --- 2. Precompute passenger destinations and lists ---
        self.destin = {}
        self.all_passengers = set()
        self.goal_passengers = set()

        # Passengers from 'destin' (static)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'destin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.destin[p] = f
                self.all_passengers.add(p)

        # Ensure all passengers from initial state 'origin' are known
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'origin' and len(parts) == 3:
                 self.all_passengers.add(parts[1])

        # Identify passengers required by the goal and ensure they are tracked
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == 'served' and len(parts) == 2:
                p = parts[1]
                self.goal_passengers.add(p)
                self.all_passengers.add(p) # Make sure goal passengers are in the main set


    def __call__(self, node):
        state = node.state

        # --- 1. Extract current state information ---
        lift_floor = None
        boarded = set()
        waiting = {} # passenger -> origin_floor
        served = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0] if parts else None
            # Check length to avoid index errors on malformed facts
            if predicate == 'lift-at' and len(parts) == 2:
                lift_floor = parts[1]
            elif predicate == 'boarded' and len(parts) == 2:
                boarded.add(parts[1])
            elif predicate == 'origin' and len(parts) == 3:
                waiting[parts[1]] = parts[2]
            elif predicate == 'served' and len(parts) == 2:
                served.add(parts[1])

        # --- 2. Goal Check ---
        # Check if all passengers specifically mentioned in the goal are served
        if self.goal_passengers.issubset(served):
            # Simple check assumes goals are only (served p) predicates
            # A more robust check could verify all goal literals against the state
            return 0

        # --- 3. Sanity Checks & Initialization ---
        if lift_floor is None:
            print("Error: Lift location ('lift-at') not found in state.")
            return float('inf') # Cannot proceed without lift location

        # Check if floor levels are available, provide fallback if not
        if not self.floor_levels and self.all_passengers:
             print("Error: Floor levels unavailable. Cannot compute heuristic accurately.")
             # Fallback: Estimate based on number of unserved goal passengers
             num_unserved_goal = len(self.goal_passengers - served)
             return num_unserved_goal * 2 # Basic estimate: 2 actions per passenger

        try:
            lift_level = self.floor_levels[lift_floor]
            # Check if the level is valid (not -1, which indicates unreached/unknown)
            if lift_level == -1:
                 print(f"Error: Lift floor '{lift_floor}' has unknown level (-1). Cannot compute heuristic.")
                 return float('inf')
        except KeyError:
             # Lift is at a floor that wasn't in the levels dictionary
             print(f"Error: Lift floor '{lift_floor}' not found in precomputed levels.")
             return float('inf')

        # --- 4. Calculate heuristic cost ---
        h_cost = 0
        # Iterate through all passengers known in the problem
        for p in self.all_passengers:
            if p in served:
                continue # Skip passengers already served

            # Get passenger's destination floor and level
            dest_floor = self.destin.get(p)
            if dest_floor is None:
                # If passenger is required for goal but has no destination, it's an error
                if p in self.goal_passengers:
                     print(f"Error: Goal passenger {p} has no destination ('destin') defined.")
                     h_cost += 1000 # Add large penalty
                # Otherwise, skip passengers without a defined destination
                continue

            try:
                dest_level = self.floor_levels[dest_floor]
                if dest_level == -1:
                     print(f"Error: Destination floor '{dest_floor}' for passenger {p} has unknown level (-1).")
                     h_cost += 1000; continue # Penalize and skip passenger
            except KeyError:
                 print(f"Error: Destination floor '{dest_floor}' for passenger {p} not found in levels.")
                 h_cost += 1000; continue # Penalize and skip passenger

            # Calculate cost based on passenger's current state
            if p in boarded:
                # State: Boarded. Needs: depart + move lift L -> D
                h_cost += 1 # depart cost
                h_cost += abs(lift_level - dest_level) # movement cost L -> D
            elif p in waiting:
                # State: Waiting at origin. Needs: move L -> O, board, move O -> D, depart
                origin_floor = waiting[p]
                try:
                    origin_level = self.floor_levels[origin_floor]
                    if origin_level == -1:
                         print(f"Error: Origin floor '{origin_floor}' for passenger {p} has unknown level (-1).")
                         h_cost += 1000; continue # Penalize and skip
                except KeyError:
                     print(f"Error: Origin floor '{origin_floor}' for passenger {p} not found in levels.")
                     h_cost += 1000; continue # Penalize and skip

                h_cost += 1 # board cost
                h_cost += 1 # depart cost
                h_cost += abs(lift_level - origin_level) # movement cost L -> O
                h_cost += abs(origin_level - dest_level) # movement cost O -> D
            else:
                # State: Unserved, but not boarded and not waiting at origin.
                # This state should ideally not occur for an unserved passenger.
                print(f"Warning: Unserved passenger {p} in unexpected state (not boarded, not waiting).")
                # Fallback estimate: Assume needs pickup from current lift pos + dropoff.
                h_cost += 2 # board + depart estimate
                h_cost += abs(lift_level - dest_level) # move L -> D estimate

        # --- 5. Return Value ---
        # Ensure the heuristic value is non-negative
        return max(0, h_cost)

