import math
from fnmatch import fnmatch
# Assuming the heuristic base class is correctly located at heuristics.heuristic_base
# If the project structure is different, this import path might need adjustment.
from heuristics.heuristic_base import Heuristic


# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extract the components of a PDDL fact string (e.g., "(predicate obj1 obj2)").
    Removes parentheses and splits the string by spaces.
    Returns a list of strings (predicate name and arguments).
    Handles potential empty strings or malformed facts gracefully by returning an empty list.
    """
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed or empty facts
        return []
    # Remove parentheses and split by space
    return fact[1:-1].split()


# Helper function to match facts against patterns with wildcards
def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern using fnmatch for wildcard support.
    - `fact`: The complete fact as a string, e.g., "(lift-at f1)".
    - `args`: A tuple representing the expected pattern (e.g., ("lift-at", "*")).
              Wildcards `*` can be used to match any single element.
    - Returns `True` if the fact matches the pattern (same number of elements and each
      element matches the corresponding pattern element via fnmatch), `False` otherwise.
    """
    parts = get_parts(fact)
    # The number of elements in the fact must match the number of elements in the pattern
    if len(parts) != len(args):
         return False
    # Check if each part of the fact matches the corresponding part of the pattern
    # fnmatch allows for wildcard matching (e.g., '*' matches anything)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It calculates the sum of:
    1. The number of 'board' actions needed for waiting passengers.
    2. The number of 'depart' actions needed for waiting and boarded passengers.
    3. An estimate of the lift movement cost required to visit all necessary floors.

    The movement cost considers the distance to the nearest target floor (origin or
    destination) and the total vertical span of all target floors.

    # Assumptions
    - The PDDL predicate `(above f1 f2)` signifies that floor `f1` is directly
      one level above floor `f2`.
    - The floors form a single, linear sequence accessible by the elevator.
    - The cost of moving between adjacent floors (via `up` or `down` actions) is 1.

    # Heuristic Initialization (`__init__`)
    - Stores the goal conditions (`task.goals`).
    - Extracts passenger destinations from static `destin` facts and stores them
      in `self.destinations`.
    - Determines the numerical level (height) of each floor based on the static
      `above` facts. It assumes a linear structure, finds the bottom-most floor
      (level 0), and assigns levels incrementally upwards. It includes fallbacks
      and warnings for ambiguous or unexpected floor structures.
    - Creates a distance function `self.dist(f1, f2)` that calculates the
      absolute difference between the levels of two floors.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse Current State:** Get the current state's facts from the input `node`.
    2.  **Goal Check:** If the current state satisfies all goal conditions (`self.goals <= state`),
        the heuristic value is 0, indicating the goal is reached.
    3.  **Identify Lift Location:** Find the current floor of the lift from `(lift-at ?f)` facts.
        If the lift's location is unknown or the floor is invalid, return infinity.
    4.  **Identify Passenger Status:** Determine which passengers are waiting at their origin
        (`origin p f` facts) and which are currently inside the lift (`boarded p` facts).
    5.  **Count Board Actions:** The number of required `board` actions is equal to the number
        of passengers currently waiting (`origin`).
    6.  **Count Depart Actions:** The number of required `depart` actions is the sum of
        passengers currently waiting (`origin`) and passengers currently boarded (`boarded`),
        as all of them will eventually need to depart at their destination.
    7.  **Handle Terminal State:** If there are no waiting or boarded passengers, the state
        should be a goal state (caught in step 2). If it's somehow not, return infinity
        to signify an unexpected/invalid state.
    8.  **Estimate Movement Cost:**
        a.  Determine the set of `target_floors` the lift must visit. This set includes:
            - The origin floors of all waiting passengers.
            - The destination floors of all currently boarded passengers.
            - The destination floors of all waiting passengers (as they will eventually
              board and need transport to their destination).
        b.  Filter `target_floors` to include only valid floors for which levels were determined
            during initialization.
        c.  If there are no valid `target_floors`, the movement cost is 0.
        d.  Otherwise, calculate the movement cost estimate:
            i.  Find the distance from the lift's current floor to the *nearest* valid target floor.
            ii. Find the highest and lowest floor levels among all valid `target_floors`.
                Calculate the `span` as the difference between the maximum and minimum levels.
            iii.The estimated movement cost is `distance_to_nearest_target + span`. This heuristic
                approximates the cost to first reach the closest required floor and then cover the
                entire vertical range needed for service. Handle potential errors (e.g., infinite
                distance) by returning infinity.
    9.  **Total Heuristic Value:** Sum the counts for board actions, depart actions, and the
        estimated movement cost. If any component resulted in infinity, the total is infinity.
        Return the final integer value.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class if it has its own init logic
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            # Use match helper for pattern matching with potential wildcards
            if match(fact, "destin", "*", "*"):
                 parts = get_parts(fact)
                 # Basic validation for expected structure (predicate, passenger, floor)
                 if len(parts) == 3:
                     self.destinations[parts[1]] = parts[2]

        # 2. Determine floor levels and precompute distance function
        self.floor_levels = {}
        adj = {} # Stores f_above -> f_below mapping from (above f_above f_below) facts
        floors = set() # Collect all unique floor names encountered

        # Parse static facts to build floor structure and collect floor names
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "above" and len(parts) == 3:
                f_above, f_below = parts[1], parts[2]
                adj[f_above] = f_below
                floors.add(f_above)
                floors.add(f_below)
            elif predicate == "destin" and len(parts) == 3:
                # Collect floors mentioned in destinations
                floors.add(parts[2])

        # Also consider floors mentioned in the initial state for completeness
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == "lift-at" and len(parts) == 2:
                 floors.add(parts[1])
             elif predicate == "origin" and len(parts) == 3:
                 floors.add(parts[2])

        # Assign levels based on the 'above' relationships
        if not floors:
            print("Warning: No floors found in the problem definition.")
        elif not adj: # Floors exist, but no 'above' relations (e.g., single floor building)
            if len(floors) == 1:
                 # Single floor, level 0
                 self.floor_levels[list(floors)[0]] = 0
            else:
                 # Multiple floors but no ordering? This is unusual for this domain.
                 # Assign arbitrary levels based on sorted names as a fallback.
                 print(f"Warning: Multiple floors ({len(floors)}) defined but no 'above' relations found. "
                       "Assigning arbitrary levels based on name sort.")
                 for i, f in enumerate(sorted(list(floors))):
                     self.floor_levels[f] = i
        else:
            # Try to determine the unique bottom floor (appears as f_below, never as f_above)
            bottom_floors = set(adj.values()) - set(adj.keys())

            if len(bottom_floors) == 1:
                # Found a unique bottom floor, proceed with level assignment upwards
                curr_floor = list(bottom_floors)[0]
                # Create reverse adjacency map (below -> above) for easier upward traversal
                rev_adj = {v: k for k, v in adj.items()}
                q = [(curr_floor, 0)] # Queue for BFS: (floor, level)
                visited_levels = {} # Store assigned levels: floor -> level

                processed_count = 0 # Safety counter against infinite loops
                max_iterations = len(floors) * 2 # Heuristic limit

                while q:
                    processed_count += 1
                    if processed_count > max_iterations:
                        print("Error: Exceeded maximum iterations during floor level assignment. "
                              "Possible cycle or unexpected structure in 'above' facts.")
                        # Fallback to arbitrary levels if structure is broken
                        self.floor_levels = {f: i for i, f in enumerate(sorted(list(floors)))}
                        break

                    f, lvl = q.pop(0)
                    if f in visited_levels: continue # Already processed this floor
                    visited_levels[f] = lvl

                    # Find the floor directly above the current floor 'f'
                    if f in rev_adj:
                        f_above = rev_adj[f]
                        if f_above not in visited_levels:
                             q.append((f_above, lvl + 1)) # Add the floor above to the queue

                if not self.floor_levels: # Check if fallback occurred
                    self.floor_levels = visited_levels

                # Verify if all floors involved in 'above' relationships were processed
                connected_floors = set(adj.keys()) | set(adj.values())
                if set(self.floor_levels.keys()) != connected_floors:
                     print(f"Warning: Floor structure might be disjoint or incomplete. "
                           f"Processed floors: {len(self.floor_levels)}, Expected connected: {len(connected_floors)}")

                # Assign level 0 to any floors found but not connected via 'above'
                for f in floors:
                    if f not in self.floor_levels:
                        print(f"Warning: Floor '{f}' seems unconnected by 'above' relations. Assigning level 0.")
                        self.floor_levels[f] = 0

            else: # Ambiguous structure (0 or >1 bottom floors)
                 print(f"Warning: Could not determine a unique bottom floor (found {len(bottom_floors)} candidates: {bottom_floors}). "
                       "Assigning arbitrary levels based on name sort.")
                 # Fallback to arbitrary levels based on sorted names
                 self.floor_levels = {f: i for i, f in enumerate(sorted(list(floors)))}

        # Define the distance function based on the computed floor levels
        def calculate_distance(f1, f2):
            """Calculates the distance (number of moves) between two floors."""
            level1 = self.floor_levels.get(f1)
            level2 = self.floor_levels.get(f2)
            # Handle cases where one or both floors might not have a level assigned
            if level1 is None or level2 is None:
                print(f"Warning: Floor level lookup failed for '{f1}' or '{f2}'. Returning large distance.")
                # Use a large finite number instead of math.inf for potentially better search behavior
                return 1_000_000
            return abs(level1 - level2)

        self.dist = calculate_distance


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # 1. Goal Check: If the current state satisfies all goals, heuristic is 0.
        if self.goals <= state:
            return 0

        # 2. Find Lift Location
        lift_floor = None
        for fact in state:
            if match(fact, "lift-at", "*"):
                lift_floor = get_parts(fact)[1]
                break

        # If lift location is missing or invalid, return infinity (unreachable state)
        if lift_floor is None or lift_floor not in self.floor_levels:
             # Use math.inf for potentially unreachable states
             return math.inf

        # 3. Identify Passenger Status (Waiting and Boarded)
        waiting_passengers = {} # passenger -> origin_floor
        boarded_passengers = set() # set of passenger names

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "origin" and len(parts) == 3:
                waiting_passengers[parts[1]] = parts[2]
            elif predicate == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])

        # 4. Calculate Board and Depart Action Costs
        board_actions = len(waiting_passengers)
        # Depart actions are needed for everyone currently waiting or boarded
        depart_actions = len(waiting_passengers) + len(boarded_passengers)

        # 5. Handle Terminal State Check (if no actions seem needed but not goal)
        if board_actions == 0 and depart_actions == 0:
             # This implies no passengers are waiting or boarded.
             # If it's not the goal state (checked earlier), it's an anomaly.
             return math.inf # Indicate an unexpected non-goal terminal state

        # 6. Calculate Estimated Movement Cost
        move_cost = 0
        # Determine all floors the lift needs to visit
        pickup_floors = set(waiting_passengers.values())
        # Get destinations, handling potential missing entries in self.destinations gracefully
        current_dest_floors = {self.destinations.get(p) for p in boarded_passengers} - {None}
        future_dest_floors = {self.destinations.get(p) for p in waiting_passengers} - {None}

        target_floors = pickup_floors | current_dest_floors | future_dest_floors

        # Filter out any target floors that are invalid (don't have a level)
        valid_target_floors = {f for f in target_floors if f in self.floor_levels}

        if not valid_target_floors:
            # If no valid targets, no movement cost is associated with passengers
            move_cost = 0
        else:
            try:
                # Calculate distance from the lift's current floor to the nearest target floor
                distances_to_targets = [self.dist(lift_floor, f) for f in valid_target_floors]
                min_dist_to_target = min(distances_to_targets)

                # Calculate the vertical span covering all target floors
                target_levels = [self.floor_levels[f] for f in valid_target_floors]
                min_target_level = min(target_levels)
                max_target_level = max(target_levels)
                span_cost = max_target_level - min_target_level

                # Check if the nearest target is effectively unreachable (large distance)
                if min_dist_to_target >= 1_000_000:
                    move_cost = math.inf # Treat as unreachable
                else:
                    # Movement cost is distance to nearest + span of targets
                    move_cost = min_dist_to_target + span_cost

            except ValueError: # Likely min() called on empty sequence, should not happen if valid_target_floors is checked
                 move_cost = math.inf # Indicate an error state
                 print(f"Warning: ValueError during movement cost calculation. State: {state}, Targets: {valid_target_floors}")
            except Exception as e: # Catch any other unexpected errors
                 move_cost = math.inf
                 print(f"Unexpected error calculating movement cost: {e}. State: {state}")


        # 7. Total Heuristic Value: Sum of actions and movement cost
        # If movement cost is infinite, the total heuristic value is infinite
        if move_cost == math.inf:
             heuristic_value = math.inf
        else:
             # Sum the components: board actions + depart actions + movement cost
             heuristic_value = board_actions + depart_actions + move_cost

        # Return the final heuristic value. It should be non-negative or infinity.
        # Ensure integer value if finite, as it estimates action counts.
        return heuristic_value if heuristic_value == math.inf else int(round(heuristic_value))

