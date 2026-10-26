import re
from fnmatch import fnmatch
# Assuming Heuristic base class is available from the planner's infrastructure
# e.g., from heuristics.heuristic_base import Heuristic
# If not, define a placeholder base class for standalone testing:
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Using placeholder.")
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError

# Helper functions (can be defined globally or as static methods)
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string like '(pred a b)'."""
    # Removes parentheses and splits by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact string matches a pattern with optional wildcards ('*')."""
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Use fnmatch to allow wildcard matching for each part
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class MiconicHeuristic(Heuristic):
    """
    # Summary
    Estimates the number of actions required to serve all passengers in the miconic (elevator) domain.
    The heuristic counts the necessary board and depart actions for all unserved passengers
    and adds an estimate of the required lift movement based on the range of floors
    that need to be visited. It is designed for use with greedy best-first search and does
    not need to be admissible.

    # Assumptions
    - Floors are linearly ordered and named f1, f2, ..., fn, where f1 is the highest floor.
      This naming convention is strictly required for the heuristic to determine floor levels.
    - The `(above f_i f_j)` predicate means floor `f_i` is physically higher than floor `f_j` (implies i < j).
    - The `up` action moves the lift physically down (e.g., from f1 to f2, increasing index).
    - The `down` action moves the lift physically up (e.g., from f2 to f1, decreasing index).
    - The cost of each action (move, board, depart) is 1.
    - Floor levels are assigned based on their index: level(fk) = k. Lower level number (index) means higher physical floor. Distance between floors is the absolute difference in levels.

    # Heuristic Initialization (`__init__`)
    - Parses static facts to identify all floor objects.
    - Assigns a numerical level to each floor based on its name (e.g., f1 -> level 1, f2 -> level 2). This relies strictly on the 'fK' naming convention observed in examples. If floor names deviate, initialization will raise a RuntimeError.
    - Stores the destination floor for each passenger from static `(destin p f)` facts.
    - Compiles a set of all passenger names found in static facts, initial state, or goals.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Check if the current state `node.state` satisfies the goal conditions `self.goals`. If yes, return 0.
    2. Identify the current lift location (`f_lift`) and its corresponding level (`l_lift`) using the precomputed `floor_levels`. If the lift location or its level is unknown, return infinity.
    3. Identify waiting passengers (`P_waiting`, stored as `{passenger: origin_floor}`) and boarded passengers (`P_boarded`, stored as a set `{passenger}`) by iterating through the current state facts.
    4. Handle the edge case where the goal is not met, but no passengers are waiting or boarded. Return 1 as a minimal cost estimate.
    5. Calculate `base_cost`: This represents the minimum number of `board` and `depart` actions required.
       - Each waiting passenger needs one `board` and one `depart` action (cost = 2).
       - Each boarded passenger needs one `depart` action (cost = 1).
       - `base_cost = 2 * len(P_waiting) + len(P_boarded)`.
    6. Determine the set of floor levels the lift needs to visit (`L_targets`):
       - Start with the lift's current level (`l_lift`).
       - Add the origin floor level for every passenger in `P_waiting`.
       - Add the destination floor level for every passenger in `P_waiting`.
       - Add the destination floor level for every passenger in `P_boarded`.
       - If any required floor level cannot be determined (e.g., due to missing data), return infinity.
    7. Calculate `movement_cost`: This estimates the minimum number of `up`/`down` actions.
       - If `L_targets` contains only one level (or is empty), no movement is needed, `movement_cost = 0`.
       - Otherwise, find the minimum (`min_level`) and maximum (`max_level`) floor levels in `L_targets`.
       - Calculate the total vertical span the lift must cover: `span = max_level - min_level`.
       - Calculate the distance the lift needs to travel to reach the closest end of this span if it's currently outside the span: `dist_to_range = max(0, l_lift - max_level) + max(0, min_level - l_lift)`.
       - `movement_cost = span + dist_to_range`.
    8. Compute the total heuristic value: `h = base_cost + movement_cost`.
    9. Return the calculated heuristic value `h`, ensuring it's non-negative.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.
        Extracts floor levels, passenger destinations, and passenger names.
        Requires floors to be named 'fK' (e.g., f1, f2).
        """
        super().__init__(task) # Call base class constructor if necessary
        self.goals = task.goals
        static_facts = task.static

        # --- Floor Level Extraction ---
        self.floors = set()
        potential_floors = set()
        # Gather all potential floor names from relevant static facts
        for fact in static_facts:
            parts = get_parts(fact)
            # Predicates mentioning floors: above, origin, destin
            if parts[0] == 'above':
                if len(parts) > 2:
                    potential_floors.add(parts[1])
                    potential_floors.add(parts[2])
            elif parts[0] in ('origin', 'destin'):
                 if len(parts) > 2: potential_floors.add(parts[2]) # Floor is the last argument

        # Also check init/goal states for floors not mentioned in static facts
        # (e.g., lift-at in init)
        for fact in task.initial_state.union(task.goals):
             parts = get_parts(fact)
             # Check predicates that mention floors as the last argument
             if parts[0] in ('lift-at', 'origin', 'destin'):
                 if len(parts) > 1: potential_floors.add(parts[-1])

        self.floors = potential_floors
        self.floor_levels = {} # Map: floor_name -> level (int)

        if not self.floors:
            # If there are no floors, the problem might be trivial or invalid.
            print("Warning: No floors found in the problem definition.")
        else:
            # Attempt to assign levels based on 'fK' naming convention
            try:
                for f in self.floors:
                    # Use regex to extract the number K from floor name 'fK'
                    match_result = re.match(r'f(\d+)', f)
                    if not match_result:
                        # If any floor doesn't match, this heuristic cannot proceed
                        raise ValueError(f"Floor name '{f}' does not match required 'fK' pattern.")
                    # Assign level K to floor fK (lower K = higher physical floor)
                    self.floor_levels[f] = int(match_result.group(1))

                # Sanity check: Ensure all found floors were assigned a level
                if len(self.floor_levels) != len(self.floors):
                     # This might happen if a floor object exists but wasn't parsed correctly
                     raise ValueError("Internal error: Not all identified floors were assigned levels.")

            except ValueError as e:
                print(f"Error during heuristic initialization: {e}")
                # This heuristic critically depends on the 'fK' floor naming.
                # Raise an error to indicate incompatibility with the problem instance.
                raise RuntimeError("Miconic heuristic requires floors named 'fK' (e.g., f1, f2) to determine levels.") from e

        # --- Passenger Information Extraction ---
        self.destinations = {} # Map: passenger_name -> destination_floor_name
        self.passengers = set() # Set of all passenger names

        # Extract destinations and passengers from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'destin':
                # Ensure fact has correct structure (predicate, passenger, floor)
                if len(parts) > 2:
                    p, f = parts[1], parts[2]
                    self.destinations[p] = f
                    self.passengers.add(p)
            elif parts[0] == 'origin':
                 # Ensure fact has correct structure (predicate, passenger, floor)
                 if len(parts) > 2:
                     self.passengers.add(parts[1]) # Ensure passenger is recorded

        # Add passengers found in initial state or goals if not already known
        # This covers cases where passengers might start boarded or are only mentioned in goals.
        for fact in task.initial_state.union(task.goals):
             parts = get_parts(fact)
             # Check predicates involving passengers (usually the second element)
             if parts[0] in ('origin', 'boarded', 'served'):
                 if len(parts) > 1:
                     self.passengers.add(parts[1])
             elif parts[0] == 'destin': # Should be static, but handle defensively
                 if len(parts) > 2:
                     p, f = parts[1], parts[2]
                     self.passengers.add(p)
                     # Store destination if missing (e.g., if only defined in init)
                     if p not in self.destinations:
                         self.destinations[p] = f


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        Returns an estimate of the remaining actions to reach the goal.
        """
        state = node.state

        # 1. Check if goal is reached
        if self.goals <= state:
             return 0

        # 2. Find current lift location and level
        lift_at_fact = next((fact for fact in state if match(fact, "lift-at", "*")), None)
        if not lift_at_fact:
            # If lift location is unknown, cannot compute heuristic accurately.
            print("Warning: Lift location predicate 'lift-at' not found in state.")
            return float('inf')

        f_lift = get_parts(lift_at_fact)[1]
        if f_lift not in self.floor_levels:
             # This might happen if floor levels failed initialization or state is inconsistent.
             print(f"Warning: Lift is at floor '{f_lift}' which has no assigned level.")
             return float('inf')
        l_lift = self.floor_levels[f_lift]

        # 3. Identify waiting and boarded passengers
        waiting_passengers = {} # passenger -> origin_floor
        boarded_passengers = set() # passenger
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            # Check facts involving known passengers
            # Ensure the fact has at least a predicate and one argument (passenger)
            if len(parts) > 1 and parts[1] in self.passengers:
                p = parts[1]
                if predicate == 'origin':
                    # Ensure origin fact has floor argument
                    if len(parts) > 2: waiting_passengers[p] = parts[2]
                elif predicate == 'boarded':
                    boarded_passengers.add(p)
            # Note: 'served' passengers are implicitly handled by the goal check.

        num_waiting = len(waiting_passengers)
        num_boarded = len(boarded_passengers)

        # 4. Handle edge case: goal not met, but no passengers need service
        if num_waiting == 0 and num_boarded == 0:
             # This state implies the goal involves something other than passenger service,
             # or there's an inconsistency. Return 1 as a minimal cost estimate.
             return 1

        # 5. Calculate base_cost (board/depart actions)
        base_cost = 2 * num_waiting + num_boarded

        # 6. Determine target floor levels for movement
        target_levels = set()
        if not self.floor_levels:
             # Should have been caught by error in init or lift check, but safeguard here.
             print("Error: Floor levels not available for calculation.")
             return float('inf')

        target_levels.add(l_lift) # Always include current lift level

        # Add levels for waiting passengers (origin + destination)
        try:
            for p, f_origin in waiting_passengers.items():
                if f_origin not in self.floor_levels:
                    raise ValueError(f"Origin floor '{f_origin}' for passenger '{p}' has no level.")
                target_levels.add(self.floor_levels[f_origin])

                dest_floor = self.destinations.get(p)
                if not dest_floor:
                    raise ValueError(f"Destination not found for waiting passenger '{p}'.")
                if dest_floor not in self.floor_levels:
                    raise ValueError(f"Destination floor '{dest_floor}' for passenger '{p}' has no level.")
                target_levels.add(self.floor_levels[dest_floor])

            # Add levels for boarded passengers (destination)
            for p in boarded_passengers:
                dest_floor = self.destinations.get(p)
                if not dest_floor:
                    raise ValueError(f"Destination not found for boarded passenger '{p}'.")
                if dest_floor not in self.floor_levels:
                    raise ValueError(f"Destination floor '{dest_floor}' for passenger '{p}' has no level.")
                target_levels.add(self.floor_levels[dest_floor])
        except ValueError as e:
            # If any required floor level is missing, calculation fails.
            print(f"Warning: {e}")
            return float('inf')

        # 7. Calculate movement_cost
        movement_cost = 0
        # Movement is needed only if there are multiple distinct target levels
        if len(target_levels) > 1:
            min_level = min(target_levels)
            max_level = max(target_levels)

            span = max_level - min_level # Min distance covering the range
            dist_to_range = 0 # Extra distance if lift is outside the range
            if l_lift > max_level:
                dist_to_range = l_lift - max_level
            elif l_lift < min_level:
                dist_to_range = min_level - l_lift

            movement_cost = span + dist_to_range
        # If len(target_levels) <= 1, movement_cost remains 0.

        # 8. Compute total heuristic value
        h_value = base_cost + movement_cost

        # 9. Return the value (ensure non-negative)
        return max(0, h_value)
