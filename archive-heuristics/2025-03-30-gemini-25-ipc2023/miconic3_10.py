import collections
from fnmatch import fnmatch
# Assuming the Heuristic base class is available at this path
# If the environment uses a different structure, adjust the import path accordingly.
from heuristics.heuristic_base import Heuristic

# Helper functions (defined outside the class for clarity)
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string.
    Removes surrounding parentheses and splits by space.
    Returns an empty list if the fact format is unexpected.

    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    # Basic validation for typical PDDL fact format
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Log warning or return empty list based on expected robustness
        # print(f"Warning: Unexpected fact format: {fact}")
        return []
    return fact[1:-1].split()

def match(fact, *pattern):
    """Checks if a fact string matches a given pattern tuple.
    Uses fnmatch for wildcard ('*') matching in pattern elements.
    Requires the number of parts in the fact to match the pattern length.

    Example: match("(at ball1 rooma)", "at", "*", "rooma") -> True
             match("(lift-at f1)", "lift-at", "f?") -> True
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the pattern length
    if len(parts) != len(pattern):
        return False
    # Check each part against the corresponding pattern element using fnmatch
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    Estimates the remaining cost to reach the goal state (all passengers served)
    for use in a Greedy Best-First Search. The cost is estimated by summing:
    1. The number of 'board' and 'depart' actions required for passengers not yet served.
    2. An estimate of the lift movement ('up'/'down') actions needed, calculated
       based on the vertical distance (levels) between relevant floors.

    # Assumptions
    - Floors are arranged vertically, and the 'above' predicate defines direct adjacency
      between floors (e.g., (above f1 f2) means f1 is one level directly above f2).
    - The cost of each action (board, depart, up, down) is 1.
    - The heuristic aims for informativeness (guiding the search effectively) rather
      than admissibility (which is not required for Greedy Best-First Search).
    - The floor structure is connected and acyclic as defined by 'above' predicates.

    # Heuristic Initialization
    - Parses static facts from the task definition (`task.static`).
    - Stores the destination floor for each passenger in `self.destin`.
    - Parses static 'above' facts to build a representation of the floor structure.
    - Calculates a numerical level (height) for each floor based on the 'above'
      relationships, starting from level 0 for the lowest floor(s). This mapping
      (`self.floor_to_level`) allows calculating vertical distances. It includes
      fallbacks for potentially incomplete floor structure information.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Parse Current State:** In the `__call__` method, analyze the current `state` set:
       - Find the lift's current floor (`lift_f`) from the `(lift-at ?f)` fact.
       - Identify passengers waiting at their origin (`waiting_passengers` map: p -> origin_floor).
       - Identify passengers currently boarded on the lift (`boarded_passengers` set).
       - Identify passengers already served (`served_passengers` set).
    2. **Calculate Board/Depart Actions:**
       - Initialize `board_depart_cost = 0`.
       - Iterate through all passengers defined in `self.destin`.
       - For each passenger `p` that is not in `served_passengers`:
         - Increment a counter `num_unserved`.
         - If `p` is in `waiting_passengers`: Add 2 to `board_depart_cost` (1 'board' action + 1 'depart' action).
         - If `p` is in `boarded_passengers`: Add 1 to `board_depart_cost` (1 'depart' action).
    3. **Identify Target Floors:** Determine the set of all floors (`target_floors`) the lift must potentially visit to serve the remaining passengers:
       - Collect the origin floors of all passengers in `waiting_passengers`.
       - Collect the destination floors (from `self.destin`) of all passengers in `boarded_passengers`.
       - Collect the destination floors (from `self.destin`) of all passengers in `waiting_passengers`.
    4. **Estimate Movement Actions:**
       - Get the numerical level of the lift's current floor (`lift_level`) using `self.floor_to_level`, with a fallback (e.g., level 0) if the floor level is unknown.
       - If `target_floors` is empty (which implies `num_unserved == 0`), set `movement_cost = 0`.
       - Otherwise:
         - Convert `target_floors` to a set of numerical levels (`target_levels`), using fallbacks for floors without pre-calculated levels.
         - If `target_levels` is not empty:
           - Find the minimum (`min_level`) and maximum (`max_level`) level among the targets.
           - Find the target level closest to the lift's current level (`nearest_target_level`).
           - Estimate movement cost as: `abs(lift_level - nearest_target_level) + (max_level - min_level)`.
             This estimates the travel distance to reach the nearest required floor plus the distance needed to cover the full vertical range of all required floors in one sweep.
         - If `target_levels` becomes empty (e.g., all target floors lacked levels), set `movement_cost = 0` as a fallback.
    5. **Total Heuristic Value:** The final heuristic value is the sum `board_depart_cost + movement_cost`. A safeguard ensures the heuristic returns 0 if and only if the state is a goal state (checked at the beginning and end). If the calculation yields 0 for a non-goal state, it returns 1 to ensure progress.
    """

    def __init__(self, task):
        """Initializes the heuristic by processing static information from the task."""
        self.goals = task.goals
        static_facts = task.static

        # 1. Parse static facts: destinations and floor structure
        self.destin = {} # passenger -> destination_floor
        # Adjacency maps based on 'above' predicate:
        adj = collections.defaultdict(set) # floor -> {floors directly below it, i.e., g where (above floor g)}
        rev_adj = collections.defaultdict(set) # floor -> {floors directly above it, i.e., g where (above g floor)}
        all_floors = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed or empty facts

            predicate = parts[0]
            if predicate == "destin" and len(parts) == 3:
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
                all_floors.add(floor) # Ensure destination floors are registered
            elif predicate == "above" and len(parts) == 3:
                f1, f2 = parts[1], parts[2] # f1 is above f2
                all_floors.add(f1)
                all_floors.add(f2)
                adj[f1].add(f2)
                rev_adj[f2].add(f1)

        # Consider floors mentioned in initial state origins if needed, although typically
        # floors are defined by objects or 'above' facts.

        # 2. Calculate floor levels using BFS starting from bottom floors
        self.floor_to_level = {}
        queue = collections.deque()

        # Identify bottom floors: those that are not above any other floor
        bottom_floors = {f for f in all_floors if f not in adj or not adj[f]}

        if not bottom_floors and all_floors:
             # Handle cases like a single floor or potentially disconnected structure
             if len(all_floors) == 1:
                 bottom_floors = all_floors # The single floor is the bottom floor
             else:
                 # Log a warning if structure is unclear (e.g., no floor is clearly lowest)
                 print(f"Warning: Could not determine bottom floor(s) unambiguously from 'above' facts for floors: {all_floors}. Level calculation might be incomplete.")
                 # As a fallback, could try finding top floors or assigning default levels later.

        processed_floors = set()
        for f in bottom_floors:
            if f not in self.floor_to_level: # Process each bottom floor once
                self.floor_to_level[f] = 0
                queue.append(f)
                processed_floors.add(f)

        # Perform BFS upwards to assign levels
        while queue:
            curr_f = queue.popleft()
            curr_level = self.floor_to_level[curr_f]

            # Find floors directly above curr_f (parents in the level hierarchy)
            for parent_f in rev_adj.get(curr_f, set()):
                if parent_f not in processed_floors:
                    self.floor_to_level[parent_f] = curr_level + 1
                    processed_floors.add(parent_f)
                    queue.append(parent_f)

        # Check for unprocessed floors and assign a fallback level (e.g., 0)
        if len(processed_floors) != len(all_floors):
             unprocessed = all_floors - processed_floors
             print(f"Warning: Not all floors received a level. Total: {len(all_floors)}, Processed: {len(processed_floors)}. Assigning fallback level 0 to unprocessed: {unprocessed}")
             for f in unprocessed:
                 if f not in self.floor_to_level: # Assign only if truly missing
                     self.floor_to_level[f] = 0 # Fallback level


    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state

        # Goal check: If goal is met, heuristic value is 0.
        is_goal = self.goals <= state
        if is_goal:
            return 0

        # 1. Get current state information
        lift_f = None
        waiting_passengers = {} # passenger -> origin_floor
        boarded_passengers = set() # set of passenger names
        served_passengers = set() # set of passenger names

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip potential empty parts from get_parts

            predicate = parts[0]
            # Using direct comparison for common predicates for efficiency,
            # assuming consistent formatting. Match could be used for more flexibility.
            if predicate == "lift-at" and len(parts) == 2:
                lift_f = parts[1]
            elif predicate == "origin" and len(parts) == 3:
                waiting_passengers[parts[1]] = parts[2]
            elif predicate == "boarded" and len(parts) == 2:
                boarded_passengers.add(parts[1])
            elif predicate == "served" and len(parts) == 2:
                served_passengers.add(parts[1])

        if lift_f is None:
            # This indicates an invalid state if it occurs during search
            print("Error: Lift location ('lift-at') not found in state. Returning infinity.")
            return float('inf') # Return infinity to avoid selecting this node

        # Get lift's current level, using fallback level 0 if floor is unknown
        lift_level = self.floor_to_level.get(lift_f, 0)
        if lift_f not in self.floor_to_level:
             print(f"Warning: Lift floor '{lift_f}' not found in level map. Using fallback level 0.")


        # 2. Calculate board/depart costs and identify target floors
        board_depart_cost = 0
        origin_floors = set()
        dest_floors_boarded = set()
        dest_floors_waiting = set()
        num_unserved = 0

        # Iterate through all passengers known to have destinations
        all_passengers = self.destin.keys()

        for p in all_passengers:
            if p not in served_passengers:
                num_unserved += 1
                dest_f = self.destin.get(p) # Destination should exist if p is in keys
                if dest_f is None:
                     # This case should not happen if self.destin is populated correctly
                     print(f"Internal Warning: Passenger {p} missing destination in self.destin. Skipping.")
                     continue

                if p in waiting_passengers:
                    # Passenger is waiting, needs board + depart
                    board_depart_cost += 2
                    origin_f = waiting_passengers[p]
                    origin_floors.add(origin_f)
                    dest_floors_waiting.add(dest_f)
                elif p in boarded_passengers:
                    # Passenger is boarded, needs depart
                    board_depart_cost += 1
                    dest_floors_boarded.add(dest_f)
                # else: Passenger is unserved but neither waiting nor boarded.
                # This implies an inconsistent state or a passenger status not captured
                # by origin/boarded/served. We ignore this case assuming valid states.


        # 3. Calculate movement cost estimate
        target_floors = origin_floors | dest_floors_boarded | dest_floors_waiting
        movement_cost = 0

        if target_floors:
            target_levels = set()
            valid_levels_found = False
            for f in target_floors:
                level = self.floor_to_level.get(f)
                if level is not None:
                    target_levels.add(level)
                    valid_levels_found = True
                else:
                    # Floor level unknown, use fallback level 0 and log warning
                    print(f"Warning: Target floor '{f}' not found in level map. Using fallback level 0 for movement calculation.")
                    target_levels.add(0) # Add fallback level

            if valid_levels_found or target_levels: # Proceed if we have any levels (even fallbacks)
                min_level = min(target_levels)
                max_level = max(target_levels)
                # Find the target level closest to the lift's current level
                nearest_target_level = min(target_levels, key=lambda level: abs(level - lift_level))
                # Estimate movement: cost to nearest target + cost to span the range
                movement_cost = abs(lift_level - nearest_target_level) + (max_level - min_level)
            # else: If target_levels remains empty (e.g., all target floors were unknown and no fallback added), movement_cost stays 0.

        # 4. Total Heuristic Value
        heuristic_value = board_depart_cost + movement_cost

        # Final safeguard: Ensure heuristic is non-zero for non-goal states.
        # If the calculation resulted in 0 but the state is not a goal, return 1.
        if heuristic_value == 0 and not is_goal:
             # This condition implies num_unserved > 0 but calculated cost is 0.
             # This might happen in edge cases (e.g., passenger waiting at destination, lift already there).
             # Returning 1 ensures the search makes progress.
             return 1

        return heuristic_value
