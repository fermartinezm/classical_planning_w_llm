import re
from fnmatch import fnmatch
from math import inf
# Assuming Heuristic base class is available from the planner's library
# If not, a simple placeholder like the one below would be needed.
# class Heuristic:
#     def __init__(self, task):
#         self.task = task
#     def __call__(self, node):
#         raise NotImplementedError
from heuristics.heuristic_base import Heuristic


def get_parts(fact_string):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(at obj loc)" -> ["at", "obj", "loc"]
    """
    return fact_string[1:-1].split()

def match_parts(fact_parts, *pattern):
    """
    Checks if fact parts (list of strings) match a pattern tuple.
    Allows '*' as a wildcard in the pattern using fnmatch.
    Example: match_parts(["at", "p1", "f1"], "at", "*", "f1") -> True
             match_parts(["at", "p1", "f1"], "at", "p2", "*") -> False
    """
    # Check if the number of parts matches the pattern length
    if len(fact_parts) != len(pattern):
        return False
    # Check if each part matches the corresponding pattern element (allowing wildcards)
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (Elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    (where all passengers are served) from the current state. It is designed for
    use with Greedy Best-First Search, prioritizing informativeness and efficient
    computation over admissibility. The estimate sums three components:
    1. The number of 'board' actions needed for passengers currently waiting at their origin.
    2. The number of 'depart' actions needed for all passengers not yet served (both waiting and boarded).
    3. An estimate of the lift's movement cost ('up'/'down' actions), calculated based on
       the distance from the lift's current position to the nearest required floor (origin or destination)
       plus the total vertical range (span) of all required floors.

    # Assumptions
    - Floors are named using the pattern 'f' followed by a number (e.g., 'f1', 'f10').
      This number directly represents the floor's level, with higher numbers indicating higher floors.
    - The cost of moving the lift one level up or down is 1.
    - The PDDL task description is consistent (e.g., passengers have destinations, floors exist).

    # Heuristic Initialization
    - The constructor (`__init__`) preprocesses information from the task definition:
        - It parses static facts to build a map `self.destinations` storing the destination floor for each passenger (`destin p f`).
        - It identifies all unique floor objects mentioned in the task's facts (initial, static, goal, general facts).
        - It extracts the level for each floor based on the numeric suffix in its name (e.g., 'f5' -> level 5) and stores this in `self.floor_levels`. It raises an error if floor names do not follow the expected 'f<number>' pattern, as this is critical for distance calculation.
        - It stores the set of all passengers (`self.passengers`) derived from the destination facts.
        - It defines helper methods `_get_level(f)` to retrieve a floor's level and `_dist(f1, f2)` to calculate the vertical distance between two floors based on their levels.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Receive the current planning `node` containing the `state`.
    2.  Identify all passengers already served by finding `(served p)` facts in the `state`.
    3.  Check if all passengers (`self.passengers`) are in the `served_passengers` set. If yes, the goal is reached, return 0.
    4.  Initialize the heuristic value `h = 0`.
    5.  Find the lift's current floor (`current_lift_f`) from the `(lift-at ?f)` fact. Raise an error if this fact is missing.
    6.  Identify waiting passengers (`P_wait`): Find `(origin p ?f)` facts for passengers `p` who are not yet served. Store their origin floors in a set `F_origins`.
    7.  Identify boarded passengers (`P_boarded`): Find `(boarded p)` facts for passengers `p` who are not yet served.
    8.  Ensure consistency: If a passenger is listed as both boarded and waiting at origin (which shouldn't happen in valid states), prioritize the 'boarded' status and remove them from the waiting list.
    9.  Determine the set of all unserved passengers (`P_unserved = P_wait | P_boarded`).
    10. Add boarding cost: Increment `h` by the number of waiting passengers (`len(P_wait)`), as each needs one 'board' action.
    11. Add departing cost: Increment `h` by the total number of unserved passengers (`len(P_unserved)`), as each needs one 'depart' action eventually.
    12. Estimate movement cost:
        a. If there are no unserved passengers (`num_unserved == 0`), skip movement cost calculation (goal already handled).
        b. Determine the set of destination floors (`F_destins`) for all `P_unserved` using the precomputed `self.destinations`. Raise an error if any destination is missing.
        c. Combine the origin floors (`F_origins`) and destination floors (`F_destins`) into the set of target floors (`F_targets`) that the lift must visit.
        d. If `F_targets` is empty, movement cost is 0.
        e. Otherwise (if `F_targets` is not empty):
            i.   Use `_get_level` and `_dist` to calculate distances. These methods will raise `ValueError` if a floor's level cannot be determined (e.g., due to bad name format).
            ii.  Find the minimum distance from the `current_lift_f` to any floor in `F_targets`. This is `dist_to_closest`.
            iii. Calculate the vertical range (span) of the target floors: `span = max_level(F_targets) - min_level(F_targets)`. If `F_targets` has only one floor, `span` is 0.
            iv.  Estimate the total movement cost as `movement_cost = dist_to_closest + span`.
            v.   If a `ValueError` occurs during level retrieval or distance calculation (e.g., floor name invalid), catch the error and return `float('inf')` for the heuristic value to signal that this state is problematic or the heuristic cannot be computed reliably.
        f. Add the calculated `movement_cost` to `h`.
    13. Return the final heuristic value `h`. This value is guaranteed to be non-negative and should be positive for any non-goal state.
    """

    def __init__(self, task):
        # Store the task object if needed by the base class or for later reference
        self.task = task
        self.goals = task.goals
        static_facts = task.static

        # 1. Store passenger destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Check if the fact matches the pattern (destin ?p - passenger ?f - floor)
            if match_parts(parts, "destin", "*", "*"):
                passenger, floor = parts[1], parts[2]
                self.destinations[passenger] = floor

        # 2. Determine floor levels from floor names (assuming format f<number>)
        self.floor_levels = {}
        floors = set()
        # Define predicates that involve floor objects
        relevant_predicates = {"lift-at", "origin", "destin", "above"}
        # Collect all known facts from the task definition
        all_fact_sources = [task.facts, task.initial_state, task.goals, static_facts]
        for fact_set in all_fact_sources:
            if fact_set: # Ensure the fact set exists and is not empty
                for fact in fact_set:
                    parts = get_parts(fact)
                    predicate = parts[0]
                    # Extract floor names based on predicate structure
                    if predicate in relevant_predicates:
                        if predicate == "lift-at" and len(parts) == 2: floors.add(parts[1])
                        elif predicate == "origin" and len(parts) == 3: floors.add(parts[2])
                        elif predicate == "destin" and len(parts) == 3: floors.add(parts[2])
                        elif predicate == "above" and len(parts) == 3:
                            floors.add(parts[1])
                            floors.add(parts[2])

        if not floors:
             print("Warning: No floor objects identified from task facts. Heuristic may fail if passengers/goals exist.")

        # Compile regex pattern to extract number from floor name 'f<number>'
        floor_pattern = re.compile(r"f(\d+)")
        floors_without_level = set()
        for floor in floors:
            match_obj = floor_pattern.match(floor)
            if match_obj:
                # Extract the number and convert to integer level
                level = int(match_obj.group(1))
                self.floor_levels[floor] = level
            else:
                 # Keep track of floors that don't match the pattern
                 floors_without_level.add(floor)

        # Report or raise error if some floors don't match the required naming convention
        if floors_without_level:
            error_message = (f"Critical Error: Cannot determine levels for floors: {floors_without_level}. "
                             f"Floor names must match 'f<number>' pattern for heuristic calculation.")
            # Raise an error because distance calculation is fundamental to the heuristic
            raise ValueError(error_message)

        # Raise error if floors exist but no levels could be determined at all
        if not self.floor_levels and floors:
             raise ValueError("Could not determine level for any floor based on 'f<number>' pattern, but floor objects exist.")

        # 3. Store the set of all passengers based on who has a destination
        self.passengers = set(self.destinations.keys())
        if not self.passengers:
            print("Warning: No passengers found with destinations defined in static facts. Goal might be trivial.")


    def _get_level(self, floor):
        """
        Retrieves the precomputed level of a floor.
        Raises ValueError if the level for the floor was not determined during initialization.
        """
        level = self.floor_levels.get(floor)
        if level is None:
            # This should only happen if a floor appears in the state but wasn't in the initial facts
            # or if initialization failed to parse its level (which should have raised an error earlier).
            raise ValueError(f"Level for floor '{floor}' not found. Check if it was defined and matched 'f<number>' pattern.")
        return level

    def _dist(self, f1, f2):
        """
        Calculates the vertical distance (number of 'up'/'down' moves) between two floors
        based on their absolute difference in levels.
        Raises ValueError if the level for either floor cannot be retrieved.
        """
        if f1 == f2:
            return 0
        # _get_level will raise ValueError if a level is missing, propagating the error
        level1 = self._get_level(f1)
        level2 = self._get_level(f2)
        return abs(level1 - level2)


    def __call__(self, node):
        """
        Calculates the heuristic value for the given state in the planning node.
        """
        state = node.state

        # --- State Parsing ---
        served_passengers = set()
        current_lift_f = None
        waiting_passengers = {} # Map: passenger -> origin_floor
        boarded_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "served" and len(parts) == 2:
                # Record served passengers, ensuring they are known passengers
                if parts[1] in self.passengers:
                    served_passengers.add(parts[1])
            elif predicate == "lift-at" and len(parts) == 2:
                current_lift_f = parts[1]
            elif predicate == "origin" and len(parts) == 3:
                passenger = parts[1]
                # Consider as waiting only if known and not already served
                if passenger in self.passengers and passenger not in served_passengers:
                     waiting_passengers[passenger] = parts[2]
            elif predicate == "boarded" and len(parts) == 2:
                 passenger = parts[1]
                 # Consider as boarded only if known and not already served
                 if passenger in self.passengers and passenger not in served_passengers:
                     boarded_passengers.add(passenger)

        # --- Goal Check ---
        if served_passengers == self.passengers:
            return 0 # Goal state reached

        # --- Sanity Checks ---
        if current_lift_f is None:
             # This indicates an invalid state if the search process allows it
             raise ValueError("Lift location ('lift-at') not found in the current state.")

        # Ensure consistency: remove passengers from waiting if they are boarded
        passengers_to_remove_from_waiting = {p for p in waiting_passengers if p in boarded_passengers}
        for p in passengers_to_remove_from_waiting:
            del waiting_passengers[p]

        # --- Heuristic Calculation ---
        h = 0 # Initialize heuristic value

        P_wait = set(waiting_passengers.keys())
        P_boarded = boarded_passengers
        P_unserved = P_wait | P_boarded
        num_unserved = len(P_unserved)

        # 1. Add cost for required 'board' actions
        h += len(P_wait)

        # 2. Add cost for required 'depart' actions
        h += num_unserved

        # 3. Estimate movement cost
        movement_cost = 0
        if num_unserved > 0: # Only needed if passengers remain to be served
            F_origins = set(waiting_passengers.values())
            F_destins = set()
            for p in P_unserved:
                dest = self.destinations.get(p)
                if dest is None:
                    # Should not happen if initialization was correct and task is valid
                    raise ValueError(f"Destination for unserved passenger '{p}' not found.")
                F_destins.add(dest)

            F_targets = F_origins | F_destins # Set of all floors lift needs to visit

            if F_targets:
                try:
                    # Ensure levels can be retrieved for all relevant floors before calculation
                    current_level = self._get_level(current_lift_f)
                    target_levels = [self._get_level(f) for f in F_targets]

                    # Calculate distance to the nearest target floor
                    # Use abs difference of levels directly for efficiency
                    min_dist_to_target = min(abs(current_level - lvl) for lvl in target_levels)

                    # Calculate the vertical span of target floors
                    if len(F_targets) == 1:
                         span = 0
                    else:
                         min_target_level = min(target_levels)
                         max_target_level = max(target_levels)
                         span = max_target_level - min_target_level

                    # Combine distance to nearest and span for movement estimate
                    movement_cost = min_dist_to_target + span

                except ValueError as e:
                     # Catch errors from _get_level (e.g., floor name invalid)
                     print(f"Heuristic calculation error for state: {e}. Returning infinity.")
                     # Return infinity to strongly discourage exploring this problematic state/branch
                     return float('inf')

        h += movement_cost

        # The logic ensures h > 0 for non-goal states (since num_unserved >= 1)
        return h

