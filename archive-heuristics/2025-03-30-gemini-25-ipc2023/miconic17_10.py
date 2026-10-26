import math
from fnmatch import fnmatch
# Assuming the planner environment provides this base class in the specified path
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact string (e.g., "(pred obj1 obj2)").
    Removes the surrounding parentheses and splits the string by spaces.

    Args:
        fact (str): The PDDL fact string.

    Returns:
        list[str]: A list of strings representing the predicate name and its arguments.
                   Returns an empty list if the fact format is invalid (e.g., not enclosed in parentheses).
    """
    if not fact or not (fact.startswith("(") and fact.endswith(")")):
        # Return empty list or raise error for invalid format
        return []
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern using fnmatch for wildcards.

    Args:
        fact (str): The complete fact as a string, e.g., "(lift-at f1)".
        *args: A variable number of strings representing the pattern components
               (predicate name, arguments). Wildcards like '*' can be used.

    Returns:
        bool: True if the fact matches the pattern (correct arity and matching components),
              False otherwise.
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the number of pattern components
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern component using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers.
    It counts the necessary board (1 action) and depart (1 action) actions for each
    passenger based on their current state (waiting at origin or boarded in the lift).
    It adds an estimate for the lift movement cost (up/down actions) based on the
    vertical range of floors the lift needs to visit and its current position
    relative to that range.

    # Assumptions
    - The `(above f1 f2)` predicate means floor `f1` is immediately above floor `f2`,
      defining a single, linear floor structure for all floors involved. Problems with
      disconnected floor groups or cycles might lead to inaccurate level computation
      and heuristic values.
    - The goal is to achieve `(served p)` for all relevant passengers `p`.
    - There is exactly one lift in the problem.
    - Floor names (e.g., f1, f2) do not necessarily correspond to their level/height.

    # Heuristic Initialization
    - Parses static facts to store the destination floor for each passenger in
      `self.destinations`.
    - Parses static `(above f1 f2)` facts to compute the numerical level (height)
      of each floor, assuming a linear structure starting from level 0 at the
      bottom-most floor. Stores this mapping in `self.levels`. Issues warnings if
      the structure seems invalid (disconnected, cycles, multiple bottoms).
    - Identifies the set of all passengers involved in the problem instance
      by looking at `origin`, `destin`, `boarded`, `served` predicates in the
      initial state and static facts. Stores this set in `self.passengers`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize heuristic value `h = 0`.
    2.  Check if the current state satisfies all goal conditions (`self.goals <= state`).
        If yes, return 0.
    3.  Identify the current floor of the lift `f_lift` by finding the fact
        `(lift-at ?f)`. If not found, return infinity (error state).
    4.  Check if floor levels were successfully computed (`self.levels` is populated).
        If not, return a fallback estimate (sum of actions per passenger state, ignoring movement).
    5.  Initialize sets to store required floors: `waiting_origin_floors`,
        `boarded_destinations`, `waiting_destinations`.
    6.  Parse the state to find which passengers are boarded (`boarded_passengers` set),
        which are waiting at their origin (`waiting_passengers_map`: p -> origin_floor),
        and which are already served (`served_passengers` set).
    7.  Iterate through all passengers `p` in `self.passengers`:
        a. If `p` is in `served_passengers`, skip this passenger.
        b. If `p` is in `boarded_passengers`:
           i. Increment `h` by 1 (cost of the `depart` action).
           ii. Add the passenger's destination floor (from `self.destinations`)
               to the `boarded_destinations` set. Handle missing destinations with a warning.
        c. If `p` is in `waiting_passengers_map` (i.e., waiting at origin):
           i. Increment `h` by 2 (cost of `board` + `depart` actions).
           ii. Add the passenger's origin floor (from `waiting_passengers_map`)
               to `waiting_origin_floors`.
           iii. Add the passenger's destination floor (from `self.destinations`)
               to `waiting_destinations`. Handle missing destinations with a warning.
        d. Handle any unexpected passenger states (e.g., unserved but not boarded
           and not at origin) with a warning and potentially a penalty to `h`.
    8.  Calculate the lift movement cost estimate (`move_cost`):
        a. Collect all floors the lift might need to visit:
           `all_required_floors_with_names = {f_lift} U waiting_origin_floors U boarded_destinations U waiting_destinations`.
        b. Filter this set to include only floors for which a level is known in `self.levels`
           (`valid_required_floors`).
        c. Check if `f_lift` itself has a known level. If not, return fallback estimate.
        d. If `valid_required_floors` contains 0 or 1 floor, `move_cost = 0` (unless the lift is not at the single required floor).
        e. If `valid_required_floors` contains only one floor (`target_floor`), and `f_lift != target_floor`,
           `move_cost` is the distance `abs(self.levels[f_lift] - self.levels[target_floor])`.
        f. If `valid_required_floors` contains multiple floors:
           i. Find the minimum (`min_req_level`) and maximum (`max_req_level`)
              floor levels among `valid_required_floors`.
           ii. Calculate the total vertical range: `range_span = max_req_level - min_req_level`.
           iii. Find the distance from the lift's current level (`lift_level`) to the
               nearest end of this required range:
               `dist_to_range_ends = min(abs(lift_level - min_req_level), abs(lift_level - max_req_level))`.
           iv. Estimate movement cost: `move_cost = range_span + dist_to_range_ends`.
        g. Handle potential errors during calculation (e.g., `KeyError` if a floor
           level is missing mid-calculation) by returning a fallback estimate.
    9.  Add the estimated `move_cost` to the total heuristic value `h`.
    10. Return the final heuristic value `h` as an integer.
    """

    def __init__(self, task):
        super().__init__(task)
        static_facts = task.static

        # 1. Parse destinations from static facts
        self.destinations = {}
        for fact in static_facts:
            # Use match helper for consistency
            if match(fact, "destin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    # passenger name is parts[1], destination floor is parts[2]
                    self.destinations[parts[1]] = parts[2]

        # 2. Compute floor levels from static 'above' facts
        self.levels = self._compute_floor_levels(static_facts)
        # Check if any 'above' facts existed but level computation failed
        if not self.levels and any(match(f, "above", "*", "*") for f in static_facts):
             print("Warning: Heuristic initialization failed to compute floor levels despite 'above' facts present.")

        # 3. Get all passengers involved in the problem
        self.passengers = set()
        # Combine initial state and static facts to find all passengers mentioned
        all_facts = task.initial_state | static_facts
        for fact in all_facts:
             parts = get_parts(fact)
             # Passengers appear as the first argument (index 1) in these predicates
             if len(parts) > 1 and parts[0] in ["origin", "destin", "boarded", "served"]:
                 # Check arity for safety, though PDDL structure is usually fixed
                 if (parts[0] in ["origin", "destin"] and len(parts) == 3) or \
                    (parts[0] in ["boarded", "served"] and len(parts) == 2):
                     self.passengers.add(parts[1])

    def _compute_floor_levels(self, static_facts):
        """
        Computes the level (height) of each floor based on 'above' facts.
        Assumes 'above f1 f2' means f1 is immediately above f2, forming a linear chain.
        Returns a dictionary mapping floor name to integer level (0 = bottom).
        Returns empty dict if structure is invalid or no 'above' facts exist.
        """
        floors = set()
        successors = {}  # floor -> floor immediately below
        predecessors = {} # floor -> floor immediately above

        for fact in static_facts:
            if match(fact, "above", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    f1, f2 = parts[1], parts[2]
                    floors.add(f1)
                    floors.add(f2)
                    # Check for conflicting definitions before assigning
                    if f1 in successors and successors[f1] != f2:
                         print(f"Warning: Conflicting 'above' definition for {f1}. Floor already above {successors[f1]}, cannot also be above {f2}.")
                         return {} # Inconsistent definition
                    if f2 in predecessors and predecessors[f2] != f1:
                         print(f"Warning: Conflicting 'above' definition for {f2}. Floor already below {predecessors[f2]}, cannot also be below {f1}.")
                         return {} # Inconsistent definition
                    successors[f1] = f2
                    predecessors[f2] = f1

        if not floors:
            # No 'above' facts found. Check if there's maybe only one floor total defined elsewhere?
            # If task involves floors but no 'above', levels are undefined. Return empty.
            return {}

        # Find the bottom floor (has no successor link originating from it, i.e., not in successors keys)
        possible_bottoms = floors - set(successors.keys())
        if len(possible_bottoms) == 1:
             bottom_floor = list(possible_bottoms)[0]
        elif len(possible_bottoms) == 0 and len(floors) == 1:
             # Handle single floor case explicitly if it wasn't caught by 'not floors'
             return {list(floors)[0]: 0}
        elif len(possible_bottoms) > 1:
             print(f"Warning: Multiple possible bottom floors found: {possible_bottoms}. Structure might be disconnected.")
             return {} # Cannot reliably compute levels for all floors
        else: # No possible bottoms found, but multiple floors exist? Cycle?
             print(f"Warning: Could not find a unique bottom floor among {floors}. Check for cycles.")
             return {} # Cannot compute levels reliably

        # Traverse up from the bottom floor to assign levels
        levels = {}
        curr_f = bottom_floor
        level = 0
        visited_count = 0
        # Safety break for unexpected cycles, slightly larger than num floors
        max_iterations = len(floors) + 2

        while curr_f is not None and max_iterations > 0:
            if curr_f in levels:
                 # Should not happen in a valid linear structure if cycle check above worked
                 print(f"Error: Cycle detected during level assignment involving {curr_f}.")
                 return {} # Cannot compute levels reliably
            levels[curr_f] = level
            visited_count += 1
            # Move up to the floor immediately above the current one
            curr_f = predecessors.get(curr_f)
            level += 1
            max_iterations -= 1

        if max_iterations <= 0:
             print("Error: Max iterations reached during level assignment, likely due to unexpected structure.")
             return {}

        if visited_count != len(floors):
            # This implies some floors defined in 'above' facts were not reached from the bottom floor.
            # This indicates a disconnected structure.
            print(f"Warning: Floor level computation covered {visited_count} of {len(floors)} floors defined by 'above'. Structure might be disconnected.")
            # Return potentially incomplete levels. The heuristic __call__ must handle missing keys.

        return levels

    def __call__(self, node):
        """
        Calculates the heuristic estimate for the given state node.
        """
        state = node.state

        # Goal check: If all goal facts are present in the state, heuristic is 0.
        if self.goals <= state:
            return 0

        h = 0
        lift_at_floor = None
        boarded_passengers = set()
        waiting_passengers_map = {} # passenger -> origin_floor
        served_passengers = set()

        # Parse current state for dynamic information
        for fact in state:
            if match(fact, "lift-at", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: lift_at_floor = parts[1]
            elif match(fact, "boarded", "*"):
                parts = get_parts(fact)
                if len(parts) == 2: boarded_passengers.add(parts[1])
            elif match(fact, "origin", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: waiting_passengers_map[parts[1]] = parts[2]
            elif match(fact, "served", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 2: served_passengers.add(parts[1])

        # Sanity check: lift must have a location
        if lift_at_floor is None:
            print("Error: Lift location predicate (lift-at) not found in state.")
            # Return a large value to indicate an error or unreachable state
            return float('inf')

        # --- Fallback if levels computation failed or is unavailable ---
        num_unserved = len(self.passengers - served_passengers)
        if not self.levels:
             # Calculate heuristic based only on passenger actions if levels are unavailable
             fallback_h = 0
             for p in self.passengers:
                 if p not in served_passengers:
                     if p in boarded_passengers: fallback_h += 1 # depart needed
                     # Assume waiting if not served and not boarded
                     elif p in waiting_passengers_map: fallback_h += 2 # board + depart needed
                     else: fallback_h += 2 # Assume waiting if state is unclear
             # print("Debug: Using fallback heuristic (no levels).")
             return fallback_h
        # --- End Fallback ---

        boarded_destinations = set()
        waiting_origin_floors = set()
        waiting_destinations = set()

        # Calculate base cost from passenger actions (board/depart)
        for p in self.passengers:
            if p in served_passengers:
                continue # Skip served passengers

            if p in boarded_passengers:
                h += 1 # Cost for 'depart' action
                dest = self.destinations.get(p)
                if dest:
                    boarded_destinations.add(dest)
                else:
                    # This indicates an issue with the problem definition or parsing
                    print(f"Warning: Destination unknown for boarded passenger {p}")

            elif p in waiting_passengers_map:
                h += 2 # Cost for 'board' + 'depart' actions
                origin_floor = waiting_passengers_map[p]
                waiting_origin_floors.add(origin_floor)
                dest = self.destinations.get(p)
                if dest:
                    waiting_destinations.add(dest)
                else:
                    print(f"Warning: Destination unknown for waiting passenger {p}")
            else:
                # Passenger 'p' is unserved, but not boarded and not found at origin.
                # This might indicate an intermediate state during planning search or an error.
                print(f"Warning: Passenger {p} in unexpected state (unserved but not boarded/waiting). Adding penalty.")
                # Add a penalty assuming both board and depart are needed.
                h += 3


        # Calculate movement cost estimate
        move_cost = 0
        try:
            # Check if the lift's current floor has a computed level
            if lift_at_floor not in self.levels:
                 print(f"Warning: Lift's current floor '{lift_at_floor}' has no level assigned. Cannot compute move cost accurately.")
                 # Return heuristic based only on passenger actions + crude move estimate
                 return h + num_unserved

            # Combine all floors the lift might need to visit (including current)
            all_required_floors_with_names = {lift_at_floor} | waiting_origin_floors | boarded_destinations | waiting_destinations

            # Filter this set to include only floors for which a level is known
            valid_required_floors = {f for f in all_required_floors_with_names if f in self.levels}

            if not valid_required_floors:
                 # This case should not happen if lift_at_floor is in self.levels
                 move_cost = 0
            elif len(valid_required_floors) == 1:
                 # Only one relevant floor with a known level.
                 # If this is just the lift's current floor, no movement needed relative to known targets.
                 # If it's a target floor different from lift's current floor, calculate distance.
                 target_floor = next(iter(valid_required_floors))
                 if lift_at_floor != target_floor:
                      # Lift needs to move to the single target
                      move_cost = abs(self.levels[lift_at_floor] - self.levels[target_floor])
                 else:
                      move_cost = 0 # Lift already at the only required known floor
            else:
                # Multiple floors need visiting among those with known levels
                lift_level = self.levels[lift_at_floor]
                # Get levels only for floors that are in self.levels
                req_levels = [self.levels[f] for f in valid_required_floors]
                min_req_level = min(req_levels)
                max_req_level = max(req_levels)

                # Calculate the span of the required vertical travel
                range_span = max_req_level - min_req_level

                # Calculate distance from current lift level to the nearest end of the required range
                dist_to_range_ends = min(abs(lift_level - min_req_level), abs(lift_level - max_req_level))

                # Estimate total movement cost as the range plus distance to nearest end
                move_cost = range_span + dist_to_range_ends

        except KeyError as e:
            # This might happen if a required floor (origin/dest) was not in 'above' facts
            # and thus not in self.levels, but was missed by the filtering step somehow.
            print(f"Warning: Floor '{e}' needed for travel not found in computed levels. Cannot calculate move cost accurately.")
            return h + num_unserved # Fallback move cost
        except Exception as e:
            # Catch any other unexpected errors during calculation
            print(f"Error during move cost calculation: {type(e).__name__} - {e}")
            return h + num_unserved # Fallback move cost

        # Add the estimated movement cost to the total heuristic value
        h += move_cost

        # Return the final heuristic value as an integer
        # Ensure non-negativity (should hold by construction, but safety check)
        final_h = max(0, int(round(h)))

        # Ensure heuristic is 0 iff goal state (initial check handles goal state -> 0)
        # If final_h is 0 but it's not a goal state, something is wrong.
        # This could happen if h=0 and move_cost=0.
        # h=0 means no unserved passengers, which implies goal state.
        # So, final_h=0 should only occur for goal states.

        return final_h
