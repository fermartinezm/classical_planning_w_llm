from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions from examples
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
    # Ensure we have the same number of parts or the pattern uses wildcards appropriately
    if len(parts) != len(args) and '*' not in args:
         return False
    # Use zip to handle cases where parts might be longer than args (if args has *)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all
    goal conditions. It counts the necessary core actions (sample, take_image,
    calibrate, communicate) for each unachieved goal and adds a simplified
    cost for navigation and store management (drop), counting drop and
    calibration costs at most once if needed for any relevant goal.

    # Assumptions
    - The heuristic assumes solvable instances; it does not return infinity
      for potentially unreachable goals (e.g., no equipped rover, no sample at waypoint).
    - Navigation cost is simplified: a fixed cost (1) is added for each
      required move to a sample site, image viewpoint, calibration target,
      or communication point, if the corresponding data/calibration is not yet achieved for that specific goal.
    - Drop cost is simplified: a fixed cost (1) is added *once* if soil sampling is
      needed for *any* unachieved soil goal and *any* equipped soil rover has a full store.
      Similarly for rock sampling.
    - Calibration cost is simplified: a fixed cost (1) is added *once* if imaging
      is needed for *any* unachieved image goal and no camera supporting the required
      mode on an equipped imaging rover is currently calibrated.

    # Heuristic Initialization
    - Extracts static facts from the task, including:
        - Mapping of stores to rovers (`store_of`).
        - Mapping of cameras to rovers (`on_board`).
        - Mapping of cameras to supported modes (`supports`).
        - Mapping of cameras to calibration targets (`calibration_target`).
        - Mapping of objectives to visible waypoints (`visible_from`).
        - Lander location(s) (`at_lander`).
        - Visible waypoint pairs (`visible`) for communication points.
        - Equipped rovers for each task type (soil, rock, imaging).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. If the goal is reached, return 0.
    2. Initialize total estimated cost to 0.
    3. Identify rovers with full stores based on the current state.
    4. Determine if soil sampling is required for *any* unachieved soil goal (check if any uncommunicated soil goal `w` lacks `have_soil_analysis` for all equipped soil rovers).
    5. Determine if rock sampling is required for *any* unachieved rock goal (similar check).
    6. Determine if imaging is required for *any* unachieved image goal (check if any uncommunicated image goal `o, m` lacks `have_image` for all equipped imaging rovers).
    7. If soil sampling is required (step 4) AND any equipped soil rover has a full store (step 3), add 1 to cost (for a drop action).
    8. If rock sampling is required (step 5) AND any equipped rock rover has a full store (step 3), add 1 to cost (for a drop action).
    9. If imaging is required (step 6):
        - Check if calibration is needed for *any* mode `m` required by an unachieved image goal (i.e., no camera supporting mode `m` on an equipped imaging rover is currently calibrated).
        - If calibration is needed for *any* such mode: Add 1 for the `calibrate` action and 1 for navigation to a calibration target viewpoint.
    10. Iterate through each goal condition in the task:
        - If the goal is `(communicated_soil_data ?w)` and it's not in the current state:
            - Add 1 for the `communicate_soil_data` action.
            - Add 1 for navigation to a communication point.
            - Check if `(have_soil_analysis ?r ?w)` is true for any equipped soil rover `r`.
            - If NO: Add 1 for the `sample_soil` action and 1 for navigation to waypoint `w`.
        - If the goal is `(communicated_rock_data ?w)` and it's not in the current state:
            - Add 1 for the `communicate_rock_data` action.
            - Add 1 for navigation to a communication point.
            - Check if `(have_rock_analysis ?r ?w)` is true for any equipped rock rover `r`.
            - If NO: Add 1 for the `sample_rock` action and 1 for navigation to waypoint `w`.
        - If the goal is `(communicated_image_data ?o ?m)` and it's not in the current state:
            - Add 1 for the `communicate_image_data` action.
            - Add 1 for navigation to a communication point.
            - Check if `(have_image ?r ?o ?m)` is true for any equipped imaging rover `r`.
            - If NO:
                - Add 1 for the `take_image` action.
                - Add 1 for navigation to a viewpoint for objective `o`.
                # Calibration cost is handled globally before the loop
    11. Return the calculated total cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Precompute static mappings from task.static
        self.store_of = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, "store_of", "*", "*")}
        self.on_board = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, "on_board", "*", "*")} # camera -> rover
        self.supports = {} # camera -> {mode, ...}
        for fact in self.static:
             if match(fact, "supports", "*", "*"):
                  c, m = get_parts(fact)[1:3]
                  self.supports.setdefault(c, set()).add(m)
        self.calibration_target = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static if match(fact, "calibration_target", "*", "*")} # camera -> objective
        self.objective_viewpoints = {} # objective -> {waypoint, ...}
        for fact in self.static:
             if match(fact, "visible_from", "*", "*"):
                  o, w = get_parts(fact)[1:3]
                  self.objective_viewpoints.setdefault(o, set()).add(w)
        self.lander_waypoints = {get_parts(fact)[2] for fact in self.static if match(fact, "at_lander", "*", "*")}
        # We don't strictly need the full visible graph for this simplified navigation cost,
        # just the existence of communication points. We can assume they exist if landers exist.

        # Precompute equipped rovers (equipment is static)
        self.static_equipped_soil_rovers = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_soil_analysis", "*")}
        self.static_equipped_rock_rovers = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_rock_analysis", "*")}
        self.static_equipped_imaging_rovers = {get_parts(fact)[1] for fact in self.static if match(fact, "equipped_for_imaging", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        total_cost = 0

        # --- Check for global costs (drop, calibrate) ---

        # Identify full stores
        full_stores = {s for fact in state if match(fact, "full", s)}

        # Check if soil sampling is needed for ANY unachieved soil goal
        any_soil_sampling_needed = False
        for goal in self.goals:
             if match(goal, "communicated_soil_data", "*"):
                  w = get_parts(goal)[1]
                  if goal not in state: # Goal not communicated
                       # Check if soil data is already collected by any equipped rover
                       is_sampled = any(f"(have_soil_analysis {r} {w})" in state for r in self.static_equipped_soil_rovers)
                       if not is_sampled:
                            any_soil_sampling_needed = True
                            break # Found one goal needing sampling

        # Check if rock sampling is needed for ANY unachieved rock goal
        any_rock_sampling_needed = False
        for goal in self.goals:
             if match(goal, "communicated_rock_data", "*"):
                  w = get_parts(goal)[1]
                  if goal not in state: # Goal not communicated
                       # Check if rock data is already collected by any equipped rover
                       is_sampled = any(f"(have_rock_analysis {r} {w})" in state for r in self.static_equipped_rock_rovers)
                       if not is_sampled:
                            any_rock_sampling_needed = True
                            break # Found one goal needing sampling

        # Add drop costs if needed (once per sample type)
        if any_soil_sampling_needed:
             # Check if any equipped soil rover has a full store
             for r in self.static_equipped_soil_rovers:
                  store = self.store_of.get(r)
                  if store and store in full_stores:
                       total_cost += 1 # for one drop action before soil sampling
                       break # Added drop cost for soil, move on

        if any_rock_sampling_needed:
             # Check if any equipped rock rover has a full store
             for r in self.static_equipped_rock_rovers:
                  store = self.store_of.get(r)
                  if store and store in full_stores:
                       total_cost += 1 # for one drop action before rock sampling
                       break # Added drop cost for rock, move on

        # Check if imaging is needed for ANY unachieved image goal
        any_imaging_needed = False
        required_modes_for_imaging = set()
        for goal in self.goals:
             if match(goal, "communicated_image_data", "*", "*"):
                  o, m = get_parts(goal)[1:3]
                  if goal not in state: # Goal not communicated
                       # Check if image data is already collected by any equipped rover
                       is_imaged = any(f"(have_image {r} {o} {m})" in state for r in self.static_equipped_imaging_rovers)
                       if not is_imaged:
                            any_imaging_needed = True
                            required_modes_for_imaging.add(m)

        # Add calibration cost if needed (once + nav)
        if any_imaging_needed:
             calibration_needed_globally = False
             # Check if for ANY mode required by imaging goals, NO camera supporting it
             # on ANY equipped imaging rover is calibrated.
             for m in required_modes_for_imaging:
                  is_mode_calibrated_by_any_rover = False
                  for r in self.static_equipped_imaging_rovers:
                       # Find cameras on this rover that support mode m
                       cameras_on_rover_for_mode = {c for c in self.on_board if self.on_board[c] == r and m in self.supports.get(c, set())}
                       for c in cameras_on_rover_for_mode:
                            if f"(calibrated {c} {r})" in state:
                                 is_mode_calibrated_by_any_rover = True
                                 break # Found a calibrated camera for this mode on an equipped rover
                       if is_mode_calibrated_by_any_rover:
                            break # This mode is covered, check next mode

                  if not is_mode_calibrated_by_any_rover:
                       # Found a mode needed for imaging that is not calibrated by anyone
                       calibration_needed_globally = True
                       break # Calibration is needed globally

             if calibration_needed_globally:
                  total_cost += 1 # for calibrate action
                  total_cost += 1 # for navigate to calibration point


        # --- Iterate through goals and add costs for unachieved ones (per goal) ---
        for goal in self.goals:
            predicate, *args = get_parts(goal)

            if predicate == "communicated_soil_data":
                w = args[0]
                if goal not in state:
                    total_cost += 1 # communicate_soil_data
                    total_cost += 1 # navigate to comm point

                    # Check if soil data is already collected by any equipped rover
                    is_sampled = any(f"(have_soil_analysis {r} {w})" in state for r in self.static_equipped_soil_rovers)

                    if not is_sampled:
                        total_cost += 1 # sample_soil
                        total_cost += 1 # navigate to w

            elif predicate == "communicated_rock_data":
                w = args[0]
                if goal not in state:
                    total_cost += 1 # communicate_rock_data
                    total_cost += 1 # navigate to comm point

                    # Check if rock data is already collected by any equipped rover
                    is_sampled = any(f"(have_rock_analysis {r} {w})" in state for r in self.static_equipped_rock_rovers)

                    if not is_sampled:
                        total_cost += 1 # sample_rock
                        total_cost += 1 # navigate to w

            elif predicate == "communicated_image_data":
                o, m = args
                if goal not in state:
                    total_cost += 1 # communicate_image_data
                    total_cost += 1 # navigate to comm point

                    # Check if image data is already collected by any equipped rover
                    is_imaged = any(f"(have_image {r} {o} {m})" in state for r in self.static_equipped_imaging_rovers)

                    if not is_imaged:
                        total_cost += 1 # take_image
                        total_cost += 1 # navigate to viewpoint for o
                        # Calibration cost is handled globally before the loop

        return total_cost
