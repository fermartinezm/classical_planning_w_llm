from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections # Used for BFS, though the simple heuristic doesn't use it yet

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    unmet goal conditions. It breaks down the goals into sub-tasks (sampling,
    imaging, communicating) and sums the estimated minimum actions for each,
    including necessary navigation, store management (drops), and camera
    calibration. It is non-admissible and designed for greedy best-first search.

    # Heuristic Initialization
    - Pre-processes static facts to quickly access information like rover
      equipment, store ownership, camera properties, objective visibility,
      waypoint connectivity, and lander location.
    - Stores the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic (for a given state)
    1. Initialize heuristic value `h = 0`.
    2. Identify all goal predicates that are NOT true in the current state.
    3. Categorize unachieved goals into:
       - Communicated soil data for waypoint W: `(communicated_soil_data W)`
       - Communicated rock data for waypoint W: `(communicated_rock_data W)`
       - Communicated image data for objective O in mode M: `(communicated_image_data O M)`
    4. For each unachieved goal:
       - **Communication:** Add 1 action for the `communicate_*_data` action itself. Add 1 action for navigation to a waypoint visible from the lander. (This is a simplification; multiple communications might use the same navigation).
       - **Sampling (Soil/Rock):** If the corresponding `(have_*_analysis R W)` predicate is NOT true for *any* rover R:
         - Add 1 action for the `sample_*` action.
         - Add 1 action for navigation to the sample waypoint W.
         - Track the total number of *needed* samples (soil + rock) across all unachieved goals.
       - **Imaging:** If the corresponding `(have_image R O M)` predicate is NOT true for *any* rover R:
         - Add 1 action for the `take_image` action.
         - Add 1 action for navigation to a waypoint P visible from objective O.
         - Add 1 action for the `calibrate` action (assuming calibration is needed before each image).
         - Add 1 action for navigation to a waypoint W visible from the camera's calibration target.
    5. **Store Management (Drops):** Estimate the number of `drop` actions needed. Count the total number of samples required (from step 4). Count the number of equipped rovers that currently have an empty store. If the number of needed samples exceeds the number of currently empty stores on equipped rovers, add the difference to `h` as an estimate of necessary drop actions. This is a rough estimate.
    6. Sum up all costs calculated in steps 4 and 5.
    7. Return `h`.

    # Simplifications and Assumptions:
    - Navigation cost is a fixed +1 per required move type (to sample, to image, to calibrate, to communicate), ignoring actual distance or path sharing.
    - Drop cost is estimated based on total samples needed vs initial empty stores, ignoring intermediate drops.
    - Calibration is assumed necessary before *each* `take_image` action for an unachieved image goal.
    - The heuristic assumes that necessary objects (equipped rovers, cameras, visible waypoints, etc.) exist to make the goals achievable. If a critical resource is missing based on static facts, the heuristic might return infinity.
    - It ignores potential conflicts or dependencies between actions beyond the immediate preconditions/effects counted.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Pre-processes static facts for efficient lookup.
        """
        self.goals = task.goals
        static_facts = task.static

        # Pre-process static facts into useful data structures
        self.store_of_rover = {}
        self.rover_of_store = {}
        for f in static_facts:
            if match(f, 'store_of', '*', '*'):
                parts = get_parts(f)
                store, rover = parts[1], parts[2]
                self.store_of_rover[rover] = store
                self.rover_of_store[store] = rover

        self.equipped_soil = {get_parts(f)[1] for f in static_facts if match(f, 'equipped_for_soil_analysis', '*')}
        self.equipped_rock = {get_parts(f)[1] for f in static_facts if match(f, 'equipped_for_rock_analysis', '*')}
        self.equipped_imaging = {get_parts(f)[1] for f in static_facts if match(f, 'equipped_for_imaging', '*')}

        self.on_board = {(get_parts(f)[1], get_parts(f)[2]) for f in static_facts if match(f, 'on_board', '*', '*')}
        self.supports = {(get_parts(f)[1], get_parts(f)[2]) for f in static_facts if match(f, 'supports', '*', '*')}
        self.cal_target = {get_parts(f)[1]: get_parts(f)[2] for f in static_facts if match(f, 'calibration_target', '*', '*')}

        self.obj_visible_from_wps = collections.defaultdict(set) # Map objective -> set of waypoints
        self.caltarget_visible_from_wps = collections.defaultdict(set) # Map cal target -> set of waypoints
        for f in static_facts:
             if match(f, 'visible_from', '*', '*'):
                 parts = get_parts(f)
                 obj, wp = parts[1], parts[2]
                 self.obj_visible_from_wps[obj].add(wp)
                 # Check if this objective is a calibration target
                 if obj in self.cal_target.values():
                     self.caltarget_visible_from_wps[obj].add(wp)


        self.lander_waypoint = next((get_parts(f)[2] for f in static_facts if match(f, 'at_lander', '*', '*')), None)
        self.lander_visible_waypoints = set()
        if self.lander_waypoint:
             self.lander_visible_waypoints = {get_parts(f)[1] for f in static_facts if match(f, 'visible', '*', self.lander_waypoint)}
             # Also add the lander waypoint itself if it's visible from anywhere (it usually is visible from itself)
             if any(match(f, 'visible', self.lander_waypoint, '*') for f in static_facts):
                  self.lander_visible_waypoints.add(self.lander_waypoint)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state
        h = 0

        # Helper to check if a 'have' predicate exists for a goal
        def has_sample(sample_type, waypoint, current_state):
             pred = f'have_{sample_type}_analysis'
             return any(match(f, pred, '*', waypoint) for f in current_state)

        def has_image(objective, mode, current_state):
             return any(match(f, 'have_image', '*', objective, mode) for f in current_state)

        # Helper to check if a rover has an empty store
        def has_empty_store(rover, current_state):
             store = self.store_of_rover.get(rover)
             # Check if the store exists for the rover and if the state contains the '(empty store)' fact
             return store is not None and f'(empty {store})' in current_state

        # --- Identify Unachieved Goals ---
        uncomm_soil_goals = set()
        uncomm_rock_goals = set()
        uncomm_image_goals = set() # Stores tuples (objective, mode)

        for goal_fact_str in self.goals:
            if goal_fact_str not in state:
                parts = get_parts(goal_fact_str)
                if not parts: continue # Skip malformed goals

                predicate = parts[0]
                if predicate == 'communicated_soil_data' and len(parts) == 2:
                    uncomm_soil_goals.add(parts[1]) # waypoint
                elif predicate == 'communicated_rock_data' and len(parts) == 2:
                    uncomm_rock_goals.add(parts[1]) # waypoint
                elif predicate == 'communicated_image_data' and len(parts) == 3:
                    uncomm_image_goals.add((parts[1], parts[2])) # (objective, mode)
                # Ignore other potential goal types not related to communication if any

        # If all communication goals are met, heuristic is 0
        if not uncomm_soil_goals and not uncomm_rock_goals and not uncomm_image_goals:
            return 0

        # --- Estimate Costs for Unachieved Goals ---

        needed_samples = 0 # Total count of sample actions needed

        # Soil Goals
        for w in uncomm_soil_goals:
            # Cost for communication
            h += 1 # communicate_soil_data action
            h += 1 # Navigation to lander view waypoint

            # Cost for sampling if needed
            if not has_sample('soil', w, state):
                needed_samples += 1
                # Check if sampling is possible at all (equipped rover exists)
                if not self.equipped_soil:
                     # This goal type is impossible if no rover is equipped
                     # In a real problem, this might indicate an unsolvable instance
                     # For heuristic, return infinity or a large value
                     return float('inf')
                # Navigation to sample location will be added later based on needed_samples

        # Rock Goals
        for w in uncomm_rock_goals:
            # Cost for communication
            h += 1 # communicate_rock_data action
            h += 1 # Navigation to lander view waypoint

            # Cost for sampling if needed
            if not has_sample('rock', w, state):
                needed_samples += 1
                # Check if sampling is possible at all (equipped rover exists)
                if not self.equipped_rock:
                     return float('inf')
                # Navigation to sample location will be added later based on needed_samples

        # Add costs related to sampling actions and navigation to sample locations
        h += needed_samples # sample_soil or sample_rock action
        h += needed_samples # Navigation to sample location

        # Estimate Drop Costs: Count how many samples are needed vs available empty stores
        # Only consider stores on rovers equipped for *any* sampling
        equipped_sampling_rovers = self.equipped_soil.union(self.equipped_rock)
        available_empty_stores = sum(1 for r in equipped_sampling_rovers if has_empty_store(r, state))

        # If more samples are needed than currently empty stores on equipped rovers,
        # estimate that a drop is needed for each additional sample.
        h += max(0, needed_samples - available_empty_stores)


        # Image Goals
        for o, m in uncomm_image_goals:
            # Cost for communication
            h += 1 # communicate_image_data action
            h += 1 # Navigation to lander view waypoint (could overlap with sample comms nav)

            # Cost for imaging if needed
            if not has_image(o, m, state):
                h += 1 # take_image action
                h += 1 # Navigation to image location (waypoint visible from objective o)

                # Find a suitable camera/rover for this objective and mode
                # We need a rover equipped for imaging, with a camera on board that supports the mode
                suitable_cameras_rovers = []
                for c, r in self.on_board:
                    if r in self.equipped_imaging and (c, m) in self.supports:
                        suitable_cameras_rovers.append((c, r))

                if not suitable_cameras_rovers:
                    # This image goal is impossible if no suitable camera/rover exists
                    return float('inf')

                # Add calibration cost for this specific image goal
                # Assumption: Each take_image requires a preceding calibrate action
                h += 1 # calibrate action

                # Find the calibration target for *a* suitable camera
                # We just need one suitable camera to estimate the calibration cost
                # Pick the first suitable camera found
                camera_for_cal = suitable_cameras_rovers[0][0]
                cal_t = self.cal_target.get(camera_for_cal)

                if cal_t is None:
                     # Camera has no calibration target, goal might be impossible?
                     # Or maybe some cameras don't need calibration? Domain says calibrate needs target.
                     # Assume this means the goal is impossible via this camera.
                     # If *all* suitable cameras lack targets, the goal is impossible.
                     # For simplicity, if *the chosen* camera lacks a target, return inf.
                     # A more robust heuristic would check all suitable cameras.
                     # Let's assume valid problems have cameras with targets if calibration is needed.
                     # If cal_t is None, it implies a problem setup issue or an impossible goal.
                     # For this heuristic, we'll proceed assuming a target exists for the chosen camera.
                     # If no target is found, it implies a problem structure we might not handle perfectly,
                     # but let's avoid returning inf unless strictly necessary based on static facts.
                     # A camera without a target cannot be calibrated according to the domain.
                     # If a goal requires an image from such a camera, and it's not pre-calibrated, it's impossible.
                     # We assume here that *some* suitable camera *can* be calibrated if needed.
                     pass # If cal_t is None, we can't add the nav cost, but don't necessarily return inf yet.

                # Navigation to calibration location
                # Need to reach a waypoint visible from the calibration target
                h += 1 # Navigation to calibration location


        return h

