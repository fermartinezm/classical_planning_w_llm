import os
import sys
from fnmatch import fnmatch

# Ensure the heuristics directory is in the Python path
# This might be needed if heuristic_base is in a parent directory structure
# Adjust the path as necessary based on your project structure
# try:
#     # Assuming the script is run from a directory where 'heuristics' is a subdirectory
#     # or accessible in the python path
#     from heuristics.heuristic_base import Heuristic
# except ImportError:
#     # If running the script directly, adjust the path to find heuristic_base
#     # This assumes the script is in 'domain_heuristics' and 'heuristic_base.py'
#     # is in the 'heuristics' directory at the same level as 'domain_heuristics'
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     sys.path.append(parent_dir)
#     from heuristics.heuristic_base import Heuristic

# If the above dynamic path adjustment is problematic, ensure Heuristic is available
# in your PYTHONPATH or place this file appropriately. For this example,
# we'll assume Heuristic is importable.
# Dummy Heuristic class if heuristic_base is not available:
class Heuristic:
    def __init__(self, task): pass
    def __call__(self, node): raise NotImplementedError

def get_parts(fact):
    """
    Extract the components of a PDDL fact string.
    Removes parentheses and splits the string by spaces.
    Example: "(at rover1 waypoint1)" -> ["at", "rover1", "waypoint1"]
    """
    return fact[1:-1].split()

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Rovers domain.

    # Summary
    This heuristic estimates the remaining cost to reach the goal state by summing
    the estimated minimum number of core actions (sampling, calibration, imaging,
    communication, dropping) required for each unsatisfied goal predicate. It ignores
    navigation costs and resource contention between rovers to ensure efficient computation.
    The heuristic is designed for Greedy Best-First Search and is not necessarily admissible.

    # Assumptions
    - Each goal predicate can be treated as an independent subproblem.
    - The cost of achieving a goal primarily depends on the core actions (sample, calibrate, take_image, communicate, drop), not navigation.
    - If data/image is already acquired by *any* rover, only communication is needed for that goal.
    - If data/image needs to be acquired, the minimum number of acquisition steps (sample/calibrate+take_image) plus communication is counted.
    - A 'drop' action cost is added only if sampling is required and *all* rovers capable of that sampling currently have full stores.
    - The problem instance is solvable (e.g., necessary equipment exists, locations are reachable).

    # Heuristic Initialization
    - The constructor (`__init__`) preprocesses the static information from the task definition.
    - It identifies rover equipment (soil, rock, imaging), camera capabilities (modes, calibration targets),
      rover stores, the lander location, and visibility information (objectives from waypoints).
    - This precomputation allows for faster lookups during heuristic evaluation.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Initialize heuristic value `h = 0`.
    2. Parse the current `state` to efficiently access dynamic information:
       - Which rovers have which soil/rock samples (`have_soil_analysis`, `have_rock_analysis`).
       - Which rovers have which images (`have_image`).
       - Which cameras are calibrated (`calibrated`).
       - Which rover stores are full (`full`).
    3. Iterate through each `goal_fact` defined in the task's goals:
    4. If `goal_fact` is already present in the current `state`, continue to the next goal (cost is 0).
    5. If `goal_fact` is `communicated_soil_data(wp)`:
       - Check if any rover has `have_soil_analysis(rover, wp)` in the state.
       - If YES: Add 1 to `h` (for the `communicate_soil_data` action).
       - If NO:
         - Base cost is 2 (for `sample_soil` + `communicate_soil_data`).
         - Check if *all* rovers equipped for soil analysis have their stores full. If yes, add 1 to the cost (for a potential `drop` action).
         - Add the calculated cost (2 or 3) to `h`.
    6. If `goal_fact` is `communicated_rock_data(wp)`:
       - Similar logic as for soil data, checking `have_rock_analysis` and rovers equipped for rock analysis. Cost is 1 (if data held) or 2/3 (if sampling needed + potential drop). Add cost to `h`.
    7. If `goal_fact` is `communicated_image_data(obj, mode)`:
       - Check if any rover has `have_image(rover, obj, mode)` in the state.
       - If YES: Add 1 to `h` (for `communicate_image_data`).
       - If NO:
         - Check if *any* capable camera (on an imaging-equipped rover, supporting the required mode) is currently `calibrated`.
         - If YES: Add 2 to `h` (for `take_image` + `communicate_image_data`).
         - If NO: Add 3 to `h` (for `calibrate` + `take_image` + `communicate_image_data`).
    8. Return the total accumulated heuristic value `h`. This value represents the estimated number of core actions remaining. It is 0 if and only if the state is a goal state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by preprocessing static information from the task.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Preprocessing Static Facts ---
        self.lander_location = None
        self.rover_stores = {} # rover -> store
        self.store_of_rover = {} # store -> rover
        self.rover_equipment = {} # rover -> {'soil': bool, 'rock': bool, 'imaging': bool}
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_supports = {} # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        # The following are not strictly needed for the action-counting heuristic
        # but might be useful for more complex heuristics (e.g., including navigation)
        # self.objective_visible_from = {} # objective -> set of waypoints
        # self.waypoint_visible = {} # waypoint -> set of visible waypoints

        all_rovers = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == 'at_lander':
                self.lander_location = parts[2]
            elif pred == 'store_of':
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
                self.store_of_rover[store] = rover
                all_rovers.add(rover)
            elif pred == 'equipped_for_soil_analysis':
                rover = parts[1]
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {}
                self.rover_equipment[rover]['soil'] = True
                all_rovers.add(rover)
            elif pred == 'equipped_for_rock_analysis':
                rover = parts[1]
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {}
                self.rover_equipment[rover]['rock'] = True
                all_rovers.add(rover)
            elif pred == 'equipped_for_imaging':
                rover = parts[1]
                if rover not in self.rover_equipment: self.rover_equipment[rover] = {}
                self.rover_equipment[rover]['imaging'] = True
                all_rovers.add(rover)
            elif pred == 'on_board':
                camera, rover = parts[1], parts[2]
                if rover not in self.rover_cameras: self.rover_cameras[rover] = set()
                self.rover_cameras[rover].add(camera)
                all_rovers.add(rover)
            elif pred == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_supports: self.camera_supports[camera] = set()
                self.camera_supports[camera].add(mode)
            elif pred == 'calibration_target':
                self.camera_calibration_target[parts[1]] = parts[2]
            # elif pred == 'visible_from':
            #     obj, wp = parts[1], parts[2]
            #     if obj not in self.objective_visible_from: self.objective_visible_from[obj] = set()
            #     self.objective_visible_from[obj].add(wp)
            # elif pred == 'visible':
            #     wp1, wp2 = parts[1], parts[2]
            #     if wp1 not in self.waypoint_visible: self.waypoint_visible[wp1] = set()
            #     self.waypoint_visible[wp1].add(wp2)

        # Initialize missing equipment flags for all known rovers
        for r in all_rovers:
            if r not in self.rover_equipment: self.rover_equipment[r] = {}
            self.rover_equipment[r].setdefault('soil', False)
            self.rover_equipment[r].setdefault('rock', False)
            self.rover_equipment[r].setdefault('imaging', False)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        """
        state = node.state
        h_value = 0

        # --- Parse current state for dynamic facts ---
        have_soil = {} # waypoint -> set of rovers that have the sample
        have_rock = {} # waypoint -> set of rovers that have the sample
        have_image = {} # (objective, mode) -> set of rovers that have the image
        calibrated_cameras = set() # set of (camera, rover) tuples
        store_is_full = set() # set of stores that are full

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]

            # No need to parse rover locations for this heuristic
            # if pred == 'at':
            #     rover_locations[parts[1]] = parts[2]
            if pred == 'have_soil_analysis':
                rover, wp = parts[1], parts[2]
                if wp not in have_soil: have_soil[wp] = set()
                have_soil[wp].add(rover)
            elif pred == 'have_rock_analysis':
                rover, wp = parts[1], parts[2]
                if wp not in have_rock: have_rock[wp] = set()
                have_rock[wp].add(rover)
            elif pred == 'have_image':
                rover, obj, mode = parts[1], parts[2], parts[3]
                key = (obj, mode)
                if key not in have_image: have_image[key] = set()
                have_image[key].add(rover)
            elif pred == 'calibrated':
                calibrated_cameras.add((parts[1], parts[2])) # (camera, rover)
            elif pred == 'full':
                store_is_full.add(parts[1]) # store name

        # --- Calculate heuristic based on unmet goals ---
        for goal_fact in self.goals:
            if goal_fact in state:
                continue # Goal already achieved

            parts = get_parts(goal_fact)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                wp = parts[1]
                if wp in have_soil and len(have_soil[wp]) > 0:
                    # Data already collected, needs 1 action (communicate)
                    h_value += 1
                else:
                    # Needs sampling (1) + communication (1) = 2 base actions
                    cost = 2
                    # Check if a drop (1) is needed because all capable rovers have full stores
                    capable_rovers = [r for r, equip in self.rover_equipment.items() if equip.get('soil', False)]
                    if not capable_rovers:
                        # No rover can achieve this goal - problem might be unsolvable
                        # Assign a large cost, though ideally planner handles unsolvability
                        return float('inf')

                    all_capable_stores_full = True
                    for r in capable_rovers:
                        store = self.rover_stores.get(r)
                        # If rover has no store or its store is not full, drop is not mandatory
                        if not store or store not in store_is_full:
                            all_capable_stores_full = False
                            break
                    if all_capable_stores_full:
                         cost += 1 # Add cost for drop
                    h_value += cost

            elif pred == 'communicated_rock_data':
                wp = parts[1]
                if wp in have_rock and len(have_rock[wp]) > 0:
                    # Data already collected, needs 1 action (communicate)
                    h_value += 1
                else:
                    # Needs sampling (1) + communication (1) = 2 base actions
                    cost = 2
                    # Check if a drop (1) is needed
                    capable_rovers = [r for r, equip in self.rover_equipment.items() if equip.get('rock', False)]
                    if not capable_rovers:
                        return float('inf') # Unsolvable

                    all_capable_stores_full = True
                    for r in capable_rovers:
                        store = self.rover_stores.get(r)
                        if not store or store not in store_is_full:
                            all_capable_stores_full = False
                            break
                    if all_capable_stores_full:
                         cost += 1 # Add cost for drop
                    h_value += cost

            elif pred == 'communicated_image_data':
                obj, mode = parts[1], parts[2]
                key = (obj, mode)
                if key in have_image and len(have_image[key]) > 0:
                    # Image already taken, needs 1 action (communicate)
                    h_value += 1
                else:
                    # Needs take_image (1) + communicate (1) = 2 base actions minimum
                    # Check if calibration (1) is also needed
                    is_any_capable_camera_calibrated = False
                    found_capable_rover = False
                    # Find rovers equipped for imaging
                    imaging_rovers = [r for r, equip in self.rover_equipment.items() if equip.get('imaging', False)]
                    for r in imaging_rovers:
                        # Find cameras on this rover that support the mode
                        cameras_on_rover = self.rover_cameras.get(r, set())
                        for c in cameras_on_rover:
                            supported_modes = self.camera_supports.get(c, set())
                            if mode in supported_modes:
                                # Found a rover+camera capable of taking this image
                                found_capable_rover = True
                                # Is this specific camera calibrated?
                                if (c, r) in calibrated_cameras:
                                    is_any_capable_camera_calibrated = True
                                    break # Found a calibrated capable camera, minimum cost applies
                        if is_any_capable_camera_calibrated:
                            break # Exit outer loop too

                    if not found_capable_rover:
                         # No rover/camera combination can take this image
                         return float('inf') # Unsolvable

                    if is_any_capable_camera_calibrated:
                        # Needs take_image + communicate = 2 actions
                        h_value += 2
                    else:
                        # Needs calibrate + take_image + communicate = 3 actions
                        h_value += 3

        # The heuristic value h_value is 0 if and only if all goals are met,
        # because we only increment h_value for unmet goals, and the increment is >= 1.
        return h_value

