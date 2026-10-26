from fnmatch import fnmatch

# Helper functions
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
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

class roversHeuristic: # Assuming Heuristic base class is implicitly handled or not strictly required for the output format
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for each unachieved goal literal.
    It breaks down the cost for each goal type (soil, rock, image) into
    collection steps (sampling or imaging) and communication steps,
    adding simple prerequisite costs for store availability and camera calibration.

    # Assumptions
    - Each navigation action between any two waypoints costs 1.
    - Each non-navigation action (sample, drop, calibrate, take_image, communicate) costs 1.
    - A rover needs to navigate to a sample location to sample (cost 1).
    - A rover needs to navigate to a viewpoint to take an image (cost 1).
    - A rover needs to navigate to a calibration target location to calibrate (cost 1).
    - A rover needs to navigate to a waypoint visible from the lander to communicate (cost 1).
    - Sampling requires an empty store. If sampling is needed and no capable rover has an empty store, one drop action is needed (cost 1). This is a simplified model of store management.
    - Taking an image requires a calibrated camera. Calibration is consumed by the take_image action. Recalibration is needed for each subsequent image taken with the same camera. Calibration requires navigating to a calibration target and performing the calibrate action (total cost 2). This heuristic adds calibration cost for *each* image goal that needs collection.

    # Heuristic Initialization
    The heuristic pre-processes static facts from the task definition to build data structures
    that provide quick access to:
    - Rover capabilities (soil, rock, imaging).
    - Mapping from store names to rover names.
    - Camera information (which rover it's on, supported modes, calibration target).
    - Waypoints from which objectives are visible.
    - The lander's location.
    - Waypoints visible from the lander's location (communication points).
    - The set of all store names.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all goal literals that are not currently true in the state (`unmet_goals`).
    2. If there are no unmet goals, the heuristic value is 0.
    3. Initialize the total estimated cost to 0.
    4. Categorize the unmet goals:
       - `needs_comm`: Goals where the data (soil, rock, or image) is already collected (e.g., `have_soil_analysis`) but not yet communicated (`communicated_soil_data`).
       - `needs_collection`: Goals where the data is not yet collected.
    5. Add cost for communication actions: For each item in `needs_comm`, add 2 (estimated cost for navigate to communication point + communicate action).
    6. Add cost for collection actions:
       - For each soil or rock waypoint in `needs_collection`, add 2 (estimated cost for navigate to sample location + sample action).
       - For each image objective/mode pair in `needs_collection`, add 2 (estimated cost for navigate to viewpoint + take_image action).
    7. Add prerequisite cost for Store:
       - Check if any soil or rock sampling is required (`needs_collection` contains soil or rock goals).
       - Check if there is at least one empty store available on any rover capable of sampling.
       - If sampling is needed AND no empty store is available on a capable rover, add 1 (estimated cost for a drop action to free up a store).
    8. Add prerequisite cost for Calibration:
       - For each image objective/mode pair in `needs_collection`:
         - Find a suitable camera on a capable rover that supports the required mode (using pre-processed static info).
         - Add 2 (estimated cost for navigate to calibration target + calibrate action). This assumes recalibration is needed for each image.
    9. The total estimated cost is the sum of costs from steps 5, 6, 7, and 8.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Pre-process static facts
        self.rover_capabilities = {} # rover -> set of capabilities ('soil', 'rock', 'imaging')
        self.store_to_rover = {}     # store -> rover
        self.camera_info = {}        # camera -> { 'rover': r, 'supports': {modes}, 'cal_target': t }
        self.objective_viewpoints = {} # objective -> {waypoints}
        self.lander_location = None
        self.all_stores = set()

        # First pass to get basic info and lander location
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'equipped_for_soil_analysis':
                self.rover_capabilities.setdefault(parts[1], set()).add('soil')
            elif parts[0] == 'equipped_for_rock_analysis':
                self.rover_capabilities.setdefault(parts[1], set()).add('rock')
            elif parts[0] == 'equipped_for_imaging':
                self.rover_capabilities.setdefault(parts[1], set()).add('imaging')
            elif parts[0] == 'store_of':
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
                self.all_stores.add(store)
            elif parts[0] == 'on_board':
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['rover'] = rover
            elif parts[0] == 'supports':
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('supports', set()).add(mode)
            elif parts[0] == 'calibration_target':
                camera, target = parts[1], parts[2]
                self.camera_info.setdefault(camera, {})['cal_target'] = target
            elif parts[0] == 'visible_from':
                objective, waypoint = parts[1], parts[2]
                self.objective_viewpoints.setdefault(objective, set()).add(waypoint)
            elif parts[0] == 'at_lander':
                self.lander_location = parts[2]

        # Second pass to find communication waypoints based on lander location
        self.comm_waypoints = set()
        if self.lander_location:
             for fact in task.static:
                parts = get_parts(fact)
                if parts[0] == 'visible':
                    wp1, wp2 = parts[1], parts[2]
                    if wp1 == self.lander_location:
                        self.comm_waypoints.add(wp2)
                    if wp2 == self.lander_location:
                        self.comm_waypoints.add(wp1)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state to find collected data, store status, calibration status
        collected_soil = set() # waypoints
        collected_rock = set() # waypoints
        collected_image = set() # (objective, mode)
        camera_calibrated = set() # (camera, rover)
        stores_full = set() # store names
        stores_empty = set() # store names

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'have_soil_analysis':
                collected_soil.add(parts[2])
            elif parts[0] == 'have_rock_analysis':
                collected_rock.add(parts[2])
            elif parts[0] == 'have_image':
                collected_image.add((parts[2], parts[3]))
            elif parts[0] == 'calibrated':
                camera_calibrated.add((parts[1], parts[2]))
            elif parts[0] == 'full':
                stores_full.add(parts[1])
            elif parts[0] == 'empty':
                stores_empty.add(parts[1])

        # Identify unachieved goals
        unmet_goals = set(self.goals) - set(state)

        # If goal is reached, heuristic is 0
        if not unmet_goals:
            return 0

        # Categorize unmet goals based on whether data is collected
        needs_comm = set() # (type, item) e.g., ('soil', wp), ('rock', wp), ('image', obj, mode)
        needs_collection = set() # (type, item)

        for goal in unmet_goals:
            parts = get_parts(goal)
            if parts[0] == 'communicated_soil_data':
                waypoint = parts[1]
                if waypoint not in collected_soil:
                    needs_collection.add(('soil', waypoint))
                needs_comm.add(('soil', waypoint))
            elif parts[0] == 'communicated_rock_data':
                waypoint = parts[1]
                if waypoint not in collected_rock:
                    needs_collection.add(('rock', waypoint))
                needs_comm.add(('rock', waypoint))
            elif parts[0] == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                if (objective, mode) not in collected_image:
                    needs_collection.add(('image', objective, mode))
                needs_comm.add(('image', objective, mode))

        total_cost = 0

        # Cost for collection actions (navigate + sample/image)
        needed_soil_waypoints = {w for type, w in needs_collection if type == 'soil'}
        needed_rock_waypoints = {w for type, w in needs_collection if type == 'rock'}
        needed_images_tuples = {(o, m) for type, o, m in needs_collection if type == 'image'}

        # Each collection needs navigation to the location (1) + the action (1) = 2
        total_cost += 2 * (len(needed_soil_waypoints) + len(needed_rock_waypoints) + len(needed_images_tuples))

        # Cost for communication actions (navigate + communicate)
        # Each communication needs navigation to a comm point (1) + the action (1) = 2
        total_cost += 2 * len(needs_comm)

        # Prerequisite cost: Store
        # If any sampling is needed and there are no empty stores on capable rovers, add cost for dropping.
        sampling_needed = len(needed_soil_waypoints) > 0 or len(needed_rock_waypoints) > 0
        capable_sampling_rovers = {r for r, caps in self.rover_capabilities.items() if 'soil' in caps or 'rock' in caps}
        stores_on_sampling_rovers = {s for s, r in self.store_to_rover.items() if r in capable_sampling_rovers}
        any_empty_store_on_sampling_rover = any(s in stores_empty for s in stores_on_sampling_rovers)

        if sampling_needed and not any_empty_store_on_sampling_rover:
            # Need to free up a store on a capable rover. Add cost for one drop action.
            total_cost += 1

        # Prerequisite cost: Calibration
        # For each image goal needing collection, add calibration cost.
        # This assumes recalibration is needed for each image action.
        for obj, mode in needed_images_tuples:
            # Find a suitable camera on a capable rover (just need existence for cost estimate)
            found_camera = False
            for cam, info in self.camera_info.items():
                rover = info.get('rover')
                if rover and mode in info.get('supports', set()) and 'imaging' in self.rover_capabilities.get(rover, set()):
                    # Found a camera/rover that *could* take this image
                    total_cost += 2 # Cost for navigate to cal target (1) + calibrate (1)
                    found_camera = True
                    break # Count calibration cost once per image goal needing collection

            # Note: If no camera/rover can achieve this image goal, the problem might be unsolvable.
            # A simple heuristic might return infinity or a large number here, but for
            # typical solvable benchmarks, we assume capabilities exist.

        return total_cost
