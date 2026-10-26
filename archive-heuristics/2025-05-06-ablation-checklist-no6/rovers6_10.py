from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Define helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Basic check for valid fact format
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
         # Handle unexpected input, though PDDL facts should be well-formed strings
         return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve missing goal conditions.
    It counts the necessary steps (sample/image, navigate, communicate, drop, calibrate)
    for each unachieved goal fact independently, without considering resource conflicts
    or optimal pathfinding. Navigation is estimated as a fixed cost (1) if the rover
    is not already at a suitable location for the next step.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static information
        self.lander_at = None
        self.equipped = {} # rover -> {soil, rock, imaging}
        self.store_of = {} # store -> rover
        self.visible = {} # waypoint -> {waypoint, ...} (visibility graph)
        self.camera_on_board = {} # camera -> rover
        self.camera_supports = {} # camera -> {mode, ...}
        self.camera_calib_target = {} # camera -> objective
        self.visible_from = {} # objective/target -> {waypoint, ...}
        self.soil_at_init = set() # {waypoint, ...} - Initial sample locations
        self.rock_at_init = set() # {waypoint, ...} - Initial sample locations

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at_lander':
                # (at_lander ?x - lander ?y - waypoint)
                # Assuming only one lander and its location is static
                if len(parts) == 3:
                    self.lander_at = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis ?r - rover)
                if len(parts) == 2:
                    rover = parts[1]
                    self.equipped.setdefault(rover, set()).add('soil')
            elif predicate == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis ?r - rover)
                if len(parts) == 2:
                    rover = parts[1]
                    self.equipped.setdefault(rover, set()).add('rock')
            elif predicate == 'equipped_for_imaging':
                # (equipped_for_imaging ?r - rover)
                if len(parts) == 2:
                    rover = parts[1]
                    self.equipped.setdefault(rover, set()).add('imaging')
            elif predicate == 'store_of':
                # (store_of ?s - store ?r - rover)
                if len(parts) == 3:
                    store, rover = parts[1], parts[2]
                    self.store_of[store] = rover
            elif predicate == 'visible':
                # (visible ?w - waypoint ?p - waypoint)
                if len(parts) == 3:
                    w1, w2 = parts[1], parts[2]
                    self.visible.setdefault(w1, set()).add(w2)
                    # Assuming visibility is symmetric based on domain example
                    self.visible.setdefault(w2, set()).add(w1)
            elif predicate == 'on_board':
                # (on_board ?i - camera ?r - rover)
                if len(parts) == 3:
                    camera, rover = parts[1], parts[2]
                    self.camera_on_board[camera] = rover
            elif predicate == 'supports':
                # (supports ?c - camera ?m - mode)
                if len(parts) == 3:
                    camera, mode = parts[1], parts[2]
                    self.camera_supports.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target':
                # (calibration_target ?i - camera ?t - objective)
                if len(parts) == 3:
                    camera, target = parts[1], parts[2]
                    self.camera_calib_target[camera] = target # Assuming one target per camera
            elif predicate == 'visible_from':
                # (visible_from ?o - objective ?w - waypoint)
                if len(parts) == 3:
                    obj, waypoint = parts[1], parts[2]
                    self.visible_from.setdefault(obj, set()).add(waypoint)
            elif predicate == 'at_soil_sample':
                 # (at_soil_sample ?w - waypoint) - These are initial sample locations
                 if len(parts) == 2:
                    self.soil_at_init.add(parts[1])
            elif predicate == 'at_rock_sample':
                 # (at_rock_sample ?w - waypoint) - These are initial sample locations
                 if len(parts) == 2:
                    self.rock_at_init.add(parts[1])
            # Ignore can_traverse for this simple heuristic's navigation cost

        # Parse goal facts to identify what needs to be communicated
        self.goal_soil = set()
        self.goal_rock = set()
        self.goal_image = set() # set of (objective, mode) tuples

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'communicated_soil_data':
                # (communicated_soil_data ?w - waypoint)
                if len(parts) == 2:
                    self.goal_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                # (communicated_rock_data ?w - waypoint)
                if len(parts) == 2:
                    self.goal_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                # (communicated_image_data ?o - objective ?m - mode)
                if len(parts) == 3:
                    self.goal_image.add((parts[1], parts[2]))

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach a goal.
        """
        state = node.state
        total_cost = 0

        # Data structures for state-dependent information
        rover_at = {} # rover -> waypoint
        store_status = {} # store -> status ('empty', 'full')
        have_sample = {} # rover -> set of (waypoint, type)
        have_image = {} # rover -> set of (objective, mode)
        calibrated_camera = set() # set of (camera, rover)
        communicated_soil = set() # {waypoint, ...}
        communicated_rock = set() # {waypoint, ...}
        communicated_image = set() # set of (objective, mode)
        # soil_at_state = set() # {waypoint, ...} # Not strictly needed for this heuristic logic
        # rock_at_state = set() # {waypoint, ...} # Not strictly needed for this heuristic logic

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'at':
                # (at ?x - rover ?y - waypoint)
                if len(parts) == 3:
                    rover_at[parts[1]] = parts[2]
            elif predicate == 'empty':
                # (empty ?s - store)
                if len(parts) == 2:
                    store_status[parts[1]] = 'empty'
            elif predicate == 'full':
                # (full ?s - store)
                if len(parts) == 2:
                    store_status[parts[1]] = 'full'
            elif predicate == 'have_soil_analysis':
                # (have_soil_analysis ?r - rover ?w - waypoint)
                if len(parts) == 3:
                    rover, waypoint = parts[1], parts[2]
                    have_sample.setdefault(rover, set()).add((waypoint, 'soil'))
            elif predicate == 'have_rock_analysis':
                # (have_rock_analysis ?r - rover ?w - waypoint)
                if len(parts) == 3:
                    rover, waypoint = parts[1], parts[2]
                    have_sample.setdefault(rover, set()).add((waypoint, 'rock'))
            elif predicate == 'calibrated':
                # (calibrated ?c - camera ?r - rover)
                if len(parts) == 3:
                    calibrated_camera.add((parts[1], parts[2]))
            elif predicate == 'have_image':
                # (have_image ?r - rover ?o - objective ?m - mode)
                if len(parts) == 4:
                    rover, obj, mode = parts[1], parts[2], parts[3]
                    have_image.setdefault(rover, set()).add((obj, mode))
            elif predicate == 'communicated_soil_data':
                # (communicated_soil_data ?w - waypoint)
                if len(parts) == 2:
                    communicated_soil.add(parts[1])
            elif predicate == 'communicated_rock_data':
                # (communicated_rock_data ?w - waypoint)
                if len(parts) == 2:
                    communicated_rock.add(parts[1])
            elif predicate == 'communicated_image_data':
                # (communicated_image_data ?o - objective ?m - mode)
                if len(parts) == 3:
                    communicated_image.add((parts[1], parts[2]))
            # Ignore at_soil_sample and at_rock_sample in state for this heuristic,
            # as the goal is about communicating data *from* the initial sample location.

        # Check communication point availability for any rover
        lander_waypoint = self.lander_at
        rover_at_comm_point = any(
            rover_at.get(r) is not None and # Ensure rover location is known
            rover_at[r] in self.visible and # Ensure rover location is in the visible graph keys
            lander_waypoint in self.visible.get(rover_at[r], set()) # Check visibility from rover to lander
            for r in rover_at
        )

        # Heuristic calculation for each missing goal
        for w in self.goal_soil:
            if w not in communicated_soil:
                cost_w = 0
                # Step 3: Communicate data
                cost_w += 1 # communicate_soil_data action
                if not rover_at_comm_point:
                     cost_w += 1 # navigate to comm point

                # Step 2: Have sample analysis
                # Check if *any* rover has the sample analysis for this waypoint
                sample_collected = any((w, 'soil') in samples for samples in have_sample.values()) if have_sample else False
                if not sample_collected:
                    # Step 1: Sample soil
                    cost_w += 1 # sample_soil action
                    # Need an equipped rover at the sample location with an empty store
                    # Find any rover equipped for soil
                    equipped_soil_rovers = {r for r, eq in self.equipped.items() if 'soil' in eq}
                    if equipped_soil_rovers: # Assume solvable, so at least one exists
                        # Is any equipped rover at the sample waypoint?
                        rover_at_w = any(rover_at.get(r) == w for r in equipped_soil_rovers)
                        if not rover_at_w:
                            cost_w += 1 # navigate to w

                        # Does any equipped rover have an empty store?
                        rover_with_empty_store = any(
                            store_status.get(s) == 'empty' and self.store_of.get(s) in equipped_soil_rovers
                            for s in self.store_of
                        )
                        if not rover_with_empty_store:
                            cost_w += 1 # drop action (assumes drop is needed and possible by an equipped rover)
                    # Note: Not checking if at_soil_sample is still true in state, assuming initial samples are the targets.

                total_cost += cost_w

        for w in self.goal_rock:
            if w not in communicated_rock:
                cost_w = 0
                # Step 3: Communicate data
                cost_w += 1 # communicate_rock_data action
                if not rover_at_comm_point:
                     cost_w += 1 # navigate to comm point

                # Step 2: Have sample analysis
                # Check if *any* rover has the sample analysis for this waypoint
                sample_collected = any((w, 'rock') in samples for samples in have_sample.values()) if have_sample else False
                if not sample_collected:
                    # Step 1: Sample rock
                    cost_w += 1 # sample_rock action
                    # Need an equipped rover at the sample location with an empty store
                    # Find any rover equipped for rock
                    equipped_rock_rovers = {r for r, eq in self.equipped.items() if 'rock' in eq}
                    if equipped_rock_rovers: # Assume solvable
                        # Is any equipped rover at the sample waypoint?
                        rover_at_w = any(rover_at.get(r) == w for r in equipped_rock_rovers)
                        if not rover_at_w:
                            cost_w += 1 # navigate to w

                        # Does any equipped rover have an empty store?
                        rover_with_empty_store = any(
                            store_status.get(s) == 'empty' and self.store_of.get(s) in equipped_rock_rovers
                            for s in self.store_of
                        )
                        if not rover_with_empty_store:
                            cost_w += 1 # drop action
                    # Note: Not checking if at_rock_sample is still true.

                total_cost += cost_w

        for (o, m) in self.goal_image:
            if (o, m) not in communicated_image:
                cost_om = 0
                # Step 3: Communicate data
                cost_om += 1 # communicate_image_data action
                if not rover_at_comm_point:
                     cost_om += 1 # navigate to comm point

                # Step 2: Have image
                # Check if *any* rover has the image
                image_taken = any((o_img, m_img) == (o, m) for images in have_image.values() for (o_img, m_img) in images) if have_image else False
                if not image_taken:
                    # Step 1: Take image
                    cost_om += 1 # take_image action
                    # Need equipped rover with camera supporting mode, at image point, with calibrated camera

                    # Find *a* suitable rover/camera combo that can take this image
                    suitable_rover_camera = None
                    for camera, rover in self.camera_on_board.items():
                        if 'imaging' in self.equipped.get(rover, set()) and m in self.camera_supports.get(camera, set()):
                            suitable_rover_camera = (rover, camera)
                            break # Found a suitable one, stop searching

                    if suitable_rover_camera: # Assume solvable
                        r_img, i_img = suitable_rover_camera

                        # Check if rover is at a waypoint visible from the objective
                        # Need to check if *any* waypoint visible from objective 'o' is the rover's current location
                        image_points = self.visible_from.get(o, set())
                        rover_at_image_point = rover_at.get(r_img) in image_points

                        if not rover_at_image_point:
                            cost_om += 1 # navigate to image point

                        # Check if camera is calibrated
                        camera_is_calibrated = (i_img, r_img) in calibrated_camera
                        if not camera_is_calibrated:
                            cost_om += 1 # calibrate action
                            # Need rover at calibration point visible from target
                            calib_target = self.camera_calib_target.get(i_img)
                            if calib_target: # Assume solvable
                                # Need to check if *any* waypoint visible from target 'calib_target' is the rover's current location
                                calib_points = self.visible_from.get(calib_target, set())
                                rover_at_calib_point = rover_at.get(r_img) in calib_points

                                if not rover_at_calib_point:
                                    cost_om += 1 # navigate to calib point
                            # Note: Not checking if camera is on board or rover is equipped for imaging again, assumed by suitable_rover_camera

                total_cost += cost_om

        return total_cost
