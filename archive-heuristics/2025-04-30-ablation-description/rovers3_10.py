from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing up
    the estimated costs of achieving each unsatisfied goal condition. It counts
    necessary actions (communication, sampling, imaging, calibration, dropping)
    and adds a penalty for movement if no suitable rover is currently positioned
    favorably for the next major step towards achieving the remaining goals.

    # Assumptions
    - Each action (communicate, sample, image, calibrate, drop, move) has a cost of 1.
    - Movement between any two waypoints is possible with a cost of 1 (relaxed graph).
    - The heuristic does not consider resource constraints beyond the need for an empty store for sampling.
    - The heuristic assumes that if a sample exists initially at a waypoint, it remains there until sampled.
    - The heuristic assumes that if a camera is needed for an image goal, it exists on an imaging-equipped rover and supports the required mode.

    # Heuristic Initialization
    The constructor extracts static information from the task definition, including:
    - Which rovers are equipped for soil, rock, and imaging.
    - Store-to-rover and rover-to-store mappings.
    - Camera-to-rover and rover-to-camera mappings.
    - Camera-to-mode support mappings.
    - Camera-to-calibration target mappings.
    - Objective-to-image waypoint mappings.
    - Calibration target-to-waypoint mappings.
    - Lander-to-waypoint mappings.
    - Initial soil and rock sample locations.
    - Visible waypoint pairs to determine communication waypoints.
    - Sets of all objects by type (rovers, stores, cameras, etc.) based on their appearance in static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a given state is calculated as follows:

    1.  Initialize `cost = 0`.
    2.  Check if the current state is the goal state. If yes, return 0.
    3.  Identify all goal conditions that are not met in the current state.
    4.  For each unsatisfied goal condition (e.g., `(communicated_soil_data ?w)`):
        -   Add 1 to `cost` (representing the final communication action).
    5.  Identify which soil samples (`?w`), rock samples (`?w`), and images (`?o`, `?m`) are required by the unsatisfied goals but have not yet been acquired (`have_soil_analysis`, `have_rock_analysis`, `have_image` facts are missing for all rovers).
    6.  For each required but unacquired sample/image:
        -   Add 1 to `cost` (representing the sample or take_image action).
    7.  Identify which cameras are needed for the required but unacquired images and are not currently calibrated.
    8.  For each camera that needs calibration:
        -   Add 1 to `cost` (representing the calibrate action).
    9.  Check if any sampling is needed (soil or rock) and if any rover equipped for sampling has a full store. If both are true, add 1 to `cost` (representing a necessary drop action).
    10. Calculate movement costs (simplified): Add 1 to `cost` for each *type* of location visit that is needed if *no* suitable rover is currently positioned at *any* waypoint of that type. The types of locations considered are:
        -   Soil sample locations (if soil sampling is needed and samples exist).
        -   Rock sample locations (if rock sampling is needed and samples exist).
        -   Image locations (if imaging is needed).
        -   Calibration locations (if calibration is needed).
        -   Communication locations (if any communication is needed).
    11. Return the total `cost`. The cost is 0 if and only if the state is a goal state.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # --- Static Information Extraction ---
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_rover = {}
        self.rover_store = {}
        self.camera_rover = {}
        self.rover_camera = defaultdict(set)
        self.camera_modes = defaultdict(set)
        self.camera_cal_target = {}
        self.obj_image_waypoints = defaultdict(set)
        self.cal_target_waypoints = defaultdict(set)
        self.lander_waypoint = {}
        self.at_soil_sample_init = set()
        self.at_rock_sample_init = set()
        self.visible_map = defaultdict(set)

        # Sets to store all objects of each type found in static facts
        self.all_rovers = set()
        self.all_stores = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_waypoints = set()
        self.all_landers = set()

        # Parse static facts
        for fact in self.static:
            parts = self._get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "equipped_for_soil_analysis": self.equipped_soil.add(parts[1])
            elif pred == "equipped_for_rock_analysis": self.equipped_rock.add(parts[1])
            elif pred == "equipped_for_imaging": self.equipped_imaging.add(parts[1])
            elif pred == "store_of":
                if len(parts) > 2:
                    self.store_rover[parts[1]] = parts[2]
                    self.rover_store[parts[2]] = parts[1]
            elif pred == "on_board":
                if len(parts) > 2:
                    self.camera_rover[parts[1]] = parts[2]
                    self.rover_camera[parts[2]].add(parts[1])
            elif pred == "supports":
                if len(parts) > 2:
                    self.camera_modes[parts[1]].add(parts[2])
            elif pred == "calibration_target":
                if len(parts) > 2:
                    self.camera_cal_target[parts[1]] = parts[2]
            elif pred == "visible_from":
                if len(parts) > 2:
                    self.obj_image_waypoints[parts[1]].add(parts[2])
            elif pred == "visible":
                if len(parts) > 2:
                    self.visible_map[parts[1]].add(parts[2])
            elif pred == "at_lander":
                if len(parts) > 2:
                    self.lander_waypoint[parts[1]] = parts[2]
            elif pred == "at_soil_sample":
                if len(parts) > 1:
                    self.at_soil_sample_init.add(parts[1])
            elif pred == "at_rock_sample":
                if len(parts) > 1:
                    self.at_rock_sample_init.add(parts[1])

            # Populate object sets based on predicate arguments (best effort without type info)
            # This is a simplified approach assuming objects appear as arguments in relevant predicates
            if len(parts) > 1:
                if pred in ["equipped_for_soil_analysis", "equipped_for_rock_analysis", "equipped_for_imaging"]:
                    self.all_rovers.add(parts[1])
                elif pred in ["empty", "full"]:
                    self.all_stores.add(parts[1])
                elif pred in ["calibrated", "on_board", "supports", "calibration_target"]:
                    self.all_cameras.add(parts[1])
                elif pred in ["visible_from", "calibration_target", "have_image", "communicated_image_data"]:
                    if len(parts) > 2: self.all_objectives.add(parts[2])
                elif pred in ["supports", "have_image", "communicated_image_data"]:
                    if len(parts) > 3: self.all_modes.add(parts[3])
                elif pred == "at_lander":
                    self.all_landers.add(parts[1])
                    if len(parts) > 2: self.all_waypoints.add(parts[2])
                elif pred in ["at", "can_traverse", "have_rock_analysis", "have_soil_analysis"]:
                    self.all_rovers.add(parts[1])
                    if len(parts) > 2: self.all_waypoints.add(parts[2])
                    if len(parts) > 3: self.all_waypoints.add(parts[3])
                elif pred in ["communicated_soil_data", "communicated_rock_data", "at_soil_sample", "at_rock_sample"]:
                    if len(parts) > 1: self.all_waypoints.add(parts[1])
                elif pred == "visible":
                    if len(parts) > 2:
                        self.all_waypoints.add(parts[1])
                        self.all_waypoints.add(parts[2])
                elif pred == "store_of":
                    self.all_stores.add(parts[1])
                    if len(parts) > 2: self.all_rovers.add(parts[2])


        # Calculate communication waypoints: waypoints visible *from* any lander waypoint
        self.comm_waypoints = set()
        lander_wps = set(self.lander_waypoint.values())
        for fact in self.static:
            if self._match(fact, "visible", "*", "*"):
                wp1, wp2 = self._get_parts(fact)[1], self._get_parts(fact)[2]
                if wp2 in lander_wps:
                    self.comm_waypoints.add(wp1)


    def _get_parts(self, fact):
        """Extract the components of a PDDL fact."""
        if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
            return []
        return fact[1:-1].split()

    def _match(self, fact, *args):
        """
        Check if a PDDL fact matches a given pattern.
        """
        parts = self._get_parts(fact)
        if len(parts) != len(args):
            return False
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))


    def __call__(self, node):
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Dynamic Information Extraction ---
        rover_location = {}
        store_empty = set()
        store_full = set()
        have_soil = set()
        have_rock = set()
        have_image = set()
        calibrated_cameras = set()
        at_soil_sample_state = set()
        at_rock_sample_state = set()

        for fact in state:
            parts = self._get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at":
                 if len(parts) > 2: rover_location[parts[1]] = parts[2]
            elif pred == "empty":
                 if len(parts) > 1: store_empty.add(parts[1])
            elif pred == "full":
                 if len(parts) > 1: store_full.add(parts[1])
            elif pred == "have_soil_analysis":
                 if len(parts) > 2: have_soil.add((parts[1], parts[2]))
            elif pred == "have_rock_analysis":
                 if len(parts) > 2: have_rock.add((parts[1], parts[2]))
            elif pred == "have_image":
                 if len(parts) > 3: have_image.add((parts[1], parts[2], parts[3]))
            elif pred == "calibrated":
                 if len(parts) > 2: calibrated_cameras.add((parts[1], parts[2]))
            elif pred == "at_soil_sample":
                 if len(parts) > 1: at_soil_sample_state.add(parts[1])
            elif pred == "at_rock_sample":
                 if len(parts) > 1: at_rock_sample_state.add(parts[1])

        # --- Heuristic Calculation ---
        cost = 0

        uncomm_soil_goals = {w for goal in self.goals if self._match(goal, "communicated_soil_data", "*") and goal not in state}
        uncomm_rock_goals = {w for goal in self.goals if self._match(goal, "communicated_rock_data", "*") and goal not in state}
        uncomm_image_goals = {(self._get_parts(goal)[1], self._get_parts(goal)[2]) for goal in self.goals if self._match(goal, "communicated_image_data", "*", "*") and goal not in state}

        # Cost for communication actions (1 per uncommunicated goal)
        cost += len(uncomm_soil_goals)
        cost += len(uncomm_rock_goals)
        cost += len(uncomm_image_goals)

        # Cost for sample/image actions (1 per item that needs acquiring)
        need_sample_soil = {w for w in uncomm_soil_goals if not any((r, w) in have_soil for r in self.all_rovers)}
        need_sample_rock = {w for w in uncomm_rock_goals if not any((r, w) in have_rock for r in self.all_rovers)}
        need_take_image = {(o, m) for (o, m) in uncomm_image_goals if not any((r, o, m) in have_image for r in self.all_rovers)}

        cost += len(need_sample_soil)
        cost += len(need_sample_rock)
        cost += len(need_take_image)

        # Cost for calibrate actions (1 per camera needed for an unacquired image that is not calibrated)
        needed_cameras_for_image = {
            i for (o, m) in need_take_image
            for i in self.all_cameras # Iterate through all cameras known from static facts
            if m in self.camera_modes.get(i, set()) # Check if camera supports mode
            and self.camera_rover.get(i) in self.equipped_imaging # Check if camera is on imaging rover
        }
        need_calibrate_cameras = {
            i for i in needed_cameras_for_image
            if not any((i, r) in calibrated_cameras for r in self.all_rovers) # Check if camera is calibrated on any rover
        }
        cost += len(need_calibrate_cameras)

        # Cost for drop action (simplified: 1 if any sampling is needed and any equipped rover has a full store)
        sampling_needed_any = len(need_sample_soil) > 0 or len(need_sample_rock) > 0
        # Check if any equipped rover that might be used for sampling has a full store
        any_equipped_rover_store_full = any(
            (self.rover_store.get(r) is not None and (self.rover_store[r] not in store_empty)) # Check if store exists and is not empty
            for r in self.equipped_soil | self.equipped_rock # Consider rovers equipped for sampling
        )
        cost += 1 if sampling_needed_any and any_equipped_rover_store_full else 0


        # Movement costs (simplified: +1 if any relevant rover is not at a suitable location type)
        # Note: This counts distinct *types* of location visits needed if no rover is currently positioned.

        # Soil sampling movement: needed if samples need taking and exist at waypoints
        soil_sampling_locations_needed = {w for w in need_sample_soil if w in at_soil_sample_state}
        if soil_sampling_locations_needed:
             # Check if *all* soil-equipped rovers are *not* at any needed soil location
             all_soil_rovers_misplaced = all(
                 r not in rover_location or rover_location[r] not in soil_sampling_locations_needed
                 for r in self.equipped_soil
             )
             cost += 1 if all_soil_rovers_misplaced else 0

        # Rock sampling movement: needed if samples need taking and exist at waypoints
        rock_sampling_locations_needed = {w for w in need_sample_rock if w in at_rock_sample_state}
        if rock_sampling_locations_needed:
             # Check if *all* rock-equipped rovers are *not* at any needed rock location
             all_rock_rovers_misplaced = all(
                 r not in rover_location or rover_location[r] not in rock_sampling_locations_needed
                 for r in self.equipped_rock
             )
             cost += 1 if all_rock_rovers_misplaced else 0

        # Imaging movement: needed if images need taking and there are waypoints to take them from
        image_locations_needed = {p for (o, m) in need_take_image for p in self.obj_image_waypoints.get(o, set())}
        if image_locations_needed:
             # Check if *all* imaging-equipped rovers are *not* at any needed image location
             all_imaging_rovers_misplaced = all(
                 r not in rover_location or rover_location[r] not in image_locations_needed
                 for r in self.equipped_imaging
             )
             cost += 1 if all_imaging_rovers_misplaced else 0

        # Calibration movement: needed if calibration is required and there are waypoints to calibrate from
        calibration_locations_needed = {
            w for i in need_calibrate_cameras
            for t in [self.camera_cal_target.get(i)] if t is not None
            for w in self.cal_target_waypoints.get(t, set())
        }
        if calibration_locations_needed:
             # Check if *all* rovers with cameras needing calibration are *not* at any needed calibration location
             all_cal_rovers_misplaced = all(
                 r not in rover_location or rover_location[r] not in calibration_locations_needed
                 for i in need_calibrate_cameras for r in [self.camera_rover.get(i)] if r is not None
             )
             cost += 1 if all_cal_rovers_misplaced else 0

        # Communication movement: needed if any communication is required and there are communication waypoints
        if (uncomm_soil_goals or uncomm_rock_goals or uncomm_image_goals) and self.comm_waypoints:
             # Check if *all* rovers are *not* at any communication location
             all_rovers_misplaced_for_comm = all(
                 r not in rover_location or rover_location[r] not in self.comm_waypoints
                 for r in self.all_rovers
             )
             cost += 1 if all_rovers_misplaced_for_comm else 0

        return cost
