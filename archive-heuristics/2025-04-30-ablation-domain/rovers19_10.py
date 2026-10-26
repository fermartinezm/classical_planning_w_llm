from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Remove outer parentheses and split by whitespace
    content = fact.strip()[1:-1].strip()
    if not content: # Handle empty fact string "()"
        return []
    return content.split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern using fnmatch.

    - `fact`: The complete fact as a string, e.g., "(at rover1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed), e.g., "at", "*", "waypoint1".
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy each
    individual goal condition that is not yet met. The total heuristic value
    is the sum of these individual goal costs. It uses a relaxed view where
    movement to a required location costs 1 if no suitable rover is already
    there, and resource constraints (like empty stores or calibrated cameras)
    are checked in a simplified way.

    # Assumptions
    - The problem instance is solvable.
    - Static facts (like equipment, camera support, calibration targets,
      visible_from, at_lander, at_soil_sample, at_rock_sample) do not change.
    - The structure of store names is 'rover_name' + 'store'.
    - Action costs are uniform (implicitly 1).

    # Heuristic Initialization
    The constructor extracts key static information from the task:
    - The lander's location.
    - Which rovers have which capabilities (soil, rock, imaging).
    - Which cameras are on board which rovers.
    - Which modes each camera supports.
    - Which objective each camera is a calibration target for.
    - Which waypoints are visible from each objective.
    - Which store belongs to which rover.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is computed by summing the estimated cost for each
    goal condition that is not currently true in the state.

    For each unsatisfied goal `(communicated_soil_data waypointX)`:
    1. Add 1 for the `communicate_soil_data` action.
    2. If no soil-equipped rover has `(soil_analysis r waypointX)`, add 1 for the `analyze_soil_sample` action.
    3. If analysis is needed and no soil-equipped rover has `(have_soil_sample r waypointX)`, add 1 for the `take_soil_sample` action.
    4. If taking a sample is needed and no soil-equipped rover is at `waypointX`, add 1 for the `drive` action to `waypointX`.
    5. If taking a sample is needed and no soil-equipped rover has an `(empty rstore)`, add 1 for the `drop_sample` action (to free a store).
    6. If analysis, communication, or dropping a sample (if needed) requires a rover at the lander, and no soil-equipped rover is at the lander location, add 1 for the `drive` action to the lander.

    For each unsatisfied goal `(communicated_rock_data waypointX)`:
    (Similar logic as soil data, replacing soil with rock)

    For each unsatisfied goal `(communicated_image_data objectiveX modeY)`:
    1. Add 1 for the `communicate_image_data` action.
    2. If no imaging-equipped rover has `(have_image r objectiveX modeY)`, add 1 for the `take_image` action.
    3. If taking an image is needed and no suitable camera (on an imaging rover, supporting the mode, and targeting the objective) is calibrated on `objectiveX`, add 1 for the `calibrate` action.
    4. If calibration is needed and no imaging-equipped rover is at a waypoint visible from `objectiveX`, add 1 for the `drive` action to a visible location for calibration.
    5. If taking an image is needed and no imaging-equipped rover is at a waypoint visible from `objectiveX`, add 1 for the `drive` action to a visible location for image taking. (Note: This might double count if calibration and image taking happen at the same location, but the heuristic is relaxed).
    6. If communication requires a rover at the lander, and no imaging-equipped rover is at the lander location, add 1 for the `drive` action to the lander.

    The total heuristic is the sum of these costs for all unsatisfied goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.lander_location = None
        self.rovers_by_capability = {'soil': set(), 'rock': set(), 'image': set()}
        self.cameras_on_board = {} # rover -> set of cameras
        self.camera_supports = {} # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective
        self.visible_from_objective = {} # objective -> set of waypoints
        self.rover_store = {} # rover -> store

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == 'at_lander':
                # (at_lander general waypointL)
                if len(parts) == 3:
                     self.lander_location = parts[2]
            elif predicate == 'equipped_for_soil_analysis':
                # (equipped_for_soil_analysis rover)
                if len(parts) == 2:
                     self.rovers_by_capability['soil'].add(parts[1])
            elif predicate == 'equipped_for_rock_analysis':
                # (equipped_for_rock_analysis rover)
                if len(parts) == 2:
                     self.rovers_by_capability['rock'].add(parts[1])
            elif predicate == 'equipped_for_imaging':
                # (equipped_for_imaging rover)
                if len(parts) == 2:
                     self.rovers_by_capability['image'].add(parts[1])
            elif predicate == 'on_board':
                # (on_board camera rover)
                if len(parts) == 3:
                     camera, rover = parts[1], parts[2]
                     self.cameras_on_board.setdefault(rover, set()).add(camera)
            elif predicate == 'supports':
                # (supports camera mode)
                if len(parts) == 3:
                     camera, mode = parts[1], parts[2]
                     self.camera_supports.setdefault(camera, set()).add(mode)
            elif predicate == 'calibration_target':
                # (calibration_target camera objective)
                if len(parts) == 3:
                     camera, objective = parts[1], parts[2]
                     self.camera_calibration_target[camera] = objective
            elif predicate == 'visible_from':
                # (visible_from objective waypoint)
                if len(parts) == 3:
                     objective, waypoint = parts[1], parts[2]
                     self.visible_from_objective.setdefault(objective, set()).add(waypoint)
            elif predicate == 'store_of':
                # (store_of store rover)
                if len(parts) == 3:
                     store, rover = parts[1], parts[2]
                     self.rover_store[rover] = store

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Pre-calculate dynamic states for efficiency
        # Using dictionaries/sets for faster lookups
        rover_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'at', '*', '*')}
        rover_has_soil_analysis = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'soil_analysis', '*', '*')}
        rover_has_rock_analysis = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'rock_analysis', '*', '*')}
        rover_has_soil_sample = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'have_soil_sample', '*', '*')}
        rover_has_rock_sample = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'have_rock_sample', '*', '*')}
        rover_has_image = {(get_parts(fact)[1], get_parts(fact)[2], get_parts(fact)[3]) for fact in state if match(fact, 'have_image', '*', '*', '*')} # (rover, objective, mode)
        camera_calibrated = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, 'calibrated', '*', '*')} # (camera, rover)
        store_status = {get_parts(fact)[1]: get_parts(fact)[0] for fact in state if match(fact, 'empty', '*') or match(fact, 'full', '*')} # store -> 'empty' or 'full'


        for goal in self.goals:
            if goal in state:
                continue # Goal already satisfied

            parts = get_parts(goal)
            if not parts: continue # Skip empty goals

            predicate = parts[0]

            if predicate == 'communicated_soil_data' and len(parts) == 2:
                waypointX = parts[1]
                goal_cost = 0

                # Find relevant rovers
                R_soil = self.rovers_by_capability.get('soil', set())
                if not R_soil: continue # Should not happen in solvable instances

                # 1. Communicate (always needed if goal not met)
                goal_cost += 1

                # 2. Analyze (needed if no rover has analysis)
                has_analysis = any(r in rover_has_soil_analysis and rover_has_soil_analysis[r] == waypointX for r in R_soil)
                if not has_analysis:
                    goal_cost += 1 # analyze_soil_sample

                    # 3. Take Sample (needed if analysis is needed and no rover has sample)
                    has_sample = any(r in rover_has_soil_sample and rover_has_soil_sample[r] == waypointX for r in R_soil)
                    if not has_sample:
                        goal_cost += 1 # take_soil_sample

                        # 4. Move to Sample Location (needed if taking sample and no rover is there)
                        at_sample_loc = any(rover_locations.get(r) == waypointX for r in R_soil)
                        if not at_sample_loc:
                            goal_cost += 1 # drive to waypointX

                        # 5. Free Store (if needed for take_sample and no empty store)
                        # Need empty store on *some* soil rover to take sample
                        has_empty_store = any(store_status.get(self.rover_store.get(r)) == 'empty' for r in R_soil if self.rover_store.get(r))
                        if not has_empty_store:
                            # Need to drop a sample. This requires being at the lander.
                            # Add cost for drop. Move to lander cost is handled below.
                            goal_cost += 1 # drop_sample

                # 6. Move to Lander (needed for analyze, communicate, drop)
                # If analysis is needed OR communication is needed OR drop was needed, a rover must go to lander
                need_rover_at_lander = (not has_analysis) or (not goal in state) or ((not has_sample) and (not has_empty_store))
                at_lander_loc = any(rover_locations.get(r) == self.lander_location for r in R_soil)
                if need_rover_at_lander and not at_lander_loc:
                     goal_cost += 1 # drive to lander

                total_cost += goal_cost

            elif predicate == 'communicated_rock_data' and len(parts) == 2:
                waypointX = parts[1]
                goal_cost = 0

                # Find relevant rovers
                R_rock = self.rovers_by_capability.get('rock', set())
                if not R_rock: continue # Should not happen

                # 1. Communicate
                goal_cost += 1 # communicate_rock_data

                # 2. Analyze
                has_analysis = any(r in rover_has_rock_analysis and rover_has_rock_analysis[r] == waypointX for r in R_rock)
                if not has_analysis:
                    goal_cost += 1 # analyze_rock_sample

                    # 3. Take Sample
                    has_sample = any(r in rover_has_rock_sample and rover_has_rock_sample[r] == waypointX for r in R_rock)
                    if not has_sample:
                        goal_cost += 1 # take_rock_sample

                        # 4. Move to Sample Location
                        at_sample_loc = any(rover_locations.get(r) == waypointX for r in R_rock)
                        if not at_sample_loc:
                            goal_cost += 1 # drive to waypointX

                        # 5. Free Store (if needed for take_sample)
                        has_empty_store = any(store_status.get(self.rover_store.get(r)) == 'empty' for r in R_rock if self.rover_store.get(r))
                        if not has_empty_store:
                            goal_cost += 1 # drop_sample

                # 6. Move to Lander (needed for analyze, communicate, drop)
                need_rover_at_lander = (not has_analysis) or (not goal in state) or ((not has_sample) and (not has_empty_store))
                at_lander_loc = any(rover_locations.get(r) == self.lander_location for r in R_rock)
                if need_rover_at_lander and not at_lander_loc:
                     goal_cost += 1 # drive to lander

                total_cost += goal_cost

            elif predicate == 'communicated_image_data' and len(parts) == 3:
                objectiveX = parts[1]
                modeY = parts[2]
                goal_cost = 0

                # Find relevant rovers
                R_image = self.rovers_by_capability.get('image', set())
                if not R_image: continue # Should not happen

                # 1. Communicate
                goal_cost += 1 # communicate_image_data

                # 2. Take Image (needed if no rover has image)
                has_image = (objectiveX, modeY) in [(obj, mode) for r, obj, mode in rover_has_image if r in R_image]
                if not has_image:
                    goal_cost += 1 # take_image

                    # 3. Calibrate Camera (if needed for take_image)
                    # Find cameras on image rovers supporting this mode and targeting this objective
                    suitable_target_cameras = {c for r in R_image for c in self.cameras_on_board.get(r, set()) if modeY in self.camera_supports.get(c, set()) and self.camera_calibration_target.get(c) == objectiveX}

                    # Check if any image rover has a calibrated suitable target camera
                    has_calibrated_camera = any((c, r) in camera_calibrated for c in suitable_target_cameras for r in R_image if c in self.cameras_on_board.get(r, set()))

                    if not has_calibrated_camera:
                        goal_cost += 1 # calibrate

                        # 4. Move to Visible Location for Calibration (objectiveX is the target)
                        visible_waypoints = self.visible_from_objective.get(objectiveX, set())
                        at_visible_loc_cal = any(rover_locations.get(r) in visible_waypoints for r in R_image)
                        if not at_visible_loc_cal:
                            goal_cost += 1 # drive to visible location for calibration

                    # 5. Move to Visible Location for Image (objectiveX is the objective)
                    # Note: This might be the same location as calibration, but we count it separately in this relaxation
                    visible_waypoints_image = self.visible_from_objective.get(objectiveX, set())
                    at_visible_loc_image = any(rover_locations.get(r) in visible_waypoints_image for r in R_image)
                    if not at_visible_loc_image:
                         goal_cost += 1 # drive to visible location for image


                # 6. Move to Lander (needed for communicate)
                at_lander_loc = any(rover_locations.get(r) == self.lander_location for r in R_image)
                if not at_lander_loc:
                    goal_cost += 1 # drive to lander

                total_cost += goal_cost

        return total_cost
