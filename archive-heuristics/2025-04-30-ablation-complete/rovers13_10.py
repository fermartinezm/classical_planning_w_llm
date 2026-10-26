from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
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

    # Summary
    This heuristic estimates the number of actions required to achieve each
    unmet goal condition (communicating soil data, rock data, or image data)
    by summing up fixed costs for simplified action sequences. It assumes
    independence between goals and the availability of necessary resources
    (equipped rovers, communication waypoints, etc.) through simplified cost
    estimates.

    # Assumptions
    - The heuristic estimates costs for three main types of goal achievements:
      communicating soil data, communicating rock data, and communicating image data.
    - Each type of achievement is broken down into sequential steps (e.g., sample,
      move, communicate; or calibrate, image, move, communicate).
    - Fixed costs are assigned to these steps or sequences, approximating the
      minimum number of actions (including navigation).
    - Navigation between relevant waypoints (sample/image location, calibration
      target location, communication waypoint) is assumed to cost a fixed amount (1 action per required move).
    - The heuristic assumes that if a sample or image is required for a goal
      and not yet obtained, the necessary resources (equipped rover, available
      sample/objective, visible locations) are available somewhere in the domain.
    - Store management (dropping samples) adds a cost of 1 if *all* rovers
      equipped for the required analysis type have full stores.
    - Camera calibration is assumed to be required before taking an image unless
      a suitable camera on a suitable rover is currently calibrated.

    # Heuristic Initialization
    - Extracts goal conditions.
    - Extracts static facts from the task definition:
        - Lander location.
        - Waypoint visibility for identifying communication waypoints.
        - Rover capabilities (equipped for soil, rock, imaging).
        - Store ownership.
        - Camera information (on board which rover, supported modes, calibration target).
        - Visibility from objectives and calibration targets.
    - Precomputes sets of communication waypoints based on lander location and visibility.
    - Precomputes mappings for camera properties and objective/calibration target visibility.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for each goal fact
    that is not yet true in the current state.

    For each goal fact `G` not in the current state:

    1.  **If `G` is `(communicated_soil_data W)`:**
        *   Check if `(have_soil_analysis R W)` is true for any rover `R`.
        *   If yes: Add 2 to the total cost (estimated cost for `navigate` to a communication waypoint + `communicate_soil_data`).
        *   If no:
            *   Add 4 to the total cost (estimated cost for `navigate` to W + `sample_soil` + `navigate` to a communication waypoint + `communicate_soil_data`).
            *   Check if all rovers equipped for soil analysis have full stores. If yes, add 1 (for `drop`).

    2.  **If `G` is `(communicated_rock_data W)`:**
        *   Check if `(have_rock_analysis R W)` is true for any rover `R`.
        *   If yes: Add 2 to the total cost (estimated cost for `navigate` to a communication waypoint + `communicate_rock_data`).
        *   If no:
            *   Add 4 to the total cost (estimated cost for `navigate` to W + `sample_rock` + `navigate` to a communication waypoint + `communicate_rock_data`).
            *   Check if all rovers equipped for rock analysis have full stores. If yes, add 1 (for `drop`).

    3.  **If `G` is `(communicated_image_data O M)`:**
        *   Check if `(have_image R O M)` is true for any rover `R`.
        *   If yes: Add 2 to the total cost (estimated cost for `navigate` to a communication waypoint + `communicate_image_data`).
        *   If no:
            *   Find a rover `R` with a camera `C` on board that supports mode `M`. Assume one exists.
            *   Check if `(calibrated C R)` is true in the current state for such a suitable camera/rover combo.
            *   If yes: Add 4 to the total cost (estimated cost for `navigate` to image location + `take_image` + `navigate` to a communication waypoint + `communicate_image_data`).
            *   If no: Add 6 to the total cost (estimated cost for `navigate` to calibration target location + `calibrate` + `navigate` to image location + `take_image` + `navigate` to a communication waypoint + `communicate_image_data`).

    The total cost accumulated across all unachieved goal facts is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # --- Extract Static Information ---
        static_facts = task.static

        self.lander_location = None
        # We don't strictly need comm_waypoint_pairs for this simple heuristic,
        # just the knowledge that *some* comm waypoint exists and is reachable (cost 1 nav).
        # Keeping the parsing just in case, but it's not used in __call__ cost logic.
        self.comm_waypoint_pairs = set() # Pairs (rover_at_wp, lander_at_wp) where visible

        self.rover_capabilities = {} # rover -> set of {'soil', 'rock', 'imaging'}
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'on_board_rover': rover, 'supported_modes': set(), 'calibration_target': obj}
        self.objective_visibility = {} # objective -> set of waypoints visible from
        self.calibration_target_visibility = {} # cal_target -> set of waypoints visible from

        # First pass to get basic info like lander location and calibration targets
        calibration_targets_set = set()
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at_lander":
                self.lander_location = parts[2] # (at_lander lander waypoint)
            elif predicate == "equipped_for_soil_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('soil')
            elif predicate == "equipped_for_rock_analysis":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('rock')
            elif predicate == "equipped_for_imaging":
                rover = parts[1]
                self.rover_capabilities.setdefault(rover, set()).add('imaging')
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'on_board_rover': rover})
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('supported_modes', set()).add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'calibration_target': objective})
                calibration_targets_set.add(objective)


        # Second pass for visible_from and comm_waypoint_pairs using info from first pass
        for fact in static_facts:
             parts = get_parts(fact)
             predicate = parts[0]

             if predicate == "visible":
                 wp1, wp2 = parts[1], parts[2]
                 # (visible ?x ?y) where rover at ?x, lander at ?y
                 if self.lander_location and wp2 == self.lander_location:
                     self.comm_waypoint_pairs.add((wp1, wp2)) # (rover_wp, lander_wp)

             elif predicate == "visible_from":
                 obj_or_target, waypoint = parts[1], parts[2]
                 if obj_or_target in calibration_targets_set:
                     self.calibration_target_visibility.setdefault(obj_or_target, set()).add(waypoint)
                 else:
                     self.objective_visibility.setdefault(obj_or_target, set()).add(waypoint)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Extract Dynamic Information from State ---
        # We only need info relevant to checking intermediate goal states or resources
        rover_store_status = {} # rover -> 'empty' or 'full'
        rover_soil_samples = {} # rover -> set of waypoints
        rover_rock_samples = {} # rover -> set of waypoints
        rover_images = {} # rover -> set of (objective, mode)
        rover_calibrated_cameras = {} # rover -> set of cameras
        # soil_samples_at_waypoint and rock_samples_at_waypoint are not strictly needed
        # if we assume samples are available on the ground if not yet sampled/held.

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "empty":
                 store = parts[1]
                 # Find which rover this store belongs to
                 for rover, s in self.rover_stores.items():
                     if s == store:
                         rover_store_status[rover] = 'empty'
                         break
            elif predicate == "full":
                 store = parts[1]
                 # Find which rover this store belongs to
                 for rover, s in self.rover_stores.items():
                     if s == store:
                         rover_store_status[rover] = 'full'
                         break
            elif predicate == "have_soil_analysis":
                rover, waypoint = parts[1], parts[2]
                rover_soil_samples.setdefault(rover, set()).add(waypoint)
            elif predicate == "have_rock_analysis":
                rover, waypoint = parts[1], parts[2]
                rover_rock_samples.setdefault(rover, set()).add(waypoint)
            elif predicate == "have_image":
                rover, objective, mode = parts[1], parts[2], parts[3]
                rover_images.setdefault(rover, set()).add((objective, mode))
            elif predicate == "calibrated":
                camera, rover = parts[1], parts[2]
                rover_calibrated_cameras.setdefault(rover, set()).add(camera)


        # --- Estimate Cost for Each Ungoaled Item ---
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            predicate = parts[0]

            if predicate == "communicated_soil_data":
                waypoint = parts[1]
                # Check if any rover has the sample
                has_sample = any(waypoint in rover_soil_samples.get(rover, set()) for rover in self.rover_capabilities)

                if has_sample:
                    # Already sampled, need to communicate
                    total_cost += 2 # navigate to comm + communicate
                else:
                    # Need to sample and communicate
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'soil' in caps]
                    needs_drop = False
                    if equipped_rovers:
                        # Check if *all* equipped rovers have full stores
                        all_equipped_full = True
                        for rover in equipped_rovers:
                            store = self.rover_stores.get(rover)
                            # If a rover has no store or its store is empty, then not all are full
                            if not store or rover_store_status.get(rover) == 'empty':
                                all_equipped_full = False
                                break
                        if all_equipped_full:
                            needs_drop = True

                    total_cost += 4 # navigate to W + sample + navigate to comm + communicate
                    if needs_drop:
                        total_cost += 1 # Add cost for drop action


            elif predicate == "communicated_rock_data":
                waypoint = parts[1]
                # Check if any rover has the sample
                has_sample = any(waypoint in rover_rock_samples.get(rover, set()) for rover in self.rover_capabilities)

                if has_sample:
                    # Already sampled, need to communicate
                    total_cost += 2 # navigate to comm + communicate
                else:
                    # Need to sample and communicate
                    equipped_rovers = [r for r, caps in self.rover_capabilities.items() if 'rock' in caps]
                    needs_drop = False
                    if equipped_rovers:
                        # Check if *all* equipped rovers have full stores
                        all_equipped_full = True
                        for rover in equipped_rovers:
                            store = self.rover_stores.get(rover)
                            # If a rover has no store or its store is empty, then not all are full
                            if not store or rover_store_status.get(rover) == 'empty':
                                all_equipped_full = False
                                break
                        if all_equipped_full:
                            needs_drop = True

                    total_cost += 4 # navigate to W + sample + navigate to comm + communicate
                    if needs_drop:
                        total_cost += 1 # Add cost for drop action


            elif predicate == "communicated_image_data":
                objective, mode = parts[1], parts[2]
                # Check if any rover has the image
                has_image = any((objective, mode) in rover_images.get(rover, set()) for rover in self.rover_capabilities)

                if has_image:
                    # Already imaged, need to communicate
                    total_cost += 2 # navigate to comm + communicate
                else:
                    # Need to take image and communicate
                    is_calibrated = False
                    suitable_rovers = [r for r, caps in self.rover_capabilities.items() if 'imaging' in caps]
                    for rover in suitable_rovers:
                         for camera, info in self.camera_info.items():
                              if info.get('on_board_rover') == rover and mode in info.get('supported_modes', set()):
                                   # Found a suitable camera/rover combo
                                   if camera in rover_calibrated_cameras.get(rover, set()):
                                        is_calibrated = True
                                        break # Found a calibrated camera for this task
                         if is_calibrated:
                              break # Found a suitable calibrated rover/camera

                    if is_calibrated:
                        total_cost += 4 # navigate to image spot + take image + navigate to comm + communicate
                    else:
                        total_cost += 6 # navigate to cal spot + calibrate + navigate to image spot + take image + navigate to comm + communicate

        return total_cost
