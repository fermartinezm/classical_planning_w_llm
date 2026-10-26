from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all
    communication goals (soil data, rock data, images). It counts the core
    actions (sample, take_image, calibrate, drop, communicate) and adds a
    navigation cost (simplified to 1) if a rover is not positioned correctly
    for a required action type. It sums the estimated costs for each
    unachieved goal.

    # Assumptions
    - All required equipment, samples, and calibration targets exist if they
      are part of a goal in a solvable instance.
    - Any equipped rover can perform the necessary actions (sampling, imaging,
      calibrating, communicating) if positioned correctly and other
      preconditions (like empty store or calibration) are met.
    - Navigation cost between any two relevant points (sample site, image point,
      calibration point, communication point) is a constant 1, if a rover
      is not already at *any* suitable location of that type for the specific
      task component (sampling, imaging, calibrating).
    - A single navigation action is sufficient to move *a* rover to *a* communication
      point if any communication goal is pending and no rover is currently
      at a communication point.
    - Dropping a sample costs 1 action if an equipped rover needs an empty store
      for sampling and no equipped rover has an empty store.
    - Calibrating a camera costs 1 action if a suitable camera is not calibrated.

    # Heuristic Initialization
    - Extract goal conditions to identify required soil data waypoints, rock data
      waypoints, and images (objective, mode).
    - Extract static facts: lander location, waypoint visibility, rover
      capabilities (equipped_for_*), store ownership, camera properties
      (on_board, supports, calibration_target), and objective visibility
      from waypoints.
    - Identify all objects of relevant types (rovers, waypoints, cameras, stores, objectives, modes, landers)
      from the initial state and static facts to facilitate lookups.
    - Pre-calculate communication points (waypoints visible from the lander).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1. Initialize the total heuristic cost `h` to 0.
    2. Extract dynamic information from the current state: rover locations,
       store fullness, collected data (soil, rock, image), sample presence
       at waypoints, and camera calibration status.
    3. Initialize a flag `needs_communication_nav` to False. This flag tracks
       if any communication goal is pending, to add the communication navigation
       cost only once globally if needed.
    4. Iterate through each waypoint `w` in `self.goal_soil_waypoints`:
       - If `(communicated_soil_data w)` is NOT in the current state:
         - Add 1 to `h` for the `communicate_soil_data` action.
         - Set `needs_communication_nav` to True.
         - Check if `(have_soil_analysis r w)` exists for any rover `r`.
         - If not (data needs sampling):
           - Add 1 to `h` for the `sample_soil` action.
           - Check if any equipped soil rover is currently at waypoint `w`. If not, add 1 for `navigate` to the sample point.
           - Check if any equipped soil rover has an empty store. If not, add 1 for the `drop` action (to free up a store).
    5. Iterate through each waypoint `w` in `self.goal_rock_waypoints`:
       - If `(communicated_rock_data w)` is NOT in the current state:
         - Add 1 to `h` for the `communicate_rock_data` action.
         - Set `needs_communication_nav` to True.
         - Check if `(have_rock_analysis r w)` exists for any rover `r`.
         - If not (data needs sampling):
           - Add 1 to `h` for the `sample_rock` action.
           - Check if any equipped rock rover is currently at waypoint `w`. If not, add 1 for `navigate` to the sample point.
           - Check if any equipped rock rover has an empty store. If not, add 1 for the `drop` action.
    6. Iterate through each image goal `(o, m)` in `self.goal_images`:
       - If `(communicated_image_data o m)` is NOT in the current state:
         - Add 1 to `h` for the `communicate_image_data` action.
         - Set `needs_communication_nav` to True.
         - Check if `(have_image r o m)` exists for any rover `r`.
         - If not (image needs taking):
           - Add 1 to `h` for the `take_image` action.
           - Find waypoints visible from objective `o`. Check if any equipped imaging rover is at any of these waypoints. If not, add 1 for `navigate` to an image point.
           - Find suitable cameras (on equipped imaging rovers, supporting mode `m`). Check if any of these suitable cameras are calibrated. If not, add 1 for `calibrate`.
             - If calibration is needed: Find the calibration target for a suitable camera. Find waypoints visible from this target. Check if *any* equipped imaging rover with a suitable camera is at *any* valid calibration point for *its* camera. If not, add 1 for `navigate` to a calibrate point.
    7. After processing all goals, if `needs_communication_nav` is True:
       - Check if any rover is currently at a waypoint visible from the lander (a communication point). If not, add 1 for `navigate` to a communication point.
    8. Return the total cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Data structures for static information
        self.goal_soil_waypoints = set()
        self.goal_rock_waypoints = set()
        self.goal_images = set() # set of (objective, mode) tuples

        self.lander_loc = None
        self.visible_waypoints = set() # set of (w1, w2) tuples (w1 visible from w2)
        self.comm_points = set() # waypoints visible from lander (where rover needs to be)

        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.equipped_imaging_rovers = set()

        self.rover_to_store = {} # rover -> store

        self.camera_on_rover = {} # camera -> rover
        self.camera_modes = {} # camera -> set of modes
        self.camera_cal_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set of waypoints (objective visible from waypoint)

        # Collect all objects by type (useful for iterating and checking types)
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_stores = set()
        self.all_cameras = set()
        self.all_objectives = set()
        self.all_modes = set()
        self.all_landers = set()

        # Helper to collect objects from fact parts
        def collect_objects(parts):
             if not parts: return
             predicate = parts[0]
             if predicate == 'at': # (at ?x ?y) - assuming ?x is object, ?y is location
                 # Need to infer type from context or rely on specific predicates below
                 pass
             elif predicate == 'at_lander': # (at_lander ?x - lander ?y - waypoint)
                 self.all_landers.add(parts[1])
                 self.all_waypoints.add(parts[2])
             elif predicate == 'can_traverse': # (can_traverse ?r - rover ?x - waypoint ?y - waypoint)
                 self.all_rovers.add(parts[1])
                 self.all_waypoints.add(parts[2])
                 self.all_waypoints.add(parts[3])
             elif predicate in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging']: # (?r - rover)
                 self.all_rovers.add(parts[1])
             elif predicate in ['empty', 'full']: # (?s - store)
                 self.all_stores.add(parts[1])
             elif predicate in ['have_rock_analysis', 'have_soil_analysis']: # (?r - rover ?w - waypoint)
                 self.all_rovers.add(parts[1])
                 self.all_waypoints.add(parts[2])
             elif predicate == 'calibrated': # (?c - camera ?r - rover)
                 self.all_cameras.add(parts[1])
                 self.all_rovers.add(parts[2])
             elif predicate == 'supports': # (?c - camera ?m - mode)
                 self.all_cameras.add(parts[1])
                 self.all_modes.add(parts[2])
             elif predicate == 'visible': # (?w1 - waypoint ?w2 - waypoint)
                 self.all_waypoints.add(parts[1])
                 self.all_waypoints.add(parts[2])
             elif predicate == 'have_image': # (?r - rover ?o - objective ?m - mode)
                 self.all_rovers.add(parts[1])
                 self.all_objectives.add(parts[2])
                 self.all_modes.add(parts[3])
             elif predicate in ['communicated_soil_data', 'communicated_rock_data']: # (?w - waypoint)
                 self.all_waypoints.add(parts[1])
             elif predicate == 'communicated_image_data': # (?o - objective ?m - mode)
                 self.all_objectives.add(parts[1])
                 self.all_modes.add(parts[2])
             elif predicate in ['at_soil_sample', 'at_rock_sample']: # (?w - waypoint)
                 self.all_waypoints.add(parts[1])
             elif predicate == 'visible_from': # (?o - objective ?w - waypoint)
                 self.all_objectives.add(parts[1])
                 self.all_waypoints.add(parts[2])
             elif predicate == 'store_of': # (?s - store ?r - rover)
                 self.all_stores.add(parts[1])
                 self.all_rovers.add(parts[2])
             elif predicate == 'calibration_target': # (?i - camera ?o - objective)
                 self.all_cameras.add(parts[1])
                 self.all_objectives.add(parts[2])
             elif predicate == 'on_board': # (?i - camera ?r - rover)
                 self.all_cameras.add(parts[1])
                 self.all_rovers.add(parts[2])


        # Parse goals and collect objects
        for goal in self.goals:
            parts = get_parts(goal)
            collect_objects(parts) # Collect objects from goals
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "communicated_soil_data" and len(parts) == 2:
                self.goal_soil_waypoints.add(parts[1])
            elif predicate == "communicated_rock_data" and len(parts) == 2:
                self.goal_rock_waypoints.add(parts[1])
            elif predicate == "communicated_image_data" and len(parts) == 3:
                self.goal_images.add((parts[1], parts[2])) # (objective, mode)

        # Parse static facts and collect objects
        for fact in static_facts:
            parts = get_parts(fact)
            collect_objects(parts) # Collect objects from static facts
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "at_lander" and len(parts) == 3:
                # Assuming only one lander for simplicity, store its location
                self.lander_loc = parts[2]
            elif predicate == "equipped_for_soil_analysis" and len(parts) == 2:
                self.equipped_soil_rovers.add(parts[1])
            elif predicate == "equipped_for_rock_analysis" and len(parts) == 2:
                self.equipped_rock_rovers.add(parts[1])
            elif predicate == "equipped_for_imaging" and len(parts) == 2:
                self.equipped_imaging_rovers.add(parts[1])
            elif predicate == "store_of" and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.rover_to_store[rover] = store
            elif predicate == "visible" and len(parts) == 3:
                self.visible_waypoints.add((parts[1], parts[2])) # w1 visible from w2
            elif predicate == "on_board" and len(parts) == 3:
                self.camera_on_rover[parts[1]] = parts[2]
            elif predicate == "supports" and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target" and len(parts) == 3:
                self.camera_cal_target[parts[1]] = parts[2]
            elif predicate == "visible_from" and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)

        # Determine communication points (waypoints ?x such that (visible ?x lander_loc) is true)
        if self.lander_loc:
             self.comm_points = {w1 for w1, w2 in self.visible_waypoints if w2 == self.lander_loc}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Extract dynamic state information
        rover_locations = {} # rover -> waypoint
        store_empty = {} # store -> bool
        have_soil = {} # (rover, waypoint) -> bool
        have_rock = {} # (rover, waypoint) -> bool
        have_image = {} # (rover, objective, mode) -> bool
        # soil_sample_at = set() # waypoint - Not strictly needed for heuristic logic
        # rock_sample_at = set() # waypoint - Not strictly needed for heuristic logic
        calibrated_cameras = set() # (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                # Assuming it's a rover location based on domain structure
                if parts[1] in self.all_rovers:
                    rover_locations[parts[1]] = parts[2]
            elif predicate == 'empty' and len(parts) == 2:
                store_empty[parts[1]] = True
            elif predicate == 'full' and len(parts) == 2:
                 store_empty[parts[1]] = False # Explicitly mark as not empty
            elif predicate == 'have_soil_analysis' and len(parts) == 3:
                have_soil[(parts[1], parts[2])] = True
            elif predicate == 'have_rock_analysis' and len(parts) == 3:
                have_rock[(parts[1], parts[2])] = True
            elif predicate == 'have_image' and len(parts) == 4:
                have_image[(parts[1], parts[2], parts[3])] = True
            # elif predicate == 'at_soil_sample' and len(parts) == 2:
            #     soil_sample_at.add(parts[1])
            # elif predicate == 'at_rock_sample' and len(parts) == 2:
            #     rock_sample_at.add(parts[1])
            elif predicate == 'calibrated' and len(parts) == 3:
                calibrated_cameras.add((parts[1], parts[2]))

        needs_communication_nav = False

        # Estimate cost for soil data goals
        for w in self.goal_soil_waypoints:
            if f"(communicated_soil_data {w})" not in state:
                h += 1 # communicate action
                needs_communication_nav = True

                data_exists = any(have_soil.get((r, w), False) for r in self.all_rovers)

                if not data_exists:
                    h += 1 # sample action
                    # Need equipped soil rover at waypoint w
                    equipped_rover_at_w = any(rover_locations.get(r) == w for r in self.equipped_soil_rovers)
                    if not equipped_rover_at_w:
                        h += 1 # navigate to sample point

                    # Need empty store on an equipped soil rover
                    # Check if *any* equipped soil rover has an empty store
                    equipped_rover_has_empty_store = any(store_empty.get(self.rover_to_store.get(r), False) for r in self.equipped_soil_rovers)
                    if not equipped_rover_has_empty_store:
                        h += 1 # drop action


        # Estimate cost for rock data goals
        for w in self.goal_rock_waypoints:
             if f"(communicated_rock_data {w})" not in state:
                h += 1 # communicate action
                needs_communication_nav = True

                data_exists = any(have_rock.get((r, w), False) for r in self.all_rovers)

                if not data_exists:
                    h += 1 # sample action
                    # Need equipped rock rover at waypoint w
                    equipped_rover_at_w = any(rover_locations.get(r) == w for r in self.equipped_rock_rovers)
                    if not equipped_rover_at_w:
                        h += 1 # navigate to sample point

                    # Need empty store on an equipped rock rover
                    # Check if *any* equipped rock rover has an empty store
                    equipped_rover_has_empty_store = any(store_empty.get(self.rover_to_store.get(r), False) for r in self.equipped_rock_rovers)
                    if not equipped_rover_has_empty_store:
                        h += 1 # drop action


        # Estimate cost for image data goals
        for o, m in self.goal_images:
            if f"(communicated_image_data {o} {m})" not in state:
                h += 1 # communicate action
                needs_communication_nav = True

                image_exists = any(have_image.get((r, o, m), False) for r in self.all_rovers)

                if not image_exists:
                    h += 1 # take_image action

                    # Need equipped imaging rover at a waypoint visible from objective o
                    waypoints_visible_from_o = self.objective_visible_from.get(o, set())
                    equipped_imaging_rover_at_image_point = any(rover_locations.get(r) in waypoints_visible_from_o for r in self.equipped_imaging_rovers)
                    if not equipped_imaging_rover_at_image_point:
                        h += 1 # navigate to image point

                    # Need a suitable camera calibrated
                    suitable_cameras = {c for c in self.camera_on_rover if self.camera_on_rover[c] in self.equipped_imaging_rovers and m in self.camera_modes.get(c, set())}
                    suitable_camera_calibrated = any((c, self.camera_on_rover[c]) in calibrated_cameras for c in suitable_cameras)

                    if not suitable_camera_calibrated:
                        h += 1 # calibrate action

                        # Need rover with suitable camera at a waypoint visible from calibration target
                        any_rover_at_calibrate_point = False
                        for c in suitable_cameras:
                            r = self.camera_on_rover.get(c)
                            t = self.camera_cal_target.get(c)
                            if r and t: # Ensure rover and target exist for this camera
                                waypoints_visible_from_t = self.objective_visible_from.get(t, set())
                                if rover_locations.get(r) in waypoints_visible_from_t:
                                    any_rover_at_calibrate_point = True
                                    break # Found one, no need to check others

                        if not any_rover_at_calibrate_point:
                            h += 1 # navigate to calibrate point

        # Add navigation cost for communication if needed and no rover is at a comm point
        if needs_communication_nav:
            any_rover_at_comm_point = any(r_loc in self.comm_points for r_loc in rover_locations.values())
            if not any_rover_at_comm_point:
                h += 1 # navigate to comm point

        return h
