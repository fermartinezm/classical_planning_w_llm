from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by summing up the estimated costs for achieving each uncommunicated goal fact.
    It breaks down the cost for each goal type (soil, rock, image) into
    communication, sampling/imaging, calibration (for images), and associated
    movement and simplified resource management (store).

    # Assumptions
    - Each uncommunicated goal fact requires a final communication action (cost 1).
    - Achieving a 'have_analysis' (soil/rock) fact requires a sampling action (cost 1).
    - Achieving a 'have_image' fact requires a take_image action (cost 1).
    - Achieving a 'calibrated' fact requires a calibrate action (cost 1).
    - Movement to the required location (sample site, imaging point, calibration point) is estimated as a fixed cost (1 action).
    - Store management (dropping a sample) is estimated as a fixed cost (1 action) per sampling task if any suitable rover currently has a full store. This is a rough estimate.
    - The heuristic does not track specific rovers or cameras for tasks, assuming any suitable one can be used if available.
    - Assumes necessary objects (equipped rovers, cameras, samples, visible points) exist based on the initial state/statics to make goals achievable.

    # Heuristic Initialization
    The constructor precomputes static information from the task definition and
    initial state, such as:
    - The set of goal facts.
    - Lander location(s) and communication points.
    - Initial locations of soil and rock samples (since samples are consumed).
    - Rover equipment, stores, and camera information (on-board, calibration target, supported modes).
    - Waypoints visible from objectives (imaging points) and calibration targets (calibration points).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Parse the current state facts into easily queryable data structures (dictionaries/sets).
    2. Initialize the heuristic value `h` to 0.
    3. Initialize sets to track unique intermediate facts needed: `needed_have_soil_waypoints`, `needed_have_rock_waypoints`, `needed_have_image_oms`, `needed_calibrated_ir`.
    4. Iterate through the goal facts defined in the task:
       - If a goal fact `(communicated_soil_data W)` is not in the current state:
         - Add 1 to `h` (cost for `communicate_soil_data`).
         - Add `W` to `needed_have_soil_waypoints`.
       - If a goal fact `(communicated_rock_data W)` is not in the current state:
         - Add 1 to `h` (cost for `communicate_rock_data`).
         - Add `W` to `needed_have_rock_waypoints`.
       - If a goal fact `(communicated_image_data O M)` is not in the current state:
         - Add 1 to `h` (cost for `communicate_image_data`).
         - Add `(O, M)` to `needed_have_image_oms`.
    5. Determine if a 'drop' action might be needed for sampling tasks: Check if any soil-equipped or rock-equipped rover currently has a full store. Set `drop_cost_per_sample` to 1 if true, 0 otherwise. This is a simplified check.
    6. Process `needed_have_soil_waypoints`: For each waypoint `W` in this set, if no rover currently has `(have_soil_analysis * W)`:
       - Check if a soil sample initially existed at `W`. If not, this goal might be impossible, skip or add large cost (skipping for simplicity).
       - Add 1 to `h` (cost for `sample_soil`).
       - Add 1 to `h` (estimated cost to move a rover to `W`).
       - Add `drop_cost_per_sample` to `h`.
    7. Process `needed_have_rock_waypoints`: For each waypoint `W` in this set, if no rover currently has `(have_rock_analysis * W)`:
       - Check if a rock sample initially existed at `W`. If not, skip.
       - Add 1 to `h` (cost for `sample_rock`).
       - Add 1 to `h` (estimated cost to move a rover to `W`).
       - Add `drop_cost_per_sample` to `h`.
    8. Process `needed_have_image_oms`: For each `(O, M)` pair in this set, if no rover currently has `(have_image * O M)`:
       - Check if any imaging point exists for objective `O`. If not, skip.
       - Add 1 to `h` (cost for `take_image`).
       - Add 1 to `h` (estimated cost to move a rover to an imaging point for `O`).
       - Identify all `(camera, rover)` pairs that could potentially take this image (rover is imaging-equipped, camera is on board, supports mode `M`, has a calibration target visible from some waypoint, and objective `O` is visible from some waypoint). For each such pair `(I, R)`, add `(I, R)` to `needed_calibrated_ir`.
    9. Process `needed_calibrated_ir`: For each `(I, R)` pair in this set, if `(calibrated I R)` is not in the current state:
       - Check if any calibration point exists for camera `I`. If not, skip.
       - Add 1 to `h` (cost for `calibrate`).
       - Add 1 to `h` (estimated cost to move rover `R` to a calibration point for camera `I`).
    10. Return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goal_facts = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state for sample locations

        # Precompute static information
        lander_waypoints = set() # Handle multiple landers
        self.visible_map = {} # waypoint -> set of visible waypoints
        self.initial_soil_samples = set()
        self.initial_rock_samples = set()
        self.rover_equipment = {} # rover -> set of equipment types ('soil', 'rock', 'imaging')
        self.rover_stores = {} # rover -> store
        self.camera_info = {} # camera -> {'rover': r, 'cal_target': t, 'modes': {m1, m2, ...}}
        self.objective_imaging_points = {} # objective -> set of visible waypoints
        self.camera_calibration_points = {} # camera -> set of visible waypoints from its cal target
        self.calibration_targets = {} # camera -> calibration_target

        # Parse static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            pred = parts[0]
            if pred == "at_lander":
                lander_waypoints.add(parts[2])
            elif pred == "visible":
                w1, w2 = parts[1], parts[2]
                self.visible_map.setdefault(w1, set()).add(w2)
                # visible is symmetric in this domain
                self.visible_map.setdefault(w2, set()).add(w1)
            elif pred == "equipped_for_soil_analysis":
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('soil')
            elif pred == "equipped_for_rock_analysis":
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('rock')
            elif pred == "equipped_for_imaging":
                rover = parts[1]
                self.rover_equipment.setdefault(rover, set()).add('imaging')
            elif pred == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif pred == "on_board":
                camera, rover = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'rover': rover})
            elif pred == "calibration_target":
                camera, target = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).update({'cal_target': target})
                self.calibration_targets[camera] = target
            elif pred == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_info.setdefault(camera, {}).setdefault('modes', set()).add(mode)
            elif pred == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_imaging_points.setdefault(objective, set()).add(waypoint)

        # Parse initial state for sample locations (they are consumed by sampling)
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred == "at_soil_sample":
                 self.initial_soil_samples.add(parts[1])
             elif pred == "at_rock_sample":
                 self.initial_rock_samples.add(parts[1])

        # Precompute communication points (waypoints visible from any lander waypoint)
        self.comm_points = set()
        for lander_wp in lander_waypoints:
             self.comm_points.update(self.visible_map.get(lander_wp, set()))

        # Precompute calibration points for each camera
        # Calibration points for camera i are waypoints visible from i's calibration target
        visible_from_target_map = {} # target -> set of visible waypoints
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == "visible_from":
                  target, waypoint = parts[1], parts[2]
                  visible_from_target_map.setdefault(target, set()).add(waypoint)

        for camera, target in self.calibration_targets.items():
             if target in visible_from_target_map:
                  self.camera_calibration_points[camera] = visible_from_target_map[target]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state facts into dictionaries/sets for quick lookup
        state_at_rover = {} # rover -> waypoint
        state_store_empty = {} # store -> bool
        state_have_soil = {} # (rover, waypoint) -> True
        state_have_rock = {} # (rover, waypoint) -> True
        state_have_image = {} # (rover, objective, mode) -> True
        state_calibrated = {} # (camera, rover) -> True
        state_communicated_soil = {} # waypoint -> True
        state_communicated_rock = {} # waypoint -> True
        state_communicated_image = {} # (objective, mode) -> True

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            pred = parts[0]
            if pred == "at":
                # Could be rover or lander, but we only care about rovers here
                if len(parts) == 3 and parts[1].startswith('rover'):
                     state_at_rover[parts[1]] = parts[2]
            elif pred == "empty":
                state_store_empty[parts[1]] = True
            elif pred == "full":
                 state_store_empty[parts[1]] = False # Store is not empty
            elif pred == "have_soil_analysis":
                state_have_soil[(parts[1], parts[2])] = True
            elif pred == "have_rock_analysis":
                state_have_rock[(parts[1], parts[2])] = True
            elif pred == "have_image":
                state_have_image[(parts[1], parts[2], parts[3])] = True
            elif pred == "calibrated":
                state_calibrated[(parts[1], parts[2])] = True
            elif pred == "communicated_soil_data":
                state_communicated_soil[parts[1]] = True
            elif pred == "communicated_rock_data":
                state_communicated_rock[parts[1]] = True
            elif pred == "communicated_image_data":
                state_communicated_image[(parts[1], parts[2])] = True

        h = 0
        needed_have_soil_waypoints = set()
        needed_have_rock_waypoints = set()
        needed_have_image_oms = set()
        needed_calibrated_ir = set()

        # 4. Process goals
        for g_str in self.goal_facts:
            parts = get_parts(g_str)
            if not parts: continue

            pred = parts[0]
            if pred == "communicated_soil_data":
                w = parts[1]
                if not state_communicated_soil.get(w, False):
                    h += 1 # communicate cost
                    needed_have_soil_waypoints.add(w)
            elif pred == "communicated_rock_data":
                w = parts[1]
                if not state_communicated_rock.get(w, False):
                    h += 1 # communicate cost
                    needed_have_rock_waypoints.add(w)
            elif pred == "communicated_image_data":
                o, m = parts[1], parts[2]
                if not state_communicated_image.get((o, m), False):
                    h += 1 # communicate cost
                    needed_have_image_oms.add((o, m))

        # 5. Determine if a 'drop' action might be needed for sampling tasks
        drop_cost_per_sample = 0
        # Check if any soil/rock equipped rover has a full store
        for rover, equipment in self.rover_equipment.items():
            if 'soil' in equipment or 'rock' in equipment:
                store = self.rover_stores.get(rover)
                # Check if the store exists and is NOT empty (i.e., full)
                if store and not state_store_empty.get(store, True):
                     drop_cost_per_sample = 1
                     break # Only need one such rover to potentially need a drop

        # 6. Process needed_have_soil_waypoints
        for w in needed_have_soil_waypoints:
            # Check if any rover has the analysis
            has_analysis = any(state_have_soil.get((r, w), False) for r in self.rover_equipment if 'soil' in self.rover_equipment.get(r, set()))
            if not has_analysis:
                # Check if the sample actually exists initially
                if w in self.initial_soil_samples:
                    h += 1 # sample cost
                    h += 1 # move to w cost
                    h += drop_cost_per_sample # potential drop cost
                # else: goal is likely impossible, heuristic doesn't add cost

        # 7. Process needed_have_rock_waypoints
        for w in needed_have_rock_waypoints:
            # Check if any rover has the analysis
            has_analysis = any(state_have_rock.get((r, w), False) for r in self.rover_equipment if 'rock' in self.rover_equipment.get(r, set()))
            if not has_analysis:
                 # Check if the sample actually exists initially
                if w in self.initial_rock_samples:
                    h += 1 # sample cost
                    h += 1 # move to w cost
                    h += drop_cost_per_sample # potential drop cost
                # else: goal likely impossible

        # 8. Process needed_have_image_oms
        for o, m in needed_have_image_oms:
            # Check if any rover has the image
            has_image = any(state_have_image.get((r, o, m), False) for r in self.rover_equipment if 'imaging' in self.rover_equipment.get(r, set()))
            if not has_image:
                # Check if there is any imaging point for this objective
                if self.objective_imaging_points.get(o):
                    h += 1 # take_image cost
                    h += 1 # move to imaging point cost (assume one exists)

                    # Identify cameras/rovers that could take this image and need calibration
                    for cam, info in self.camera_info.items():
                        r = info.get('rover')
                        cal_target = info.get('cal_target')
                        supported_modes = info.get('modes', set())

                        # Check if this camera/rover pair is suitable for this image goal
                        if (r and 'imaging' in self.rover_equipment.get(r, set()) and
                            m in supported_modes and
                            cal_target # Camera must have a calibration target
                           ):
                            # Check if calibration points exist for this camera's target
                            if self.camera_calibration_points.get(cam):
                                # Check if imaging points exist for the objective
                                if self.objective_imaging_points.get(o):
                                     # This (cam, r) pair *could* potentially take image (o, m)
                                     needed_calibrated_ir.add((cam, r))
                # else: goal likely impossible (no imaging point for objective)


        # 9. Process needed_calibrated_ir
        for i, r in needed_calibrated_ir:
            if not state_calibrated.get((i, r), False):
                 # Check if calibration points exist for this camera
                 if self.camera_calibration_points.get(i):
                    h += 1 # calibrate cost
                    h += 1 # move to calibration point cost (assume one exists)
                 # else: calibration likely impossible

        return h
