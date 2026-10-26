from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
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
    # Check if each part matches the corresponding arg pattern up to the length of args
    # This works correctly for PDDL facts where the number of parts matches the predicate arity.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to satisfy all goal
    conditions. It breaks down the problem into achieving individual goal
    predicates (communicating soil data, rock data, or image data) and sums
    the estimated minimum actions for each unsatisfied goal.

    # Assumptions
    - Navigation between any two relevant waypoints (sample location,
      calibration location, image location, communication location) costs a
      fixed amount (simplified to 1 action for efficiency).
    - For any required task (sampling, imaging), there exists at least one
      rover capable of performing it (equipped, has camera/store) and that
      rover can reach the necessary locations. The heuristic does not check
      specific rover-waypoint reachability beyond existence.
    - Soil and rock samples required by goals are assumed to be present at
      their respective waypoints in the initial state if not already sampled.
    - Imaging tasks are assumed to be possible if the necessary static
      conditions (equipped rover, camera, modes, calibration target,
      visibility) exist in the problem definition.
    - The cost of a 'drop' action is added only if a sampling task is needed
      and all equipped rovers capable of that task have full stores. This drop
      cost is added at most once for soil sampling needs and at most once for
      rock sampling needs per state evaluation.

    # Heuristic Initialization
    The heuristic extracts static information from the task definition,
    including rover capabilities (equipped for soil/rock/imaging), store
    ownership, camera properties (on-board, supported modes, calibration
    target), objective visibility from waypoints, lander location, and
    initial sample locations. This information is stored in dictionaries and
    sets for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the cost as follows:

    1.  Identify all goal predicates that are not yet satisfied in the current state.
    2.  For each unsatisfied goal predicate, estimate the minimum number of
        actions required to achieve it, based on the current state:
        -   **`(communicated_soil_data ?w)`:**
            -   If the soil sample from `?w` is already collected by any rover (`(have_soil_analysis ?r ?w)` is true): Cost is 2 actions (navigate to communication point + communicate).
            -   If the sample is not collected: Cost is 4 actions (navigate to `?w` + sample + navigate to communication point + communicate). Mark that soil sampling is needed.
        -   **`(communicated_rock_data ?w)`:**
            -   If the rock sample from `?w` is already collected by any rover (`(have_rock_analysis ?r ?w)` is true): Cost is 2 actions (navigate to communication point + communicate).
            -   If the sample is not collected: Cost is 4 actions (navigate to `?w` + sample + navigate to communication point + communicate). Mark that rock sampling is needed.
        -   **`(communicated_image_data ?o ?m)`:**
            -   If the image of `?o` in mode `?m` is already taken by any rover (`(have_image ?r ?o ?m)` is true): Cost is 2 actions (navigate to communication point + communicate).
            -   If the image is not taken: Cost is 6 actions (navigate to calibration point + calibrate + navigate to image point + take image + navigate to communication point + communicate). This cost is added only if there exists a rover/camera combination capable of this task based on static facts.
    3.  After summing costs for individual goals, check if soil sampling was needed. If yes, and all equipped soil rovers have full stores, add 1 for a drop action.
    4.  Similarly, check if rock sampling was needed. If yes, and all equipped rock rovers have full stores, add 1 for a drop action.
    5.  The total heuristic value is the sum of all calculated costs.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and relevant initial state facts.
        """
        self.goals = task.goals  # Goal conditions (frozenset of strings)
        static_facts = task.static # Static facts (frozenset of strings)
        initial_state = task.initial_state # Initial state facts (frozenset of strings)

        # --- Extract Static Information ---
        self.equipped_soil_rovers = set()
        self.equipped_rock_rovers = set()
        self.equipped_imaging_rovers = set()
        self.store_to_rover = {}
        self.rover_to_stores = {} # Map rover to set of its stores
        self.rover_to_cameras = {} # Map rover to set of its cameras
        self.camera_to_modes = {} # Map camera to set of supported modes
        self.camera_to_target = {} # Map camera to calibration target objective
        self.target_to_visible_wps = {} # Map objective (target) to set of visible waypoints
        self.objective_to_visible_wps = {} # Map objective (image target) to set of visible waypoints
        self.lander_location = {} # Map lander to waypoint

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == 'equipped_for_soil_analysis' and len(parts) == 2:
                self.equipped_soil_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_rock_analysis' and len(parts) == 2:
                self.equipped_rock_rovers.add(parts[1])
            elif parts[0] == 'equipped_for_imaging' and len(parts) == 2:
                self.equipped_imaging_rovers.add(parts[1])
            elif parts[0] == 'store_of' and len(parts) == 3:
                store, rover = parts[1], parts[2]
                self.store_to_rover[store] = rover
                self.rover_to_stores.setdefault(rover, set()).add(store)
            elif parts[0] == 'on_board' and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                self.rover_to_cameras.setdefault(rover, set()).add(camera)
            elif parts[0] == 'supports' and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                self.camera_to_modes.setdefault(camera, set()).add(mode)
            elif parts[0] == 'calibration_target' and len(parts) == 3:
                camera, objective = parts[1], parts[2]
                self.camera_to_target[camera] = objective
            elif parts[0] == 'visible_from' and len(parts) == 3:
                objective, waypoint = parts[1], parts[2]
                self.objective_to_visible_wps.setdefault(objective, set()).add(waypoint)
            elif parts[0] == 'at_lander' and len(parts) == 3:
                 lander, waypoint = parts[1], parts[2]
                 self.lander_location[lander] = waypoint

        # Populate target_to_visible_wps based on camera_to_target and objective_to_visible_wps
        for camera, target_objective in self.camera_to_target.items():
             if target_objective in self.objective_to_visible_wps:
                 self.target_to_visible_wps[target_objective] = self.objective_to_visible_wps[target_objective]


        # --- Extract Relevant Initial State Information ---
        # We only need initial sample locations to check if sampling is possible
        self.initial_at_soil_sample = set()
        self.initial_at_rock_sample = set()
        for fact in initial_state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'at_soil_sample' and len(parts) == 2:
                 self.initial_at_soil_sample.add(parts[1])
             elif parts[0] == 'at_rock_sample' and len(parts) == 2:
                 self.initial_at_rock_sample.add(parts[1])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings)

        # --- Extract Relevant State Information ---
        self.have_soil = set() # set of (rover, waypoint)
        self.have_rock = set() # set of (rover, waypoint)
        self.have_image = set() # set of (rover, objective, mode)
        self.communicated_soil = set() # set of waypoint
        self.communicated_rock = set() # set of waypoint
        self.communicated_image = set() # set of (objective, mode)
        self.full_stores = set() # set of store names

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'have_soil_analysis' and len(parts) == 3:
                self.have_soil.add((parts[1], parts[2]))
            elif parts[0] == 'have_rock_analysis' and len(parts) == 3:
                self.have_rock.add((parts[1], parts[2]))
            elif parts[0] == 'have_image' and len(parts) == 4:
                self.have_image.add((parts[1], parts[2], parts[3]))
            elif parts[0] == 'communicated_soil_data' and len(parts) == 2:
                self.communicated_soil.add(parts[1])
            elif parts[0] == 'communicated_rock_data' and len(parts) == 2:
                self.communicated_rock.add(parts[1])
            elif parts[0] == 'communicated_image_data' and len(parts) == 3:
                self.communicated_image.add((parts[1], parts[2]))
            elif parts[0] == 'full' and len(parts) == 2:
                self.full_stores.add(parts[1])


        total_cost = 0  # Initialize action cost counter.

        # Identify unsatisfied goals
        uncomm_soil = {w for g in self.goals if match(g, "communicated_soil_data", w := "*") and w not in self.communicated_soil}
        uncomm_rock = {w for g in self.goals if match(g, "communicated_rock_data", w := "*") and w not in self.communicated_rock}
        uncomm_image = {(o, m) for g in self.goals if match(g, "communicated_image_data", o := "*", m := "*") and (o, m) not in self.communicated_image}

        soil_sampling_needed = False
        rock_sampling_needed = False

        # Calculate cost for each unsatisfied soil goal
        for w in uncomm_soil:
            # Check if sample is already collected by any rover
            # We check against rovers that have stores, as only they can collect samples
            sample_collected = any((r, w) in self.have_soil for r in self.rover_to_stores)

            if sample_collected:
                total_cost += 2 # Nav to comm point + Communicate
            else:
                # Sample is needed. Cost: Nav(w) + Sample + Nav(comm) + Comm
                total_cost += 4
                soil_sampling_needed = True


        # Calculate cost for each unsatisfied rock goal
        for w in uncomm_rock:
            # Check if sample is already collected by any rover
            # We check against rovers that have stores, as only they can collect samples
            sample_collected = any((r, w) in self.have_rock for r in self.rover_to_stores)

            if sample_collected:
                total_cost += 2 # Nav to comm point + Communicate
            else:
                # Sample is needed. Cost: Nav(w) + Sample + Nav(comm) + Comm
                total_cost += 4
                rock_sampling_needed = True


        # Calculate cost for each unsatisfied image goal
        for (o, m) in uncomm_image:
            # Check if image is already taken by any rover
            # We check against rovers that have cameras, as only they can take images
            image_taken = any((r, o, m) in self.have_image for r in self.rover_to_cameras)

            if image_taken:
                total_cost += 2 # Nav to comm point + Communicate
            else:
                # Image needs to be taken. Cost: Nav(cal) + Cal + Nav(img) + Img + Nav(comm) + Comm
                # Check if imaging is possible in principle based on static facts
                imaging_possible = False
                for r in self.equipped_imaging_rovers:
                    for camera in self.rover_to_cameras.get(r, set()):
                        if m in self.camera_to_modes.get(camera, set()):
                            if camera in self.camera_to_target:
                                target = self.camera_to_target[camera]
                                # Check if there's *any* waypoint visible from the target AND *any* waypoint visible from the objective
                                if target in self.target_to_visible_wps and o in self.objective_to_visible_wps:
                                    # Found a rover/camera combo that *could* do this image task
                                    imaging_possible = True
                                    break # Found a possibility, no need to check other cameras/rovers for this goal
                    if imaging_possible:
                        break # Found a possibility, no need to check other rovers for this goal

                if imaging_possible:
                    total_cost += 6
                # else: # If imaging is not possible based on static facts, the goal is unreachable.
                #      # For solvable problems, this case should not occur for goals in the problem.
                #      # If it does, returning a large number or infinity is appropriate.
                #      # For this heuristic, we assume solvable problems and add 0 cost if statically impossible.
                #      # The search will eventually fail if it's truly impossible.


        # Add drop cost if sampling is needed and all relevant rovers have full stores
        if soil_sampling_needed:
            equipped_soil_rover_exists = len(self.equipped_soil_rovers) > 0
            # Check if *all* equipped soil rovers have *all* their stores full
            all_equipped_soil_rovers_have_full_stores = equipped_soil_rover_exists # Assume true initially if rovers exist
            if equipped_soil_rover_exists:
                for r in self.equipped_soil_rovers:
                    # If rover 'r' has any store that is NOT full, then not all equipped rovers have full stores
                    # A rover might have no stores, check for that too.
                    rover_stores = self.rover_to_stores.get(r, set())
                    if rover_stores and any(store not in self.full_stores for store in rover_stores):
                         all_equipped_soil_rovers_have_full_stores = False
                         break
                    # If rover has stores but all are full, the loop continues.
                    # If rover has no stores, it cannot sample anyway, but the check `equipped_soil_rover_exists` handles the impossibility.
            # If equipped_soil_rover_exists is False, all_equipped_soil_rovers_have_full_stores remains False

            if equipped_soil_rover_exists and all_equipped_soil_rovers_have_full_stores:
                 total_cost += 1 # Add cost for one drop action for soil sampling needs

        if rock_sampling_needed:
            equipped_rock_rover_exists = len(self.equipped_rock_rovers) > 0
            all_equipped_rock_rovers_have_full_stores = equipped_rock_rover_exists
            if equipped_rock_rover_exists:
                for r in self.equipped_rock_rovers:
                     rover_stores = self.rover_to_stores.get(r, set())
                     if rover_stores and any(store not in self.full_stores for store in rover_stores):
                         all_equipped_rock_rovers_have_full_stores = False
                         break
            # If equipped_rock_rover_exists is False, all_equipped_rock_rovers_have_full_stores remains False

            if equipped_rock_rover_exists and all_equipped_rock_rovers_have_full_stores:
                 total_cost += 1 # Add cost for one drop action for rock sampling needs


        return total_cost
