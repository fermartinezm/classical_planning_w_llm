from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string or invalid format gracefully, though PDDL facts are structured.
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to achieve all goal conditions.
    The heuristic sums the estimated costs for each unachieved goal independently.

    For each unachieved goal (communicated_X_data ...):
    1. Estimate the minimum cost to collect the required data (if not already collected).
       - Soil/Rock: Navigate to sample WP + (Drop if store full) + Sample.
       - Image: Navigate to Image WP + (Navigate to Cal WP + Calibrate if needed) + TakeImage.
       - Minimum cost is taken over all capable rovers/cameras.
    2. Estimate the minimum cost to communicate the data.
       - Navigate to Communication WP + Communicate.
       - Minimum cost is taken over all rovers that have/can collect the data.
    3. The total cost for the goal is the sum of collection and communication costs.
    4. The heuristic value is the sum of costs for all unachieved goals.

    Navigation cost is estimated as 1 if the rover is not currently at a suitable location, 0 otherwise.
    Other actions (Sample, Drop, Calibrate, TakeImage, Communicate) cost 1.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Parse static facts
        self.lander_location = None
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_store = {} # rover -> store
        self.visible_graph = {} # wp -> set of visible wps (wp_from -> {wp_to, ...})
        self.camera_on_rover = {} # camera -> rover
        self.camera_supports = {} # camera -> set of modes
        self.camera_calibration_target = {} # camera -> objective (target)
        self.objective_visible_from = {} # objective -> set of wps (objective is visible from these wps)

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_lander":
                if len(parts) == 3:
                    self.lander_location = parts[2]
            elif predicate == "equipped_for_soil_analysis":
                if len(parts) == 2:
                    self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                if len(parts) == 2:
                    self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                if len(parts) == 2:
                    self.equipped_imaging.add(parts[1])
            elif predicate == "store_of":
                if len(parts) == 3:
                    self.rover_store[parts[2]] = parts[1] # rover -> store
            elif predicate == "visible":
                if len(parts) == 3:
                    wp1, wp2 = parts[1], parts[2]
                    self.visible_graph.setdefault(wp1, set()).add(wp2)
            elif predicate == "on_board":
                if len(parts) == 3:
                    self.camera_on_rover[parts[1]] = parts[2] # camera -> rover
            elif predicate == "supports":
                if len(parts) == 3:
                    self.camera_supports.setdefault(parts[1], set()).add(parts[2]) # camera -> mode
            elif predicate == "calibration_target":
                if len(parts) == 3:
                    self.camera_calibration_target[parts[1]] = parts[2] # camera -> target objective
            elif predicate == "visible_from":
                if len(parts) == 3:
                    obj, wp = parts[1], parts[2]
                    self.objective_visible_from.setdefault(obj, set()).add(wp)


        # Pre-calculate communication waypoints (visible from lander)
        self.comm_wps = set()
        if self.lander_location:
             # Find all waypoints W such that (visible W lander_location) is true
             for wp_from, wps_to in self.visible_graph.items():
                 if self.lander_location in wps_to:
                     self.comm_wps.add(wp_from)

        # Identify all rovers from static info
        self.all_rovers = self.equipped_soil | self.equipped_rock | self.equipped_imaging


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse state facts
        rover_location = {} # rover -> wp
        store_status = {} # store -> 'empty' or 'full'
        have_soil = set() # (rover, wp)
        have_rock = set() # (rover, wp)
        have_image = set() # (rover, obj, mode)
        communicated_soil = set() # wp
        communicated_rock = set() # wp
        communicated_image = set() # (obj, mode)
        soil_samples_at = set() # wp
        rock_samples_at = set() # wp
        calibrated_cameras = set() # (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                if len(parts) == 3:
                    obj, loc = parts[1], parts[2]
                    if obj in self.all_rovers:
                         rover_location[obj] = loc
            elif predicate == "empty":
                if len(parts) == 2:
                    store_status[parts[1]] = 'empty'
            elif predicate == "full":
                if len(parts) == 2:
                    store_status[parts[1]] = 'full'
            elif predicate == "have_soil_analysis":
                if len(parts) == 3:
                    have_soil.add((parts[1], parts[2]))
            elif predicate == "have_rock_analysis":
                if len(parts) == 3:
                    have_rock.add((parts[1], parts[2]))
            elif predicate == "have_image":
                if len(parts) == 4:
                    have_image.add((parts[1], parts[2], parts[3]))
            elif predicate == "communicated_soil_data":
                if len(parts) == 2:
                    communicated_soil.add(parts[1])
            elif predicate == "communicated_rock_data":
                if len(parts) == 2:
                    communicated_rock.add(parts[1])
            elif predicate == "communicated_image_data":
                if len(parts) == 3:
                    communicated_image.add((parts[1], parts[2]))
            elif predicate == "at_soil_sample":
                if len(parts) == 2:
                    soil_samples_at.add(parts[1])
            elif predicate == "at_rock_sample":
                if len(parts) == 2:
                    rock_samples_at.add(parts[1])
            elif predicate == "calibrated":
                if len(parts) == 3:
                    calibrated_cameras.add((parts[1], parts[2]))

        total_cost = 0

        # Process each goal
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]

            if predicate == "communicated_soil_data":
                if len(parts) != 2: continue
                wp = parts[1]
                if wp not in communicated_soil:
                    goal_cost = 0
                    data_collected = any((r, wp) in have_soil for r in self.equipped_soil)

                    # Cost to collect (if needed)
                    if not data_collected:
                        min_coll_cost = float('inf')
                        if wp in soil_samples_at: # Sample must be available to collect
                            for r in self.equipped_soil:
                                if r in rover_location: # Rover must exist and have a location
                                    current_coll_cost = 0
                                    if rover_location[r] != wp:
                                        current_coll_cost += 1 # Navigate to sample WP
                                    rover_s = self.rover_store.get(r)
                                    if rover_s and store_status.get(rover_s, 'empty') == 'full':
                                        current_coll_cost += 1 # Drop
                                    current_coll_cost += 1 # Sample
                                    min_coll_cost = min(min_coll_cost, current_coll_cost)

                        # If sample is gone but data not collected, assume it was collected
                        if wp not in soil_samples_at:
                             data_collected = True

                        if not data_collected: # If data still not considered collected
                             if min_coll_cost == float('inf'):
                                 # Cannot collect - should not happen in solvable problems if sample is there
                                 continue # Skip goal contribution
                             goal_cost += min_coll_cost

                    # Cost to communicate
                    # Candidate rovers are those that have the data or could collect it
                    rovers_for_comm = set()
                    if data_collected:
                         # If data is collected (either initially or because sample is gone),
                         # only rovers that *have* the data can communicate it.
                         rovers_for_comm = {r for r, current_wp in have_soil if current_wp == wp}
                    else:
                         # If data needs collecting, any equipped rover could do it and then communicate.
                         rovers_for_comm = self.equipped_soil

                    if not rovers_for_comm:
                         # Cannot communicate - should not happen in solvable problems
                         continue # Skip goal contribution

                    comm_cost = 2 # Navigate + Communicate
                    if any(r in rover_location and rover_location[r] in self.comm_wps for r in rovers_for_comm):
                         comm_cost = 1 # Just Communicate

                    goal_cost += comm_cost
                    total_cost += goal_cost

            elif predicate == "communicated_rock_data":
                if len(parts) != 2: continue
                wp = parts[1]
                if wp not in communicated_rock:
                    goal_cost = 0
                    data_collected = any((r, wp) in have_rock for r in self.equipped_rock)
                    rovers_for_comm = set()

                    # Cost to collect (if needed)
                    if not data_collected:
                        min_coll_cost = float('inf')
                        if wp in rock_samples_at: # Sample must be available
                            for r in self.equipped_rock:
                                if r in rover_location:
                                    current_coll_cost = 0
                                    if rover_location[r] != wp:
                                        current_coll_cost += 1 # Navigate to sample WP
                                    rover_s = self.rover_store.get(r)
                                    if rover_s and store_status.get(rover_s, 'empty') == 'full':
                                        current_coll_cost += 1 # Drop
                                    current_coll_cost += 1 # Sample
                                    min_coll_cost = min(min_coll_cost, current_coll_cost)

                        if wp not in rock_samples_at:
                             data_collected = True

                        if not data_collected:
                             if min_coll_cost == float('inf'): continue
                             goal_cost += min_coll_cost

                    # Cost to communicate
                    if data_collected:
                         rovers_for_comm = {r for r, current_wp in have_rock if current_wp == wp}
                    else:
                         rovers_for_comm = self.equipped_rock

                    if not rovers_for_comm: continue

                    comm_cost = 2 # Navigate + Communicate
                    if any(r in rover_location and rover_location[r] in self.comm_wps for r in rovers_for_comm):
                         comm_cost = 1 # Just Communicate

                    goal_cost += comm_cost
                    total_cost += goal_cost

            elif predicate == "communicated_image_data":
                if len(parts) != 3: continue
                obj, mode = parts[1], parts[2]
                if (obj, mode) not in communicated_image:
                    goal_cost = 0
                    data_collected = any((r, obj, mode) in have_image for r in self.equipped_imaging)
                    rovers_for_comm = set()

                    # Cost to collect (if needed)
                    if not data_collected:
                        min_coll_cost = float('inf')
                        image_wps = self.objective_visible_from.get(obj, set())
                        if not image_wps:
                            # Cannot image this objective (no visible_from location)
                            continue # Skip this goal's contribution

                        for r in self.equipped_imaging:
                            if r in rover_location:
                                for i in self.camera_on_rover:
                                    if self.camera_on_rover[i] == r and mode in self.camera_supports.get(i, set()):
                                        current_coll_cost = 0

                                        # Navigation to image WP
                                        if rover_location[r] not in image_wps:
                                            current_coll_cost += 1

                                        # Calibration cost (if needed)
                                        cal_target = self.camera_calibration_target.get(i)
                                        if cal_target is None:
                                            # Camera cannot be calibrated (no target)
                                            continue

                                        if (i, r) not in calibrated_cameras:
                                            cal_wps = self.objective_visible_from.get(cal_target, set()) # Calibration targets are also objectives
                                            if not cal_wps:
                                                # Cannot calibrate this camera (no visible target wp)
                                                continue
                                            if rover_location[r] not in cal_wps:
                                                current_coll_cost += 1 # Navigate to calibration WP
                                            current_coll_cost += 1 # Calibrate

                                        current_coll_cost += 1 # Take image
                                        min_coll_cost = min(min_coll_cost, current_coll_cost)

                        if min_coll_cost == float('inf'):
                            # Cannot collect (e.g., no equipped rover, no suitable camera, no visible_from for obj/target)
                            continue # Skip this goal's contribution
                        else:
                            goal_cost += min_coll_cost

                    # Cost to communicate
                    if data_collected:
                         rovers_for_comm = {r for r, current_obj, current_mode in have_image if current_obj == obj and current_mode == mode}
                    else:
                         rovers_for_comm = self.equipped_imaging

                    if not rovers_for_comm: continue

                    comm_cost = 2 # Navigate + Communicate
                    if any(r in rover_location and rover_location[r] in self.comm_wps for r in rovers_for_comm):
                         comm_cost = 1 # Just Communicate

                    goal_cost += comm_cost
                    total_cost += goal_cost

        return total_cost
