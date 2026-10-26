from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    Estimates the number of actions required to reach the goal state.
    It sums up the estimated costs for each unachieved goal fact,
    considering necessary intermediate steps like sampling, imaging,
    calibration, and navigation.

    It is not admissible but aims to be informative for greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.lander_waypoint = None
        self.comm_points = set()
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # rover -> store
        self.rover_cameras = {} # rover -> set of cameras
        self.camera_modes = {} # camera -> set of modes
        self.camera_cal_targets = {} # camera -> objective (calibration target)
        self.obj_visible_waypoints = {} # objective -> set of waypoints

        # First pass to find lander waypoint
        for fact in static_facts:
            if match(fact, "at_lander", "*", "*"):
                self.lander_waypoint = get_parts(fact)[1]
                break # Assuming only one lander

        # Second pass to parse other static facts, including comm points now that lander_waypoint is known
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == "visible":
                 # (visible ?w - waypoint ?p - waypoint)
                 w1, w2 = args
                 # A waypoint X is a communication point if a rover at X can see the lander at Y.
                 # Precondition: (visible ?x ?y) where ?x is rover loc, ?y is lander loc.
                 # So, X is a comm point if (visible X lander_waypoint).
                 if self.lander_waypoint and w2 == self.lander_waypoint:
                      self.comm_points.add(w1)
                 # Also, if the rover is AT the lander waypoint, it can communicate if (visible lander_waypoint lander_waypoint)
                 # is true. Let's add lander_waypoint itself if it's visible to itself.
                 if self.lander_waypoint and w1 == self.lander_waypoint and w2 == self.lander_waypoint:
                      self.comm_points.add(self.lander_waypoint)

            elif predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])
            elif predicate == "store_of":
                # (store_of ?s - store ?r - rover)
                store, rover = args
                self.rover_stores[rover] = store
            elif predicate == "on_board":
                # (on_board ?i - camera ?r - rover)
                camera, rover = args
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif predicate == "supports":
                # (supports ?c - camera ?m - mode)
                camera, mode = args
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                # (calibration_target ?i - camera ?o - objective)
                camera, objective = args
                self.camera_cal_targets[camera] = objective
            elif predicate == "visible_from":
                # (visible_from ?o - objective ?w - waypoint)
                objective, waypoint = args
                self.obj_visible_waypoints.setdefault(objective, set()).add(waypoint)

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state
        goals = self.goals
        h = 0

        # Track needs that require specific resources/locations
        needed_soil_sample_waypoints = set() # Waypoints W where soil sample is needed
        needed_rock_sample_waypoints = set() # Waypoints W where rock sample is needed
        needed_image_objectives_modes = set() # (O, M) pairs where image is needed

        # First pass: Identify missing communication goals and required intermediate steps
        for g in goals:
            if g in state:
                continue # Goal already achieved

            h += 1 # Count the final communication action
            h += 1 # Navigation to comm point (estimate)

            if match(g, "communicated_soil_data", "*"):
                W = get_parts(g)[1]
                # Check if (have_soil_analysis R W) exists for any equipped rover R
                have_fact_exists = False
                for rover in self.equipped_soil:
                    if f"(have_soil_analysis {rover} {W})" in state:
                        have_fact_exists = True
                        break
                if not have_fact_exists:
                    needed_soil_sample_waypoints.add(W)

            elif match(g, "communicated_rock_data", "*"):
                W = get_parts(g)[1]
                # Check if (have_rock_analysis R W) exists for any equipped rover R
                have_fact_exists = False
                for rover in self.equipped_rock:
                    if f"(have_rock_analysis {rover} {W})" in state:
                        have_fact_exists = True
                        break
                if not have_fact_exists:
                    needed_rock_sample_waypoints.add(W)

            elif match(g, "communicated_image_data", "*", "*"):
                O, M = get_parts(g)[1:]
                # Check if (have_image R O M) exists for any equipped rover R with camera I supporting M
                have_fact_exists = False
                for rover in self.equipped_imaging:
                    if rover in self.rover_cameras:
                        for camera in self.rover_cameras[rover]:
                            if camera in self.camera_modes and M in self.camera_modes[camera]:
                                if f"(have_image {rover} {O} {M})" in state:
                                    have_fact_exists = True
                                    break
                    if have_fact_exists: break # Found the image

                if not have_fact_exists:
                    needed_image_objectives_modes.add((O, M))

        # Second pass: Process intermediate needs (sampling, imaging, calibration)

        # Sample costs
        N_samples_needed = len(needed_soil_sample_waypoints) + len(needed_rock_sample_waypoints)
        h += N_samples_needed # `sample_soil` or `sample_rock` action for each needed sample
        h += N_samples_needed # Navigation to sample point for each needed sample (estimate)

        # Drop costs for samples
        if N_samples_needed > 0:
            # Count equipped rovers with empty stores
            N_equipped_rovers_with_empty_store = 0
            for rover in self.equipped_soil | self.equipped_rock:
                 if rover in self.rover_stores:
                     store = self.rover_stores[rover]
                     if f"(empty {store})" in state:
                         N_equipped_rovers_with_empty_store += 1

            # Estimate drops needed: total samples needed minus initially empty stores.
            # This assumes optimal assignment and that communications happen after all samples by a rover.
            # This is a lower bound on drops needed to free up store capacity for sampling.
            h += max(0, N_samples_needed - N_equipped_rovers_with_empty_store)


        # Image and Calibration costs
        for O, M in needed_image_objectives_modes:
            h += 1 # `take_image` action
            h += 1 # Navigation to image point (waypoint visible from O) (estimate)

            # Check if calibration is needed for this specific image need (O, M)
            # Find a suitable camera I on an equipped rover R supporting M
            calibration_needed_for_this_image = True
            for rover in self.equipped_imaging:
                 if rover in self.rover_cameras:
                     for camera in self.rover_cameras[rover]:
                         if camera in self.camera_modes and M in self.camera_modes[camera]:
                             # Found a suitable camera I on rover R for this image (O, M)
                             # Check if (calibrated I R) exists in state
                             if f"(calibrated {camera} {rover})" in state:
                                 calibration_needed_for_this_image = False
                                 break # Found a calibrated camera for this image need
                     if not calibration_needed_for_this_image: break # Found a suitable calibrated camera on this rover
            if calibration_needed_for_this_image:
                 h += 1 # `calibrate` action
                 h += 1 # Navigation to calibration target point (estimate)


        # Ensure heuristic is 0 only at goal
        # The initial check `if g in state: continue` ensures h is 0 if all goals are met.
        # If goals are not met, h will be >= 1 (at least the communication action).
        # So h is 0 iff goals <= state.

        return h
