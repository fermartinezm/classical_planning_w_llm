from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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

    Estimates the number of actions needed to achieve the goal conditions.
    It sums up the estimated cost for each unachieved goal fact.
    The cost for each goal is estimated based on the missing intermediate
    conditions (having data/image, being calibrated) and adds a simplified
    navigation cost (1) if the agent is not currently at a suitable location
    type for the next step required for that goal.

    This heuristic is non-admissible and aims to guide a greedy best-first search.
    It simplifies navigation costs and resource constraints for efficiency.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        super().__init__(task)

        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_stores = {} # rover -> store
        self.lander_waypoint = None
        self.visible_map = {} # waypoint -> set(waypoint)
        self.lander_communication_waypoints = set()
        self.rover_cameras = {} # rover -> set(cameras)
        self.camera_modes = {} # camera -> set(modes)
        self.camera_calibration_target = {} # camera -> objective
        self.objective_visible_from = {} # objective -> set(waypoints)

        # Extract static information
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "equipped_for_soil_analysis":
                self.equipped_soil.add(parts[1])
            elif predicate == "equipped_for_rock_analysis":
                self.equipped_rock.add(parts[1])
            elif predicate == "equipped_for_imaging":
                self.equipped_imaging.add(parts[1])
            elif predicate == "store_of":
                store, rover = parts[1], parts[2]
                self.rover_stores[rover] = store
            elif predicate == "at_lander":
                # Assuming only one lander based on examples
                lander, waypoint = parts[1], parts[2]
                self.lander_waypoint = waypoint
            elif predicate == "visible":
                wp1, wp2 = parts[1], parts[2]
                self.visible_map.setdefault(wp1, set()).add(wp2)
                # Assuming visibility is symmetric, add the reverse edge
                self.visible_map.setdefault(wp2, set()).add(wp1)
            elif predicate == "on_board":
                camera, rover = parts[1], parts[2]
                self.rover_cameras.setdefault(rover, set()).add(camera)
            elif predicate == "supports":
                camera, mode = parts[1], parts[2]
                self.camera_modes.setdefault(camera, set()).add(mode)
            elif predicate == "calibration_target":
                camera, objective = parts[1], parts[2]
                self.camera_calibration_target[camera] = objective
            elif predicate == "visible_from":
                objective, waypoint = parts[1], parts[2]
                self.objective_visible_from.setdefault(objective, set()).add(waypoint)

        # Compute lander communication waypoints
        # A rover at waypoint X can communicate if (visible X lander_waypoint) is true.
        if self.lander_waypoint:
             self.lander_communication_waypoints = self.visible_map.get(self.lander_waypoint, set())


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state
        h = 0

        # Extract relevant dynamic information from the state
        rover_locations = {} # rover -> waypoint
        store_status = {} # store -> 'empty' or 'full'
        have_soil_data = set() # (rover, waypoint)
        have_rock_data = set() # (rover, waypoint)
        have_image_data = set() # (rover, objective, mode)
        calibrated_cameras = set() # (camera, rover)

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at":
                obj, waypoint = parts[1], parts[2]
                # Assuming only rovers have 'at' facts we need to track dynamically
                if obj.startswith("rover"):
                    rover_locations[obj] = waypoint
            elif predicate == "empty":
                store_status[parts[1]] = 'empty'
            elif predicate == "full":
                store_status[parts[1]] = 'full'
            elif predicate == "have_soil_analysis":
                rover, waypoint = parts[1], parts[2]
                have_soil_data.add((rover, waypoint))
            elif predicate == "have_rock_analysis":
                rover, waypoint = parts[1], parts[2]
                have_rock_data.add((rover, waypoint))
            elif predicate == "have_image":
                rover, objective, mode = parts[1], parts[2], parts[3]
                have_image_data.add((rover, objective, mode))
            elif predicate == "calibrated":
                camera, rover = parts[1], parts[2]
                calibrated_cameras.add((camera, rover))


        # Calculate cost for each unachieved goal
        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            # Goal is not achieved, it will eventually require communication.
            # Add cost for communication and navigation to comm point if needed.
            h += 1 # communicate action
            any_rover_at_comm_point = any(loc in self.lander_communication_waypoints for loc in rover_locations.values())
            if not any_rover_at_comm_point:
                h += 1 # navigate to comm point

            # Now add costs for collecting the data/image if not already collected.
            goal_parts = get_parts(goal)
            if not goal_parts: continue # Should not happen for valid goals

            goal_predicate = goal_parts[0]

            if goal_predicate == "communicated_soil_data":
                waypoint_W = goal_parts[1]
                has_soil_data_W = any(w == waypoint_W for r, w in have_soil_data)
                if not has_soil_data_W:
                    h += 1 # sample_soil
                    # Need rover at W for sampling
                    any_rover_at_sample_waypoint = any(loc == waypoint_W for loc in rover_locations.values())
                    if not any_rover_at_sample_waypoint:
                         h += 1 # navigate to W
                    # Need empty store for sampling
                    all_stores_full = all(status == 'full' for status in store_status.values())
                    if all_stores_full:
                         h += 1 # drop

            elif goal_predicate == "communicated_rock_data":
                waypoint_W = goal_parts[1]
                has_rock_data_W = any(w == waypoint_W for r, w in have_rock_data)
                if not has_rock_data_W:
                    h += 1 # sample_rock
                    # Need rover at W for sampling
                    any_rover_at_sample_waypoint = any(loc == waypoint_W for loc in rover_locations.values())
                    if not any_rover_at_sample_waypoint:
                         h += 1 # navigate to W
                    # Need empty store for sampling
                    all_stores_full = all(status == 'full' for status in store_status.values())
                    if all_stores_full:
                         h += 1 # drop

            elif goal_predicate == "communicated_image_data":
                objective_O, mode_M = goal_parts[1], goal_parts[2]
                has_image_OM = any(o == objective_O and m == mode_M for r, o, m in have_image_data)
                if not has_image_OM:
                    h += 1 # take_image
                    # Need rover at image waypoint visible from O
                    image_waypoints_for_O = self.objective_visible_from.get(objective_O, set())
                    any_rover_at_image_waypoint = any(loc in image_waypoints_for_O for loc in rover_locations.values())
                    if not any_rover_at_image_waypoint:
                        h += 1 # navigate to image point

                    # Need calibrated camera
                    # Find any camera I on any equipped rover R that supports M
                    suitable_cameras_rovers = [(c, r) for r in self.equipped_imaging for c in self.rover_cameras.get(r, set()) if mode_M in self.camera_modes.get(c, set())]
                    is_any_suitable_camera_calibrated = any((c, r) in calibrated_cameras for c, r in suitable_cameras_rovers)

                    if not is_any_suitable_camera_calibrated:
                        h += 1 # calibrate
                        # Need rover at calibration waypoint
                        any_rover_at_calib_waypoint = False
                        for c, r in suitable_cameras_rovers:
                             calib_target = self.camera_calibration_target.get(c)
                             if calib_target:
                                 calib_waypoints = self.objective_visible_from.get(calib_target, set())
                                 if any(loc in calib_waypoints for loc in rover_locations.values()):
                                     any_rover_at_calib_waypoint = True
                                     break
                        if not any_rover_at_calib_waypoint:
                             h += 1 # navigate to calib point

        return h
