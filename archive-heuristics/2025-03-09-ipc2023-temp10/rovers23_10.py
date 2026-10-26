from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at package5 city3-2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class rovers23Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goals in the Rovers domain.
    It considers the number of soil samples, rock samples, and images that need to be collected and communicated,
    as well as the need to calibrate cameras and navigate between waypoints.

    # Assumptions
    - Each rover can perform only one task at a time (e.g., sample, image, communicate).
    - Rovers must navigate to a waypoint to perform sampling or imaging.
    - Rovers must be at a waypoint visible to the lander to communicate data.
    - Camera calibration is required before taking images.

    # Heuristic Initialization
    - Extract the goal predicates from the task.
    - Identify the locations of soil and rock samples.
    - Identify the objectives and modes for imaging.
    - Identify the lander location.
    - Store the visibility information between waypoints and objectives.
    - Store the rover equipment information.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Identify the unmet goals related to soil data communication.
       - For each unmet goal, find the closest rover equipped for soil analysis.
       - Estimate the cost as the sum of:
         - Navigation cost to the soil sample location.
         - Cost of sampling soil.
         - Navigation cost to a waypoint visible from the lander.
         - Cost of communicating soil data.
    3. Identify the unmet goals related to rock data communication.
       - For each unmet goal, find the closest rover equipped for rock analysis.
       - Estimate the cost as the sum of:
         - Navigation cost to the rock sample location.
         - Cost of sampling rock.
         - Navigation cost to a waypoint visible from the lander.
         - Cost of communicating rock data.
    4. Identify the unmet goals related to image data communication.
       - For each unmet goal, find the closest rover equipped for imaging.
       - If the camera is not calibrated, estimate the cost of calibration:
         - Navigation cost to a waypoint visible from the objective.
         - Cost of calibrating the camera.
       - Estimate the cost as the sum of:
         - Navigation cost to a waypoint visible from the objective.
         - Cost of taking the image.
         - Navigation cost to a waypoint visible from the lander.
         - Cost of communicating image data.
    5. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract soil and rock sample locations
        self.soil_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_soil_sample", "*")
        }
        self.rock_samples = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "at_rock_sample", "*")
        }

        # Extract objectives and their visibility from waypoints
        self.objective_visibility = {}
        for fact in static_facts:
            if match(fact, "visible_from", "*", "*"):
                objective = get_parts(fact)[1]
                waypoint = get_parts(fact)[2]
                if objective not in self.objective_visibility:
                    self.objective_visibility[objective] = set()
                self.objective_visibility[objective].add(waypoint)

        # Extract rover equipment information
        self.rovers_equipped_for_soil = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_soil_analysis", "*")
        }
        self.rovers_equipped_for_rock = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_rock_analysis", "*")
        }
        self.rovers_equipped_for_imaging = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "equipped_for_imaging", "*")
        }

        # Extract lander location
        self.lander_location = next(
            (get_parts(fact)[2] for fact in static_facts if match(fact, "at_lander", "*", "*")), None
        )

        # Extract waypoint visibility
        self.waypoint_visibility = {}
        for fact in static_facts:
            if match(fact, "visible", "*", "*"):
                waypoint1 = get_parts(fact)[1]
                waypoint2 = get_parts(fact)[2]
                if waypoint1 not in self.waypoint_visibility:
                    self.waypoint_visibility[waypoint1] = set()
                self.waypoint_visibility[waypoint1].add(waypoint2)

        # Extract camera calibration targets
        self.camera_calibration_targets = {}
        for fact in static_facts:
            if match(fact, "calibration_target", "*", "*"):
                camera = get_parts(fact)[1]
                objective = get_parts(fact)[2]
                self.camera_calibration_targets[camera] = objective

        # Extract camera onboard information
        self.camera_onboard = {}
        for fact in static_facts:
            if match(fact, "on_board", "*", "*"):
                camera = get_parts(fact)[1]
                rover = get_parts(fact)[2]
                self.camera_onboard[camera] = rover

    def __call__(self, node):
        """Estimate the cost to reach the goal state from the current state."""
        state = node.state

        if self.goals <= state:
            return 0

        cost = 0

        # Unmet soil data communication goals
        unmet_soil_goals = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "communicated_soil_data", "*") and goal not in state
        }
        for waypoint in unmet_soil_goals:
            best_rover = None
            min_dist = float('inf')
            for rover in self.rovers_equipped_for_soil:
                if f"(at {rover} {waypoint})" in state:
                    best_rover = rover
                    min_dist = 0
                    break
                else:
                    current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
                    if current_location:
                        dist = self.shortest_path(current_location, waypoint)
                        if dist < min_dist:
                            min_dist = dist
                            best_rover = rover
            if best_rover:
                current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", best_rover, "*")), None)
                if current_location:
                    cost += self.shortest_path(current_location, waypoint)
                    cost += 1  # Sample soil
                    lander_wp = next((wp for wp in self.waypoint_visibility.get(waypoint, []) if wp == self.lander_location), None)
                    if not lander_wp:
                        cost += 5 # TODO: Find closest waypoint visible from lander
                    cost += 1  # Communicate soil data

        # Unmet rock data communication goals
        unmet_rock_goals = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "communicated_rock_data", "*") and goal not in state
        }
        for waypoint in unmet_rock_goals:
            best_rover = None
            min_dist = float('inf')
            for rover in self.rovers_equipped_for_rock:
                if f"(at {rover} {waypoint})" in state:
                    best_rover = rover
                    min_dist = 0
                    break
                else:
                    current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
                    if current_location:
                        dist = self.shortest_path(current_location, waypoint)
                        if dist < min_dist:
                            min_dist = dist
                            best_rover = rover
            if best_rover:
                current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", best_rover, "*")), None)
                if current_location:
                    cost += self.shortest_path(current_location, waypoint)
                    cost += 1  # Sample rock
                    lander_wp = next((wp for wp in self.waypoint_visibility.get(waypoint, []) if wp == self.lander_location), None)
                    if not lander_wp:
                        cost += 5 # TODO: Find closest waypoint visible from lander
                    cost += 1  # Communicate rock data

        # Unmet image data communication goals
        unmet_image_goals = set()
        for goal in self.goals:
            if match(goal, "communicated_image_data", "*", "*") and goal not in state:
                unmet_image_goals.add(goal)

        for goal in unmet_image_goals:
            objective = get_parts(goal)[1]
            mode = get_parts(goal)[2]

            best_rover = None
            min_dist = float('inf')
            for rover in self.rovers_equipped_for_imaging:
                current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", rover, "*")), None)
                if current_location:
                    dist = self.shortest_path(current_location, next(iter(self.objective_visibility[objective])), current_location)
                    if dist < min_dist:
                        min_dist = dist
                        best_rover = rover

            if best_rover:
                camera = next((cam for cam, rov in self.camera_onboard.items() if rov == best_rover), None)
                if camera:
                    if f"(calibrated {camera} {best_rover})" not in state:
                        # Need to calibrate the camera
                        cost += 1  # Navigate to calibration waypoint
                        cost += 1  # Calibrate camera

                    current_location = next((get_parts(fact)[2] for fact in state if match(fact, "at", best_rover, "*")), None)
                    if current_location:
                        objective_wp = next(iter(self.objective_visibility[objective]))
                        cost += self.shortest_path(current_location, objective_wp)  # Navigate to objective
                        cost += 1  # Take image
                        lander_wp = next((wp for wp in self.waypoint_visibility.get(objective_wp, []) if wp == self.lander_location), None)
                        if not lander_wp:
                            cost += 5 # TODO: Find closest waypoint visible from lander
                        cost += 1  # Communicate image data

        return cost

    def shortest_path(self, start, end, current_location=None):
        """
        Estimate the shortest path between two waypoints.
        This is a very basic heuristic and can be improved.
        """
        if start == end:
            return 0

        if current_location is None:
            return 10

        if end in self.waypoint_visibility.get(start, []):
            return 1

        return 5  # A higher cost if not directly visible

