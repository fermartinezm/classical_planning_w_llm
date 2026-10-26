from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the remaining effort by counting the number of
    unmet goal conditions, adding costs for the necessary actions to achieve
    them, specifically focusing on communication, data collection (sampling/imaging),
    and a simplified view of calibration requirements. It ignores navigation costs
    and store management for simplicity.

    # Assumptions
    - Each unmet communication goal requires one communication action.
    - If the required data (soil, rock, image) for a communication goal is not
      yet collected, one data collection action (sample_soil, sample_rock,
      take_image) is needed.
    - Taking an image requires a calibrated camera. Calibration cost is added
      once for each image mode required by an un-taken image goal, if no camera
      capable of that mode is currently calibrated on an imaging-equipped rover.
    - Navigation costs are ignored.
    - Store capacity and dropping samples are ignored.
    - The existence of suitable rovers, cameras, calibration targets, and
      visible waypoints for actions is assumed if the static facts allow it.

    # Heuristic Initialization
    - Extracts static information about camera capabilities (modes supported),
      camera placement on rovers, and which rovers are equipped for imaging.
      This information is used to determine which cameras can take images in
      specific modes and which rovers can perform imaging tasks.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Identify the set of goal facts and the set of facts true in the current state.
    3. Pre-process static facts during initialization to build lookups for
       camera capabilities, camera-rover assignments, and rover imaging equipment.
    4. In the heuristic call:
       a. Identify all 'have_soil_analysis', 'have_rock_analysis', and 'have_image'
          facts currently true in the state.
       b. Identify all 'calibrated' camera-rover pairs currently true in the state.
       c. Determine which image modes are supported by currently calibrated cameras
          that are on imaging-equipped rovers, considering static facts.
       d. Initialize a set to track image modes required by unmet, un-taken image goals.
       e. Iterate through each goal fact:
          i. If the goal fact is already in the current state, it contributes 0 to the heuristic.
          ii. If the goal is '(communicated_soil_data W)' and not in state:
              - Add 1 (for the 'communicate_soil_data' action).
              - If '(have_soil_analysis R W)' is not true for any rover R, add 1
                (for the 'sample_soil' action).
          iii. If the goal is '(communicated_rock_data W)' and not in state:
              - Add 1 (for the 'communicated_rock_data' action).
              - If '(have_rock_analysis R W)' is not true for any rover R, add 1
                (for the 'sample_rock' action).
          iv. If the goal is '(communicated_image_data O M)' and not in state:
              - Add 1 (for the 'communicated_image_data' action).
              - Check if '(have_image R O M)' is true for any rover R.
              - If not true for any R:
                  - Add 1 (for the 'take_image' action).
                  - Add mode M to the set of modes required by unmet, un-taken image goals.
       f. After iterating through all goals, determine which modes in the set
          from step 4d do *not* have a currently calibrated camera capable of
          supporting them (based on static facts and current calibrated status).
       g. Add the count of these modes to the total heuristic cost (each represents
          a required calibration effort for that mode).
    5. Return the total calculated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        # The task object is expected to have 'goals' and 'static' attributes
        self.goals = task.goals
        self.static_facts = task.static # Store static facts for potential future use, although processed below

        # Static lookups
        self.camera_supports_mode = {} # camera -> set of modes
        self.camera_on_rover = {}      # camera -> rover
        self.rover_equipped_for_imaging = set() # rover names

        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts or len(parts) < 2: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "supports" and len(parts) == 3:
                camera, mode = parts[1], parts[2]
                if camera not in self.camera_supports_mode:
                    self.camera_supports_mode[camera] = set()
                self.camera_supports_mode[camera].add(mode)
            elif predicate == "on_board" and len(parts) == 3:
                camera, rover = parts[1], parts[2]
                self.camera_on_rover[camera] = rover
            elif predicate == "equipped_for_imaging" and len(parts) == 2:
                rover = parts[1]
                self.rover_equipped_for_imaging.add(rover)

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Identify current state facts relevant to heuristic calculation
        have_soil_analysis_waypoints = set()
        have_rock_analysis_waypoints = set()
        have_image_goals = set() # (objective, mode) pairs
        calibrated_cameras_on_rovers = set() # (camera, rover) pairs

        for fact in state:
            parts = get_parts(fact)
            if not parts or len(parts) < 2: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "have_soil_analysis" and len(parts) == 3:
                # fact is like (have_soil_analysis rover1 waypoint4)
                waypoint = parts[2]
                have_soil_analysis_waypoints.add(waypoint)
            elif predicate == "have_rock_analysis" and len(parts) == 3:
                 # fact is like (have_rock_analysis rover1 waypoint4)
                waypoint = parts[2]
                have_rock_analysis_waypoints.add(waypoint)
            elif predicate == "have_image" and len(parts) == 4:
                 # fact is like (have_image rover1 objective1 colour)
                objective, mode = parts[2], parts[3]
                have_image_goals.add((objective, mode))
            elif predicate == "calibrated" and len(parts) == 3:
                 # fact is like (calibrated camera1 rover1)
                camera, rover = parts[1], parts[2]
                calibrated_cameras_on_rovers.add((camera, rover))

        total_cost = 0
        modes_needed_for_unmet_images = set()

        for goal in self.goals:
            # Check if the goal is already satisfied
            if goal in state:
                continue

            # Parse the goal fact
            parts = get_parts(goal)
            if not parts or len(parts) < 2: continue # Skip malformed goals

            predicate = parts[0]

            if predicate == "communicated_soil_data" and len(parts) == 2:
                # Goal: (communicated_soil_data W)
                waypoint = parts[1]
                total_cost += 1 # Cost for communicate action
                # Check if soil analysis data is available for this waypoint
                if waypoint not in have_soil_analysis_waypoints:
                    total_cost += 1 # Cost for sample_soil action

            elif predicate == "communicated_rock_data" and len(parts) == 2:
                # Goal: (communicated_rock_data W)
                waypoint = parts[1]
                total_cost += 1 # Cost for communicate action
                # Check if rock analysis data is available for this waypoint
                if waypoint not in have_rock_analysis_waypoints:
                    total_cost += 1 # Cost for sample_rock action

            elif predicate == "communicated_image_data" and len(parts) == 3:
                # Goal: (communicated_image_data O M)
                objective, mode = parts[1], parts[2]
                total_cost += 1 # Cost for communicate action
                # Check if the image is already taken
                if (objective, mode) not in have_image_goals:
                    total_cost += 1 # Cost for take_image action
                    # Add this mode to the set of modes needed for un-taken images
                    modes_needed_for_unmet_images.add(mode)

        # Calculate calibration cost
        # Find all modes supported by currently calibrated cameras on imaging-equipped rovers
        modes_with_calibrated_camera = set()
        for camera, rover in calibrated_cameras_on_rovers:
            # Check if the rover is equipped for imaging (static fact)
            # and if the camera is on board that rover (static fact) - the latter is implicit in calibrated_cameras_on_rovers
            # and if the camera supports any modes (static fact)
            if rover in self.rover_equipped_for_imaging and camera in self.camera_supports_mode:
                 modes_with_calibrated_camera.update(self.camera_supports_mode[camera])

        # We need to consider only modes that *can* be supported by *some* camera
        # on *some* imaging-equipped rover, and are needed for unmet images.
        all_possible_imaging_modes = set()
        for camera, modes in self.camera_supports_mode.items():
             if camera in self.camera_on_rover:
                 rover = self.camera_on_rover[camera]
                 if rover in self.rover_equipped_for_imaging:
                     all_possible_imaging_modes.update(modes)

        # We only care about modes needed for unmet images that are actually possible
        # to achieve through imaging.
        relevant_modes_needed = modes_needed_for_unmet_images.intersection(all_possible_imaging_modes)

        # Calibration is needed for modes in relevant_modes_needed that are NOT
        # covered by currently calibrated cameras.
        modes_actually_requiring_calibration = relevant_modes_needed - modes_with_calibrated_camera

        total_cost += len(modes_actually_requiring_calibration)

        return total_cost
