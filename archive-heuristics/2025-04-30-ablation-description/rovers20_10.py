from fnmatch import fnmatch
# Assuming Heuristic base class exists as in examples
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError
        def __repr__(self):
             return f"DummyHeuristic(goals={len(self.goals)})"


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully, though PDDL facts are structured.
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The number of parts must match the number of args for fnmatch to work element by element.
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated costs
    for each uncommunicated goal fact. The cost for each goal fact is estimated based
    on whether the necessary sample/image has been acquired and whether calibration
    is needed for images. Movement costs are largely ignored, except implicitly
    by counting steps like 'sample', 'take_image', 'calibrate', 'communicate'.

    # Assumptions
    - The heuristic counts the number of key actions needed for each uncommunicated goal,
      ignoring the rover's current location and the need to navigate.
    - Store capacity is ignored.
    - Camera calibration state is only checked globally (is *any* suitable camera calibrated?),
      not per specific camera instance needed for a goal, and the heuristic doesn't
      account for calibration being consumed by the take_image action.
    - Assumes all necessary objects (equipped rovers, cameras, calibration targets,
      visible waypoints, lander) exist to make the goals achievable in principle.
    - Assumes soil/rock samples initially exist at waypoints specified by `at_soil_sample` / `at_rock_sample`
      facts if they are part of a goal that requires sampling.

    # Heuristic Initialization
    - Extracts static information from the task, such as which rovers have which equipment,
      which cameras are on which rovers and support which modes. This information
      is used to determine if a rover/camera combination is "suitable" for an image goal.
      Other static facts like calibration targets, objective visibility, and lander/communication
      points are not strictly needed for this simplified heuristic logic and are ignored
      during initialization.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each goal fact that is not yet true
    in the current state.

    For each uncommunicated goal `(communicated_soil_data ?w)`:
    1. Add 1 to the heuristic (representing the final `communicate_soil_data` action).
    2. Check if `(have_soil_analysis ?r ?w)` is true for *any* rover `?r` in the current state.
    3. If no rover has the soil analysis for `?w`, add 1 to the heuristic (representing the `sample_soil` action).

    For each uncommunicated goal `(communicated_rock_data ?w)`:
    1. Add 1 to the heuristic (representing the final `communicate_rock_data` action).
    2. Check if `(have_rock_analysis ?r ?w)` is true for *any* rover `?r` in the current state.
    3. If no rover has the rock analysis for `?w`, add 1 to the heuristic (representing the `sample_rock` action).

    For each uncommunicated goal `(communicated_image_data ?o ?m)`:
    1. Add 1 to the heuristic (representing the final `communicate_image_data` action).
    2. Check if `(have_image ?r ?o ?m)` is true for *any* rover `?r` in the current state.
    3. If no rover has the image for `?o` in mode `?m`, add 1 to the heuristic (representing the `take_image` action).
    4. To take an image, a suitable camera must be calibrated. A camera `?i` on rover `?r` is suitable for this goal if `?r` is equipped for imaging, `?i` is on board `?r`, and `?i` supports mode `?m`.
    5. Check if `(calibrated ?i ?r)` is true in the current state for *any* suitable camera `?i` on *any* rover `?r` that could potentially take this image (i.e., `?r` is equipped for imaging, `?i` is on board `?r`, and `?i` supports `?m`).
    6. If no such suitable camera is calibrated, add 1 to the heuristic (representing the `calibrate` action).

    The total heuristic value is the sum of these costs over all uncommunicated goals.
    A heuristic value of 0 means all goal facts are present in the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        # Ensure the base class is initialized correctly
        super().__init__(task)

        # Pre-process static facts for quick lookups
        self.equipped_for_imaging = set()
        self.on_board = {} # camera -> rover
        self.supports = {} # camera -> set of modes

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            if predicate == 'equipped_for_imaging':
                self.equipped_for_imaging.add(parts[1])
            elif predicate == 'on_board':
                self.on_board[parts[1]] = parts[2] # camera -> rover
            elif predicate == 'supports':
                camera, mode = parts[1], parts[2]
                if camera not in self.supports:
                    self.supports[camera] = set()
                self.supports[camera].add(mode)
            # Other static facts like calibration_target, visible_from, at_lander, visible
            # are not used in this simplified heuristic logic.


    def is_suitable_camera_rover(self, camera, rover, mode):
        """
        Checks if a camera on a rover is suitable for taking an image
        in a specific mode, based on static facts.
        Suitability for a *specific* objective/waypoint is ignored here,
        we only check if the camera/rover *can* support the mode and imaging.
        """
        # Check if the rover is equipped for imaging
        if rover not in self.equipped_for_imaging:
            return False
        # Check if the camera is on board the rover
        if self.on_board.get(camera) != rover:
            return False
        # Check if the camera supports the mode
        if camera not in self.supports or mode not in self.supports[camera]:
            return False
        return True


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state
        h_value = 0

        # Check each goal fact
        for goal in self.goals:
            # If the goal fact is already true, it contributes 0 to the heuristic
            if goal in state:
                continue

            # Goal is not in state, calculate its contribution
            # Base cost for the final communication action
            h_value += 1

            parts = get_parts(goal)
            if not parts: continue # Should not happen for goal facts

            predicate = parts[0]

            if predicate == 'communicated_soil_data':
                waypoint = parts[1]
                # Check if soil analysis for this waypoint exists on any rover
                sample_taken = False
                # Iterate through state facts to find (have_soil_analysis ?r waypoint)
                for fact in state:
                    if match(fact, 'have_soil_analysis', '*', waypoint):
                        sample_taken = True
                        break
                if not sample_taken:
                    # Cost for sample_soil action
                    h_value += 1

            elif predicate == 'communicated_rock_data':
                waypoint = parts[1]
                # Check if rock analysis for this waypoint exists on any rover
                sample_taken = False
                # Iterate through state facts to find (have_rock_analysis ?r waypoint)
                for fact in state:
                    if match(fact, 'have_rock_analysis', '*', waypoint):
                        sample_taken = True
                        break
                if not sample_taken:
                    # Cost for sample_rock action
                    h_value += 1

            elif predicate == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                # Check if the image exists on any rover
                image_taken = False
                # Iterate through state facts to find (have_image ?r objective mode)
                for fact in state:
                    if match(fact, 'have_image', '*', objective, mode):
                        image_taken = True
                        break

                if not image_taken:
                    # Cost for take_image action
                    h_value += 1

                    # To take the image, a suitable camera must be calibrated.
                    # Check if *any* suitable camera/rover combination is calibrated in the current state.
                    # A camera/rover is suitable if it can *potentially* take this image based on static facts.
                    any_suitable_camera_calibrated = False
                    # Iterate through all cameras known from static facts (those listed in on_board or supports)
                    # Using self.on_board.keys() gives us cameras that are on a rover.
                    for camera in self.on_board:
                        rover = self.on_board[camera]
                        # Check if this camera/rover is suitable for the current image goal (mode)
                        if self.is_suitable_camera_rover(camera, rover, mode):
                             # Check if this specific camera on this specific rover is calibrated in the current state
                             if f'(calibrated {camera} {rover})' in state:
                                 any_suitable_camera_calibrated = True
                                 break # Found a calibrated suitable camera, no need to check others

                    if not any_suitable_camera_calibrated:
                        # Cost for calibrate action
                        h_value += 1

            # No other goal types defined in the domain

        return h_value
