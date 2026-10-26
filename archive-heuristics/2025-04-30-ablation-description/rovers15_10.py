from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Helper functions
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


# Define a dummy Heuristic base class if not provided by the environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            # Assume task object has initial_state, goals, static

        def __call__(self, node):
            # node.state is the current state (frozenset of facts)
            pass


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated costs
    for each unsatisfied goal condition. The estimated cost for a single goal condition
    is based on the necessary actions and required locations to achieve that condition,
    adding a cost of 1 for each major action step (sample, take_image, calibrate, communicate)
    and 1 for each required move if a capable agent is not already at a suitable location.

    # Assumptions
    - The heuristic assumes a cost of 1 for any move action and any other action.
    - It simplifies complex interactions (like store management beyond needing an empty one for sampling,
      or camera calibration state across multiple image goals) by checking for the *existence*
      of a required state (e.g., *any* equipped rover has an empty store, *any* suitable camera is calibrated)
      rather than tracking specific objects.
    - It checks for static unreachability (e.g., objective not visible from any waypoint) and returns infinity.
    - It assumes that if data/image is not yet acquired, any capable rover can acquire it and move to the communication point.
    - It assumes that if data/image is acquired, the specific rover holding it must move to the communication point if not already there.

    # Heuristic Initialization
    The constructor extracts and stores static information from the task, including:
    - The set of all rovers, waypoints, stores, cameras, modes, landers, and objectives.
    - The lander's location.
    - Static predicate facts like `equipped_for_soil_analysis`, `equipped_for_rock_analysis`,
      `equipped_for_imaging`, `store_of`, `visible`, `can_traverse`, `calibration_target`,
      `on_board`, `supports`, `visible_from`, `at_lander`. This information is used
      during the heuristic calculation to check capabilities and potential locations.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value for a state is the sum of costs for each goal fact that is not true
    in the current state.

    For each unsatisfied goal `g`:
    1.  Initialize the cost for this goal to 0.
    2.  **Communication Cost:** Add 1 for the `communicate` action itself.
    3.  **Data/Image Acquisition Cost:**
        -   If `g` is `(communicated_soil_data ?w)` and `(have_soil_analysis ?r ?w)` is not true for any rover `?r`:
            -   Add 1 for the `sample_soil` action.
            -   Check if any soil-equipped rover is currently at waypoint `?w`. If not, add 1 for a `navigate` action to get there.
            -   Check if any soil-equipped rover has an empty store. If not, add 1 for a `drop` action (assuming one drop is sufficient).
        -   If `g` is `(communicated_rock_data ?w)` and `(have_rock_analysis ?r ?w)` is not true for any rover `?r`:
            -   Add 1 for the `sample_rock` action.
            -   Check if any rock-equipped rover is currently at waypoint `?w`. If not, add 1 for a `navigate` action.
            -   Check if any rock-equipped rover has an empty store. If not, add 1 for a `drop` action.
        -   If `g` is `(communicated_image_data ?o ?m)` and `(have_image ?r ?o ?m)` is not true for any rover `?r`:
            -   Add 1 for the `take_image` action.
            -   Identify rovers with cameras supporting mode `?m` and equipped for imaging.
            -   Check if any such rover is at a waypoint from which objective `?o` is visible. If not, add 1 for a `navigate` action to an image waypoint. If no such waypoint exists statically, the goal is unreachable (return infinity).
            -   Check if any such rover/camera pair is currently calibrated. If not:
                -   Add 1 for the `calibrate` action.
                -   Identify calibration targets for the suitable cameras.
                -   Check if any such rover/camera pair is at a waypoint from which any of its calibration targets are visible. If not, add 1 for a `navigate` action to a calibration waypoint. If no such waypoint exists statically, the goal is unreachable (return infinity).
    4.  **Communication Location Cost:**
        -   Check if any rover that *has* the required data/image (if acquired) or *any capable rover* (if not acquired) is currently at a waypoint visible from the lander. If not, add 1 for a `navigate` action to a communication waypoint. If no waypoint is visible from the lander statically, the goal is unreachable (return infinity).
    5.  Add the calculated cost for this goal to the total heuristic value.

    The total heuristic value is the sum of costs for all unsatisfied goals.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract objects by type from static and initial facts
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()
        self.lander_location = None

        # Combine initial state and static facts to find all objects
        all_facts = set(task.initial_state) | set(task.static)

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            pred = parts[0]
            args = parts[1:]

            # Infer object types based on predicate arguments
            if pred == 'at' and len(args) == 2:
                 self.rovers.add(args[0])
                 self.waypoints.add(args[1])
            elif pred == 'at_lander' and len(args) == 2:
                 self.landers.add(args[0])
                 self.waypoints.add(args[1])
                 self.lander_location = args[1] # Assuming only one lander
            elif pred == 'can_traverse' and len(args) == 3:
                 self.rovers.add(args[0])
                 self.waypoints.add(args[1])
                 self.waypoints.add(args[2])
            elif pred in ['equipped_for_soil_analysis', 'equipped_for_rock_analysis', 'equipped_for_imaging'] and len(args) == 1:
                 self.rovers.add(args[0])
            elif pred in ['empty', 'full'] and len(args) == 1:
                 self.stores.add(args[0])
            elif pred in ['have_rock_analysis', 'have_soil_analysis'] and len(args) == 2:
                 self.rovers.add(args[0])
                 self.waypoints.add(args[1])
            elif pred in ['calibrated', 'on_board'] and len(args) == 2:
                 self.cameras.add(args[0])
                 self.rovers.add(args[1])
            elif pred == 'supports' and len(args) == 2:
                 self.cameras.add(args[0])
                 self.modes.add(args[1])
            elif pred == 'visible' and len(args) == 2:
                 self.waypoints.add(args[0])
                 self.waypoints.add(args[1])
            elif pred == 'have_image' and len(args) == 3:
                 self.rovers.add(args[0])
                 self.objectives.add(args[1])
                 self.modes.add(args[2])
            elif pred in ['communicated_soil_data', 'communicated_rock_data'] and len(args) == 1:
                 self.waypoints.add(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                 self.objectives.add(args[0])
                 self.modes.add(args[1])
            elif pred in ['at_soil_sample', 'at_rock_sample'] and len(args) == 1:
                 self.waypoints.add(args[0])
            elif pred == 'visible_from' and len(args) == 2:
                 self.objectives.add(args[0])
                 self.waypoints.add(args[1])
            elif pred == 'store_of' and len(args) == 2:
                 self.stores.add(args[0])
                 self.rovers.add(args[1])
            elif pred == 'calibration_target' and len(args) == 2:
                 self.cameras.add(args[0])
                 self.objectives.add(args[1])

        # Precompute static lookups for efficiency
        self._equipped_soil = {r for r in self.rovers if f'(equipped_for_soil_analysis {r})' in self.static}
        self._equipped_rock = {r for r in self.rovers if f'(equipped_for_rock_analysis {r})' in self.static}
        self._equipped_imaging = {r for r in self.rovers if f'(equipped_for_imaging {r})' in self.static}
        self._store_owner = {s: r for s in self.stores for r in self.rovers if f'(store_of {s} {r})' in self.static}
        self._camera_on_board = {i: r for i in self.cameras for r in self.rovers if f'(on_board {i} {r})' in self.static}
        self._camera_supports = {i: {m for m in self.modes if f'(supports {i} {m})' in self.static} for i in self.cameras}
        self._camera_cal_target = {i: {t for t in self.objectives if f'(calibration_target {i} {t})' in self.static} for i in self.cameras}
        self._objective_visible_from = {o: {w for w in self.waypoints if f'(visible_from {o} {w})' in self.static} for o in self.objectives}
        self._waypoint_visible = {w1: {w2 for w2 in self.waypoints if f'(visible {w1} {w2})' in self.static} for w1 in self.waypoints}

        # Find waypoints visible from the lander
        self._comm_waypoints = self._waypoint_visible.get(self.lander_location, set())
        # Lander waypoint is also a comm waypoint if visible from itself (usually true)
        if self.lander_location in self._waypoint_visible.get(self.lander_location, set()):
             self._comm_waypoints.add(self.lander_location)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # Check if goal is already reached
        if self.goals <= state:
            return 0

        # Check for static unreachability of communication points
        if not self._comm_waypoints:
             return float('inf') # No waypoint visible from lander

        for goal in self.goals:
            if goal in state:
                continue # This goal is already satisfied

            parts = get_parts(goal)
            pred = parts[0]

            if pred == 'communicated_soil_data':
                waypoint = parts[1]
                # Cost for communication action
                total_cost += 1

                # Check if soil data is already collected by any rover
                has_data = any(f'(have_soil_analysis {r} {waypoint})' in state for r in self.rovers)

                if not has_data:
                    # Cost for sample action
                    total_cost += 1
                    # Check if any equipped rover is at the sample waypoint
                    equipped_rovers = self._equipped_soil
                    rover_at_sample_point = any(f'(at {r} {waypoint})' in state for r in equipped_rovers)
                    if not rover_at_sample_point:
                        total_cost += 1 # Cost to move to sample point

                    # Check if any equipped rover has an empty store
                    has_empty_store = any(f'(empty {s})' in state for s in self.stores if self._store_owner.get(s) in equipped_rovers)
                    if not has_empty_store:
                         total_cost += 1 # Cost for drop action

                # Check if a rover with the data is at a communication point
                rover_with_data_at_comm_point = any(
                    f'(have_soil_analysis {r} {waypoint})' in state and f'(at {r} {wx})' in state and wx in self._comm_waypoints
                    for r in self.rovers for wx in self.waypoints
                )
                # If data exists, we need the specific rover with data at comm point.
                # If data doesn't exist, any capable rover can go to comm point after sampling.
                # The heuristic simplifies: if data exists, check rover with data. If not, check any capable rover.
                if not rover_with_data_at_comm_point:
                     if not has_data:
                         capable_rovers = self._equipped_soil
                         capable_rover_at_comm_point = any(
                             f'(at {r} {wx})' in state and wx in self._comm_waypoints
                             for r in capable_rovers for wx in self.waypoints
                         )
                         if not capable_rover_at_comm_point:
                             total_cost += 1 # Cost to move to comm point
                     else: # Data exists, but no rover with data is at comm point
                         total_cost += 1 # Cost to move the rover with data to comm point


            elif pred == 'communicated_rock_data':
                waypoint = parts[1]
                # Cost for communication action
                total_cost += 1

                # Check if rock data is already collected by any rover
                has_data = any(f'(have_rock_analysis {r} {waypoint})' in state for r in self.rovers)

                if not has_data:
                    # Cost for sample action
                    total_cost += 1
                    # Check if any equipped rover is at the sample waypoint
                    equipped_rovers = self._equipped_rock
                    rover_at_sample_point = any(f'(at {r} {waypoint})' in state for r in equipped_rovers)
                    if not rover_at_sample_point:
                        total_cost += 1 # Cost to move to sample point

                    # Check if any equipped rover has an empty store
                    has_empty_store = any(f'(empty {s})' in state for s in self.stores if self._store_owner.get(s) in equipped_rovers)
                    if not has_empty_store:
                         total_cost += 1 # Cost for drop action

                # Check if a rover with the data is at a communication point
                rover_with_data_at_comm_point = any(
                    f'(have_rock_analysis {r} {waypoint})' in state and f'(at {r} {wx})' in state and wx in self._comm_waypoints
                    for r in self.rovers for wx in self.waypoints
                )
                if not rover_with_data_at_comm_point:
                     if not has_data:
                         capable_rovers = self._equipped_rock
                         capable_rover_at_comm_point = any(
                             f'(at {r} {wx})' in state and wx in self._comm_waypoints
                             for r in capable_rovers for wx in self.waypoints
                         )
                         if not capable_rover_at_comm_point:
                             total_cost += 1 # Cost to move to comm point
                     else: # Data exists, but no rover with data is at comm point
                         total_cost += 1 # Cost to move the rover with data to comm point


            elif pred == 'communicated_image_data':
                objective, mode = parts[1], parts[2]
                # Cost for communication action
                total_cost += 1

                # Check if image is already taken by any rover
                has_image = any(f'(have_image {r} {objective} {mode})' in state for r in self.rovers)

                if not has_image:
                    # Cost for take_image action
                    total_cost += 1

                    # Find suitable rover/camera pairs (equipped for imaging, camera on board, supports mode)
                    suitable_rover_camera_pairs = [(r, i) for r in self._equipped_imaging for i in self.cameras if self._camera_on_board.get(i) == r and mode in self._camera_supports.get(i, set())]

                    # Check if an image waypoint exists for this objective
                    image_waypoint_exists = len(self._objective_visible_from.get(objective, set())) > 0

                    if not image_waypoint_exists:
                        # Objective not visible from any waypoint - goal unreachable
                        return float('inf')

                    # Check if any suitable rover is at an image waypoint
                    image_waypoint_occupied = any(
                        f'(at {r} {p})' in state
                        for r, i in suitable_rover_camera_pairs
                        for p in self._objective_visible_from.get(objective, set())
                    )
                    if not image_waypoint_occupied:
                        total_cost += 1 # Cost to move to image point

                    # Check if any suitable camera on a rover is calibrated
                    is_calibrated = any(f'(calibrated {i} {r})' in state for r, i in suitable_rover_camera_pairs)

                    if not is_calibrated:
                        # Cost for calibrate action
                        total_cost += 1

                        # Find calibration targets for suitable cameras
                        suitable_cameras = {i for r, i in suitable_rover_camera_pairs}
                        calibration_targets = {t for i in suitable_cameras for t in self._camera_cal_target.get(i, set())}

                        # Check if a calibration waypoint exists for any of these targets
                        calibration_waypoint_exists = any(len(self._objective_visible_from.get(t, set())) > 0 for t in calibration_targets)

                        if not calibration_waypoint_exists:
                             # No calibration target visible from any waypoint - goal unreachable
                             return float('inf')

                        # Check if any suitable rover is at a calibration waypoint
                        calibration_waypoint_occupied = any(
                            f'(at {r} {w})' in state
                            for r, i in suitable_rover_camera_pairs
                            for t in self._camera_cal_target.get(i, set())
                            for w in self._objective_visible_from.get(t, set())
                        )
                        if not calibration_waypoint_occupied:
                            total_cost += 1 # Cost to move to calibration point

                # Check if a rover with the image is at a communication point
                rover_with_image_at_comm_point = any(
                    f'(have_image {r} {objective} {mode})' in state and f'(at {r} {wx})' in state and wx in self._comm_waypoints
                    for r in self.rovers for wx in self.waypoints
                )

                if not rover_with_image_at_comm_point:
                     if not has_image:
                         capable_rovers = self._equipped_imaging
                         capable_rover_at_comm_point = any(
                             f'(at {r} {wx})' in state and wx in self._comm_waypoints
                             for r in capable_rovers for wx in self.waypoints
                         )
                         if not capable_rover_at_comm_point:
                             total_cost += 1 # Cost to move to comm point
                     else: # Image exists, but no rover with image is at comm point
                         total_cost += 1 # Cost to move the rover with image to comm point


            # If an unknown goal type appears, maybe return infinity? Or just ignore?
            # Assuming only the three communicated_* goals are possible.

        return total_cost
