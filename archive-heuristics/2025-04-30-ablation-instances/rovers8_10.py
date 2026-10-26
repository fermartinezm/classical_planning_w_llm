from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at_lander lander1 waypoint1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class roversHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Rovers domain.

    # Summary
    This heuristic estimates the number of actions required to achieve all goal conditions.
    It sums up the estimated cost for each unachieved goal fact independently.
    The cost for each goal fact is estimated by counting the necessary domain actions
    (like sample, take_image, calibrate, communicate, drop) and adding a simplified
    cost for navigation if the required agent is not currently at a location
    suitable for performing the next step towards that goal.

    # Assumptions
    - The problem instance is solvable.
    - For sampling goals, the soil/rock sample exists at the target waypoint initially.
    - For imaging goals, the objective is visible from at least one waypoint initially.
    - Navigation between any two waypoints is possible (though not necessarily directly)
      and costs a simplified fixed amount (e.g., 1 action) if movement is needed.
    - Any equipped rover can perform the necessary actions (sampling, imaging, communicating)
      if positioned correctly and other preconditions are met (e.g., empty store, calibrated camera).
    - The `have_soil_analysis` and `have_rock_analysis` facts persist until communication.
    - The `have_image` fact persists until communication.

    # Heuristic Initialization
    The heuristic extracts static information from the task definition, which does not change
    during planning. This includes:
    - The set of goal facts.
    - Lists of all rovers, stores, cameras, waypoints, objectives, modes.
    - The lander's location.
    - Visibility relationships between waypoints.
    - Rover equipment capabilities (soil, rock, imaging).
    - Store ownership by rovers.
    - Cameras on board rovers, supported modes, and calibration targets.
    - Visibility relationships from waypoints to objectives/targets.
    This information is stored in dictionaries and sets for efficient lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the total heuristic cost `h` to 0.
    2. Iterate through each goal fact specified in the problem.
    3. For each goal fact `g`:
        a. If `g` is already present in the current state, this goal is achieved for now; continue to the next goal fact.
        b. If `g` is `(communicated_soil_data W)` for some waypoint `W`:
            - Add 1 to `h` for the `communicate_soil_data` action.
            - Check if `(have_soil_analysis R W)` is true for any rover `R` in the current state.
            - If no rover has the soil analysis for `W`:
                - Add 1 to `h` for the `sample_soil` action.
                - Check if `(at_soil_sample W)` is still in the state (meaning it hasn't been sampled yet).
                - If `(at_soil_sample W)` is in the state:
                    - Check if there is an equipped rover currently at waypoint `W` with an empty store. If not:
                        - Add 1 to `h` for navigation to waypoint `W`.
                        - Check if there is any equipped rover (for soil analysis) with a full store (indicating a potential need to drop). If yes:
                            - Add 1 to `h` for the `drop` action.
            - Check if any rover is currently at a waypoint visible from the lander's location (a communication waypoint). If not:
                - Add 1 to `h` for navigation to a communication waypoint.
        c. If `g` is `(communicated_rock_data W)` for some waypoint `W`:
            - Add 1 to `h` for the `communicate_rock_data` action.
            - Check if `(have_rock_analysis R W)` is true for any rover `R` in the current state.
            - If no rover has the rock analysis for `W`:
                - Add 1 to `h` for the `sample_rock` action.
                - Check if `(at_rock_sample W)` is still in the state.
                - If `(at_rock_sample W)` is in the state:
                    - Check if there is an equipped rover currently at waypoint `W` with an empty store. If not:
                        - Add 1 to `h` for navigation to waypoint `W`.
                        - Check if there is any equipped rover (for rock analysis) with a full store. If yes:
                            - Add 1 to `h` for the `drop` action.
            - Check if any rover is currently at a communication waypoint. If not:
                - Add 1 to `h` for navigation to a communication waypoint.
        d. If `g` is `(communicated_image_data O M)` for some objective `O` and mode `M`:
            - Add 1 to `h` for the `communicate_image_data` action.
            - Check if `(have_image R O M)` is true for any rover `R` in the current state.
            - If no rover has the image for `O` and `M`:
                - Add 1 to `h` for the `take_image` action.
                - Identify waypoints `P` from which `O` is visible. Check if there is an equipped rover (for imaging) with a camera supporting mode `M` currently at any such waypoint `P`. If not:
                    - Add 1 to `h` for navigation to an imaging waypoint.
                - Check if there is a camera on an equipped rover (for imaging) supporting mode `M` that is currently calibrated. If not:
                    - Add 1 to `h` for the `calibrate` action.
                    - Identify waypoints `W` from which the calibration target for a relevant camera is visible. Check if there is an equipped rover (for imaging) with a camera supporting mode `M` currently at any such waypoint `W`. If not:
                        - Add 1 to `h` for navigation to a calibration waypoint.
            - Check if any rover is currently at a communication waypoint. If not:
                - Add 1 to `h` for navigation to a communication waypoint.
    4. Return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Initialize sets and dictionaries for static information
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        self.cameras = set()
        self.modes = set()
        self.landers = set()
        self.objectives = set()

        self.lander_location = None
        self.visible_waypoints = set() # (wp1, wp2)
        self.equipped_for_soil = set() # rovers
        self.equipped_for_rock = set() # rovers
        self.equipped_for_imaging = set() # rovers
        self.store_of = {} # store -> rover
        self.on_board = {} # rover -> [cameras]
        self.camera_modes = {} # camera -> {modes}
        self.calibration_targets = {} # camera -> objective
        self.visible_from_objective = set() # (waypoint, objective)

        # Parse static facts to populate objects and relationships
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'at_lander':
                if len(args) == 2:
                    self.landers.add(args[0])
                    self.waypoints.add(args[1])
                    self.lander_location = args[1]
            elif predicate == 'can_traverse':
                 if len(args) == 3:
                    self.rovers.add(args[0])
                    self.waypoints.add(args[1])
                    self.waypoints.add(args[2])
            elif predicate == 'equipped_for_soil_analysis':
                 if len(args) == 1: self.rovers.add(args[0]); self.equipped_for_soil.add(args[0])
            elif predicate == 'equipped_for_rock_analysis':
                 if len(args) == 1: self.rovers.add(args[0]); self.equipped_for_rock.add(args[0])
            elif predicate == 'equipped_for_imaging':
                 if len(args) == 1: self.rovers.add(args[0]); self.equipped_for_imaging.add(args[0])
            elif predicate == 'store_of':
                 if len(args) == 2: self.stores.add(args[0]); self.rovers.add(args[1]); self.store_of[args[0]] = args[1]
            elif predicate == 'visible':
                 if len(args) == 2: self.waypoints.add(args[0]); self.waypoints.add(args[1]); self.visible_waypoints.add((args[0], args[1]))
            elif predicate == 'on_board':
                 if len(args) == 2: self.cameras.add(args[0]); self.rovers.add(args[1]); self.on_board.setdefault(args[1], []).append(args[0])
            elif predicate == 'supports':
                 if len(args) == 2: self.cameras.add(args[0]); self.modes.add(args[1]); self.camera_modes.setdefault(args[0], set()).add(args[1])
            elif predicate == 'calibration_target':
                 if len(args) == 2: self.cameras.add(args[0]); self.objectives.add(args[1]); self.calibration_targets[args[0]] = args[1]
            elif predicate == 'visible_from':
                 # Store as (waypoint, objective) for easier lookup
                 if len(args) == 2: self.objectives.add(args[0]); self.waypoints.add(args[1]); self.visible_from_objective.add((args[1], args[0]))

        # Add objects mentioned in goals that might not be in static facts (less likely but safe)
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue
             predicate = parts[0]
             args = parts[1:]
             if predicate == 'communicated_soil_data' or predicate == 'communicated_rock_data':
                 if len(args) == 1: self.waypoints.add(args[0])
             elif predicate == 'communicated_image_data':
                 if len(args) == 2: self.objectives.add(args[0]); self.modes.add(args[1])

        # Precompute communication waypoints (visible from lander)
        self.comm_waypoints = set()
        if self.lander_location:
            self.comm_waypoints = {wp1 for wp1, wp2 in self.visible_waypoints if wp2 == self.lander_location} | \
                                  {wp2 for wp1, wp2 in self.visible_waypoints if wp1 == self.lander_location}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        for goal in self.goals:
            if goal in state:
                continue # Goal already achieved

            parts = get_parts(goal)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'communicated_soil_data':
                if len(args) != 1: continue # Malformed goal
                waypoint = args[0]
                h += 1 # Cost for communicate_soil_data

                # Check if soil analysis is available
                sample_analyzed = any(f"(have_soil_analysis {r} {waypoint})" in state for r in self.rovers)
                if not sample_analyzed:
                    h += 1 # Cost for sample_soil

                    # Check if sample needs taking and if a suitable rover is ready
                    if f"(at_soil_sample {waypoint})" in state:
                        # Need equipped rover at waypoint with empty store
                        suitable_rover_ready = any(
                            f"(at {r} {waypoint})" in state and
                            r in self.equipped_for_soil and
                            any(f"(store_of {s} {r})" in self.static_facts and f"(empty {s})" in state for s in self.stores)
                            for r in self.rovers
                        )
                        if not suitable_rover_ready:
                            h += 1 # Cost for navigation to waypoint
                            # Check if any equipped rover has a full store (potential drop needed)
                            equipped_rover_has_full_store = any(
                                r in self.equipped_for_soil and
                                any(f"(store_of {s} {r})" in self.static_facts and f"(full {s})" in state for s in self.stores)
                                for r in self.rovers
                            )
                            if equipped_rover_has_full_store:
                                h += 1 # Cost for drop

                # Check if any rover is at a communication waypoint
                rover_at_comm_waypoint = any(f"(at {r} {wp})" in state for r in self.rovers for wp in self.comm_waypoints)
                if not rover_at_comm_waypoint:
                    h += 1 # Cost for navigation to communication waypoint

            elif predicate == 'communicated_rock_data':
                if len(args) != 1: continue # Malformed goal
                waypoint = args[0]
                h += 1 # Cost for communicate_rock_data

                # Check if rock analysis is available
                sample_analyzed = any(f"(have_rock_analysis {r} {waypoint})" in state for r in self.rovers)
                if not sample_analyzed:
                    h += 1 # Cost for sample_rock

                    # Check if sample needs taking and if a suitable rover is ready
                    if f"(at_rock_sample {waypoint})" in state:
                         # Need equipped rover at waypoint with empty store
                        suitable_rover_ready = any(
                            f"(at {r} {waypoint})" in state and
                            r in self.equipped_for_rock and
                            any(f"(store_of {s} {r})" in self.static_facts and f"(empty {s})" in state for s in self.stores)
                            for r in self.rovers
                        )
                        if not suitable_rover_ready:
                            h += 1 # Cost for navigation to waypoint
                            # Check if any equipped rover has a full store (potential drop needed)
                            equipped_rover_has_full_store = any(
                                r in self.equipped_for_rock and
                                any(f"(store_of {s} {r})" in self.static_facts and f"(full {s})" in state for s in self.stores)
                                for r in self.rovers
                            )
                            if equipped_rover_has_full_store:
                                h += 1 # Cost for drop

                # Check if any rover is at a communication waypoint
                rover_at_comm_waypoint = any(f"(at {r} {wp})" in state for r in self.rovers for wp in self.comm_waypoints)
                if not rover_at_comm_waypoint:
                    h += 1 # Cost for navigation to communication waypoint

            elif predicate == 'communicated_image_data':
                if len(args) != 2: continue # Malformed goal
                objective = args[0]
                mode = args[1]
                h += 1 # Cost for communicate_image_data

                # Check if image is available
                image_taken = any(f"(have_image {r} {objective} {mode})" in state for r in self.rovers)
                if not image_taken:
                    h += 1 # Cost for take_image

                    # Find imaging waypoints for this objective
                    imaging_waypoints = {wp for wp, obj in self.visible_from_objective if obj == objective}

                    # Check if a suitable rover is at an imaging waypoint
                    # Suitable rover: equipped for imaging, has camera supporting mode
                    suitable_rover_at_imaging_waypoint = any(
                        f"(at {r} {wp})" in state and
                        r in self.equipped_for_imaging and
                        any(i in self.on_board.get(r, []) and i in self.camera_modes and mode in self.camera_modes[i] for i in self.cameras) # Check cameras on THIS rover
                        for r in self.rovers for wp in imaging_waypoints
                    )
                    if not suitable_rover_at_imaging_waypoint:
                        h += 1 # Cost for navigation to imaging waypoint

                    # Check if a camera on a suitable rover is calibrated
                    suitable_rovers = [r for r in self.rovers if r in self.equipped_for_imaging and any(i in self.on_board.get(r, []) and i in self.camera_modes and mode in self.camera_modes[i] for i in self.cameras)] # Find suitable rovers first
                    camera_calibrated = any(f"(calibrated {i} {r})" in state for r in suitable_rovers for i in self.on_board.get(r, []) if i in self.camera_modes and mode in self.camera_modes[i]) # Check calibration for cameras on suitable rovers

                    if not camera_calibrated:
                        h += 1 # Cost for calibrate

                        # Find calibration waypoints for relevant cameras on suitable rovers
                        calibration_waypoints = set()
                        for r in suitable_rovers:
                            for i in self.on_board.get(r, []):
                                if i in self.camera_modes and mode in self.camera_modes[i]:
                                    target = self.calibration_targets.get(i)
                                    if target:
                                        calibration_waypoints.update({wp for wp, obj in self.visible_from_objective if obj == target})

                        # Check if a suitable rover is at a calibration waypoint
                        suitable_rover_at_calibration_waypoint = any(
                            f"(at {r} {wp})" in state and
                            r in self.equipped_for_imaging and
                            any(i in self.on_board.get(r, []) and i in self.camera_modes and mode in self.camera_modes[i] for i in self.cameras) # Check cameras on THIS rover
                            for r in self.rovers for wp in calibration_waypoints
                        )
                        if not suitable_rover_at_calibration_waypoint:
                            h += 1 # Cost for navigation to calibration waypoint

                # Check if any rover is at a communication waypoint
                rover_at_comm_waypoint = any(f"(at {r} {wp})" in state for r in self.rovers for wp in self.comm_waypoints)
                if not rover_at_comm_waypoint:
                    h += 1 # Cost for navigation to communication waypoint

            # Note: Assuming only communication goals based on typical rovers benchmarks.
            # If other goal types exist (e.g., (at rover1 waypointX)), they would need
            # separate handling here.

        return h
