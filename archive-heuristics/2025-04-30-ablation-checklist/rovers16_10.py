# Assuming heuristics.heuristic_base provides a base class
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided externally, define a simple one:
class Heuristic:
    """Base class for domain-dependent heuristics."""
    def __init__(self, task):
        """Initialize the heuristic with task information."""
        pass # Placeholder
    def __call__(self, node):
        """Compute the heuristic value for a given state node."""
        return 0 # Placeholder


from fnmatch import fnmatch
from collections import defaultdict # deque is not used in this version

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

    # Summary
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for unachieved goals. It counts necessary
    actions like sampling, imaging, calibrating, and communicating, and adds
    simplified costs for navigation and dropping samples.

    # Assumptions
    - Each unachieved goal requires at least one communication action.
    - If a sample or image needed for a goal is not held by any suitable rover,
      it must be collected (sampling/imaging actions).
    - If an image is needed and the camera/rover pair that would take it is not
      calibrated, a calibration action is needed.
    - If any sample is needed and any equipped rover's store is full, at least
      one drop action is needed.
    - Each rover involved in any remaining task requires at least one navigation
      action.
    - Equipment, cameras, and stores are static and assigned to specific rovers.
    - The existence of equipped rovers and suitable cameras is checked; if a task
      requires one and none exist, the heuristic returns infinity.

    # Heuristic Initialization
    - Extracts static facts like rover equipment, camera assignments and modes,
      calibration targets, store ownership, and lander location.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Initialize total cost to 0.
    2.  Identify the set of goal predicates that are not currently true in the state (`goals_to_achieve`).
    3.  Initialize sets to track necessary tasks: `needed_soil_samples` (waypoints), `needed_rock_samples` (waypoints), `needed_images` ((objective, mode) tuples), `needed_calibrations` ((camera, rover) tuples), and `active_rovers` (rovers involved in any remaining task).
    4.  Track current state facts relevant to task completion: `have_soil`, `have_rock`, `have_image`, `stores_full`.
    5.  Iterate through each `goal` in `goals_to_achieve`:
        *   If `goal` is `(communicated_soil_data ?w)`:
            *   Add 1 to `cost` for the `communicate_soil_data` action.
            *   Check if `(have_soil_analysis ?r ?w)` is true for *any* soil-equipped rover `r` in the current state.
            *   If not, add 1 to `cost` for the `sample_soil` action and add `w` to `needed_soil_samples`. Add all soil-equipped rovers to `active_rovers` (as any of them might perform the task). If no soil-equipped rovers exist, return `float('inf')`.
            *   If the sample is held, add all soil-equipped rovers to `active_rovers` (as any of them might perform the communication). If no soil-equipped rovers exist, return `float('inf')`.
        *   If `goal` is `(communicated_rock_data ?w)`: (Similar logic as soil data)
            *   Add 1 to `cost` for the `communicate_rock_data` action.
            *   Check if `(have_rock_analysis ?r ?w)` is true for *any* rock-equipped rover `r`.
            *   If not, add 1 to `cost` for the `sample_rock` action and add `w` to `needed_rock_samples`. Add all rock-equipped rovers to `active_rovers`. If no rock-equipped rovers exist, return `float('inf')`.
            *   If the sample is held, add all rock-equipped rovers to `active_rovers`. If no rock-equipped rovers exist, return `float('inf')`.
        *   If `goal` is `(communicated_image_data ?o ?m)`:
            *   Add 1 to `cost` for the `communicate_image_data` action.
            *   Find all imaging-equipped rovers `r` that have a camera `i` on board supporting mode `m`. If none exist, return `float('inf')`.
            *   Check if `(have_image ?r ?o ?m)` is true for *any* of these suitable rovers `r`.
            *   If not, add 1 to `cost` for the `take_image` action and add `(o, m)` to `needed_images`. Pick *one* suitable rover/camera pair `(r, i)`. If `(calibrated i r)` is not true in the state, add 1 to `cost` for the `calibrate` action and add `(i, r)` to `needed_calibrations`. Add this specific rover `r` to `active_rovers`.
            *   If the image is held, add all suitable rovers to `active_rovers`.
    7.  Estimate cost for `drop` actions: If there are any waypoints in `needed_soil_samples` or `needed_rock_samples`, check if *any* rover equipped for soil or rock analysis has a full store (`(full ?s)` for its store `s`). If this condition is met, add 1 to `cost` for a `drop` action.
    8.  Estimate cost for `navigate` actions: Add the number of unique rovers in the `active_rovers` set to the total `cost`. This is a simplified estimate that each rover involved in remaining tasks will require at least one navigation action.
    9.  Return the total estimated `cost`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.equipped_soil = {get_parts(f)[1] for f in static_facts if match(f, "equipped_for_soil_analysis", "*")}
        self.equipped_rock = {get_parts(f)[1] for f in static_facts if match(f, "equipped_for_rock_analysis", "*")}
        self.equipped_imaging = {get_parts(f)[1] for f in static_facts if match(f, "equipped_for_imaging", "*")}
        self.camera_on_rover = {get_parts(f)[1]: get_parts(f)[2] for f in static_facts if match(f, "on_board", "*", "*")}
        self.camera_supports_mode = defaultdict(set)
        for f in static_facts:
            if match(f, "supports", "*", "*"):
                camera, mode = get_parts(f)[1:3]
                self.camera_supports_mode[camera].add(mode)
        self.calibration_target = {get_parts(f)[1]: get_parts(f)[2] for f in static_facts if match(f, "calibration_target", "*", "*")}
        self.store_of_rover = {get_parts(f)[2]: get_parts(f)[1] for f in static_facts if match(f, "store_of", "*", "*")}
        # lander_location and lander_visible_wps are not used in this simplified heuristic
        # self.lander_location = next((get_parts(f)[2] for f in static_facts if match(f, "at_lander", "*", "*")), None)
        # self.lander_visible_wps = {get_parts(f)[1] for f in static_facts if match(f, "visible", "*", self.lander_location)} if self.lander_location else set()


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        cost = 0

        goals_to_achieve = set(self.goals) - set(state)

        needed_soil_samples = set() # waypoints
        needed_rock_samples = set() # waypoints
        needed_images = set() # (objective, mode)
        needed_calibrations = set() # (camera, rover)
        active_rovers = set() # rovers involved in any remaining task

        # Track which rovers hold which samples/images in the current state
        have_soil = defaultdict(set) # rover -> {waypoint}
        have_rock = defaultdict(set) # rover -> {waypoint}
        have_image = defaultdict(set) # rover -> {(objective, mode)}
        stores_full = set() # stores

        for fact in state:
            if match(fact, "have_soil_analysis", "*", "*"):
                r, w = get_parts(fact)[1:3]
                have_soil[r].add(w)
            elif match(fact, "have_rock_analysis", "*", "*"):
                r, w = get_parts(fact)[1:3]
                have_rock[r].add(w)
            elif match(fact, "have_image", "*", "*", "*"):
                r, o, m = get_parts(fact)[1:4]
                have_image[r].add((o, m))
            elif match(fact, "full", "*"):
                stores_full.add(get_parts(fact)[1])

        # Process each unachieved goal
        for goal in goals_to_achieve:
            if match(goal, "communicated_soil_data", "*"):
                w = get_parts(goal)[1]
                cost += 1 # communicate action

                # Check if sample is held by any equipped rover
                sample_held = any(w in have_soil.get(r, set()) for r in self.equipped_soil)

                if not sample_held:
                    cost += 1 # sample action
                    needed_soil_samples.add(w)
                    if not self.equipped_soil: return float('inf') # Cannot sample if no equipped rover
                    active_rovers.update(self.equipped_soil) # Any equipped soil rover might do it
                else: # Sample is held, need to communicate
                     # Add any equipped soil rover if the goal is not met.
                     if not self.equipped_soil: return float('inf')
                     active_rovers.update(self.equipped_soil)


            elif match(goal, "communicated_rock_data", "*"):
                w = get_parts(goal)[1]
                cost += 1 # communicate action

                # Check if sample is held by any equipped rover
                sample_held = any(w in have_rock.get(r, set()) for r in self.equipped_rock)

                if not sample_held:
                    cost += 1 # sample action
                    needed_rock_samples.add(w)
                    if not self.equipped_rock: return float('inf') # Cannot sample if no equipped rover
                    active_rovers.update(self.equipped_rock) # Any equipped rock rover might do it
                else: # Sample is held, need to communicate
                     # Add any equipped rock rover if the goal is not met.
                     if not self.equipped_rock: return float('inf')
                     active_rovers.update(self.equipped_rock)


            elif match(goal, "communicated_image_data", "*", "*"):
                o, m = get_parts(goal)[1:3]
                cost += 1 # communicate action

                # Find equipped imaging rovers with cameras supporting mode m
                suitable_rovers_cameras = set() # Store (rover, camera)
                for r in self.equipped_imaging:
                    for camera, rover_on_board in self.camera_on_rover.items():
                        if rover_on_board == r and m in self.camera_supports_mode.get(camera, set()):
                            suitable_rovers_cameras.add((r, camera))

                if not suitable_rovers_cameras: return float('inf') # Cannot take image if no suitable rover/camera

                # Check if image is held by any suitable rover
                image_held = any((o, m) in have_image.get(r, set()) for r, c in suitable_rovers_cameras)

                if not image_held:
                    cost += 1 # take_image action
                    needed_images.add((o, m))

                    # Need calibration? Pick *one* suitable rover/camera pair.
                    # This assumes one calibration per camera per plan, which is not strictly true
                    # (calibration expires), but is a reasonable heuristic simplification.
                    rover, camera = next(iter(suitable_rovers_cameras)) # Pick the first one
                    if f"(calibrated {camera} {rover})" not in state:
                        cost += 1 # calibrate action
                        needed_calibrations.add((camera, rover)) # Add (camera, rover) pair
                    active_rovers.add(rover) # This specific rover is needed for imaging/calibration
                else: # Image is held, need to communicate
                    # Add any suitable rover if the goal is not met.
                    active_rovers.update({r for r, c in suitable_rovers_cameras})

            # else: # Unknown goal type? Ignore or return inf? Ignore for now.

        # Cost for drop: If any soil/rock sample is needed AND any equipped rover for soil/rock has a full store.
        # This is a rough estimate, assuming one drop might be needed if sampling is required and a store is full.
        if needed_soil_samples or needed_rock_samples:
            equipped_rovers_for_sample = self.equipped_soil | self.equipped_rock
            needs_drop = False
            for r in equipped_rovers_for_sample:
                store = self.store_of_rover.get(r)
                if store and store in stores_full:
                    needs_drop = True
                    break
            if needs_drop:
                cost += 1 # drop action

        # Cost for navigation: Add 1 navigation action for each rover involved in any remaining task.
        # This is a simplified estimate, assuming each active rover needs at least one move
        # to get to a location where it can perform its required tasks (sample, image, calibrate, communicate).
        cost += len(active_rovers)

        return cost
