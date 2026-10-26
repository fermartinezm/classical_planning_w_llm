from fnmatch import fnmatch
# Assuming Heuristic base class is available
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # Handle potential non-string facts or malformed facts gracefully
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

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number
    of unachieved "milestones" for each goal condition. It considers the steps
    needed to gather data (sample soil/rock, take image), calibrate cameras,
    and communicate the data, adding a penalty for full stores if sampling is required.
    It ignores navigation costs.

    # Assumptions
    - The heuristic assumes that if a sample exists at a waypoint (`at_soil_sample`, `at_rock_sample`),
      it can be sampled by an equipped rover with an empty store.
    - It assumes that if an objective is visible from a waypoint (`visible_from`),
      an imaging-equipped rover at that waypoint with a calibrated camera can take an image.
    - It assumes that if data is held by a rover (`have_soil_analysis`, `have_rock_analysis`, `have_image`),
      it can eventually be communicated if a lander exists (communication waypoint reachability is ignored).
    - It assumes that calibration is possible if a calibration target exists for the camera
      and the rover is at a waypoint visible from the target (waypoint reachability is ignored).
    - Navigation costs are not included in the heuristic value.
    - If a soil/rock sample is no longer at its initial waypoint (`at_soil_sample`, `at_rock_sample` is false),
      and the corresponding `have_soil_analysis` or `have_rock_analysis` fact is also false,
      the heuristic assumes the goal requiring this data is unachievable from this point forward
      via sampling from that specific waypoint and does not add costs for sampling/dropping for this specific waypoint's data.

    # Heuristic Initialization
    - Extracts static information about rover equipment, store ownership, camera properties (on_board, supports, calibration_target).
      This information is used to identify which rovers/cameras are capable of performing certain actions
      when assessing missing prerequisites.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each unachieved goal fact.
    For each goal fact `g` in `task.goals`:

    1.  If `g` is already in the current state, its contribution is 0.

    2.  If `g` is `(communicated_soil_data W)` and not in state:
        - Add 1 to cost (for the `communicate_soil_data` action).
        - Check if `(have_soil_analysis R W)` is true for *any* rover `R` in the state.
        - If no such fact exists:
            # Need to sample. This is only possible if the sample is still there.
            sample_still_at_waypoint = any(match(fact, "at_soil_sample", W) for fact in state)
            if sample_still_at_waypoint:
                cost += 1 # Cost for the sample step
                # Check if a drop is needed before sampling.
                # A drop is needed if *all* soil-equipped rovers with stores have full stores.
                # First, find all soil-equipped rovers that have a store.
                soil_rovers_with_stores = [r for r in self.equipped_soil if r in self.rover_stores]
                if soil_rovers_with_stores: # Only check if there are rovers capable of sampling with stores
                    # Check if *any* soil-equipped rover with a store has an empty store.
                    has_empty_store_among_soil_rovers = False
                    for rover in soil_rovers_with_stores:
                        store = self.rover_stores[rover]
                        if f"(empty {store})" in state:
                             has_empty_store_among_soil_rovers = True
                             break
                    # If none of them have an empty store, a drop is needed by one of them.
                    if not has_empty_store_among_soil_rovers:
                         cost += 1 # Cost for a drop action

    3.  If `g` is `(communicated_rock_data W)` and not in state:
        - Add 1 to cost (for the `communicate_rock_data` action).
        - Check if `(have_rock_analysis R W)` is true for *any* rover `R` in the state.
        - If no such fact exists:
            # Need to sample. This is only possible if the sample is still there.
            sample_still_at_waypoint = any(match(fact, "at_rock_sample", W) for fact in state)
            if sample_still_at_waypoint:
                cost += 1 # Cost for the sample step
                # Check if a drop is needed before sampling.
                rock_rovers_with_stores = [r for r in self.equipped_rock if r in self.rover_stores]
                if rock_rovers_with_stores:
                    has_empty_store_among_rock_rovers = False
                    for rover in rock_rovers_with_stores:
                        store = self.rover_stores[rover]
                        if f"(empty {store})" in state:
                             has_empty_store_among_rock_rovers = True
                             break
                    if not has_empty_store_among_rock_rovers:
                         cost += 1 # Cost for a drop action

    4.  If `g` is `(communicated_image_data O M)` and not in state:
        - Add 1 to cost (for the `communicate_image_data` action).
        - Check if `(have_image R O M)` is true for *any* rover `R` in the state.
        - If no such fact exists:
            cost += 1 # Cost for the take_image step
            # Check if a suitable camera is calibrated on any rover.
            # A suitable camera is one that is on board an imaging-equipped rover and supports the required mode.
            is_calibrated = False
            # Iterate through cameras that support the mode
            for camera, supported_modes in self.camera_supports.items():
                if M in supported_modes:
                    # Check if this camera is on board any imaging-equipped rover
                    if camera in self.camera_on_board:
                        rover = self.camera_on_board[camera]
                        if rover in self.equipped_imaging:
                            # Check if this specific camera on this specific rover is calibrated in the current state
                            if f"(calibrated {camera} {rover})" in state:
                                is_calibrated = True
                                break # Found a calibrated suitable camera

            if not is_calibrated:
                cost += 1 # Cost for the calibrate step

    # The heuristic is 0 if and only if all goals are in the state.
    # The loop structure ensures this: if all goals are in state, the loop finishes with cost = 0.
    # If any goal is not in state, cost will be at least 1.
    # The heuristic is finite for solvable states because it only counts a maximum of 3 or 4 steps per goal.
    # If a sample is gone, it stops counting sample/drop steps, implicitly assuming unachievability for that part.

    return cost

