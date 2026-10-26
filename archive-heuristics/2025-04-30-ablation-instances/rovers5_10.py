from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
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

    # Summary
    This heuristic estimates the number of actions required to satisfy the goal
    conditions, which involve communicating collected data (soil, rock, images)
    from specific locations/objectives. It counts the number of remaining
    communication tasks, the necessary data collection tasks (sampling/imaging),
    required setup actions (dropping samples, calibrating cameras), and a
    simplified estimate for navigation.

    # Assumptions
    - All goal conditions are of the form `(communicated_soil_data ?w)`,
      `(communicated_rock_data ?w)`, or `(communicated_image_data ?o ?m)`.
    - Rovers are statically equipped for certain tasks (`equipped_for_*`).
    - Camera capabilities (`supports`, `on_board`, `calibration_target`) are static.
    - Lander location is static (`at_lander`).
    - Navigation costs are simplified (a fixed cost is added for each *category*
      of navigation needed: sampling, imaging, calibrating, communicating).
    - Dropping a sample costs 1 action if any sampling is needed and any
      equipped rover's store is full.
    - Calibrating costs 1 action if any imaging is needed and no suitable
      camera is currently calibrated on an equipped rover. Calibration is only
      counted if it is possible to achieve in the domain instance.

    # Heuristic Initialization
    The constructor extracts the following information from the task:
    - The set of all objects by type (rovers, waypoints, stores, cameras, modes, landers, objectives).
    - The specific goal conditions related to communication.
    - Static facts like equipment, store ownership, camera properties, lander location, and connectivity (though connectivity is only used implicitly for possibility checks, not pathfinding).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state `s`:
    1. Initialize heuristic value `h = 0`.
    2. Identify unsatisfied communication goals: Iterate through the goals and
       collect the waypoints (`?w`) for soil/rock goals and (objective, mode)
       pairs (`?o`, `?m`) for image goals that are not present in the state `s`.
    3. Add cost for communication actions: Add the total count of unsatisfied
       communication goals to `h`. (Each needs one `communicate_*_data` action).
    4. Identify required data collection tasks (sampling/imaging): For each
       unsatisfied communication goal, check if the corresponding `have_*`
       predicate is true in state `s` for *any* rover. If not, the data needs
       to be collected.
       - For missing soil data at `?w`: Add 1 to `h` (for `sample_soil`).
       - For missing rock data at `?w`: Add 1 to `h` (for `sample_rock`).
       - For missing image data for (`?o`, `?m`): Add 1 to `h` (for `take_image`).
    5. Add cost for setup (dropping): If any sampling task is required (step 4)
       AND there is at least one rover equipped for sampling that has a full
       store in state `s`, add 1 to `h` (for a `drop` action).
    6. Add cost for setup (calibrating): If any imaging task is required (step 4)
       AND there is no suitable camera (onboard an equipped rover supporting
       the required mode) that is currently calibrated in state `s`, add 1 to `h`
       (for a `calibrate` action). This cost is only added if calibration is
       possible at all in the domain instance (checked during initialization).
    7. Add cost for navigation: Add 1 to `h` for each *category* of navigation
       that is still required based on the missing tasks identified in steps 4 and 6,
       and the unsatisfied communication goals from step 2.
       - If any sampling is needed (step 4): Add 1 (navigation to sample location).
       - If any imaging is needed (step 4): Add 1 (navigation to image location).
       - If calibration is needed (step 6): Add 1 (navigation to calibration location).
       - If any communication is needed (step 2): Add 1 (navigation to communication location).
    8. Return the total heuristic value `h`. If `h` is 0, it means all goals are satisfied.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all objects by type from static facts
        self.all_rovers = set()
        self.all_waypoints = set()
        self.all_stores = set()
        self.all_cameras = set()
        self.all_modes = set()
        self.all_landers = set()
        self.all_objectives = set()

        # Collect objects from static facts
        for fact in self.static:
             parts = get_parts(fact)
             if not parts: continue
             pred = parts[0]
             if pred in ["equipped_for_soil_analysis", "equipped_for_rock_analysis", "equipped_for_imaging"]:
                 if len(parts) > 1: self.all_rovers.add(parts[1])
             elif pred == "store_of":
                 if len(parts) > 1: self.all_stores.add(parts[1])
                 if len(parts) > 2: self.all_rovers.add(parts[2])
             elif pred == "on_board":
                 if len(parts) > 1: self.all_cameras.add(parts[1])
                 if len(parts) > 2: self.all_rovers.add(parts[2])
             elif pred == "supports":
                 if len(parts) > 1: self.all_cameras.add(parts[1])
                 if len(parts) > 2: self.all_modes.add(parts[2])
             elif pred == "calibration_target":
                 if len(parts) > 1: self.all_cameras.add(parts[1])
                 if len(parts) > 2: self.all_objectives.add(parts[2])
             elif pred == "visible_from":
                 if len(parts) > 1: self.all_objectives.add(parts[1])
                 if len(parts) > 2: self.all_waypoints.add(parts[2])
             elif pred == "at_lander":
                 if len(parts) > 1: self.all_landers.add(parts[1])
                 if len(parts) > 2: self.all_waypoints.add(parts[2])
             elif pred == "can_traverse":
                 if len(parts) > 1: self.all_rovers.add(parts[1])
                 if len(parts) > 2: self.all_waypoints.add(parts[2])
                 if len(parts) > 3: self.all_waypoints.add(parts[3])
             elif pred == "visible":
                 if len(parts) > 1: self.all_waypoints.add(parts[1])
                 if len(parts) > 2: self.all_waypoints.add(parts[2])
             # Note: Objects might also appear in initial state facts (e.g., at_soil_sample, at_rock_sample, at rover).
             # This heuristic relies on static facts providing a comprehensive list of objects.

        # Extract goal requirements
        self.goal_soil_waypoints = {get_parts(g)[1] for g in self.goals if match(g, "communicated_soil_data", "*")}
        self.goal_rock_waypoints = {get_parts(g)[1] for g in self.goals if match(g, "communicated_rock_data", "*")}
        self.goal_image_objectives_modes = {(get_parts(g)[1], get_parts(g)[2]) for g in self.goals if match(g, "communicated_image_data", "*", "*")}

        # Extract static relationships needed for setup/navigation checks
        self.equipped_for_soil = {get_parts(f)[1] for f in self.static if match(f, "equipped_for_soil_analysis", "*")}
        self.equipped_for_rock = {get_parts(f)[1] for f in self.static if match(f, "equipped_for_rock_analysis", "*")}
        self.equipped_for_imaging = {get_parts(f)[1] for f in self.static if match(f, "equipped_for_imaging", "*")}
        self.store_owners = {get_parts(f)[1]: get_parts(f)[2] for f in self.static if match(f, "store_of", "*", "*")} # store -> rover
        self.camera_on_board = {(get_parts(f)[1], get_parts(f)[2]) for f in self.static if match(f, "on_board", "*", "*")} # (camera, rover)
        self.camera_supports = {(get_parts(f)[1], get_parts(f)[2]) for f in self.static if match(f, "supports", "*", "*")} # (camera, mode)
        self.calibration_targets = {(get_parts(f)[1], get_parts(f)[2]) for f in self.static if match(f, "calibration_target", "*", "*")} # (camera, objective)
        self.visible_from = {(get_parts(f)[1], get_parts(f)[2]) for f in self.static if match(f, "visible_from", "*", "*")} # (objective, waypoint)
        self.lander_location = next((get_parts(f)[2] for f in self.static if match(f, "at_lander", "*", "*")), None) # Assuming one lander

        # Pre-check if calibration is possible at all in this domain instance
        self.is_calibration_possible_in_domain = False
        for r in self.all_rovers:
            if r in self.equipped_for_imaging:
                for i in self.all_cameras:
                    if (i, r) in self.camera_on_board:
                        for t in self.all_objectives:
                            if (i, t) in self.calibration_targets:
                                # Calibration target exists for this camera
                                # Check if there's a waypoint visible from this target
                                for w in self.all_waypoints:
                                    if (t, w) in self.visible_from:
                                        self.is_calibration_possible_in_domain = True
                                        break # Found a possible calibration path
                                if self.is_calibration_possible_in_domain: break
                            if self.is_calibration_possible_in_domain: break
                        if self.is_calibration_possible_in_domain: break
                    if self.is_calibration_possible_in_domain: break
                if self.is_calibration_possible_in_domain: break


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # 1. Communication Goals
        unsat_soil_goals = {w for w in self.goal_soil_waypoints if f'(communicated_soil_data {w})' not in state}
        unsat_rock_goals = {w for w in self.goal_rock_waypoints if f'(communicated_rock_data {w})' not in state}
        unsat_image_goals = {(o, m) for (o, m) in self.goal_image_objectives_modes if f'(communicated_image_data {o} {m})' not in state}

        h += len(unsat_soil_goals) + len(unsat_rock_goals) + len(unsat_image_goals)

        # 2. Have Data Requirements (Sampling/Imaging)
        sample_soil_tasks = {w for w in unsat_soil_goals if not any(f'(have_soil_analysis {r} {w})' in state for r in self.all_rovers)}
        sample_rock_tasks = {w for w in unsat_rock_goals if not any(f'(have_rock_analysis {r} {w})' in state for r in self.all_rovers)}
        take_image_tasks = {(o, m) for (o, m) in unsat_image_goals if not any(f'(have_image {r} {o} {m})' in state for r in self.all_rovers)}

        h += len(sample_soil_tasks) + len(sample_rock_tasks) + len(take_image_tasks)

        # 3. Setup Costs (Drop)
        needs_drop = False
        if len(sample_soil_tasks) > 0 or len(sample_rock_tasks) > 0:
            # Check if any rover equipped for sampling has a full store
            sampling_rovers = self.equipped_for_soil.union(self.equipped_for_rock)
            for r in sampling_rovers:
                # Find the store for this rover
                store = next((s for s, owner in self.store_owners.items() if owner == r), None)
                if store and f'(full {store})' in state:
                    needs_drop = True
                    break
        if needs_drop:
            h += 1

        # 4. Setup Costs (Calibrate)
        needs_calibration = False
        if len(take_image_tasks) > 0 and self.is_calibration_possible_in_domain:
            # Check if ANY needed image (o, m) can be taken by ANY equipped rover
            # with an onboard camera supporting m that is already calibrated.
            can_take_any_needed_image_now = False
            for (o, m) in take_image_tasks:
                for r in self.all_rovers:
                    if r in self.equipped_for_imaging:
                        for i in self.all_cameras:
                            if (i, r) in self.camera_on_board and (i, m) in self.camera_supports:
                                if f'(calibrated {i} {r})' in state:
                                    can_take_any_needed_image_now = True
                                    break # Found suitable calibrated camera for this mode
                        if can_take_any_needed_image_now: break # Found suitable calibrated camera on this rover
                if can_take_any_needed_image_now: break # Found suitable calibrated camera for this image task

            if not can_take_any_needed_image_now:
                 h += 1 # Cost for calibrate action
                 needs_calibration = True

        # 5. Navigation Costs
        # Add 1 for each category of navigation needed
        if len(sample_soil_tasks) > 0 or len(sample_rock_tasks) > 0:
            h += 1 # Nav to sample locations
        if len(take_image_tasks) > 0:
            h += 1 # Nav to image locations
        if needs_calibration:
            h += 1 # Nav to calibration locations
        if len(unsat_soil_goals) > 0 or len(unsat_rock_goals) > 0 or len(unsat_image_goals) > 0:
            h += 1 # Nav to communication locations

        return h
