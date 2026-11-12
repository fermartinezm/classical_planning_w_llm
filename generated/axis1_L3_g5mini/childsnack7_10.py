from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the tokens of a fact string without surrounding parentheses."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `rover-2` (secret_domain) domain.
    The planner's goals in the provided instances are to make certain modes `full`.
    Two domain operators can achieve `full(mode)` directly:
      - calibrate: requires mode to be `supports`ed, `at mode rover`, some
        `equipped_for_rock_analysis(waypoint, lander)` and `can_traverse(lander, rover)`.
      - communicate_soil_data: requires `equipped_for_soil_analysis(mode)`,
        `equipped_for_rock_analysis(waypoint, lander)`, `at mode rover`,
        `visible_from(waypoint)` and `can_traverse(lander, rover)`.

    The heuristic estimates, for each goal mode that is not yet `full`, a small
    integer number of actions necessary to reach `full(mode)` from the current
    state. It prefers the short "calibrate" path when available, otherwise
    considers creating an `equipped_for_rock_analysis` via `take_image` (if a
    visible waypoint with a traversable lander exists) or via a two-step
    `sample_rock` + `take_image` pattern when only `have_soil_analysis` and
    `have_image`/`empty` are available. If none of these recognizable patterns
    apply, it falls back to a small finite upper bound.

    Assumptions
    -----------
    - The planner's goal facts are of the form "(full modeX)" and these are the
      only goals we need to reason about.
    - Some static information (e.g., `supports modeX`, `equipped_for_soil_analysis modeX`,
      and `at modeX roverY`) is provided in the task.static set and does not
      change during planning.
    - Dynamic facts such as `equipped_for_rock_analysis`, `visible`, `can_traverse`,
      `visible_from`, `have_image`, `have_soil_analysis`, and `empty` appear in
      the current state; the heuristic inspects them to estimate whether short
      action sequences are available.
    - The heuristic is not required to be admissible; it focuses on guiding
      greedy best-first search efficiently with a small, informative cost.
    - Mode names start with "mode" and can be identified from facts of the form
      "(at modeX roverY)"; lander names typically appear in `can_traverse` and
      `equipped_for_rock_analysis` facts.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts useful static information:
      - self.goals : set of goal facts (strings).
      - self.supports : set of modes that have (supports mode).
      - self.equipped_soil_modes : set of modes that have (equipped_for_soil_analysis mode).
      - self.mode_at_static : mapping mode -> rover from static (at mode rover) facts.
      - self.static_can_traverse : set of (lander, rover) pairs available in static facts.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic operates as follows:

    1. If all goal facts are already in the state, return 0.

    2. Parse dynamic state facts into convenient structures:
       - equipped_for_rock: list of (waypoint, lander) pairs present in the state.
       - visible, visible_from, have_image (cameras), have_soil_analysis (waypoints),
         empty_objectives sets for quick membership tests.
       - can_traverse is formed by union of static can_traverse and state can_traverse.
       - at_mode runtime map: if the `(at mode rover)` is present in the state,
         prefer that over static mapping.

    3. For each required mode that is not `full` in the current state, estimate
       the minimal number of actions in a small, domain-informed way:
       - If the mode is supported (supports(mode)) and there is some
         equipped_for_rock_analysis(waypoint, lander) whose lander can_traverse
         to the rover where the mode is located, then the single action
         `calibrate` can make the mode full -> cost = 1.
       - Else if the mode is equipped_for_soil_analysis and there exists an
         equipped_for_rock_analysis(waypoint, lander) with waypoint present in
         `visible_from` and the lander can_traverse to the mode's rover, then
         the single action `communicate_soil_data` can make the mode full -> cost = 1.
       - Else if there exists a visible waypoint and a lander that can_traverse
         to the mode's rover, then we can `take_image` (1 action) to produce an
         equipped_for_rock_analysis and then `calibrate`/`communicate` (1 action)
         -> cost = 2.
       - Else if there exists a waypoint with `have_soil_analysis` and the state
         has both `have_image(camera)` and some `empty` objective and some lander
         that can_traverse to the mode's rover, then we can `sample_rock` (to
         make the waypoint visible), then `take_image`, then `calibrate` -> cost = 3.
       - Otherwise, no short recognized pattern applies; return a small finite
         fallback cost (4) for this mode.

    4. Sum up the per-mode estimates to produce the final heuristic value. The
       heuristic therefore yields 0 exactly in goal states, finite positive
       integers for states that appear solvable, and makes use of static info
       extracted at initialization for fast decision-making.
    """

    def __init__(self, task):
        # Goal facts (set of strings)
        self.goals = set(task.goals)

        # Extract static facts into convenient data structures.
        self.supports = set()
        self.equipped_soil_modes = set()
        self.mode_at_static = dict()  # mode -> rover
        self.static_can_traverse = set()  # set of (lander, rover) pairs

        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "supports" and len(parts) >= 2:
                self.supports.add(parts[1])
            elif pred == "equipped_for_soil_analysis" and len(parts) >= 2:
                self.equipped_soil_modes.add(parts[1])
            elif pred == "at" and len(parts) >= 3:
                # In domain `at` of a mode: (at mode rover)
                # Identify mode names by their prefix 'mode' (consistent with domain).
                subj = parts[1]
                obj = parts[2]
                if subj.startswith("mode"):
                    self.mode_at_static[subj] = obj
            elif pred == "can_traverse" and len(parts) >= 3:
                lander = parts[1]
                rover = parts[2]
                self.static_can_traverse.add((lander, rover))

    def __call__(self, node):
        state = node.state

        # If all goal facts already true, heuristic = 0.
        if self.goals <= state:
            return 0

        # Parse dynamic facts from state.
        equipped_for_rock = []  # list of (waypoint, lander)
        visible = set()
        visible_from = set()
        have_image_cameras = set()
        have_soil_waypoints = set()
        empty_objectives = set()
        state_can_traverse = set()
        at_mode_state = dict()  # mode -> rover from dynamic state

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "equipped_for_rock_analysis" and len(parts) >= 3:
                equipped_for_rock.append((parts[1], parts[2]))
            elif pred == "visible" and len(parts) >= 2:
                visible.add(parts[1])
            elif pred == "visible_from" and len(parts) >= 2:
                visible_from.add(parts[1])
            elif pred == "have_image" and len(parts) >= 2:
                have_image_cameras.add(parts[1])
            elif pred == "have_soil_analysis" and len(parts) >= 2:
                have_soil_waypoints.add(parts[1])
            elif pred == "empty" and len(parts) >= 2:
                empty_objectives.add(parts[1])
            elif pred == "can_traverse" and len(parts) >= 3:
                state_can_traverse.add((parts[1], parts[2]))
            elif pred == "at" and len(parts) >= 3:
                subj = parts[1]
                obj = parts[2]
                if subj.startswith("mode"):
                    at_mode_state[subj] = obj

        # Union static and dynamic can_traverse for availability testing.
        all_can_traverse = self.static_can_traverse | state_can_traverse

        total_cost = 0

        # For each goal mode that is not yet full, estimate cost.
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] != "full":
                # This heuristic only handles full(mode) goals; other goals are left as 0-cost contribution.
                continue
            mode = parts[1]
            full_fact = f"(full {mode})"
            if full_fact in state:
                continue  # already achieved

            # Determine the rover where this mode is located (prefer dynamic `at` over static).
            mode_rover = at_mode_state.get(mode, self.mode_at_static.get(mode, None))

            supports_flag = mode in self.supports
            equipped_soil_flag = mode in self.equipped_soil_modes

            # 1) Check if calibrate is directly possible: supports + some equipped_for_rock whose lander can reach the same rover.
            calibrate_possible = False
            if supports_flag and mode_rover is not None:
                for wp, lander in equipped_for_rock:
                    if (lander, mode_rover) in all_can_traverse:
                        calibrate_possible = True
                        break
            if calibrate_possible:
                total_cost += 1
                continue

            # 2) Check if communicate_soil_data is directly possible:
            # requires equipped_for_soil_analysis(mode), some equipped_for_rock(waypoint, lander),
            # waypoint in visible_from and lander can reach mode_rover.
            comm_soil_possible = False
            if equipped_soil_flag and mode_rover is not None:
                for wp, lander in equipped_for_rock:
                    if wp in visible_from and (lander, mode_rover) in all_can_traverse:
                        comm_soil_possible = True
                        break
            if comm_soil_possible:
                total_cost += 1
                continue

            # 3) If there is a visible waypoint and a lander that can_traverse to mode_rover,
            # we can take_image (1) -> produce equipped_for_rock_analysis, then calibrate/communicate (1) -> total 2.
            take_then_calibrate_possible = False
            if mode_rover is not None:
                for wp in visible:
                    # Find any lander that can_traverse to mode_rover (lander might be in static or state)
                    for lander, rover in all_can_traverse:
                        if rover == mode_rover:
                            # if some lander exists that can_traverse to this rover, we can use take_image at wp with that lander
                            take_then_calibrate_possible = True
                            break
                    if take_then_calibrate_possible:
                        break
            if take_then_calibrate_possible:
                total_cost += 2
                continue

            # 4) If there's have_soil_analysis waypoints and we have a camera with have_image and an empty objective,
            # we can sample_rock (1 -> produce visible), take_image (1 -> produce equipped_for_rock_analysis), calibrate (1) -> total 3.
            sample_then_take_possible = False
            if mode_rover is not None and have_soil_waypoints and have_image_cameras and empty_objectives:
                # Also need some lander that can_traverse to mode_rover for the subsequent take_image/calibrate
                for lander, rover in all_can_traverse:
                    if rover == mode_rover:
                        sample_then_take_possible = True
                        break
            if sample_then_take_possible:
                total_cost += 3
                continue

            # 5) Fallback small finite cost when no short pattern recognized.
            total_cost += 4

        return total_cost