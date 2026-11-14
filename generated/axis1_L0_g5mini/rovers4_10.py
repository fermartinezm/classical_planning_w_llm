from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of tokens of a PDDL fact string, removing surrounding parentheses."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check whether a fact (string) matches a pattern of tokens. Wildcards '*' allowed.
    Example: match('(at rover1 waypoint2)', 'at', '*', 'waypoint2') -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the PDDL 'rover' domain. The heuristic estimates
    the (non-admissible) number of actions required to reach a goal state by
    summing cheap, relaxations-based costs for each not-yet-satisfied goal fact
    (communicated_image_data, communicated_soil_data, communicated_rock_data).
    It reasons about the typical goal pipeline:
      - imaging: calibrate -> take_image -> communicate
      - sampling: sample_soil/sample_rock -> communicate
    The estimate counts required action steps (navigate, calibrate, take_image,
    sample_*, communicate_*, drop) but uses a compact approximate model for
    navigation (0/1 moves) and resource constraints (stores/calibration) to
    keep computation efficient while providing strong pruning guidance.

    Assumptions
    -----------
    - Static facts (supports, on_board, calibration_target, visible_from, visible,
      at_lander, store_of, equipped_for_*) are available in task.static and do
      not change.
    - Navigation between any two relevant waypoints is approximated by at most
      one "navigate" action when the rover is not already at the desired waypoint.
      This underestimates true path lengths but is effective to guide greedy
      search.
    - Stores are assumed to be reusable with a single drop action if no empty
      store exists for a rover; if no rover is equipped for a sensing task, a
      moderate penalty is charged instead of returning infinity.
    - Calibrations are assumed to be required if no suitable "calibrated" fact
      exists in the current state for a camera that supports the requested mode.
    - We ignore some detailed ordering interactions (for example, camera becomes
      uncalibrated after take_image) except in so far as they affect immediate
      costs for remaining goals.

    Heuristic Initialization
    ------------------------
    In the constructor we extract static relationships into convenient maps/sets:
      - self.supports: camera -> set(modes)
      - self.on_board: camera -> rover
      - self.calibration_target: camera -> objective
      - self.visible_from: objective -> set(waypoints)
      - self.visible: waypoint -> set(waypoints)  (waypoint a visible b stored as a->b)
      - self.at_landers: set of waypoint positions where a lander is located
      - self.store_of: store -> rover
      - self.equipped_for_imaging / _soil / _rock: sets of rovers with respective capabilities
    The constructor also stores the task goals (task.goals) for guiding the per-state
    computation.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given search node (state), the heuristic computes a sum of per-goal
    costs using the following approximate reasoning:

    1) If the entire goal set is already satisfied in the state, return 0.

    2) Parse the current state into quick-access structures:
       - rover positions: where each rover currently is (from (at rover waypoint))
       - which cameras are currently calibrated (calibrated camera rover)
       - which analysis results or images already exist (have_soil_analysis,
         have_rock_analysis, have_image)
       - store emptiness/fullness (empty/full facts) to estimate if sampling is
         possible without a drop.

    3) For each unsatisfied goal:
       - communicated_image_data (objective, mode):
           Evaluate candidate (camera, rover) pairs that can produce the
           required mode (camera supports mode and is on_board some rover that
           is equipped_for_imaging). For each candidate compute:
             * take_image cost: 1
             * calibrate cost: 0 if a matching calibrated fact exists in state
               for that camera+rover; otherwise 1 (calibrate)
             * navigation to an imaging viewpoint p in visible_from[objective]:
               0 if the rover is already at some p; otherwise 1
             * navigation to a lander-visible waypoint for communication:
               0 if the rover already (or after imaging at p) is at a waypoint
               that is visible to a lander waypoint; otherwise 1
             * communication cost: 1
           Choose the minimal cost among candidates. If no candidate exists
           (no suitable camera/rover), add a moderate penalty (default 6) to
           reflect the extra cost / infeasibility in a soft way.
       - communicated_soil_data / communicated_rock_data (waypoint):
           Consider rovers that are equipped for the required analysis and have
           an associated store (store_of). For each rover compute:
             * if have_*_analysis already exists for that rover+waypoint: only
               communication (1) + navigation to lander if needed
             * otherwise:
                 - sample action: 1
                 - need empty store: 0 if an empty store_of that rover exists in state,
                   else 1 (drop) to free it
                 - navigation to sample waypoint: 0 if rover already there else 1
                 - navigation to lander-visible waypoint for communication:
                     0 if sample waypoint itself is visible to a lander waypoint
                     (so communication can happen from there) or if rover already
                     at such a waypoint; otherwise 1
                 - communication: 1
           Choose minimal cost among rovers; if none available, add penalty (6).

    4) Sum the minimal per-goal costs. This is the heuristic returned.

    Notes
    -----
    - The heuristic is intentionally approximate (not admissible) but aims to
      capture required action types and likely navigation steps without solving
      the full subproblem. It is designed to be fast to compute and to reduce
      node expansions in greedy best-first search by preferring states that
      already have parts of the sensing/communication pipeline completed.
    """

    def __init__(self, task):
        # store goals and static facts for quicker access
        self.goals = task.goals
        static_facts = task.static

        # maps and sets extracted from static facts
        # camera -> set(modes)
        self.supports = {}
        # camera -> rover
        self.on_board = {}
        # camera -> calibration_target_objective
        self.calibration_target = {}
        # objective -> set(waypoints)
        self.visible_from = {}
        # waypoint -> set(waypoints)  (which other waypoint is visible from this one)
        self.visible = {}
        # set of waypoints where a lander is located
        self.at_landers = set()
        # store -> rover
        self.store_of = {}
        # equipped sets
        self.equipped_for_imaging = set()
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()

        for fact in static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "supports":
                cam, mode = parts[1], parts[2]
                self.supports.setdefault(cam, set()).add(mode)
            elif pred == "on_board":
                cam, rover = parts[1], parts[2]
                self.on_board[cam] = rover
            elif pred == "calibration_target":
                cam, obj = parts[1], parts[2]
                self.calibration_target[cam] = obj
            elif pred == "visible_from":
                obj, wp = parts[1], parts[2]
                self.visible_from.setdefault(obj, set()).add(wp)
            elif pred == "visible":
                wp_from, wp_to = parts[1], parts[2]
                self.visible.setdefault(wp_from, set()).add(wp_to)
            elif pred == "at_lander":
                # at_lander lander waypoint
                _, wp = parts[1], parts[2]
                # store only the waypoint (communication requires visibility to this waypoint)
                self.at_landers.add(wp)
            elif pred == "store_of":
                store, rover = parts[1], parts[2]
                self.store_of[store] = rover
            elif pred == "equipped_for_imaging":
                self.equipped_for_imaging.add(parts[1])
            elif pred == "equipped_for_soil_analysis":
                self.equipped_for_soil.add(parts[1])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_for_rock.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Return 0 if goal reached
        if self.goals <= state:
            return 0

        # Parse dynamic facts from the current state
        # rover -> waypoint
        rover_at = {}
        # calibrated camera facts: set of (camera, rover)
        calibrated = set()
        # have_image facts: set of (rover, objective, mode)
        have_image = set()
        # have_soil_analysis set (rover, waypoint)
        have_soil = set()
        # have_rock_analysis set (rover, waypoint)
        have_rock = set()
        # empty stores set and full stores set
        empty_stores = set()
        full_stores = set()

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # we only record rover positions here (others exist, but we need rover positions)
                # rovers are the ones that appear in equipped sets or on_board targets,
                # but we accept all facts and store those that look like rovers by convention:
                rover_at[obj] = loc
            elif pred == "calibrated" and len(parts) == 3:
                cam, rover = parts[1], parts[2]
                calibrated.add((cam, rover))
            elif pred == "have_image" and len(parts) == 4:
                rover, obj, mode = parts[1], parts[2], parts[3]
                have_image.add((rover, obj, mode))
            elif pred == "have_soil_analysis" and len(parts) == 3:
                rover, wp = parts[1], parts[2]
                have_soil.add((rover, wp))
            elif pred == "have_rock_analysis" and len(parts) == 3:
                rover, wp = parts[1], parts[2]
                have_rock.add((rover, wp))
            elif pred == "empty":
                empty_stores.add(parts[1])
            elif pred == "full":
                full_stores.add(parts[1])

        total_cost = 0

        # Utility to check whether a waypoint is visible to any lander waypoint
        def waypoint_has_lander_visibility(wp):
            # If there exists a lander waypoint y such that visible(wp, y)
            return any(y in self.visible.get(wp, set()) for y in self.at_landers)

        # For each goal, compute minimal approximate cost to satisfy it
        for goal in self.goals:
            if goal in state:
                # already satisfied
                continue

            parts = get_parts(goal)
            pred = parts[0]

            if pred == "communicated_image_data":
                # goal: (communicated_image_data objective mode)
                objective, mode = parts[1], parts[2]

                # If any rover already has the image for this objective and mode, we only
                # need to communicate it (1 action) plus possible navigation.
                # We'll search across candidates (camera, rover) that can produce the image.
                best_cost = None

                # Quick check: if any have_image exists for this objective/mode by any rover,
                # then only communication remains
                if any((r, objective, mode) in have_image for r in rover_at.keys()):
                    # communication cost 1 plus possible navigation to lander
                    min_comm_nav = 1  # communication action
                    # If there exists a rover r with have_image that is already at a lander-visible waypoint -> nav 0
                    for (r, obj, m) in list(have_image):
                        if obj == objective and m == mode:
                            wp = rover_at.get(r)
                            if wp is not None and waypoint_has_lander_visibility(wp):
                                min_comm_nav = 1  # cost already counted is the communication action only
                                break
                            # otherwise we assume one navigation needed to reach a lander-visible waypoint
                    best_cost = min_comm_nav

                # Otherwise, consider producing the image from a suitable camera+rover candidate
                # candidate criteria: camera supports mode, camera on_board rover, rover equipped_for_imaging
                for cam, rover in self.on_board.items():
                    # camera must support required mode
                    if mode not in self.supports.get(cam, set()):
                        continue
                    if rover not in self.equipped_for_imaging:
                        continue

                    # Cost components
                    cost = 0
                    # 1 for take_image (unless already have_image, which we checked above)
                    cost += 1

                    # calibrate cost: 0 if (cam, rover) calibrated in current state, else 1
                    if (cam, rover) not in calibrated:
                        cost += 1  # calibrate

                    # navigation to an imaging viewpoint p (some p in visible_from[objective])
                    imaging_waypoints = self.visible_from.get(objective, set())
                    # if there are no imaging waypoints known, set nav_imaging = 1 conservatively
                    if not imaging_waypoints:
                        nav_imaging = 1
                    else:
                        # if rover currently at some imaging waypoint -> 0 else 1
                        cur_wp = rover_at.get(rover)
                        if cur_wp is not None and cur_wp in imaging_waypoints:
                            nav_imaging = 0
                        else:
                            nav_imaging = 1
                    cost += nav_imaging

                    # communication: 1 action always (communicate_image_data),
                    # plus possible navigation to lander-visible waypoint (0/1).
                    # However, if we plan to move to an imaging waypoint p and that p is
                    # visible to a lander waypoint, then after imaging we can communicate
                    # without extra navigation. So we check whether any imaging waypoint
                    # is lander-visible.
                    comm_action_cost = 1
                    nav_comm = 1  # default assume one navigation needed
                    # check if current rover position already lander-visible
                    cur_wp = rover_at.get(rover)
                    if cur_wp is not None and waypoint_has_lander_visibility(cur_wp):
                        nav_comm = 0
                    else:
                        # if moving to an imaging waypoint that itself has lander visibility,
                        # then nav_comm = 0 because the move for imaging suffices
                        for p in imaging_waypoints:
                            if p in self.visible and waypoint_has_lander_visibility(p):
                                # moving to p for imaging (nav_imaging=1) also enables communication.
                                # If nav_imaging was 0, we also benefit. In either case, we can
                                # avoid an extra nav.
                                if nav_imaging == 1 or (cur_wp is not None and cur_wp == p):
                                    nav_comm = 0
                                    break
                    cost += comm_action_cost + nav_comm

                    if best_cost is None or cost < best_cost:
                        best_cost = cost

                if best_cost is None:
                    # No suitable camera/rover candidate found: charge a penalty
                    best_cost = 6

                total_cost += best_cost

            elif pred == "communicated_soil_data" or pred == "communicated_rock_data":
                # goal: (communicated_soil_data waypoint) or (communicated_rock_data waypoint)
                wp = parts[1]
                is_soil = pred == "communicated_soil_data"
                # Determine which rovers are equipped for this analysis
                candidate_rovers = (
                    self.equipped_for_soil if is_soil else self.equipped_for_rock
                )

                best_cost = None

                # If some rover already has the analysis for this waypoint, we only need to communicate
                # check any have_soil/have_rock
                if is_soil:
                    already_have = [r for (r, w) in have_soil if w == wp]
                else:
                    already_have = [r for (r, w) in have_rock if w == wp]

                if already_have:
                    # communication cost 1 plus possible navigation to lander
                    min_comm = None
                    for r in already_have:
                        cur_wp = rover_at.get(r)
                        comm_cost = 1
                        # if current location visible to lander -> no navigation
                        if cur_wp is not None and waypoint_has_lander_visibility(cur_wp):
                            nav_comm = 0
                        else:
                            nav_comm = 1
                        val = comm_cost + nav_comm
                        if min_comm is None or val < min_comm:
                            min_comm = val
                    best_cost = min_comm

                # otherwise consider sampling by candidate rovers
                for r in candidate_rovers:
                    # must have a store belonging to this rover; find any store s with store_of s r
                    stores = [s for s, rr in self.store_of.items() if rr == r]
                    if not stores:
                        # rover cannot store samples -> skip
                        continue

                    # cost components
                    cost = 0
                    # sample action
                    cost += 1

                    # store availability: if any of rover's stores empty -> no drop cost, else assume one drop needed
                    store_available = any(s in empty_stores for s in stores)
                    if not store_available:
                        # if some store is full, we can drop it (1 action). If none full nor empty (unlikely),
                        # still charge 1 as a conservative estimate.
                        cost += 1

                    # navigation to sample waypoint
                    cur_wp = rover_at.get(r)
                    if cur_wp is not None and cur_wp == wp:
                        nav_sample = 0
                    else:
                        nav_sample = 1
                    cost += nav_sample

                    # communication: 1 action plus possible navigation
                    comm_cost = 1
                    # If the sample waypoint itself is visible to a lander waypoint, then communication can be done from there
                    if waypoint_has_lander_visibility(wp):
                        nav_comm = 0
                    else:
                        # else, if rover is already at a lander-visible waypoint (before or after sampling), nav may be 0
                        if cur_wp is not None and waypoint_has_lander_visibility(cur_wp):
                            nav_comm = 0
                        else:
                            # we assume one extra navigation to reach a lander-visible waypoint
                            nav_comm = 1
                    cost += comm_cost + nav_comm

                    if best_cost is None or cost < best_cost:
                        best_cost = cost

                if best_cost is None:
                    # No candidate rover available: charge penalty
                    best_cost = 6

                total_cost += best_cost

            else:
                # For any other goal type not explicitly handled, fall back to counting 1 action.
                # This keeps the heuristic finite and simple.
                total_cost += 1

        # Ensure heuristic is an integer and finite
        if total_cost is None:
            return 0
        return int(total_cost)