from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split the fact into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a fact (string) matches a pattern given by args.
    Wildcard '*' is supported via fnmatch.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-specific heuristic for the 'rover-2' (secret_domain) PDDL domain.
    The primary planning objective in the provided instances is to make certain
    modes 'full' (predicate: (full modeX)). Two operators can produce (full mode):
      - calibrate: requires the mode to be supported and requires an
        (equipped_for_rock_analysis waypoint lander) together with a can_traverse
        link to the rover where the mode is 'at'.
      - communicate_soil_data: requires the mode to be equipped_for_soil_analysis,
        an equipped_for_rock_analysis (waypoint lander), visible_from waypoint,
        and an appropriate can_traverse link to the rover where the mode is 'at'.
    Both of these require an (equipped_for_rock_analysis ...) fact. An
    (equipped_for_rock_analysis) fact is typically created by take_image,
    which itself requires the waypoint to be visible and a can_traverse relation.

    This heuristic estimates how many actions are still necessary to reach the
    goal (making all goal modes 'full'). It is not required to be admissible,
    and it uses readily available static facts and current-state facts to form
    a compact but discriminative estimate.

    Assumptions
    -----------
    - Static facts (task.static) are always true; we therefore consider them
      together with the current state when testing preconditions that are
      static in examples.
    - The goal formula contains only (full modeX) goals (as in the instances).
    - We do not attempt to perfectly account for shared effects between modes.
      Each non-full mode contributes an estimated independent cost; this keeps
      the heuristic cheap to compute.
    - All facts are represented as strings like "(predicate arg1 arg2)".

    Heuristic Initialization
    ------------------------
    The constructor extracts and stores:
    - The set of goal conditions (task.goals).
    - The static facts (task.static).
    - Useful static indexes (e.g., which modes 'support' calibration) to speed
      up checks during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node (state), the heuristic proceeds as follows:

    1. If the state already satisfies all goal facts, return 0.

    2. Build a working set of facts that is the union of the current state and
       static facts. Many operator preconditions are satisfied by static facts,
       so considering them together simplifies the checks.

    3. Extract relevant predicates from the working set:
       - which modes are already (full ...)
       - which (equipped_for_rock_analysis waypoint lander) facts exist
       - which waypoints are (visible ...) and which are (visible_from ...)
       - the mapping of (at object location) facts (used to find the rover
         where a particular mode is located)
       - which (can_traverse lander rover) relations exist
       - which modes are (equipped_for_soil_analysis ...)
       - which cameras are (have_image ...) and which waypoints have
         (have_soil_analysis ...) (used as a fallback to estimate cost to create
         a visible waypoint)

    4. For each mode that is a goal but not yet full, estimate the number of
       actions needed to make it full. Evaluate the easiest applicable patterns
       first (cheap repairs), falling back to longer sequences when few helpful
       facts are present.

       The per-mode decision rules (ordered by preference):
       - If calibrate or communicate_soil_data is already applicable in the
         working set (i.e., all their preconditions exist), estimate cost = 1.
       - Else if an (equipped_for_rock_analysis ...) fact exists (but some
         other preconditions for the final operator are missing), estimate
         cost = 2 (one action to satisfy the missing precondition(s) and one
         to perform the final operator).
       - Else if there is at least one visible waypoint and there exists a
         can_traverse(link, rover1) (the take_image operator specifically
         requires can_traverse lander rover1 in this domain encoding), then
         take_image can produce the needed equipped_for_rock_analysis in one
         action, and a calibrate/communicate action will then complete the mode:
         estimate cost = 2.
       - Else if there exist (have_image camera), (have_soil_analysis waypoint)
         and an (empty objective ...) (allowing sample_rock or communicate_image_data
         to produce a visible waypoint), estimate cost = 3 (to create a visible
         waypoint and then obtain an equipped_for_rock_analysis and finally
         the calibrate/communicate action).
       - Otherwise, conservatively estimate a small constant fallback cost (4).
         This guarantees a finite heuristic even when many helpful facts are
         missing.

    5. Sum the per-mode estimates and return the sum as the heuristic value.

    Note about admissibility and sharing:
    - The heuristic is deliberately optimistic about some costs (e.g., it
      assumes one action can quickly create a missing precondition) and does
      not model sharing of intermediate facts between different modes. This
      makes it non-admissible but fast and often effective for greedy best-
      first search guidance.
    """

    def __init__(self, task):
        """Initialize with task goals and static facts and derive useful indexes."""
        self.goals = task.goals
        self.static = task.static

        # Precompute quick static lookups to speed per-node checks.
        # Modes that support calibration
        self.supports_modes = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "supports", "*")
        }
        # Modes that are statically equipped_for_soil_analysis (rare but possible)
        self.equipped_soil_static = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "equipped_for_soil_analysis", "*")
        }
        # Precompute any static can_traverse relations
        self.static_can_traverse = {
            (get_parts(f)[1], get_parts(f)[2])
            for f in self.static
            if match(f, "can_traverse", "*", "*")
        }
        # Static at facts (object -> location)
        self.static_at = {
            get_parts(f)[1]: get_parts(f)[2]
            for f in self.static
            if match(f, "at", "*", "*")
        }

    def __call__(self, node):
        """Return an estimated number of actions to reach a goal state from node."""
        state = node.state

        # 1) Return 0 if already at a goal state.
        if self.goals <= state:
            return 0

        # 2) Combine current state with static facts to simplify applicability checks.
        combined = set(state) | set(self.static)

        # 3) Extract relevant predicates from combined.
        # Modes that are already full.
        full_modes = {
            get_parts(f)[1]
            for f in combined
            if match(f, "full", "*")
        }

        # Mapping for at facts: object -> location/rover
        at_map = {
            get_parts(f)[1]: get_parts(f)[2]
            for f in combined
            if match(f, "at", "*", "*")
        }

        # equipped_for_rock_analysis facts: list of (waypoint, lander)
        equipped_for_rock = [
            (get_parts(f)[1], get_parts(f)[2])
            for f in combined
            if match(f, "equipped_for_rock_analysis", "*", "*")
        ]

        # visible and visible_from waypoints
        visible = {
            get_parts(f)[1]
            for f in combined
            if match(f, "visible", "*")
        }
        visible_from = {
            get_parts(f)[1]
            for f in combined
            if match(f, "visible_from", "*")
        }

        # can_traverse relations (lander -> rover)
        can_traverse = {
            (get_parts(f)[1], get_parts(f)[2])
            for f in combined
            if match(f, "can_traverse", "*", "*")
        } | self.static_can_traverse

        # modes that are equipped_for_soil_analysis (could be in static or state)
        equipped_for_soil = {
            get_parts(f)[1]
            for f in combined
            if match(f, "equipped_for_soil_analysis", "*")
        } | self.equipped_soil_static

        # cameras that have images
        have_images = {
            get_parts(f)[1]
            for f in combined
            if match(f, "have_image", "*")
        }

        # waypoints that have soil analysis
        have_soil_analysis = {
            get_parts(f)[1]
            for f in combined
            if match(f, "have_soil_analysis", "*")
        }

        # empty objectives (used by sample_rock and communicate_image_data)
        empty_objectives = {
            get_parts(f)[1]
            for f in combined
            if match(f, "empty", "*")
        }

        # goal modes (assume goals are of the form (full modeX))
        goal_modes = [
            get_parts(g)[1]
            for g in self.goals
            if get_parts(g)[0] == "full"
        ]

        total_estimated = 0

        for mode in goal_modes:
            if mode in full_modes:
                continue  # already satisfied

            # Gather candidate rovers where this mode is at (could be static or state)
            mode_rover = None
            if mode in at_map:
                mode_rover = at_map[mode]  # e.g., 'rover2'
            # Note: it's possible 'at' is not present for this mode; then some preconditions
            # (like at mode rover) will be considered missing.

            # Quick checks: can we apply calibrate directly?
            calibrate_applicable = False
            if mode in self.supports_modes and mode_rover is not None:
                # need an equipped_for_rock_analysis (wp, lander) and (can_traverse lander mode_rover)
                for wp, land in equipped_for_rock:
                    if (land, mode_rover) in can_traverse:
                        calibrate_applicable = True
                        break

            # Quick checks: can we apply communicate_soil_data directly?
            communicate_applicable = False
            if mode in equipped_for_soil and mode_rover is not None:
                # need equipped_for_rock_analysis (wp, lander), visible_from wp, and can_traverse lander mode_rover
                for wp, land in equipped_for_rock:
                    if wp in visible_from and (land, mode_rover) in can_traverse:
                        communicate_applicable = True
                        break

            if calibrate_applicable or communicate_applicable:
                # One action suffices to make the mode full.
                total_estimated += 1
                continue

            # If we already have some equipped_for_rock_analysis anywhere, but missing
            # at(mode rover) or can_traverse link or other small preconditions, assume
            # one action to fix missing preconditions + one to apply calibrate/communicate.
            if equipped_for_rock:
                total_estimated += 2
                continue

            # If we have a visible waypoint and a can_traverse link for rover1 the
            # take_image operator can create an equipped_for_rock_analysis in one step.
            # Domain's take_image precondition includes (can_traverse ?v4 rover1)
            # so we check for any (land, rover1) in can_traverse and a visible waypoint.
            if visible and any(r == "rover1" for (_, r) in can_traverse):
                # take_image + calibrate/communicate => estimate 2 actions
                total_estimated += 2
                continue

            # If we can produce a visible waypoint from have_image + have_soil_analysis
            # and an empty objective (sample_rock or communicate_image_data), then we
            # estimate 3 actions: produce visible, take_image (equip), calibrate/communicate.
            if have_images and have_soil_analysis and empty_objectives:
                total_estimated += 3
                continue

            # Fallback conservative cost; ensures finite heuristic estimate.
            total_estimated += 4

        return total_estimated