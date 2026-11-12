from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return the components of a PDDL fact string by removing parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (a simplified sokoban-like domain).
    The heuristic estimates how many actions are required to make every goal box satisfy
    the predicate (adjacent-2 box). It reasons about the two-step pattern visible in the
    domain: (move) can produce (at box) from (adjacent box location) when the robot is
    at that location, and (push) produces (adjacent-2 box) from (at box) when the robot
    stands at a location that matches the static relationship (at-robot box location).
    The heuristic computes an individual estimate for each goal box and returns the sum.

    Assumptions
    -----------
    - Goals are of the form "(adjacent-2 boxX)". Only boxes mentioned in goal conditions
      are considered.
    - Static facts encode relationships like "(adjacent boxX locationY)" and
      "(at-robot boxX locationY)" while the robot's current location(s) are encoded by
      dynamic facts "(at-robot-2 locationY)" and whether a box is in the 'at' state is
      encoded by "(at boxX)".
    - Connectivity/robot movement (move-2 / push-2) is approximated by a small
      integer cost; the heuristic does not attempt full pathfinding for the robot.
    - The heuristic is allowed to be inadmissible (it can overestimate), but it should
      return 0 on goal states and remain finite for solvable states.

    Heuristic Initialization
    ------------------------
    - Extract the set of boxes that appear in goal conditions of the form (adjacent-2 BOX).
    - Extract static relationships from task.static that are useful:
        * static_adjacent: mapping box -> set(locations) from "(adjacent box loc)" facts
        * static_atrobot: mapping box -> set(locations) from "(at-robot box loc)" facts
      These static maps speed up per-state computation in __call__.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal box B:
      1. If "(adjacent-2 B)" is already true in the current state => cost 0 for B.
      2. Otherwise, inspect the dynamic state and static information to see how close B is
         to achieving (adjacent-2 B). From the domain definition one way to reach
         (adjacent-2 B) is:
           - (move) produces "(at B)" given "(adjacent B L)" and "(at-robot-2 L)".
           - (push) produces "(adjacent-2 B)" given "(at B)", "(at-robot B L)", and "(at-robot-2 L)".
      3. Using the facts available, the heuristic tries to estimate the minimum
         number of actions to get from the current state to satisfying (adjacent-2 B).
         Practical (conservative, finite) cost assignments used by the heuristic:
           - If "(at B)" is true and there exists a location L such that
             "(at-robot B L)" (static or state) and "(at-robot-2 L)" (state) are true:
               => push is immediately applicable; cost for B is 1.
           - If neither the box is "at" nor robot stands in a favorable location,
             but there exists a location L where "(adjacent B L)" and "(at-robot-2 L)"
             and "(at-robot B L)" are true:
               => move then push can be performed; cost for B is 2.
           - If only some of the preconditions are present, the heuristic accounts
             for additional robot movement and necessary actions by assigning
             small integer penalties. The concrete cost levels used are:
               * box already (at): best remaining cost is 1 (push) or 2 if robot must move.
               * box not (at): best case 2 (move then push), worse cases 3 or 4 to model
                 additional robot repositioning or missing static relations.
      4. If there is insufficient information (no known locations related to the box),
         a conservative finite fallback cost is used (4 actions per unsolved box).
    The final heuristic value is the sum of the per-box estimates.

    This heuristic is intentionally simple and fast to compute. It uses static facts to
    choose low estimated costs where the box and robot are already favorably aligned,
    and larger small integer estimates otherwise. It returns 0 exactly when all goal
    adjacent-2 facts hold in the state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic using static facts and goal extraction.

        Parameters
        ----------
        task : Task
            The planning task object containing static facts and goals.
        """
        # Goal conditions as provided by the planner (set of fact strings)
        self.goals = task.goals

        # Static facts (those not changed by operators) are useful to find
        # invariant relationships. We parse the static set once here.
        static_facts = task.static if task.static is not None else frozenset()

        # static_adjacent: box -> set(loc) from (adjacent box loc)
        self.static_adjacent = {}
        # static_atrobot: box -> set(loc) from (at-robot box loc)
        self.static_atrobot = {}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "adjacent" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.static_adjacent.setdefault(box, set()).add(loc)
            elif pred == "at-robot" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                self.static_atrobot.setdefault(box, set()).add(loc)

        # Extract goal boxes that require adjacent-2
        self.goal_boxes = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "adjacent-2" and len(parts) >= 2:
                self.goal_boxes.add(parts[1])

    def __call__(self, node):
        """
        Compute the heuristic value for the given search node.

        Parameters
        ----------
        node : Node
            A search node with attribute .state (a frozenset of fact strings).

        Returns
        -------
        int
            Estimated number of actions to satisfy all goal adjacent-2 predicates.
        """
        state = node.state
        # Fast check: if all goals already satisfied, heuristic = 0
        if self.goals <= state:
            return 0

        # Gather dynamic information useful for per-box reasoning:
        # - which locations are currently occupied by robot (at-robot-2 L)
        robot_locations = {get_parts(f)[1] for f in state if get_parts(f)[0] == "at-robot-2" and len(get_parts(f)) >= 2}
        # - which boxes are currently in 'at' state: (at box)
        boxes_at = {get_parts(f)[1] for f in state if get_parts(f)[0] == "at" and len(get_parts(f)) >= 2}
        # - dynamic adjacent facts: box -> set(loc)
        dyn_adjacent = {}
        # - dynamic at-robot facts (rarely dynamic, but consider state as well): box -> set(loc)
        dyn_atrobot = {}

        for f in state:
            parts = get_parts(f)
            if not parts:
                continue
            pred = parts[0]
            if pred == "adjacent" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                dyn_adjacent.setdefault(box, set()).add(loc)
            elif pred == "at-robot" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                dyn_atrobot.setdefault(box, set()).add(loc)

        total_cost = 0

        # For each box in the goals compute an estimated cost
        for box in self.goal_boxes:
            # If already satisfied, skip
            if f"(adjacent-2 {box})" in state:
                continue

            # Collect candidate locations from both static and dynamic facts
            locs_adj = set()
            locs_adj.update(self.static_adjacent.get(box, set()))
            locs_adj.update(dyn_adjacent.get(box, set()))

            locs_atrobot = set()
            locs_atrobot.update(self.static_atrobot.get(box, set()))
            locs_atrobot.update(dyn_atrobot.get(box, set()))

            # Whether box is currently in the 'at' state
            is_at = box in boxes_at

            # If (at BOX) and there exists a location L where both (at-robot BOX L)
            # (static or dynamic) and robot actually stands at L, then push is immediate.
            immediate_push_possible = any((L in robot_locations) for L in locs_atrobot)

            if is_at and immediate_push_possible:
                # Only push needed
                total_cost += 1
                continue

            # Try to find a location L that makes move+push achievable quickly:
            # - For move we need (adjacent BOX L) and (at-robot-2 L) (robot at that L right now)
            # - For push after move we additionally want (at-robot BOX L)
            best_cost_for_box = None

            candidate_locs = set(locs_adj) | set(locs_atrobot)
            # If no candidate locs known, fall back to a conservative finite cost
            if not candidate_locs:
                # fallback: assume move + some repositioning + push => cost 4
                total_cost += 4
                continue

            for L in candidate_locs:
                has_adj = L in locs_adj
                has_atrobotbox = L in locs_atrobot
                robot_at_L = L in robot_locations

                # Prefer lower costs; evaluate scenarios:
                if is_at:
                    # only push needed but may need robot move
                    if robot_at_L and has_atrobotbox:
                        cost = 1  # immediate push
                    elif has_atrobotbox:
                        cost = 2  # robot must move to L then push
                    elif robot_at_L:
                        cost = 2  # need (at-robot box L) relation (assume one extra op)
                    else:
                        cost = 3  # need both establishing at-robot-box relation or moves
                else:
                    # box not in at-state; we typically need move (to produce at) and then push
                    if has_adj and robot_at_L and has_atrobotbox:
                        cost = 2  # move then push (both preconditions present at L)
                    elif has_adj and robot_at_L:
                        cost = 3  # move, then need to obtain at-robot box relation or reposition
                    elif has_adj and has_atrobotbox:
                        # robot not currently at L: need to move robot (1), move box (1), push (1) => 3
                        cost = 3
                    elif has_adj:
                        # need robot relocation (>=1) then move then push -> assume 4
                        cost = 4
                    elif has_atrobotbox and robot_at_L:
                        # strange case: push would work if box already (at), but it isn't; need to make it (at)
                        # making at requires (adjacent BOX L) which we don't have for L; assume 3
                        cost = 3
                    else:
                        # worst reasonable fallback for this candidate
                        cost = 4

                if best_cost_for_box is None or cost < best_cost_for_box:
                    best_cost_for_box = cost
                    # early exit if we reached the minimum plausible cost
                    if best_cost_for_box == 2 or best_cost_for_box == 1:
                        break

            # If for some reason best_cost_for_box is still None (shouldn't happen), use fallback 4
            if best_cost_for_box is None:
                best_cost_for_box = 4

            total_cost += best_cost_for_box

        # Ensure heuristic finite and integer
        return int(total_cost)