from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a PDDL fact string like '(pred a b)' -> ['pred','a','b']"""
    return fact[1:-1].split()


def match(fact, *pattern):
    """Return True if fact matches pattern with wildcards '*' allowed."""
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    This heuristic estimates how many actions are required to reach a goal state
    in the secret_domain (a variant of blocksworld where the goals are expressed
    as arm-empty facts). The heuristic focuses on satisfying the goal arm-empty
    predicates. It counts mismatches between the current arm-empty mapping and
    the goal mapping and adds an estimate for direct conflicts that will require
    extra operations to resolve.

    Assumptions
    -----------
    - The goal conditions for this domain are (arm-empty <arm> <object>) facts.
      The planner only needs these facts to be true at the end (this matches
      the provided examples).
    - Other predicates (on, on-table, holding, clear) influence what operators
      are applicable, but deriving an exact cost from them is expensive. The
      heuristic therefore uses a compact but informative approximation:
      - One operation is often sufficient to change the second argument of an
        arm-empty fact (many domain actions toggle arm-empty relations).
      - If the goal requires a particular object to be the partner of a given
        arm but that object is currently assigned to a different arm, an extra
        operation will generally be needed to resolve the conflict.
    - The heuristic is not required to be admissible (it may overestimate),
      but should be efficiently computable.

    Heuristic Initialization
    ------------------------
    In the constructor we:
    - Extract the goal arm-empty pairs from task.goals and store them as a
      mapping goal_arm_map: arm -> desired_partner.
    - Extract static facts (task.static) and organize those facts into sets for
      quick lookup (holding, on, on-table). These are not used for exact cost
      computation here, but are kept to allow future refinements and to respect
      the requirement to extract static information.
    - No expensive preprocessing is performed.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node/state the heuristic proceeds as follows:

    1. Goal check:
       - If all goal facts are already present in the state, return 0.

    2. Read current arm-empty mapping:
       - Parse the state's facts for (arm-empty ARM OBJ) and build a mapping
         current_arm_map: ARM -> partner_OBJ.

    3. Count direct mismatches:
       - For every goal (arm -> desired_obj):
         - If the arm is absent in current_arm_map or the current partner != desired_obj,
           that is one mismatch and we add +1 to the heuristic. This models the
           (approximate) fact that at least one action is needed to change the
           arm-empty relation for that arm.

    4. Count conflicts (additional cost):
       - Some desired partners might already be assigned to other arms in the
         current state. Each distinct desired object that is currently assigned
         to a different arm represents a conflict that will require at least
         one additional action (e.g., to free that object or reassign the other arm).
       - We count how many distinct desired partner objects are currently held
         by some other arm (i.e., assigned in current_arm_map to another arm)
         and add +1 for each such distinct conflict. This reduces overly
         optimistic estimates when multiple goals compete for the same object.

    5. Combine:
       - The final heuristic is the sum of mismatches and conflicts. This is a
         simple, efficient integer estimate that is 0 iff the goal is reached,
         finite for all states, and captures both missing and contested arm-empty
         assignments to guide greedy best-first search.

    Notes
    -----
    - The heuristic purposefully avoids deep reasoning over the large 'on'
      graph and static 'holding' facts because such computation is expensive
      (and the domain instances may contain many cyclic 'on' relations).
    - The heuristic is designed to be cheap (linear in the number of facts)
      while still being informative enough to steer the search away from
      states that are far from satisfying the goal arm-empty mapping.
    """

    def __init__(self, task):
        # Store the set of goal facts; used to check goal states quickly.
        self.goals = task.goals

        # Extract goal arm-empty mapping: arm -> desired_partner
        self.goal_arm_map = {}
        for g in self.goals:
            parts = get_parts(g)
            if parts and parts[0] == "arm-empty" and len(parts) >= 3:
                arm, partner = parts[1], parts[2]
                self.goal_arm_map[arm] = partner

        # Extract static facts into useful containers for potential refinements.
        self.static_holding = {fact for fact in task.static if match(fact, "holding", "*", "*")}
        self.static_on = {fact for fact in task.static if match(fact, "on", "*", "*")}
        self.static_on_table = {fact for fact in task.static if match(fact, "on-table", "*", "*")}

    def __call__(self, node):
        state = node.state

        # 1) If the goal is already reached, heuristic is zero.
        if self.goals <= state:
            return 0

        # 2) Build current mapping of arm -> partner from arm-empty facts in state.
        current_arm_map = {}
        for fact in state:
            if match(fact, "arm-empty", "*", "*"):
                _, arm, partner = get_parts(fact)
                current_arm_map[arm] = partner

        # 3) Count mismatches: every arm in the goal that doesn't currently match
        # contributes at least 1 action.
        mismatches = 0
        # Track desired partners that are in conflict (assigned to another arm).
        conflicting_desired_partners = set()

        # Build reverse mapping partner -> arm for current state to detect conflicts.
        partner_to_arm = {}
        for a, p in current_arm_map.items():
            partner_to_arm[p] = a

        for arm, desired_partner in self.goal_arm_map.items():
            current_partner = current_arm_map.get(arm)
            if current_partner != desired_partner:
                mismatches += 1
                # If the desired partner is currently assigned to some other arm,
                # record this as a conflict (it will likely require an extra op).
                owner = partner_to_arm.get(desired_partner)
                if owner is not None and owner != arm:
                    conflicting_desired_partners.add(desired_partner)

        # 4) Conflicts: each distinct desired partner currently held by a different arm
        # will typically need an extra action to resolve.
        conflicts = len(conflicting_desired_partners)

        # Final heuristic: mismatches plus conflicts (integer).
        heuristic_value = mismatches + conflicts

        # Ensure finiteness and non-negativity.
        if heuristic_value < 0:
            heuristic_value = 0

        return heuristic_value