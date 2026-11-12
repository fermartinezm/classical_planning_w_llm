from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract parts of a PDDL fact by removing surrounding parentheses and splitting by space.

    For example: "(qxafq0yb rap6j3pu)" -> ["qxafq0yb", "rap6j3pu"]
    """
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check if a fact matches a pattern with possible wildcards "*".

    Parameters:
    - fact (str): the fact string.
    - pattern (strs): sequence of pattern elements.

    Returns:
    - bool: True if all corresponding parts match, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(pattern):
        return False  # pattern longer than fact parts
    for part, pat in zip(parts, pattern):
        if pat != "*" and part != pat:
            return False
    return True


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (aixg4uih) domain.

    Summary:
    -----------
    Estimates the number of actions needed to achieve all goal facts (qxafq0yb ?r7dv9).
    Goals require achieving qxafq0yb facts for specific objects of type z65nyce9.
    This heuristic focuses on the progress of those goal facts and tracks necessary
    intermediate conditions such as:
    - Disabling dym1mop6 predicates, which seem required to achieve qxafq0yb.
    - The presence of required (qrohx31m ?axwhj ?imdah) predicates related to s1clodkz and u59rqa6u objects.
    - The enabling and disabling of okcl3dr0, xx5ljjai, omcsgizs predicates on ggld6jmz objects.
    - Ensures proper movement of locations for ?axwhj tokens (via qrohx31m facts).
    - Actions which move qrohx31m facts between u59rqa6u objects (ss17onrb).

    Assumptions:
    -----------
    - The main goal is to get all required (qxafq0yb ...) facts true.
    - Each (qxafq0yb ?r7dv9) can only be achieved by actions that remove (dym1mop6 ?c3xfw ?axwhj)
      and that have certain preconditions involving (qrohx31m ?axwhj ?imdah), (xx5ljjai ?c3xfw),
      (zmomsdb3 ?r7dv9), and (waexgd4o ?r7dv9 ?imdah).
    - The actions are mostly about moving "tokens" ?axwhj between u59rqa6u objects, enabling 
      or disabling predicates on ggld6jmz and z65nyce9 types.
    - The static facts give invariants like (waexgd4o ?r7dv9 ?imdah) mappings.

    Heuristic Initialization:
    -------------------------
    From static facts, extract:
    - Maps of waexgd4o (?r7dv9 to ?imdah), for quick lookup.
    - Set of all relevant s1clodkz, u59rqa6u, and ggld6jmz objects from static facts.
    - The constant u59rqa6u object (s154x56h) identified in domain constants.
    From goals:
    - The list/set of goal objects of type z65nyce9 that must satisfy (qxafq0yb).

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify which goal predicates (qxafq0yb ?r7dv9) are not yet satisfied.
       Each missing predicate implies at least one corresponding goal-achieving action.
    2. For each unsatisfied goal fact (qxafq0yb r), the heuristic estimates:
       a. The number of actions to remove the (dym1mop6 ?c3xfw ?axwhj) conditions blocking qxafq0yb.
       b. The necessary movement of the s1clodkz token locations (qrohx31m facts) to appropriate u59rqa6u values.
       c. The conversion of okcl3dr0 to either xx5ljjai or omcsgizs depending on action sequences.
    3. Count:
       - Number of (dym1mop6 ...) facts currently true blocking goals.
       - Number of (qrohx31m ?axwhj ?imdah) facts at incorrect u59rqa6u locations (not matching the required waexgd4o).
       - Number of ggld6jmz objects still with okcl3dr0 or not yet transformed into xx5ljjai or omcsgizs as required.
       - Consider presence or absence of special predicates (xx5ljjai, omcsgizs) that enable key actions.
    4. Estimate minimal moves of s1clodkz tokens using (ss17onrb) to move qrohx31m facts from source u59rqa6u to target u59rqa6u.
       Each move estimated as one action per misplaced token.
    5. Combine estimates for each unsatisfied goal, summing the required conversions and moves.
    6. The heuristic value is a weighted sum of these counts since each corresponds roughly to one action to fix or progress.

    The heuristic is thus a close, efficient domain-specific approximation of the number of required fix and transport actions.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract the goal objects of type z65nyce9 that must have (qxafq0yb) true.
        self.goal_z65nyce9 = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "qxafq0yb" and len(parts) == 2:
                self.goal_z65nyce9.add(parts[1])

        # Extract the mapping of waexgd4o (z65nyce9 to u59rqa6u) from static facts.
        self.waexgd4o_map = {}
        for fact in self.static:
            if match(fact, "waexgd4o", "*", "*"):
                parts = get_parts(fact)
                z_obj = parts[1]
                u_obj = parts[2]
                self.waexgd4o_map[z_obj] = u_obj

        # Identify all relevant s1clodkz objects appearing in qrohx31m predicates
        self.s1clodkz_objects = set()
        for fact in self.static | self.goals | task.initial_state:
            if match(fact, "qrohx31m", "*", "*"):
                parts = get_parts(fact)
                self.s1clodkz_objects.add(parts[1])

        # Find the special u59rqa6u constant s154x56h from domain constants assumed fixed:
        # We assume s154x56h is known and used as the initial u59rqa6u source location.
        self.u59rqa6u_root = "s154x56h"

        # Pre-extract all ggld6jmz objects (from facts mentioning them)
        self.ggld6jmz_objects = set()
        for fact in self.static | self.goals | task.initial_state:
            parts = get_parts(fact)
            # ggld6jmz appear in predicates: okcl3dr0, xx5ljjai, omcsgizs, dym1mop6 (first arg)
            if parts[0] in {"okcl3dr0", "xx5ljjai", "omcsgizs", "dym1mop6"} and len(parts) >= 2:
                self.ggld6jmz_objects.add(parts[1])

    def __call__(self, node):
        state = node.state

        # 1) Is goal reached? heuristic = 0
        # If all goal qxafq0yb are satisfied, heuristic is zero.
        achieved_goals = {obj for obj in self.goal_z65nyce9 if f"(qxafq0yb {obj})" in state}
        if achieved_goals == self.goal_z65nyce9:
            return 0

        # Count how many goal qxafq0yb are unsatisfied.
        remaining_goals = self.goal_z65nyce9 - achieved_goals
        if not remaining_goals:
            return 0

        # Helper to count facts matching a predicate and arguments (supports wildcards).
        def count_facts(pred, *args):
            c = 0
            for fact in state:
                if match(fact, pred, *args):
                    c += 1
            return c

        # 2) Count dym1mop6 facts still true: one per not-yet fixed ggld6jmz and s1clodkz token
        n_dym = count_facts("dym1mop6", "*", "*")

        # 3) For each goal object, check if the corresponding (qxafq0yb ?) is missing.
        # We estimate moves needed to place s1clodkz tokens at correct u59rqa6u locals via qrohx31m predicates.

        # Extract current qrohx31m facts: mapping s1clodkz tokens to u59rqa6u locations
        # The goal is for each token associated with each ggld6jmz to be located at some waexgd4o ?imdah linked to the z65nyce9.
        # However, from precondition of l6w9j7tw and eav722gv, the qrohx31m must involve s154x56h or appropriate u59rqa6u.

        token_locations = {}
        for fact in state:
            if match(fact, "qrohx31m", "*", "*"):
                parts = get_parts(fact)
                token, u_obj = parts[1], parts[2]
                token_locations[token] = u_obj

        # Count misplaced tokens: s1clodkz tokens not at root u59rqa6u (s154x56h)
        # or tokens not at correct u59rqa6u locations from waexgd4o_map for corresponding z65nyce9.
        misplaced_tokens = 0
        for token in self.s1clodkz_objects:
            loc = token_locations.get(token, None)
            if loc is None:
                # Missing token location implies need to move/add it:
                misplaced_tokens += 1
            else:
                # Usually tokens should eventually be at s154x56h for generating dym1mop6
                if loc != self.u59rqa6u_root:
                    misplaced_tokens += 1

        # 4) Count ggld6jmz with okcl3dr0 still true versus xx5ljjai or omcsgizs states indicating progress
        n_okcl3dr0 = count_facts("okcl3dr0", "*")
        n_xx5ljjai = count_facts("xx5ljjai", "*")
        n_omcsgizs = count_facts("omcsgizs", "*")

        # 5) For each goal z65nyce9 (in remaining_goals), check if waexgd4o facts link them to correct u59rqa6u.
        # If waexgd4o is missing or inconsistent, add estimated cost.
        waexgd4o_missing = 0
        for z in remaining_goals:
            if z not in self.waexgd4o_map:
                waexgd4o_missing += 1

        # Heuristic value calculation:
        # - Each missing qxafq0yb requires at least one l6w9j7tw or eav722gv action => 1 action minimum per goal.
        # - Each dym1mop6 fact corresponds to one blocking 'payload' needing to be removed.
        # - Misplaced tokens require moves (ss17onrb).
        # - Remaining okcl3dr0 signals ggld6jmz not yet prepared => require steps.
        # - Missing waexgd4o facts imply more re-positioning or errors.

        # weights to reflect rough action counts (heuristic not necessarily admissible)
        w_goal = 3       # cost estimate per missing goal qxafq0yb
        w_dym = 2        # cost estimate per dym1mop6 to remove
        w_token = 2      # per misplaced token
        w_okcl3dr0 = 1   # per ggld6jmz still at okcl3dr0 state
        w_waex_missing = 2  # per missing waexgd4o

        heuristic_value = (
            len(remaining_goals) * w_goal
            + n_dym * w_dym
            + misplaced_tokens * w_token
            + n_okcl3dr0 * w_okcl3dr0
            + waexgd4o_missing * w_waex_missing
        )

        # Ensure heuristic is at least 1 if not at goal to avoid zero heuristic on non-goal states.
        return max(heuristic_value, 1)
