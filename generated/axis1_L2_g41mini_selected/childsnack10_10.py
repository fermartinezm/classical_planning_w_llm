from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern with wildcards (*).
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain.

    Summary
    -------
    This heuristic estimates the number of actions required to achieve all goal facts `(qxafq0yb ?z65nyce9)` 
    in the domain `aixg4uih`. The main goal is to enable the planner to reach states with all required
    `qxafq0yb` predicates true on specific `z65nyce9` objects.

    Assumptions
    -----------
    - The goal consists exclusively of facts of the form `(qxafq0yb ?z65nyce9)`.
    - Each `qxafq0yb` fact is typically produced by applying actions `l6w9j7tw` or `eav722gv`.
    - Key intermediate predicates to manage are:
        * `qxafq0yb` on `z65nyce9`: goal achievement predicate.
        * `zmomsdb3` marks `z65nyce9` objects linked to goal-relevant facts.
        * `mcoxi8qw` shows a subset of `z65nyce9` objects relevant for goal processing.
        * `dym1mop6` tracks `ggld6jmz` objects paired with `s1clodkz` relevant for enabling effects.
        * `omcsgizs` and `okcl3dr0` mark states of `ggld6jmz` objects controlling whether actions can be applied.
        * `xx5ljjai` and `okcl3dr0` represent toggled states of `ggld6jmz`.
        * `waexgd4o` links `z65nyce9` objects with `u59rqa6u` objects, representing a quantifiable resource or relationship needed.
        * `qrohx31m` relates `s1clodkz` with `u59rqa6u`; its change is controlled by `ss17onrb` actions.
        * `g7wohkyk`, `sl9kfffg`, `vrips567`, `usj0xp5t` are various "resource" or "availability" predicates influencing applicability.

    Heuristic Initialization
    ------------------------
    - Extract the set of goal `qxafq0yb` facts on `z65nyce9`.
    - Extract all static facts involving:
      + `waexgd4o` links between `z65nyce9` & `u59rqa6u`.
      + The constant `s154x56h` of type `u59rqa6u`.
    - Identify all `s1clodkz` objects that appear in static links `qrohx31m ?s1clodkz s154x56h`.
    - Extract `mcoxi8qw` facts associated with `z65nyce9` for heuristic progress.
    - Track `okcl3dr0`, `omcsgizs`, `xx5ljjai` states for `ggld6jmz` objects representing toggled states.
    - Recognize that toggling `okcl3dr0` ↔ `xx5ljjai` and managing `dym1mop6` facts are central to producing `qxafq0yb`.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Calculate how many goal `qxafq0yb` facts are still missing in the current state.
       If none missing, heuristic is 0.

    2. To achieve `(qxafq0yb ?z)`, one of `l6w9j7tw` or `eav722gv` must be applied:
       - Both require `(zmomsdb3 ?z)` and `(waexgd4o ?z ?u)` where `qrohx31m ?axwhj ?u` and some `dym1mop6` facts.
       - So these predicates indicate dependencies that must be satisfied.
       
    3. Count the missing `qxafq0yb` goal facts (call this `missing_qxafq0yb`).
    
    4. For each `z65nyce9` object missing `qxafq0yb`:
       - Check if `mcoxi8qw` is currently true for it. If missing, it implies the need for an `eav722gv` action to establish it. Increment cost accordingly.
       
    5. Count how many `dym1mop6` facts are currently true (representing active states for `ggld6jmz` objects and `s1clodkz` pairs).
       The heuristic assumes some number of `dym1mop6` facts are needed to eventually enable `qxafq0yb`.

    6. Count `okcl3dr0` and `xx5ljjai` toggles on `ggld6jmz` objects — toggling these states likely corresponds to performing `(ppnkvo80)`, `(aegb5ot3)`, and other toggling actions.

    7. Count facts about `(qrohx31m ?axwhj ?u59rqa6u)`, focusing on those with `s154x56h` or updated by `ss17onrb`.

    8. Each toggle or enabling is assumed to require roughly 1 action plus some adjustment for layered dependencies.

    9. For efficiency, the heuristic sums:
       - The number of missing `qxafq0yb` goals * a base cost (e.g., 1 or 2).
       - The number of needed `mcoxi8qw` facts to be restored.
       - The number of toggles required on `okcl3dr0`/`xx5ljjai`.
       - The number of `dym1mop6` facts that must be manipulated.
       - Adjust for the necessary resource switches `qrohx31m`.

    This heuristic is **not admissible**, but aims to closely estimate effort to reach the goals by counting key progress markers and dependencies in the state.

    The heuristic is guaranteed to be 0 at goal, always finite for solvable states, and quickly computable from the state representation.
    """

    def __init__(self, task):
        """
        Initialize by extracting static knowledge and goal facts relevant to the heuristic.
        """
        self.goals = task.goals  # e.g. contains all required (qxafq0yb ?z65nyce9) goal facts.

        # Extract constant of type u59rqa6u (only one known: s154x56h)
        # It appears in qrohx31m facts frequently, so we remember it for matching.
        self.fixed_u59rqa6u = "s154x56h"

        # Extract all static facts that are persistent.
        static = task.static

        # Record all waexgd4o (?z65nyce9, ?u59rqa6u) pairs - relates objects to resources.
        self.waexgd4o_pairs = set()
        for fact in static:
            if match(fact, "waexgd4o", "*", "*"):
                parts = get_parts(fact)
                self.waexgd4o_pairs.add((parts[1], parts[2]))

        # Record all qrohx31m (?s1clodkz, ?u59rqa6u) facts in static or inferred from init.
        # Especially those involving s154x56h as second param (the constant).
        self.qrohx31m_s154x56h = set()
        for fact in static:
            if match(fact, "qrohx31m", "*", self.fixed_u59rqa6u):
                parts = get_parts(fact)
                self.qrohx31m_s154x56h.add(parts[1])  # Save s1clodkz objects linked to s154x56h.

        # Extract all z65nyce9 objects required for the goals (from qxafq0yb goals).
        self.goal_z65nyce9 = set()
        for goal in self.goals:
            if match(goal, "qxafq0yb", "*"):
                parts = get_parts(goal)
                self.goal_z65nyce9.add(parts[1])

    def __call__(self, node):
        """
        Compute heuristic estimate of remaining actions to reach the goal.
        """
        state = node.state

        # Fast exit: if goal facts (qxafq0yb) all present, heuristic = 0.
        if self.goals <= state:
            return 0

        # ---------------------------------------
        # Count missing goal qxafq0yb facts.
        missing_qxafq0yb = 0
        for zobj in self.goal_z65nyce9:
            fact = f"(qxafq0yb {zobj})"
            if fact not in state:
                missing_qxafq0yb += 1

        # Count for each `z65nyce9` if mcoxi8qw is present.
        # If missing, we need one eav722gv action to establish it.
        missing_mcoxi8qw = 0
        for zobj in self.goal_z65nyce9:
            fact = f"(mcoxi8qw {zobj})"
            if fact not in state:
                missing_mcoxi8qw += 1

        # Count how many dym1mop6 facts are currently true.
        # They represent pairs (ggld6jmz, s1clodkz) enabling qxafq0yb effects.
        dym1mop6_count = sum(1 for fact in state if match(fact, "dym1mop6", "*", "*"))

        # Count how many okcl3dr0 and xx5ljjai facts for ggld6jmz objects exist.
        okcl3dr0_count = sum(1 for fact in state if match(fact, "okcl3dr0", "*"))
        xx5ljjai_count = sum(1 for fact in state if match(fact, "xx5ljjai", "*"))

        # Count how many omcsgizs facts for ggld6jmz, needed to enable dym1mop6.
        omcsgizs_count = sum(1 for fact in state if match(fact, "omcsgizs", "*"))

        # Count number of qrohx31m facts with s154x56h constant as second param.
        qrohx31m_s154x56h_count = sum(1 for fact in state if match(fact, "qrohx31m", "*", self.fixed_u59rqa6u))

        # Count how many waexgd4o facts (z65nyce9, u59rqa6u) present (should support qrohx31m).
        waexgd4o_count = sum(1 for fact in state if match(fact, "waexgd4o", "*", "*"))

        # Heuristic reasoning:
        # Each missing qxafq0yb requires the completion of:
        # - Having mcoxi8qw true (or reset it)
        # - Having dym1mop6 available to remove
        # - Toggling okcl3dr0/xx5ljjai states as needed
        # - Manipulating qrohx31m to match waexgd4o pairs
        #
        # Base cost per missing goal fact is 2 (at least 2 actions to produce qxafq0yb).
        # Add cost proportional to missing mcoxi8qw (each potentially one eav722gv action).
        # Add cost proportional to toggles needed (approximated by min okcl3dr0, xx5ljjai count).
        # Add cost for update actions on qrohx31m and waexgd4o interplay.
        # Dym1mop6 count usually indicates available intermediate enabling facts.
        #
        # We scale parts empirically to reflect difficulty and dependencies.

        base_goal_cost = 2 * missing_qxafq0yb
        mcoxi8qw_cost = 3 * missing_mcoxi8qw
        toggle_cost = abs(okcl3dr0_count - xx5ljjai_count)  # Actions to toggle states to balanced count
        omcsgizs_cost = max(0, missing_qxafq0yb - omcsgizs_count)  # Need more omcsgizs to support dym1mop6
        qrohx31m_cost = max(0, len(self.qrohx31m_s154x56h) - qrohx31m_s154x56h_count)
        waexgd4o_expected = len([p for p in self.waexgd4o_pairs if p[0] in self.goal_z65nyce9])
        waexgd4o_cost = max(0, waexgd4o_expected - waexgd4o_count)

        dym1mop6_needed = max(0, missing_qxafq0yb - dym1mop6_count)

        heuristic_value = (
            base_goal_cost
            + mcoxi8qw_cost
            + toggle_cost
            + 2 * omcsgizs_cost
            + 2 * qrohx31m_cost
            + waexgd4o_cost
            + dym1mop6_needed
        )

        # Heuristic must be at least 1 if not solved and finite.
        return max(heuristic_value, 1)
