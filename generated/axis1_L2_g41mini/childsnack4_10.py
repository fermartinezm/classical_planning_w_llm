from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate obj1 obj2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain ("aixg4uih").

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the
    goal facts (specifically, all goal facts with predicate `qxafq0yb` true)
    in the secret_domain. The domain actions often revolve around establishing
    `(qxafq0yb ?r7dv9)` goal facts by manipulating predicates related to
    `dym1mop6`, `okcl3dr0`, `xx5ljjai`, and `omcsgizs`.

    Assumptions
    -----------
    - The only goal predicates seen in the provided instances are of the form `(qxafq0yb ?r7dv9)`.
    - Establishing `(qxafq0yb ?r7dv9)` generally requires:
      * Starting from some `(mcoxi8qw ?r7dv9)` static fact (or known static facts)
      * Removing or setting `(dym1mop6 ?c3xfw ?axwhj)` facts
      * Using `okcl3dr0`, `xx5ljjai`, and `omcsgizs` predicates on ggld6jmz-type objects.
      * Possibly relabeling ownership of `s1clodkz`-type objects via `qrohx31m`.
      * The constant `s154x56h` (of type `u59rqa6u`) serves as a key fixed object for `qrohx31m`.

    Heuristic Initialization
    ------------------------
    Upon initialization, the heuristic extracts and stores:
    - The set of goal objects `r7dv9` for which `(qxafq0yb ?r7dv9)` is required.
    - The static facts:
      * Which objects have `(mcoxi8qw ?r7dv9)` true (likely needed to produce goal predicates).
      * The mapping `(waexgd4o ?r7dv9 ?imdah)` that associates `r7dv9` objects to `u59rqa6u` type objects.
      * The unique goal-related constant `s154x56h` (known from domain).
    - Sets of objects of different types extracted from static facts (optional, but useful for checks).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state:
    1. Check if the state already satisfies all goal predicates `(qxafq0yb ?r7dv9)`; if yes, heuristic is 0.
    2. For each goal object `r7dv9_goal`:
       a. If `(qxafq0yb r7dv9_goal)` is true, no cost to achieve this particular goal.
       b. Else:
          i. Inspect if the preconditions to produce `(qxafq0yb r7dv9_goal)` are met.
             According to operators "l6w9j7tw" and "eav722gv", to make `(qxafq0yb ?r7dv9)`,
             certain predicates must be true:
               - `(mcoxi8qw ?r7dv9)` (initial static condition)
               - `(waexgd4o ?r7dv9 ?imdah)` linking to some `imdah`.
               - `(qrohx31m ?axwhj ?imdah)` for some `axwhj` (slot of type `s1clodkz`)
               - `(dym1mop6 ?c3xfw ?axwhj)` for some `c3xfw` (ggld6jmz object)
               - `(xx5ljjai ?c3xfw)` or `(omcsgizs ?c3xfw)` depending on action used
          ii. If these supporting predicates are missing, estimate the cost required to add them.
              Main costs involve:
              - Moving ownership of `(qrohx31m ?axwhj ?imdah)` via action `ss17onrb`.
              - Creating/deleting `(dym1mop6 ?c3xfw ?axwhj)` via actions `nzh0a058`, `eav722gv`, `l6w9j7tw`.
              - Managing `okcl3dr0`, `xx5ljjai`, and `omcsgizs` states via actions `aegb5ot3` and `ppnkvo80`.
    3. Approximate the cost as:
       - One action for each missing `(qxafq0yb ?r7dv9)` goal fact.
       - Plus 1-3 actions for establishing the required supporting conditions per goal fact:
         * One action to ensure `(dym1mop6 ...)` is set.
         * One action to ensure `(xx5ljjai ...)` or `(omcsgizs ...)` is set.
         * Possibly one action for shifting ownership `(qrohx31m ...)`.
    4. Sum these per goal to compute heuristic.

    This heuristic is not admissible but should be efficiently computable and reasonably informative.
    """

    def __init__(self, task):
        self.goals = task.goals  # Set of goal facts.
        self.static = task.static

        # Extract all goal objects r7dv9 from goals of the form (qxafq0yb ?r7dv9).
        self.goal_objects = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts[0] == "qxafq0yb" and len(parts) == 2:
                self.goal_objects.add(parts[1])

        # Extract static facts related to r7dv9 objects:
        # (mcoxi8qw ?r7dv9) and (waexgd4o ?r7dv9 ?imdah)
        self.mcoxi8qw_set = set()  # Set of r7dv9
        self.waexgd4o_map = {}  # r7dv9 => set of imdah
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "mcoxi8qw" and len(parts) == 2:
                self.mcoxi8qw_set.add(parts[1])
            elif pred == "waexgd4o" and len(parts) == 3:
                r7 = parts[1]
                imdah = parts[2]
                if r7 not in self.waexgd4o_map:
                    self.waexgd4o_map[r7] = set()
                self.waexgd4o_map[r7].add(imdah)

        # Identify the constant s154x56h (u59rqa6u type) known from domain constants.
        self.s154x56h = "s154x56h"

    def __call__(self, node):
        state = node.state

        # Quick check: if all goal facts are true, heuristic is zero.
        if self.goals <= state:
            return 0

        # Helper: check if the goal predicate is true for a given r7dv9.
        def has_goal_predicate(r7dv9):
            return f"(qxafq0yb {r7dv9})" in state

        # For each goal object, estimate cost to achieve (qxafq0yb r7dv9)
        total_estimated_cost = 0
        for r7dv9 in self.goal_objects:
            if has_goal_predicate(r7dv9):
                # Already achieved, zero cost.
                continue

            # Base cost for producing (qxafq0yb r7dv9)
            cost = 1

            # Check if (mcoxi8qw r7dv9) is true; if not, extra cost to fulfill it (static fact usually present)
            if r7dv9 not in self.mcoxi8qw_set and f"(mcoxi8qw {r7dv9})" not in state:
                cost += 1  # assume cost to "get" it or approximate penalty

            # Check for any linked imdah via waexgd4o (static)
            possible_imdahs = self.waexgd4o_map.get(r7dv9, set())
            # As fallback, gather imdah that appear in state related to r7dv9 by waexgd4o as well:
            if not possible_imdahs:
                # Try to find any in state
                possible_imdahs = {fact.split()[2][:-1] for fact in state if fact.startswith(f"(waexgd4o {r7dv9} ")}
            # If none found, heuristic cannot do much; count cost pessimistically.
            if not possible_imdahs:
                cost += 2  # penalty for missing linking info
                total_estimated_cost += cost
                continue

            imdah = next(iter(possible_imdahs))

            # Find s1clodkz objects axwhj linked via qrohx31m to imdah (ownership)
            # Look in state for (qrohx31m ?axwhj ?imdah)
            axwhj_candidates = [get_parts(fact)[1] for fact in state if match(fact, "qrohx31m", "*", imdah)]

            if not axwhj_candidates:
                # Cost for assigning ownership via ss17onrb (one action)
                cost += 1
                # Pick a dummy axwhj for further checks to not multiply penalties
                axwhj = "unknown_axwhj"
            else:
                axwhj = axwhj_candidates[0]

            # Check if (dym1mop6 ?c3xfw ?axwhj) present for some c3xfw (ggld6jmz object)
            # Gather all (dym1mop6 ?c3xfw ?axwhj)
            dym1mop6_facts = [fact for fact in state if match(fact, "dym1mop6", "*", axwhj)]
            if not dym1mop6_facts:
                cost += 1  # need action to create this predicate (via nzh0a058)

            # Check if supporting (xx5ljjai ?c3xfw) or (omcsgizs ?c3xfw) exists for the c3xfw objects above
            # For all c3xfw with dym1mop6 to axwhj, check for xx5ljjai or omcsgizs
            c3xfw_objects = [get_parts(f)[1] for f in dym1mop6_facts]
            supporting_pred_found = False
            for c3xfw in c3xfw_objects:
                if f"(xx5ljjai {c3xfw})" in state or f"(omcsgizs {c3xfw})" in state:
                    supporting_pred_found = True
                    break
            if not supporting_pred_found:
                cost += 1  # need action to set (xx5ljjai) or (omcsgizs) (via ppnkvo80 or aegb5ot3)

            total_estimated_cost += cost

        # Return total heuristic cost (sum over all goal objects)
        # Ensure finite and nonnegative value.
        return max(total_estimated_cost, 0)
