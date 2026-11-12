from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(qxafq0yb rap6j3pu)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (aixg4uih).

    Summary:
    ----------
    This heuristic estimates the minimum number of actions needed to achieve all the goal facts of 
    the form (qxafq0yb ?r7dv9) for the secret_domain. The goal predicates (qxafq0yb) represent 
    the final target facts for certain z65nyce9-type objects. To achieve (qxafq0yb ?r7dv9),
    the domain operators require removing some predicates related to ggld6jmz and s1clodkz objects,
    possibly switching some relation of s1clodkz with u59rqa6u objects, and satisfying certain predicates
    on the involved objects.

    Assumptions:
    ------------
    - The goal is always a conjunction of facts of the form (qxafq0yb ?z65nyce9-object).
    - Each goal (qxafq0yb x) can be achieved once by applying exactly one of the actions l6w9j7tw or eav722gv
      on suitable matching objects and parameters.
    - To apply l6w9j7tw and eav722gv, certain facts must hold about predicates dym1mop6, zmomsdb3, waexgd4o,
      qrohx31m, mcoxi8qw, etc., which may require applying nzh0a058, ss17onrb, or ppnkvo80, aegb5ot3 to
      modify these predicates.
    - Fact (qxafq0yb ?r7dv9) once achieved is never undone (no operator deletes it).
    - Heuristic does not need to be admissible, so somewhat rough cost estimations are acceptable as long 
      as they guide the search well and remain finite.

    Heuristic Initialization:
    ---------------------------
    - Extract all goal objects of type z65nyce9 that must have (qxafq0yb ?r7dv9) true.
    - Extract static facts for quick reference:
      * Mapping from waexgd4o facts pairs (?r7dv9 u59rqa6u).
      * Fixed unique constant for u59rqa6u (s154x56h).
    - No precomputation of operators; heuristic relies on static info and current state to estimate costs.

    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    For each goal object (qxafq0yb ?r7dv9), if the goal fact is already true in the current state,
    zero cost is added for it. Otherwise:
    
    1. We identify that to produce (qxafq0yb ?r7dv9), either l6w9j7tw or eav722gv must be applied:
       - l6w9j7tw requires:
         * (dym1mop6 ?c3xfw ?axwhj)
         * (xx5ljjai ?c3xfw)
         * (zmomsdb3 ?r7dv9)
         * (waexgd4o ?r7dv9 ?imdah)
         * (qrohx31m ?axwhj ?imdah)
       - eav722gv requires:
         * (qrohx31m ?axwhj ?imdah)
         * (dym1mop6 ?c3xfw ?axwhj)
         * (mcoxi8qw ?r7dv9)
         * (waexgd4o ?r7dv9 ?imdah)
     
    2. To achieve dym1mop6 predicates, nzh0a058 can be applied, which requires (omcsgizs ?c3xfw) 
       and (qrohx31m ?axwhj s154x56h).
       
    3. To achieve predicates xx5ljjai and omcsgizs and okcl3dr0 and g7wohkyk, ppnkvo80 and aegb5ot3 allow toggling 
       these predicates on ggld6jmz, n4cwbwmy, vwkdymi3 objects.

    4. The predicate qrohx31m (?axwhj ?imdah) can be modified by ss17onrb that moves qrohx31m facts 
       between u59rqa6u objects.

    5. Setting up waexgd4o (?r7dv9 ?imdah) is static and fixed (does not change).

    6. Since predicates like dym1mop6 and xx5ljjai are intermediate and must be created or toggled, 
       the heuristic counts how many of the required intermediate facts and transformations are missing 
       and roughly estimate one action per missing predicate or toggle, plus one action for producing goal fact.

    7. Estimate separate costs for:
       - Preparing qrohx31m pairs via ss17onrb moves (count moves needed).
       - Producing (dym1mop6) by toggling omcsgizs and others via ppnkvo80 / aegb5ot3 and nzh0a058.
       - Toggling okcl3dr0, xx5ljjai, omcsgizs, and g7wohkyk accordingly.
       - Finally, one application of l6w9j7tw or eav722gv to achieve (qxafq0yb).
    
    The minimum cost among the possible action paths is accumulated for all goals.
    
    The heuristic sums these estimates for all goal objects not yet achieved.

    The heuristic returns 0 if all goal facts are achieved, otherwise the sum of all estimated efforts.

    This heuristic should guide the search efficiently, is finite on solvable states, and outputs 0 only 
    for goal states.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting relevant static and goal data."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal objects that require (qxafq0yb ?r7dv9)
        self.goal_objects = []
        for g in self.goals:
            if match(g, "qxafq0yb", "*"):
                _, obj = get_parts(g)
                self.goal_objects.append(obj)

        # Extract static waexgd4o relations: map z65nyce9-object -> u59rqa6u-object
        # This relation does not change state to state, so store for quick lookup.
        self.waexgd4o_map = {}
        for fact in self.static:
            if match(fact, "waexgd4o", "*", "*"):
                parts = get_parts(fact)
                # waexgd4o ?r7dv9 ?imdah
                self.waexgd4o_map[parts[1]] = parts[2]

        # The special constant s154x56h of type u59rqa6u (used in domain)
        self.special_u59rqa6u = "s154x56h"

    def __call__(self, node):
        """Estimate cost to achieve all (qxafq0yb ?r7dv9) goal facts from given state."""
        state = node.state

        # Quick lookup functions

        def has_pred(predicate, *args):
            # Check if state contains predicate with given args
            pattern = f"({predicate} " + " ".join(args) + ")"
            return pattern in state

        def count_facts(predicate, object1):
            # Count how many facts of form (predicate object1 <anything>) are in state
            # For performance this is approximate and used only if necessary
            c = 0
            prefix = f"({predicate} {object1} "
            plen = len(prefix)
            for f in state:
                if f.startswith(prefix):
                    c += 1
            return c

        # For holding total heuristic cost
        total_cost = 0

        # If all goal facts (qxafq0yb ?r7dv9) true, heuristic is 0
        done_goals = 0

        for goal_obj in self.goal_objects:
            if has_pred("qxafq0yb", goal_obj):
                done_goals += 1
                continue  # No cost for achieved goal

            # Goal not achieved yet, estimate cost

            # Extract static u59rqa6u object related to goal_obj
            imdah = self.waexgd4o_map.get(goal_obj, None)
            if imdah is None:
                # No waexgd4o pair for goal_obj. 
                # Without this, goal likely not achievable, but return high cost.
                total_cost += 10
                continue

            # Check related predicates needed for l6w9j7tw or eav722gv:

            # Check if (zmomsdb3 goal_obj) or (mcoxi8qw goal_obj) true.
            # Exactly one may be required (l6w9j7tw requires zmomsdb3, eav722gv requires mcoxi8qw)
            has_zmomsdb3 = has_pred("zmomsdb3", goal_obj)
            has_mcoxi8qw = has_pred("mcoxi8qw", goal_obj)

            # We guess it costs 1 to produce zmomsdb3 or mcoxi8qw (if missing) by any means.
            cost_zmo_mco = 0
            if not (has_zmomsdb3 or has_mcoxi8qw):
                cost_zmo_mco = 1

            # Find dym1mop6 predicates involving ggld6jmz and s1clodkz objects to satisfy preconditions.
            # Look for any (dym1mop6 ?c3xfw ?axwhj) where qrohx31m (?axwhj ?imdah).
            # To simplify, count how many (dym1mop6 *) predicates that satisfy qrohx31m (* imdah) exist.
            # We need at least one to apply l6w9j7tw or eav722gv.

            # Gather (qrohx31m ?axwhj imdah) for any ?axwhj
            qrohx31m_axwhj = set()
            prefix = f"(qrohx31m "
            for f in state:
                if f.startswith(prefix) and f.endswith(f" {imdah})"):
                    parts = get_parts(f)
                    qrohx31m_axwhj.add(parts[1])

            # Find (dym1mop6 ?c3xfw ?axwhj) with ?axwhj in qrohx31m_axwhj
            dym1mop6_pairs = []
            for f in state:
                if match(f, "dym1mop6", "*", "*"):
                    parts = get_parts(f)
                    if parts[2] in qrohx31m_axwhj:
                        dym1mop6_pairs.append((parts[1], parts[2]))

            # If no such (dym1mop6 ?c3xfw ?axwhj), likely need to create it first (nzh0a058, ppnkvo80 etc.)
            missing_dym1mop6_cost = 0 if dym1mop6_pairs else 2

            # Check if any ggld6jmz has (xx5ljjai) for l6w9j7tw precondition
            has_xx5ljjai = any(match(f, "xx5ljjai", "*") for f in state)

            # If goal requires l6w9j7tw, xx5ljjai is needed; if missing, add cost ~1
            # This is an approximation: If zmomsdb3 present (l6w9j7tw path), xx5ljjai should be present.
            cost_xx5ljjai = 0
            if has_zmomsdb3 and not has_xx5ljjai:
                cost_xx5ljjai = 1

            # Count how many times (okcl3dr0 ?c3xfw) hold, affecting toggling
            okcl3dr0_count = sum(1 for f in state if match(f, "okcl3dr0", "*"))
            # Count g7wohkyk count to approximate cost of toggling
            g7wohkyk_count = sum(1 for f in state if match(f, "g7wohkyk", "*"))
            # These predicates relate to enabling/disabling (omcsgizs), necessary for nzh0a058 and ppnkvo80 sequencing.
            # Approximate toggle cost proportional to missing expected state:
            cost_toggle = 0
            if has_zmomsdb3 and okcl3dr0_count == 0:
                cost_toggle += 1
            if has_zmomsdb3 and g7wohkyk_count == 0:
                cost_toggle += 1

            # Check if qrohx31m (?axwhj ?imdah) includes the special s154x56h (needed for nzh0a058)
            has_qrohx31m_special = any(match(f, "qrohx31m", "*", self.special_u59rqa6u) for f in state)
            cost_qrohx31m_special = 0 if has_qrohx31m_special else 1

            # Count how many ss17onrb moves are needed to get qrohx31m (?axwhj ?imdah) to (?axwhj s154x56h)
            # Approximate as 1 if imdah != s154x56h
            cost_ss17onrb = 0
            if imdah != self.special_u59rqa6u:
                cost_ss17onrb = 1

            # Final application cost of l6w9j7tw or eav722gv
            cost_final = 1

            # Sum min cost path estimates (try l6w9j7tw or eav722gv paths, choose approximate minimum)
            # l6w9j7tw path cost terms: cost_zmo_mco + missing_dym1mop6_cost + cost_xx5ljjai + cost_toggle + cost_ss17onrb + cost_final
            cost_l6w9j7tw = (
                (cost_zmo_mco if has_zmomsdb3 else 10)  # high if no zmomsdb3
                + missing_dym1mop6_cost
                + cost_xx5ljjai
                + cost_toggle
                + cost_ss17onrb
                + cost_final
            )
            # eav722gv path cost terms: cost_zmo_mco + missing_dym1mop6_cost + cost_ss17onrb + cost_final
            cost_eav722gv = (
                (cost_zmo_mco if has_mcoxi8qw else 10)  # high if no mcoxi8qw
                + missing_dym1mop6_cost
                + cost_ss17onrb
                + cost_final
            )

            estimated_cost = min(cost_l6w9j7tw, cost_eav722gv)

            # Add to total heuristic
            total_cost += estimated_cost

        # If all goals achieved, cost is zero
        if done_goals == len(self.goal_objects):
            return 0

        # Return total estimated cost for remaining goals
        return total_cost
