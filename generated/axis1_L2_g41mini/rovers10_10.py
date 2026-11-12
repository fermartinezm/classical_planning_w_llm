from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(ixsb7u9c jk58y1hv vjt56pf)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions needed to satisfy all goal conditions of the form
    (ixsb7u9c ?r6gw2 ?pmq6p), i.e., the predicate `ixsb7u9c` with parameters matching the goals.
    The heuristic is based on counting how many goal facts are not yet achieved and estimating the minimal
    transport and setup actions required to achieve them, leveraging static relationships and current states.

    Assumptions:
    - The main goal predicate to satisfy is `ixsb7u9c`, which relates two typed objects.
    - Achieving an `ixsb7u9c` goal fact is only possible after setting related intermediate predicates, especially
      `r1frsf5x` that links `(rb7qr1fs, dbk5i2a5, chxau8fy)`.
    - The domain seems to represent a transport or assignment problem involving multiple object types:
      - rb7qr1fs: primary locations/entities
      - dbk5i2a5: some container or intermediate objects
      - chxau8fy: endpoint or target objects
      - uwq7ubjf: movable entities (agents/items)
      - tom0ncko & zp2z4mnx: operational/helper objects
      - ggaeyv41: perhaps workers or special agents
    - Static facts define adjacencies and allowed relations among objects.
    - The heuristic doesn't need to be admissible but should be informative and efficiently computable.
    - Most goals require bringing together specific triples so that the final predicate holds.

    Heuristic Initialization:
    - Extract all goal `ixsb7u9c` facts to determine what must be achieved.
    - Extract static facts about:
      - The static relationships `b2mwd6x4` that probably represent allowed links between locations and agents.
      - The set of objects per type for quick membership testing.
      - Extract predicates denoting availability or special status, e.g., `a298chgk` or `bszmiquc`.
    - Extract the set of all `r1frsf5x` facts currently true (partial achievements toward the goal).
    - Extract and store other useful predicates (such as `splvr1ag` or `y01m0fhm`) that mark progress.
  
    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all goal facts of the form `(ixsb7u9c R6GW2 PMQ6P)` from the goals.
    2. For each goal fact:
       - Check if it is currently satisfied in the state; if yes, no cost needed.
       - Otherwise, infer the steps to achieve it:
         a) Requires an `r1frsf5x` fact linking a known `rb7qr1fs` and the two goal parameters `(dbk5i2a5 and chxau8fy)` that must be established first.
         b) Establishing an `r1frsf5x` fact requires an effect from `dlx472ba`, which in turn requires the predicate `wfrx4exs` to hold.
         c) The predicate `wfrx4exs` is produced by action `p8i30ct9` if certain preconditions hold, such as `efxvv9ax`, `bszmiquc`, `iaub97vi`, `cs7680d7`, and `jckjsxca`.
         d) Possibly, to establish `w1yp6zl7` (used in other actions), `kz1pxajl` and `s1f6inra` actions must be performed.
         e) Transfer of `jckjsxca` predicates among `uwq7ubjf` objects might be required, handled by action `xvldpwsb`.
    3. Use counts of unsatisfied goal facts and intermediate needed facts to approximate minimal needed actions:
       - Count how many goals are unmet.
       - Estimate how many `r1frsf5x` facts are still missing corresponding to those goals.
       - Estimate the minimal number of `wfrx4exs` facts needed.
       - Estimate needed transfers or moves of `jckjsxca` predicates among `uwq7ubjf` objects.
    4. Sum these estimates with weight approximations reflecting the minimal number of actions needed, guided by action definitions.
    5. Return 0 if all goals are satisfied.

    The heuristic thus combines partial goal satisfaction checks, progress predicates, and static structural info to provide an informative, efficiently computable estimate guiding greedy search.
    """

    def __init__(self, task):
        """Initialize heuristic data structures from the task."""
        self.goals = task.goals
        self.static = task.static

        # Extract goal ixsb7u9c facts as tuples: (dbk5i2a5, chxau8fy)
        self.goal_ixsb7u9c = []
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "ixsb7u9c" and len(parts) == 3:
                self.goal_ixsb7u9c.append((parts[1], parts[2]))

        # Precompute sets of static facts for quick membership queries
        self.b2mwd6x4 = set()  # (rb7qr1fs, uwq7ubjf, uwq7ubjf)
        for fact in self.static:
            if match(fact, "b2mwd6x4", "*", "*", "*"):
                parts = get_parts(fact)
                self.b2mwd6x4.add((parts[1], parts[2], parts[3]))

        self.a298chgk = set()  # set of rb7qr1fs with predicate a298chgk
        for fact in self.static:
            if match(fact, "a298chgk", "*"):
                parts = get_parts(fact)
                self.a298chgk.add(parts[1])

        self.bszmiquc = set()  # set of rb7qr1fs with predicate bszmiquc
        for fact in self.static:
            if match(fact, "bszmiquc", "*"):
                parts = get_parts(fact)
                self.bszmiquc.add(parts[1])

        self.p6u7tzz0 = set()  # set of zp2z4mnx objects with predicate p6u7tzz0
        for fact in self.static:
            if match(fact, "p6u7tzz0", "*"):
                parts = get_parts(fact)
                self.p6u7tzz0.add(parts[1])

        self.a8x3330w = set()  # set of uwq7ubjf objects with predicate a8x3330w
        for fact in self.static:
            if match(fact, "a8x3330w", "*"):
                parts = get_parts(fact)
                self.a8x3330w.add(parts[1])

        self.rrhxwbyt = set()  # set of (ggaeyv41, uwq7ubjf)
        for fact in self.static:
            if match(fact, "rrhxwbyt", "*", "*"):
                parts = get_parts(fact)
                self.rrhxwbyt.add((parts[1], parts[2]))

        # Static facts needed for "wx95011q" relation between uwq7ubjf objects
        self.wx95011q = set()
        for fact in self.static:
            if match(fact, "wx95011q", "*", "*"):
                parts = get_parts(fact)
                self.wx95011q.add((parts[1], parts[2]))

    def __call__(self, node):
        """Compute a heuristic estimate of steps needed to reach the goal from the current state."""
        state = node.state

        # Quick goal check: if all goal ixsb7u9c facts hold, heuristic = 0
        if all(f"(ixsb7u9c {r6gw2} {pmq6p})" in state for (r6gw2, pmq6p) in self.goal_ixsb7u9c):
            return 0

        # Count how many goal ixsb7u9c facts are not yet achieved
        missing_goals = [
            (r6gw2, pmq6p)
            for (r6gw2, pmq6p) in self.goal_ixsb7u9c
            if f"(ixsb7u9c {r6gw2} {pmq6p})" not in state
        ]

        # Collect currently true r1frsf5x facts (rb7qr1fs, dbk5i2a5, chxau8fy)
        current_r1frsf5x = set()
        for fact in state:
            if match(fact, "r1frsf5x", "*", "*", "*"):
                parts = get_parts(fact)
                current_r1frsf5x.add((parts[1], parts[2], parts[3]))

        # Mapping from dbk5i2a5/chxau8fy to r1frsf5x presence for missing goals
        r1frsf5x_needed = 0
        for (r6gw2, pmq6p) in missing_goals:
            # Check if there exists an r1frsf5x with rb7qr1fs that covers (r6gw2, pmq6p)
            covered = any((rb7qr1fs, r6gw2, pmq6p) in current_r1frsf5x for rb7qr1fs in self.bszmiquc)
            if not covered:
                r1frsf5x_needed += 1

        # Count how many wfrx4exs (p4vzsrq2 g01zhgse) or equivalents exist - prerequisites for r1frsf5x creation
        current_wfrx4exs = set()
        for fact in state:
            if match(fact, "wfrx4exs", "*", "*"):
                parts = get_parts(fact)
                current_wfrx4exs.add((parts[1], parts[2]))

        # Count how many wfrx4exs are missing needed for the r1frsf5x_needed
        # We conservatively estimate wfrx4exs needed is at least r1frsf5x_needed
        wfrx4exs_needed = max(0, r1frsf5x_needed - len(current_wfrx4exs))

        # Count the number of working agents that possess predicates jckjsxca at uwq7ubjf objects
        # Presumably needed for operations related to transferring and enabling facts
        jckjsxca_facts = [fact for fact in state if match(fact, "jckjsxca", "*", "*")]
        current_jckjsxca = set()
        for fact in jckjsxca_facts:
            parts = get_parts(fact)
            current_jckjsxca.add((parts[1], parts[2]))

        # Count how many required jckjsxca predicates needed to facilitate needed steps
        # Estimate is to cover missing 'r1frsf5x' and 'wfrx4exs' contributions
        # We approximate minimal jckjsxca needed as max of wfrx4exs_needed and some fraction of missing_goals
        jckjsxca_needed = max(0, r1frsf5x_needed + wfrx4exs_needed - len(current_jckjsxca))

        # Estimate cost weights using action decomposition:
        # - p8i30ct9 creates wfrx4exs given preconditions
        # - dlx472ba creates r1frsf5x from wfrx4exs and other preconditions
        # - Actions like xvldpwsb transfer jckjsxca predicates
        # - Actions like kz1pxajl, s1f6inra, and others prepare prerequisites (estimating ~1 action each)
        # We know each missing r1frsf5x goal requires at least 2 actions: one to create wfrx4exs, one to create r1frsf5x
        # plus around 1 extra action for preparation or transfers per jckjsxca needed.

        # Weighting estimates with minimal numbers of actions per missing fact:
        cost = 0
        cost += wfrx4exs_needed * 2     # actions to create needed wfrx4exs (e.g., p8i30ct9)
        cost += r1frsf5x_needed * 3     # actions to create needed r1frsf5x (e.g., dlx472ba + prerequisite moves)
        cost += jckjsxca_needed * 1     # actions to transfer jckjsxca predicates (xvldpwsb)
        cost += len(missing_goals)       # minimal 1 action per missing goal as approximation

        # Additionally, consider goals asking for y01m0fhm predicates (they appear in goals sometimes)
        # For states containing these goals: estimate missing y01m0fhm facts and add actions similarly
        y01m0fhm_goals = [goal for goal in self.goals if match(goal, "y01m0fhm", "*")]
        missing_y01m0fhm = 0
        for g in y01m0fhm_goals:
            parts = get_parts(g)
            if g not in state:
                missing_y01m0fhm += 1

        cost += missing_y01m0fhm * 2  # approx 2 actions per missing y01m0fhm (like ttp8j82e or sn0ysqqm)

        # Ensure heuristic is at least 1 for non-goal states (unless zero)
        if cost == 0:
            return 1
        return cost
