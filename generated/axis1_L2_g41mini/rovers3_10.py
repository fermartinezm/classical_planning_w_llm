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
    - Returns True if the fact matches the pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (rb7qr1fs) domain.

    Summary
    -------
    The heuristic estimates the number of remaining actions required to achieve the goal from a given state.
    It focuses on key goal-related predicates and attempts to approximate distances by counting unmet goals
    and prerequisites needed to fulfill those goals. The heuristic is non-admissible but efficiently computed
    and reduces node expansions by providing a meaningful estimate of progress.

    Assumptions
    -----------
    - The primary goal facts rely heavily on the predicate (ixsb7u9c ?r6gw2 ?pmq6p).
      These represent required "goal pairs" between dbk5i2a5 and chxau8fy objects.
    - Achieving (ixsb7u9c) facts seems to require that certain supporting predicates involving
      (r1frsf5x ?jdovh ?r6gw2 ?pmq6p) and (rycjvfag ?eeuld ?pmq6p) hold, and some relations linking
      tom0ncko, rb7qr1fs, uwq7ubjf objects appear relevant.
    - Static facts define some domain structure that does not change and can help match objects and relations.
    - The heuristic counts how many goal (ixsb7u9c) facts remain unmet, penalizes missing prerequisite (r1frsf5x)
      facts, and includes estimated costs for creating or enabling these prerequisite facts.
    - The heuristic ignores complex cyclical dependencies for efficiency, focusing on approximating unmet goals and
      important intermediate facts.

    Heuristic Initialization
    ------------------------
    - Extract from static facts the sets of key typed objects:
        * rb7qr1fs-type objects (jdovh)
        * dbk5i2a5-type objects (r6gw2)
        * chxau8fy-type objects (pmq6p)
        * tom0ncko-type objects (eeuld)
    - Extract the goal facts (assumed to be mostly (ixsb7u9c r6gw2 pmq6p)).
    - Store all static (unchanging) b2mwd6x4 facts (which relate rb7qr1fs to uwq7ubjf pairs) to help judge connectivity.
    - Collect initial known key predicates that describe the task’s structure.
    - Prepare fast membership tests for relevant predicates.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which goal (ixsb7u9c ?r6gw2 ?pmq6p) facts are not yet achieved.
    2. For each such missing goal fact:
       a. Check if there exists an (r1frsf5x ?jdovh ?r6gw2 ?pmq6p) to "support" this goal.
          If missing, this represents an immediate extra action cost for establishing prerequisite facts.
       b. Missing prerequisite facts increase cost.
    3. Missing (r1frsf5x ?jdovh ?r6gw2 ?pmq6p) facts are estimated to require at least one action each to create.
    4. Attempt to estimate how many (wfrx4exs ?eeuld ?jdovh) facts are missing that typically trigger creation of (r1frsf5x) facts.
       Since (wfrx4exs) is achieved by the action p8i30ct9, which requires a tom0ncko and rb7qr1fs object relation,
       each missing (r1frsf5x) may translate to at least one missing (wfrx4exs) enabling action.
    5. Finally, sum the counts: number of unmet goal facts + number of missing prerequisite (r1frsf5x) facts + 
       estimated number of missing enabling (wfrx4exs) facts.
    6. The heuristic returns 0 if all goals are met. Otherwise, it returns the sum of these counts as a proxy for
       minimal number of actions remaining.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting relevant static information and goal facts.
        """
        self.goals = task.goals

        # Extract sets of objects by type from static facts
        self.rb7qr1fs_objects = set()
        self.dbk5i2a5_objects = set()
        self.chxau8fy_objects = set()
        self.tom0ncko_objects = set()
        self.uwq7ubjf_objects = set()
        self.ggaeyv41_objects = set()

        # Classify objects by their usage in static facts' signatures (heuristic guess)
        # Because only facts are given, we infer objects from facts by types mentioned in familiar predicates.

        # Also extract static facts for quick checks
        self.static_facts = set(task.static)

        # Extract objects from static facts based on type signatures appearing in predicates in static
        for fact in self.static_facts:
            parts = get_parts(fact)
            pred = parts[0]
            objs = parts[1:]

            if pred == "a298chgk":
                # (a298chgk ?jdovh - rb7qr1fs)
                # used as rb7qr1fs object
                if len(objs) >= 1:
                    self.rb7qr1fs_objects.add(objs[0])
            elif pred == "bszmiquc":
                # (bszmiquc ?jdovh - rb7qr1fs)
                if len(objs) >= 1:
                    self.rb7qr1fs_objects.add(objs[0])
            elif pred == "dfhfijzr":
                # (dfhfijzr ?jdovh - rb7qr1fs)
                if len(objs) >= 1:
                    self.rb7qr1fs_objects.add(objs[0])
            elif pred == "z6yvwl7n":
                # (z6yvwl7n ?zmrjf - zp2z4mnx ?jdovh - rb7qr1fs)
                if len(objs) == 2:
                    self.rb7qr1fs_objects.add(objs[1])
            elif pred == "p6u7tzz0":
                # (p6u7tzz0 ?zmrjf - zp2z4mnx)
                if len(objs) == 1:
                    self.chxau8fy_objects.add(objs[0])
                    # zp2z4mnx objects seem related to chxau8fy via parameters? Use as chxau8fy or zp2z4mnx
                    # We'll treat zp2z4mnx objects mainly as chxau8fy here for heuristic estimation

            elif pred == "rycjvfag":
                # (rycjvfag ?g8yy0 - tom0ncko ?pmq6p - chxau8fy)
                if len(objs) == 2:
                    self.tom0ncko_objects.add(objs[0])
                    self.chxau8fy_objects.add(objs[1])
            elif pred == "efxvv9ax":
                # (efxvv9ax ?eeuld - tom0ncko ?jdovh - rb7qr1fs)
                if len(objs) == 2:
                    self.tom0ncko_objects.add(objs[0])
                    self.rb7qr1fs_objects.add(objs[1])
            elif pred == "b2mwd6x4":
                # (b2mwd6x4 ?jdovh - rb7qr1fs ?jehs5 - uwq7ubjf ?ijb5b - uwq7ubjf)
                if len(objs) == 3:
                    self.rb7qr1fs_objects.add(objs[0])
                    self.uwq7ubjf_objects.add(objs[1])
                    self.uwq7ubjf_objects.add(objs[2])
            elif pred == "wx95011q":
                # (wx95011q ?lt0bb - uwq7ubjf ?jeh3h - uwq7ubjf)
                if len(objs) == 2:
                    self.uwq7ubjf_objects.add(objs[0])
                    self.uwq7ubjf_objects.add(objs[1])
            elif pred == "rrhxwbyt":
                # (rrhxwbyt ?jehs5 - ggaeyv41 ?ijb5b - uwq7ubjf)
                if len(objs) == 2:
                    self.ggaeyv41_objects.add(objs[0])
                    self.uwq7ubjf_objects.add(objs[1])
            elif pred == "ipivzc5v":
                # (ipivzc5v ?lt0bb - uwq7ubjf)
                if len(objs) == 1:
                    self.uwq7ubjf_objects.add(objs[0])
            elif pred == "splvr1ag":
                # (splvr1ag ?lt0bb - uwq7ubjf)
                if len(objs) == 1:
                    self.uwq7ubjf_objects.add(objs[0])
            elif pred == "y01m0fhm":
                # (y01m0fhm ?lt0bb - uwq7ubjf)
                if len(objs) == 1:
                    self.uwq7ubjf_objects.add(objs[0])
            elif pred == "cs7680d7":
                # (cs7680d7 ?r6gw2 - dbk5i2a5 ?lt0bb - uwq7ubjf)
                if len(objs) == 2:
                    self.dbk5i2a5_objects.add(objs[0])
                    self.uwq7ubjf_objects.add(objs[1])
            elif pred == "iaub97vi":
                # (iaub97vi ?eeuld - tom0ncko ?r6gw2 - dbk5i2a5)
                if len(objs) == 2:
                    self.tom0ncko_objects.add(objs[0])
                    self.dbk5i2a5_objects.add(objs[1])
            elif pred == "r1frsf5x":
                # (r1frsf5x ?jdovh - rb7qr1fs ?r6gw2 - dbk5i2a5 ?pmq6p - chxau8fy)
                if len(objs) == 3:
                    self.rb7qr1fs_objects.add(objs[0])
                    self.dbk5i2a5_objects.add(objs[1])
                    self.chxau8fy_objects.add(objs[2])
            elif pred == "ixsb7u9c":
                # (ixsb7u9c ?r6gw2 - dbk5i2a5 ?pmq6p - chxau8fy)
                if len(objs) == 2:
                    self.dbk5i2a5_objects.add(objs[0])
                    self.chxau8fy_objects.add(objs[1])
            elif pred == "wfrx4exs":
                # (wfrx4exs ?g8yy0 - tom0ncko ?jdovh - rb7qr1fs)
                if len(objs) == 2:
                    self.tom0ncko_objects.add(objs[0])
                    self.rb7qr1fs_objects.add(objs[1])

        # Store goal ixsb7u9c facts as a set of (dbk5i2a5, chxau8fy) tuples:
        self.goal_ixsb7u9c = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "ixsb7u9c" and len(parts) == 3:
                self.goal_ixsb7u9c.add((parts[1], parts[2]))

    def __call__(self, node):
        """
        Estimate the required number of actions to reach the goal from the current node's state.

        Returns 0 if the state satisfies the goal.

        Otherwise, the heuristic counts:
        - Number of unsatisfied goal (ixsb7u9c) facts.
        - Number of missing prerequisite (r1frsf5x) facts supporting those goals.
        - Estimated missing enabling (wfrx4exs) facts that support prerequisites.
        """
        state = node.state

        # If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract currently true ixsb7u9c facts in state for quick lookup
        achieved_ixsb7u9c = set()
        for fact in state:
            if fact.startswith("(ixsb7u9c "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    achieved_ixsb7u9c.add((parts[1], parts[2]))

        # Determine unmet goal facts
        unmet_goal_pairs = self.goal_ixsb7u9c - achieved_ixsb7u9c

        # Extract current r1frsf5x facts from state
        current_r1frsf5x = set()
        for fact in state:
            if fact.startswith("(r1frsf5x "):
                parts = get_parts(fact)
                if len(parts) == 4:
                    # Store (rb7qr1fs, dbk5i2a5, chxau8fy)
                    current_r1frsf5x.add((parts[1], parts[2], parts[3]))

        # Extract current wfrx4exs facts from state
        current_wfrx4exs = set()
        for fact in state:
            if fact.startswith("(wfrx4exs "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    current_wfrx4exs.add((parts[1], parts[2]))

        # For each unmet goal (ixsb7u9c r6gw2 pmq6p), find if any r1frsf5x (jdovh r6gw2 pmq6p) exists
        # Count number of missing r1frsf5x prerequisite facts related to the unmet goals
        missing_r1frsf5x = 0
        # We'll track for fast lookup r1frsf5x facts by (dbk5i2a5, chxau8fy)
        r1frsf5x_by_pair = {}
        for (jdovh, r6gw2, pmq6p) in current_r1frsf5x:
            r1frsf5x_by_pair.setdefault((r6gw2, pmq6p), set()).add(jdovh)

        for (r6gw2, pmq6p) in unmet_goal_pairs:
            if (r6gw2, pmq6p) not in r1frsf5x_by_pair:
                missing_r1frsf5x += 1

        # Now estimate missing enabling (wfrx4exs) facts needed to produce missing r1frsf5x facts
        # We assume for each missing r1frsf5x fact, at least one corresponding (wfrx4exs ?eeuld ?jdovh) is needed
        # Extract all known pairs (jdovh, r6gw2, pmq6p) for r1frsf5x in static+state:
        # We'll approximate by counting how many jdovh’s appear in static bszmiquc or efxvv9ax linked to rb7qr1fs objects
        # For heuristic speed, just count distinct jdovh in current_r1frsf5x and in (efxvv9ax ?eeuld ?jdovh) facts in static.
        # Assume roughly 1 enabling wfrx4exs per jdovh required.

        # Collect jdovh involved in r1frsf5x facts (needed for unmet)
        jdovh_for_missing = set()

        for (r6gw2, pmq6p) in unmet_goal_pairs:
            # For missing r1frsf5x, we do not know jdovh, so conservatively estimate 1 enabling per missing prerequisite
            # For those that have r1frsf5x, count jdovh associated to (r6gw2, pmq6p)
            jdovh_set = r1frsf5x_by_pair.get((r6gw2, pmq6p), None)
            if jdovh_set is None:
                # No prerequisite: estimate 1 enabling needed
                jdovh_for_missing.add(None)  # placeholder for missing jdovh
            else:
                jdovh_for_missing.update(jdovh_set)

        # Count how many enabling wfrx4exs facts are currently true for jdovh in the state
        enabling_missing = 0
        for jdovh in jdovh_for_missing:
            if jdovh is None:
                # Missing prerequisite, count as 1 enabling missing
                enabling_missing += 1
            else:
                # Check if any wfrx4exs (eeuld jdovh) exists in current state
                # We don't know eeuld for sure, so check if any wfrx4exs with jdovh
                found_enabling = any(wjdovh == jdovh for (_, wjdovh) in current_wfrx4exs)
                if not found_enabling:
                    enabling_missing += 1

        # Heuristic sum: unmet goal facts + missing prerequisites + missing enabling facts
        heuristic_value = len(unmet_goal_pairs) + missing_r1frsf5x + enabling_missing

        return heuristic_value
