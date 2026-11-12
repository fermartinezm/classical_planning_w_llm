from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *pattern):
    """
    Check if a fact matches a given pattern with wildcards.

    Args:
        fact (str): A PDDL fact string, e.g., "(predicate obj1 obj2)".
        pattern (str): Pattern components, can include '*' as wildcard.

    Returns:
        bool: True if matches, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain.

    Summary:
    ----------
    This heuristic estimates the number of actions needed to achieve
    the goal facts related mainly to the predicates (ixsb7u9c ?r6gw2 ?pmq6p)
    and (y01m0fhm ?lt0bb) observed in goal conditions. From the provided 
    domain and instances, goals require certain (ixsb7u9c) facts to hold 
    for pairs of objects of types dbk5i2a5 and chxau8fy and certain 
    (y01m0fhm) facts to hold for objects of type uwq7ubjf.

    The heuristic calculates how many such goal facts are still missing 
    in the current state and estimates the number of actions necessary 
    to add them, using domain knowledge about how the goal can be reached 
    based on operator effects and static facts.

    Assumptions:
    ------------
    - Goal facts are conjunctions of (ixsb7u9c ?dbk5i2a5 ?chxau8fy) and/or (y01m0fhm ?uwq7ubjf).
    - Achieving an (ixsb7u9c) fact requires first having (r1frsf5x ?rb7qr1fs ?dbk5i2a5 ?chxau8fy)
      established by applying actions involving tom0ncko objects and related predicates.
    - (y01m0fhm) facts are achieved by actions that require the presence of (rpzrvk8g ?rb7qr1fs ?uwq7ubjf)
      or (w1yp6zl7 ?rb7qr1fs ?uwq7ubjf).
    - Each missing goal fact corresponds roughly to 1-3 actions to achieve it,
      based on domain operators.
    - Some static predicates (e.g., b2mwd6x4, wx95011q, efxvv9ax, etc.) represent relationships
      linking objects and types, which inform connectivity and help estimate effort
      to reach goals.

    Heuristic Initialization:
    -------------------------
    - Extract goal facts from the task.
    - Identify and store goal (ixsb7u9c) pairs (dbk5i2a5, chxau8fy).
    - Identify and store goal (y01m0fhm) objects (uwq7ubjf).
    - Extract static relation mappings between objects from static facts:
       * Mapping rb7qr1fs -> uwq7ubjf (from jckjsxca and b2mwd6x4)
       * Which tom0ncko objects relate to rb7qr1fs (efxvv9ax)
       * Which dbk5i2a5 and chxau8fy pairs relate to rb7qr1fs (r1frsf5x)
       * Relations supporting achieving (y01m0fhm) from w1yp6zl7 and rpzrvk8g
    - These maps allow quick assessment whether required intermediate predicates 
      for goals are satisfied.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. If the current state satisfies all goal facts, heuristic is 0.

    2. Count the number of missing (ixsb7u9c) goal facts:
       - For each missing (ixsb7u9c dbk5i2a5 chxau8fy) fact:
         a) Check if the supporting (r1frsf5x rb7qr1fs dbk5i2a5 chxau8fy) fact is present for some rb7qr1fs.
            This indicates the goal fact is close to being achieved.
         b) If the support is present, estimate 1 action needed.
            Otherwise, more actions are needed (estimate 2 to 3).
         c) Accumulate an estimated cost for these missing ixsb7u9c facts.

    3. Count the number of missing (y01m0fhm uwq7ubjf) goal facts:
       - For each missing (y01m0fhm uwq7ubjf):
         a) Check if the required condition w1yp6zl7 or rpzrvk8g predicates exists with some rb7qr1fs.
            If yes, estimate 1 action needed.
            Otherwise, estimate 2 actions.
         b) Accumulate these costs.

    4. Return sum of these estimates as heuristic value.

    This heuristic guides the search efficiently by focusing on direct goal-related predicates
    and their immediate supports, while tolerating some inadmissibility for speed.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal predicates for fast membership tests
        self.goal_ixsb7u9c = set()
        self.goal_y01m0fhm = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "ixsb7u9c":
                # (ixsb7u9c dbk5i2a5 chxau8fy)
                self.goal_ixsb7u9c.add((parts[1], parts[2]))
            elif parts[0] == "y01m0fhm":
                # (y01m0fhm uwq7ubjf)
                self.goal_y01m0fhm.add(parts[1])

        # Build mappings from static facts
        # jckjsxca (rb7qr1fs uwq7ubjf)
        self.jckjsxca_map = {}
        for fact in self.static:
            if match(fact, "jckjsxca", "*", "*"):
                rb7qr1fs_obj, uwq7ubjf_obj = get_parts(fact)[1:3]
                self.jckjsxca_map.setdefault(rb7qr1fs_obj, set()).add(uwq7ubjf_obj)

        # b2mwd6x4 (rb7qr1fs uwq7ubjf uwq7ubjf) - relations between uwq7ubjf for rb7qr1fs
        self.b2mwd6x4_map = {}
        for fact in self.static:
            if match(fact, "b2mwd6x4", "*", "*", "*"):
                rb7qr1fs_obj, uwq7ubjf1, uwq7ubjf2 = get_parts(fact)[1:4]
                self.b2mwd6x4_map.setdefault(rb7qr1fs_obj, set()).add((uwq7ubjf1, uwq7ubjf2))

        # w1yp6zl7 (rb7qr1fs uwq7ubjf)
        self.w1yp6zl7_map = {}
        for fact in self.static:
            if match(fact, "w1yp6zl7", "*", "*"):
                rb7qr1fs_obj, uwq7ubjf_obj = get_parts(fact)[1:3]
                self.w1yp6zl7_map.setdefault(rb7qr1fs_obj, set()).add(uwq7ubjf_obj)

        # rpzrvk8g (rb7qr1fs uwq7ubjf)
        self.rpzrvk8g_map = {}
        for fact in self.static:
            if match(fact, "rpzrvk8g", "*", "*"):
                rb7qr1fs_obj, uwq7ubjf_obj = get_parts(fact)[1:3]
                self.rpzrvk8g_map.setdefault(rb7qr1fs_obj, set()).add(uwq7ubjf_obj)

        # r1frsf5x (rb7qr1fs dbk5i2a5 chxau8fy)
        self.r1frsf5x_map = {}
        for fact in self.static:
            if match(fact, "r1frsf5x", "*", "*", "*"):
                rb7qr1fs_obj, dbk5i2a5_obj, chxau8fy_obj = get_parts(fact)[1:4]
                self.r1frsf5x_map.setdefault(rb7qr1fs_obj, set()).add((dbk5i2a5_obj, chxau8fy_obj))

    def __call__(self, node):
        state = node.state

        # Quick check if goal satisfied
        if self.goals <= state:
            return 0

        # Check current ixsb7u9c facts in state (dbk5i2a5 chxau8fy) to see which goals remain
        achieved_ixsb7u9c = set()
        for fact in state:
            if match(fact, "ixsb7u9c", "*", "*"):
                parts = get_parts(fact)
                achieved_ixsb7u9c.add((parts[1], parts[2]))

        missing_ixsb7u9c = self.goal_ixsb7u9c - achieved_ixsb7u9c

        # Similarly for y01m0fhm objects
        achieved_y01m0fhm = set()
        for fact in state:
            if match(fact, "y01m0fhm", "*"):
                parts = get_parts(fact)
                achieved_y01m0fhm.add(parts[1])

        missing_y01m0fhm = self.goal_y01m0fhm - achieved_y01m0fhm

        # Extract current r1frsf5x facts from state (rb7qr1fs dbk5i2a5 chxau8fy)
        current_r1frsf5x = {}
        for fact in state:
            if match(fact, "r1frsf5x", "*", "*", "*"):
                parts = get_parts(fact)
                rb7qr1fs_o = parts[1]
                pair = (parts[2], parts[3])
                current_r1frsf5x.setdefault(rb7qr1fs_o, set()).add(pair)

        # Extract current w1yp6zl7 and rpzrvk8g facts to estimate y01m0fhm supports
        current_w1yp6zl7 = {}
        current_rpzrvk8g = {}
        for fact in state:
            if match(fact, "w1yp6zl7", "*", "*"):
                parts = get_parts(fact)
                rb7qr1fs_o, uwq7ubjf_o = parts[1], parts[2]
                current_w1yp6zl7.setdefault(rb7qr1fs_o, set()).add(uwq7ubjf_o)
            elif match(fact, "rpzrvk8g", "*", "*"):
                parts = get_parts(fact)
                rb7qr1fs_o, uwq7ubjf_o = parts[1], parts[2]
                current_rpzrvk8g.setdefault(rb7qr1fs_o, set()).add(uwq7ubjf_o)

        # Approximate the cost to achieve missing ixsb7u9c facts
        ixsb7u9c_cost = 0
        # For each missing (dbk5i2a5, chxau8fy) pair,
        # check if some rb7qr1fs links them in r1frsf5x (close to goal)
        for (dbk, chx) in missing_ixsb7u9c:
            # Check if any rb7qr1fs currently has (r1frsf5x rb7qr1fs dbk chx)
            supported = False
            for rb7 in current_r1frsf5x:
                if (dbk, chx) in current_r1frsf5x[rb7]:
                    supported = True
                    break

            if supported:
                # Already supported, needs ~1 action to finalize (ixsb7u9c)
                ixsb7u9c_cost += 1
            else:
                # Not supported yet, higher cost due to intermediate steps
                ixsb7u9c_cost += 2  # could increase if desired

        # Approximate the cost to achieve missing y01m0fhm facts
        y01m0fhm_cost = 0
        # For each uwq7ubjf needing y01m0fhm,
        # check if either current w1yp6zl7 or rpzrvk8g facts support it,
        # indicating closeness to goal
        for uwq in missing_y01m0fhm:
            supported = False
            for rb7 in current_w1yp6zl7:
                if uwq in current_w1yp6zl7.get(rb7, set()):
                    supported = True
                    break
            if not supported:
                for rb7 in current_rpzrvk8g:
                    if uwq in current_rpzrvk8g.get(rb7, set()):
                        supported = True
                        break
            if supported:
                # Close to achieving goal predicate y01m0fhm
                y01m0fhm_cost += 1
            else:
                # Needs extra preparatory steps
                y01m0fhm_cost += 2  # could be tuned

        # Sum estimated costs
        total_estimate = ixsb7u9c_cost + y01m0fhm_cost

        # Heuristic must be at least 1 for non-goal states
        if total_estimate == 0:
            return 1

        return total_estimate
