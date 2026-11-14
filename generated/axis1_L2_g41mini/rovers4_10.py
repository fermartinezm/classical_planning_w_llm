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
    Domain-dependent heuristic for the secret_domain domain.

    Summary:
    ----------
    The heuristic estimates the minimal number of actions required to achieve all goal facts 
    that describe the predicates (ixsb7u9c ?dbk5i2a5 ?chxau8fy) and (y01m0fhm ?uwq7ubjf).
    These predicates appear as goals in the example instances, indicating that the primary 
    objective is to establish these facts for certain object pairs.

    The heuristic focuses on how many of these goal-required predicates are not yet achieved 
    and approximates costs based on the domain's structure — the transport/movement of 
    uwq7ubjf objects and establishing "ixsb7u9c" relations between dbk5i2a5 and chxau8fy.

    Assumptions:
    ------------
    - The goals always require making certain (ixsb7u9c dbk5i2a5 chxau8fy) facts true, and/or
      ensuring that certain uwq7ubjf objects satisfy (y01m0fhm uwq7ubjf).
    - The transitions to achieve (ixsb7u9c ...) facts are costly and depend on moving/interacting 
      uwq7ubjf and dbk5i2a5 objects through rb7qr1fs "locations".
    - Actions that change positions of uwq7ubjf objects (like xvldpwsb), and establish relations 
      (like dlx472ba) are critical.
    - The heuristic is not necessarily admissible but tries to estimate action counts efficiently 
      to guide greedy best-first search.

    Heuristic Initialization:
    -------------------------
    - Extract all goal facts.
    - Extract static facts, especially fixed predicates involving objects that might influence cost 
      estimates (e.g., "bszmiquc", "a298chgk" predicates describing rb7qr1fs objects).
    - Extract from static facts sets of all uwq7ubjf, dbk5i2a5, chxau8fy, zp2z4mnx, tom0ncko objects 
      (by scanning relevant predicates, especially from static).
    - Identify mappings useful for reasoning, e.g., which uwq7ubjf objects are linked or connected, 
      and which relations are static.
    - Remember goal predicates of the types (ixsb7u9c ?) and (y01m0fhm ?) for heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. If the state is a goal state (all goal facts true), return 0.

    2. Count how many goal (ixsb7u9c dbk5i2a5 chxau8fy) facts are not yet true in the current state.
       Each such missing fact implies that the domain must perform a sequence of actions to "connect" 
       the corresponding dbk5i2a5 and chxau8fy objects via rb7qr1fs instances.

    3. Count how many goal (y01m0fhm uwq7ubjf) facts are missing, which involve marking uwq7ubjf objects.

    4. Estimate number of action steps needed to achieve each missing (ixsb7u9c ...) fact:
       - Achieving one (ixsb7u9c) fact requires:
         a) Establishing (r1frsf5x rb7qr1fs dbk5i2a5 chxau8fy) — transports or deliveries, often done by dlx472ba.
         b) It depends on having effects like (wfrx4exs tom0ncko rb7qr1fs), i.e., states involving tom0ncko instances.
         c) Moving owq7ubjf objects around (like xvldpwsb) to fulfill preconditions.
       - The heuristic approximates by counting missing (ixsb7u9c) facts, multiplying by a small constant that simulates 
         the number of actions involved (suggest ~3–5 actions per missing fact).

    5. Estimate costs for missing (y01m0fhm uwq7ubjf) facts similarly:
       - Establishing (y01m0fhm uwq7ubjf) likely requires moving uwq7ubjf objects to certain states or locations.
       - Approximate cost per missing fact as ~1-2 actions.

    6. Finally, account for the current number of (jckjsxca rb7qr1fs uwq7ubjf) predicates, 
       as this indicates locations of uwq7ubjf objects connected to rb7qr1fs objects (likely needed for 
       achieving goals). If few uwq7ubjf are connected, additional movements may be needed.

    7. Sum these estimated costs and return as the heuristic value.

    This heuristic balances between accuracy and efficiency by focusing only on counting missing goal predicates
    and approximating cost based on domain relationships extracted from static facts.
    """

    def __init__(self, task):
        self.goals = task.goals  # goal predicates
        self.static = task.static  # static facts

        # Sets of objects by type extracted from static facts and goals for reference
        self.rb7qr1fs_objs = set()
        self.uwq7ubjf_objs = set()
        self.dbk5i2a5_objs = set()
        self.chxau8fy_objs = set()
        self.zp2z4mnx_objs = set()
        self.tom0ncko_objs = set()
        self.ggaeyv41_objs = set()

        # Extract objects from static facts using predicate signatures
        # We analyze static facts to extract objects by type based on predicates and their arguments
        for fact in self.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            # Based on domain types known from domain-file:
            # rb7qr1fs: often first argument in predicates like bszmiquc, a298chgk, a298chgk(?jdovh)
            # uwq7ubjf: second or third argument in b2mwd6x4, wx95011q, etc.
            # dbk5i2a5 and chxau8fy: often appear together in r1frsf5x and ixsb7u9c
            # zp2z4mnx: appears in predicates h163jybj, p6u7tzz0, z6yvwl7n
            # tom0ncko: efxvv9ax, iaub97vi, wfrx4exs
            # ggaeyv41: rrhxwbyt, sn0ysqqm

            if pred in ("bszmiquc", "a298chgk", "dfhfijzr", "rpzrvk8g", "jckjsxca", "r1frsf5x"):
                self.rb7qr1fs_objs.add(args[0])
            if pred in ("b2mwd6x4", "wx95011q", "w1yp6zl7", "splvr1ag", "ipivzc5v", "y01m0fhm", "cs7680d7", "rpzrvk8g", "jckjsxca", "ixsb7u9c"):
                for a in args:
                    if a.startswith("w") or a.startswith("x") or a.startswith("j") or a.startswith("f") or a.startswith("l") or a.startswith("d") or a.startswith("q") or a.startswith("v") or a.startswith("h") or a.startswith("y"):
                        self.uwq7ubjf_objs.add(a)
            if pred in ("r1frsf5x", "ixsb7u9c", "cs7680d7", "iaub97vi", "h163jybj"):
                # first arg is rb7qr1fs or dbk5i2a5, second dbk5i2a5 or chxau8fy
                for a in args:
                    # Heuristic guess: numerical/digit starting object names mostly dbk5i2a5 or chxau8fy
                    if a.startswith("j") or a.startswith("k") or a.startswith("m") or a.startswith("q") or a.startswith("n") or a.startswith("o") or a.startswith("r"):
                        self.dbk5i2a5_objs.add(a)
                    if a.startswith("v") or a.startswith("y") or a.startswith("p") or a.startswith("n") or a.startswith("u"):
                        self.chxau8fy_objs.add(a)
            if pred in ("h163jybj", "p6u7tzz0", "z6yvwl7n"):
                self.zp2z4mnx_objs.add(args[0])
            if pred in ("efxvv9ax", "iaub97vi", "wfrx4exs", "rycjvfag"):
                self.tom0ncko_objs.add(args[0])
            if pred in ("rrhxwbyt", "sn0ysqqm"):
                self.ggaeyv41_objs.add(args[0])

        # Extract goal predicates of interest
        self.goal_ixsb7u9c = set(
            g for g in self.goals if get_parts(g)[0] == "ixsb7u9c"
        )
        self.goal_y01m0fhm = set(
            g for g in self.goals if get_parts(g)[0] == "y01m0fhm"
        )

    def __call__(self, node):
        state = node.state

        # 1) If goal reached, heuristic is 0
        if self.goals <= state:
            return 0

        # 2) Count missing (ixsb7u9c ?dbk5i2a5 ?chxau8fy) goals
        missing_ixsb7u9c = 0
        for goal_fact in self.goal_ixsb7u9c:
            if goal_fact not in state:
                missing_ixsb7u9c += 1

        # 3) Count missing (y01m0fhm ?uwq7ubjf) goals
        missing_y01m0fhm = 0
        for goal_fact in self.goal_y01m0fhm:
            if goal_fact not in state:
                missing_y01m0fhm += 1

        # 4) Count connected uwq7ubjf objects to rb7qr1fs using (jckjsxca rb7qr1fs uwq7ubjf)
        connected_uwq7ubjf = set()
        for fact in state:
            if match(fact, "jckjsxca", "*", "*"):
                parts = get_parts(fact)
                if parts[1] in self.rb7qr1fs_objs and parts[2] in self.uwq7ubjf_objs:
                    connected_uwq7ubjf.add(parts[2])

        # 5) Estimate cost for establishing missing (ixsb7u9c ...) facts:
        # Each missing (ixsb7u9c) likely requires several coordinated moves and actions,
        # e.g. moving uwq7ubjf objects to correct place, interacting with tom0ncko and rb7qr1fs objects,
        # and performing dlx472ba actions.
        # We'll assume approx 4 actions per missing (ixsb7u9c):
        ixsb7u9c_cost = 4 * missing_ixsb7u9c

        # 6) Estimate cost for establishing missing (y01m0fhm ...) facts:
        # Each likely requires moving or interacting uwq7ubjf.
        # Assume approx 2 actions per missing (y01m0fhm) fact.
        y01m0fhm_cost = 2 * missing_y01m0fhm

        # 7) If few uwq7ubjf connected to rb7qr1fs (less than goals involving y01m0fhm),
        # assume some additional cost to move uwq7ubjf to desired locations.
        connections_diff = max(0, len(self.goal_y01m0fhm) - len(connected_uwq7ubjf))
        uwq7ubjf_move_cost = connections_diff * 2  # approx 2 actions per needed connection

        # 8) Sum and return heuristic
        return ixsb7u9c_cost + y01m0fhm_cost + uwq7ubjf_move_cost
