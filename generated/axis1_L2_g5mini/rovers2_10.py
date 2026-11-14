from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return list of tokens inside a fact string, ignoring surrounding parentheses."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A compact, domain-dependent heuristic for the secret_domain (rb7qr1fs).
    It focuses on the two kinds of goal facts that appear frequently in instances:
      - (ixsb7u9c <dbk5i2a5> <chxau8fy>)
      - (y01m0fhm <uwq7ubjf>)
    The heuristic estimates how many operator applications are required to
    produce these goal facts from the current state by inspecting the
    (shallow) causal chains encoded in the domain. The estimate is not
    admissible; it is designed to be informative and cheaply computable.

    Assumptions
    -----------
    - Goals are conjunctive and independent for the heuristic estimation.
      We sum per-goal estimates (simple additive combination).
    - The relevant goal-achieving operators and their immediate preconditions
      are:
        * ixsb7u9c(...)  <- ymbljzxy (requires r1frsf5x(...) and jckjsxca(...) and static wx95011q/rrhxwbyt)
        * r1frsf5x(...)  <- dlx472ba (requires wfrx4exs(...), efxvv9ax(...), jckjsxca(...), cs7680d7(...))
        * wfrx4exs(...)  <- p8i30ct9 (requires iaub97vi(...), cs7680d7(...), efxvv9ax(...), jckjsxca(...))
        * y01m0fhm(...)  <- sn0ysqqm (requires rpzrvk8g(...), rpzrvk8g can be created by xmjnictn)
      The heuristic uses the presence/absence of those intermediate facts
      in the current state (and of related static facts) to produce small,
      informative integer costs.
    - Static facts (task.static) are trusted: they provide necessary
      connectivity information (e.g., wx95011q, cs7680d7, z6yvwl7n, rrhxwbyt).
    - jckjsxca mappings (rb7qr1fs -> uwq7ubjf) are pivotal: if a mapping
      for a relevant rb7qr1fs object exists in the current state, achieving
      the goal tends to be cheaper.

    Heuristic Initialization
    ------------------------
    The constructor extracts:
      - self.goals: set of goal fact strings.
      - self.static_by_pred: dictionary mapping predicate names to lists of
        (arg1, arg2, ...) tuples for quick membership tests.
    The static facts used most by the heuristic are:
      - 'wx95011q', 'rrhxwbyt', 'cs7680d7', 'z6yvwl7n', 'iaub97vi',
        'p6u7tzz0', 'dfhfijzr', 'efxvv9ax', 'b2mwd6x4', 'a298chgk', 'bszmiquc'
    These are stored for lightweight pattern checks during heuristic calls.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For each goal fact not yet satisfied in the current state:
      - If the goal is (ixsb7u9c R6 PMQ):
          1) If the exact fact exists in state -> cost 0.
          2) If some r1frsf5x(JD, R6, PMQ) exists in the state:
               - If there is also a jckjsxca(JD, U) in the state and the
                 static facts wx95011q(U, V) and rrhxwbyt(S, V) exist
                 (i.e., the final operator ymbljzxy likely has its
                 supporting static facts), then we estimate cost = 1
                 (one ymbljzxy).
               - If r1frsf5x exists but the supporting mapping is missing,
                 assign cost = 2 (we assume an extra mapping/swap step).
          3) If no r1frsf5x currently exists:
               - Check whether there are static links that make creation of
                 r1frsf5x plausible in a couple of steps (iaub97vi and
                 cs7680d7 and an efxvv9ax linking). If these statics are
                 present for R6 and some JD/EEULD, assume:
                   p8i30ct9 + dlx472ba + ymbljzxy -> estimated cost = 3.
               - Otherwise use a conservative fallback cost = 5.
      - If the goal is (y01m0fhm LT0):
          1) If the fact exists in the state -> cost 0.
          2) If some rpzrvk8g(JD, JEH) exists in the state:
               - If jckjsxca(JD, JEH) exists and static wx95011q/rrhxwbyt
                 provide the other required connectivity, estimate cost = 1
                 (one sn0ysqqm).
               - Otherwise estimate cost = 2 to account for small missing
                 setup steps.
          3) If rpzrvk8g is not present:
               - If the static facts that allow creation via xmjnictn exist
                 (z6yvwl7n, p6u7tzz0, dfhfijzr) and a suitable jckjsxca is
                 available, estimate cost = 2 (xmjnictn + sn0ysqqm).
               - Otherwise use a fallback cost = 4.
      - Any other goal predicates (uncommon) receive a conservative cost = 3
        if not present (simple fallback).

    The heuristic returns the sum of per-goal costs. It is fast: the checks
    are membership queries over small dictionaries and scans over the current
    state. The numbers (1,2,3,4,5) are tuned to reflect operator-chain
    lengths observed in the domain (one-shot operators like ymbljzxy/sn0ysqqm
    are cost 1; creating prerequisite structures costs a few more steps).

    Note: The heuristic is not admissible. It is intended to be informative
    for greedy best-first search to reduce node expansions while remaining
    cheap to evaluate.
    """

    def __init__(self, task):
        self.goals = task.goals
        # Build a predicate -> list of argument tuples mapping for static facts.
        self.static_by_pred = {}
        for fact in task.static:
            parts = get_parts(fact)
            pred = parts[0]
            args = tuple(parts[1:])
            self.static_by_pred.setdefault(pred, []).append(args)

    def _static_has(self, pred, *args):
        """Fast check if the exact static fact pred(args...) exists."""
        return args in self.static_by_pred.get(pred, [])

    def __call__(self, node):
        state = node.state

        # If already a goal state, heuristic is zero.
        if self.goals <= state:
            return 0

        # Precompute quick-access structures over the current state.
        state_by_pred = {}
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            args = tuple(parts[1:])
            state_by_pred.setdefault(pred, []).append(args)

        def state_has(pred, *args):
            return args in state_by_pred.get(pred, [])

        total_cost = 0

        for goal in self.goals:
            if goal in state:
                continue  # satisfied

            parts = get_parts(goal)
            pred = parts[0]
            args = parts[1:]

            # Handle ixsb7u9c goals: require r1frsf5x(...) then ymbljzxy.
            if pred == "ixsb7u9c":
                r6, pmq = args
                # Check whether some r1frsf5x(JD, r6, pmq) exists in state
                r1_candidates = [
                    jd for (jd, r6s, pmqs) in state_by_pred.get("r1frsf5x", [])
                    if r6s == r6 and pmqs == pmq
                ]
                if r1_candidates:
                    # If there is a JD where jckjsxca(JD, U) is present and
                    # static connectivity wx95011q(U, V) and rrhxwbyt(S, V) exist,
                    # then ymbljzxy is likely applicable now.
                    satisfied = False
                    for jd in r1_candidates:
                        for (u,) in [(args2[0],) for args2 in state_by_pred.get("jckjsxca", []) if args2[0] == jd]:
                            # find all jckjsxca entries for this jd
                            # actually state_by_pred jckjsxca stores tuples (jd, u)
                            pass
                        # iterate explicit
                        for (jd2, u) in state_by_pred.get("jckjsxca", []):
                            if jd2 != jd:
                                continue
                            # need static wx95011q(u, v) and rrhxwbyt(s, v)
                            wx_list = [v for (a, v) in self.static_by_pred.get("wx95011q", []) if a == u]
                            found_conn = False
                            for v in wx_list:
                                # rrhxwbyt is likely static (maps stk->uwq)
                                # check if any rrhxwbyt(s, v) exists in static or state
                                if self._static_has("rrhxwbyt", v) or ("rrhxwbyt" in state_by_pred and any(v == t for (_, t) in state_by_pred.get("rrhxwbyt", []))):
                                    found_conn = True
                                    break
                                # rrhxwbyt typically has two args: (stk6x, ijb5b) ; v is ijb5b so check static pairs
                                for (s, ijb) in self.static_by_pred.get("rrhxwbyt", []):
                                    if ijb == v:
                                        found_conn = True
                                        break
                                if found_conn:
                                    break
                            if found_conn:
                                satisfied = True
                                break
                        if satisfied:
                            break
                    if satisfied:
                        total_cost += 1
                    else:
                        # r1 exists but supporting mapping/connectivity missing: small extra cost.
                        total_cost += 2
                else:
                    # No r1frsf5x currently: check static patterns that would allow creating it
                    # p8i30ct9 + dlx472ba + ymbljzxy ~= 3 ops if supporting statics exist
                    # Required pieces: exist static efxvv9ax(EE, JD), iaub97vi(EE, R6) and cs7680d7(R6, LT)
                    plausible = False
                    # search for efxvv9ax(ee, jd) in static or state and iaub97vi(ee, r6)
                    for (ee, jd) in self.static_by_pred.get("efxvv9ax", []) + state_by_pred.get("efxvv9ax", []):
                        if (ee, r6) in self.static_by_pred.get("iaub97vi", []) or (ee, r6) in state_by_pred.get("iaub97vi", []):
                            # check cs7680d7 exists for (r6, lt) for some lt (required by p8i30ct9)
                            if any(r6 == r for (r, lt) in self.static_by_pred.get("cs7680d7", [])) or any(r6 == r for (r, lt) in state_by_pred.get("cs7680d7", [])):
                                plausible = True
                                break
                    if plausible:
                        total_cost += 3
                    else:
                        total_cost += 5  # fallback conservative estimate

            # Handle y01m0fhm goals: require rpzrvk8g + sn0ysqqm or creation via xmjnictn.
            elif pred == "y01m0fhm":
                lt0 = args[0]
                # If rpzrvk8g(JD, LT0) already exists in the state, sn0ysqqm may be applicable.
                rpz_candidates = [ (jd, jeh) for (jd, jeh) in state_by_pred.get("rpzrvk8g", []) if jeh == lt0 or jd == lt0 ]
                # Note: rpzrvk8g signature is (?jdovh - rb7qr1fs ?jeh3h - uwq7ubjf)
                # We consider any rpzrvk8g where the second argument equals the desired LT0.
                if any(jeh == lt0 for (jd, jeh) in state_by_pred.get("rpzrvk8g", [])):
                    # check other likely preconditions (jckjsxca, wx95011q, rrhxwbyt)
                    satisfied = False
                    for (jd, jeh) in state_by_pred.get("rpzrvk8g", []):
                        if jeh != lt0:
                            continue
                        # need jckjsxca(jd, some) and static wx95011q & rrhxwbyt for that mapping
                        for (jd2, u) in state_by_pred.get("jckjsxca", []):
                            if jd2 != jd:
                                continue
                            # find static wx95011q(u, v) and rrhxwbyt(s, v)
                            wx_list = [v for (a, v) in self.static_by_pred.get("wx95011q", []) if a == u]
                            for v in wx_list:
                                for (s, ijb) in self.static_by_pred.get("rrhxwbyt", []):
                                    if ijb == v:
                                        satisfied = True
                                        break
                                if satisfied:
                                    break
                            if satisfied:
                                break
                        if satisfied:
                            break
                    if satisfied:
                        total_cost += 1
                    else:
                        total_cost += 2
                else:
                    # rpzrvk8g not present: check if xmjnictn can plausibly create it in one step
                    # xmjnictn requires: z6yvwl7n(zmrjf, jehs5), ipivzc5v jeh3h, p6u7tzz0 zmrjf,
                    # dfhfijzr jehs5 and jckjsxca jehs5 jeh3h.
                    plausible = False
                    # iterate over z6yvwl7n static entries that point to some rb7qr1fs object
                    for (zmrjf, jehs5) in self.static_by_pred.get("z6yvwl7n", []):
                        # p6u7tzz0 zmrjf should exist (static)
                        if (zmrjf,) in self.static_by_pred.get("p6u7tzz0", []):
                            # dfhfijzr jehs5 must exist in static or state
                            if (jehs5,) in self.static_by_pred.get("dfhfijzr", []) or (jehs5,) in state_by_pred.get("dfhfijzr", []):
                                # try to find a jckjsxca(jehs5, jeh3h) for some jeh3h (in state)
                                for (jehs5_s, jeh3h) in state_by_pred.get("jckjsxca", []):
                                    if jehs5_s != jehs5:
                                        continue
                                    # ipivzc5v jeh3h must be present somewhere (static or state)
                                    if (jeh3h,) in self.static_by_pred.get("ipivzc5v", []) or (jeh3h,) in state_by_pred.get("ipivzc5v", []):
                                        plausible = True
                                        break
                                if plausible:
                                    break
                    if plausible:
                        total_cost += 2
                    else:
                        total_cost += 4  # fallback

            else:
                # Generic fallback for other predicates: small constant cost
                total_cost += 3

        # Ensure finite integer
        return int(total_cost)