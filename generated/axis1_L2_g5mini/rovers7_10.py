from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the components of a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()

def build_pred_map(facts):
    """
    Build a mapping from predicate name to a set of tuples of arguments.
    facts: iterable of fact strings like '(pred a b)'.
    """
    pred_map = {}
    for f in facts:
        parts = get_parts(f)
        if not parts:
            continue
        pred = parts[0]
        args = tuple(parts[1:])
        pred_map.setdefault(pred, set()).add(args)
    return pred_map


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic crafted for the secret_domain (rb7qr1fs).
    The domain's goals in sample instances typically include predicates such as
    - ixsb7u9c <r6gw2> <pmq6p>
    - y01m0fhm <lt0bb>
    These goal facts are usually produced by multi-step operator chains:
      p8i30ct9 -> wfrx4exs -> dlx472ba -> r1frsf5x -> ymbljzxy -> ixsb7u9c
      xmjnictn -> rpzrvk8g -> sn0ysqqm -> y01m0fhm
    The heuristic estimates how many actions are required to establish each
    missing goal fact by looking for evidence that some or parts of those
    production chains are already present in the current state.

    Assumptions
    -----------
    - The heuristic is not required to be admissible; it is allowed to be
      optimistic or pessimistic so long as it is informative and efficiently
      computable.
    - Static facts are truly static (they never change) and are useful to
      detect whether prerequisites for producing a goal fact already exist.
    - The domain contains repeated structures: many operators produce intermediate
      predicates (e.g., r1frsf5x, wfrx4exs, rpzrvk8g). The heuristic exploits
      the presence of these intermediate predicates in the current state to
      reduce the estimated remaining cost.
    - If no helpful intermediate facts are present, the heuristic assigns a small
      bounded cost estimate (rather than infinity) so the heuristic remains
      finite for solvable states.

    Heuristic Initialization
    ------------------------
    - Stores the set of goal facts (self.goals).
    - Parses static facts from the task into a predicate->set(args) mapping
      (self.static_map) for quick membership checks of static prerequisites.
    - No expensive offline preprocessing is performed; initialization is O(|static|).

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given search node (state), the heuristic proceeds as follows:

    1. Quick exit:
       - If all goal facts are already true in the given state, return 0.

    2. Parse the current state into a predicate->set(args) mapping
       to allow fast lookups of "which facts with this predicate exist".

    3. For each goal fact that is not satisfied in the current state, estimate
       the number of actions required to make it true using small, domain-aware
       pattern checks. The estimator uses short causal-chains (the concrete
       chains observed in the domain) to produce a compact cost:

       - Goals of form (ixsb7u9c R6 PM):
         * If an r1frsf5x ?JDOVH R6 PM entry is already present in the state,
           then we assume one action (ymbljzxy) suffices to produce ixsb7u9c.
         * Else, if there exists an eeuld such that both
           (wfrx4exs eeuld ?JDOVH) and (rycjvfag eeuld PM) exist (these are the
           immediate prerequisites typically used to produce r1frsf5x via dlx472ba),
           assume two actions (dlx472ba + ymbljzxy).
         * Else, if (efxvv9ax eeuld ?JDOVH) together with (rycjvfag eeuld PM)
           exists for some eeuld, assume three actions (p8i30ct9 + dlx472ba + ymbljzxy).
         * Otherwise assign a small fallback cost (4) representing a longer chain.

       - Goals of form (y01m0fhm LT0BB):
         * If a matching (rpzrvk8g ?JDOVH LT0BB) is present, assume one action
           (sn0ysqqm) will produce y01m0fhm.
         * Else if there is evidence that the xmjnictn preconditions are close
           (for example, an ipivzc5v LT0BB or a p6u7tzz0 that could trigger the
           rpzrvk8g production), assume two actions to produce rpzrvk8g then one
           more to produce y01m0fhm — we therefore add 2 as the estimate.
         * Otherwise assign a small fallback cost (3).

       - Any other goal predicate (not observed in the sampled instances) receives
         a default small cost of 1 if absent.

    4. Sum the estimated costs for each missing goal fact and return the sum.

    Rationale and properties
    ------------------------
    - The heuristic is intentionally lightweight: it checks for a few
      intermediate facts (those that commonly appear in the production chains)
      and assigns small integer costs based on how close the state is to
      achieving a goal fact. This tends to guide greedy best-first search
      toward states that have already satisfied subgoals or partial chains.
    - The heuristic returns 0 precisely when all goal facts are satisfied.
    - It always returns a finite integer (small bounded estimates for unknown
      chains), so the planner will not receive infinite heuristics on
      solvable states.
    """

    def __init__(self, task):
        # Save goals and static information for quick checks.
        self.goals = task.goals  # set of goal fact strings
        # Build a predicate->set(args-tuples) map for static facts.
        self.static_map = build_pred_map(task.static)

    def __call__(self, node):
        """
        Compute an estimate of the number of actions needed from the node's state
        to reach the goal. Returns 0 iff the state already satisfies all goals.
        """
        state = node.state  # frozenset of fact strings

        # Fast check: if all goals satisfied, heuristic is 0.
        if self.goals <= state:
            return 0

        # Build predicate->args map for the current state.
        state_map = build_pred_map(state)

        total_cost = 0

        # Helper checks for r1frsf5x and related intermediate facts for ixsb7u9c.
        def has_r1_for(r6, pm):
            """Return True if there exists r1frsf5x ?jdovh r6 pm in state."""
            for args in state_map.get("r1frsf5x", ()):
                # args are (jdovh, r6gw2, pmq6p)
                if len(args) >= 2 and args[1] == r6 and args[2] == pm:
                    return True
            return False

        def exists_wfrx_and_rycj_for(pm):
            """
            Return True if there exists an eeuld such that
            (wfrx4exs eeuld ?jdovh) and (rycjvfag eeuld pm) are both present.
            This is a heuristic indicator that dlx472ba could produce r1frsf5x.
            """
            wfrx_eeulds = {args[0] for args in state_map.get("wfrx4exs", ()) if args}
            rycj_eeulds = {args[0] for args in state_map.get("rycjvfag", ()) if len(args) >= 2 and args[1] == pm}
            return len(wfrx_eeulds & rycj_eeulds) > 0

        def exists_efx_and_rycj_for(pm):
            """
            Return True if there exists an eeuld such that
            (efxvv9ax eeuld ?jdovh) and (rycjvfag eeuld pm) exist.
            This indicates an earlier point in the chain.
            """
            efx_eeulds = {args[0] for args in state_map.get("efxvv9ax", ()) if args}
            rycj_eeulds = {args[0] for args in state_map.get("rycjvfag", ()) if len(args) >= 2 and args[1] == pm}
            return len(efx_eeulds & rycj_eeulds) > 0

        # Helper checks for rpzrvk8g and related facts for y01m0fhm.
        def has_rpz_for(lt):
            """Return True if there exists rpzrvk8g ?jdovh lt in state."""
            for args in state_map.get("rpzrvk8g", ()):
                if len(args) >= 2 and args[1] == lt:
                    return True
            return False

        def has_ipiv_for(lt):
            """Return True if (ipivzc5v lt) is in state (useful for xmjnictn)."""
            return (lt,) in state_map.get("ipivzc5v", set())

        def has_p6u_for_some():
            """Return True if any p6u7tzz0 fact exists in state (helps xmjnictn)."""
            return len(state_map.get("p6u7tzz0", ())) > 0

        # Iterate missing goals and add heuristic estimate for each.
        for goal in self.goals:
            if goal in state:
                continue  # goal satisfied, cost 0

            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]

            # Heuristic for ixsb7u9c goals.
            if pred == "ixsb7u9c" and len(args) >= 2:
                r6, pm = args[0], args[1]
                if has_r1_for(r6, pm):
                    total_cost += 1
                elif exists_wfrx_and_rycj_for(pm):
                    total_cost += 2
                elif exists_efx_and_rycj_for(pm):
                    total_cost += 3
                else:
                    # fallback pessimistic but bounded estimate
                    total_cost += 4

            # Heuristic for y01m0fhm goals.
            elif pred == "y01m0fhm" and len(args) >= 1:
                lt = args[0]
                if has_rpz_for(lt):
                    total_cost += 1
                elif has_ipiv_for(lt) or has_p6u_for_some():
                    # we have some evidence xmjnictn might be applicable soon
                    total_cost += 2
                else:
                    total_cost += 3

            # Generic fallback for other goal predicates: cost 1 if missing.
            else:
                total_cost += 1

        # Ensure heuristic is an integer >= 0.
        return int(max(0, total_cost))