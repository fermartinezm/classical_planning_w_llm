from heuristics.heuristic_base import Heuristic


def _parse_fact(fact):
    """Parse a PDDL fact string into a tuple of (predicate, arg1, arg2, ...)."""
    return tuple(fact[1:-1].split())  # remove surrounding parentheses and split


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach the goal state
    by focusing on the differences between the current state and goal facts,
    particularly on the predicates involving `bclhkf4r`, `mdawkki4`, `gaijkwef`, 
    `zzar8qlk`, and `chxwa5j4`. It counts unsatisfied goal facts and adds penalties
    for 'zzar8qlk' state mismatches and for those objects that require state transitions,
    modeling the domain transitions needed to achieve the goal.

    Assumptions
    -----------
    - The only interesting objects are those appearing in goal predicates.
    - Each goal fact involving predicates with arguments is treated individually.
    - The predicate `zzar8qlk` is a global state flag that flips often to enable actions.
    - Objects either have `bclhkf4r`, `mdawkki4`, and/or `chxwa5j4` predicates,
      which represent modes or states for those objects.
    - Transitions from `bclhkf4r` to `mdawkki4` and/or `chxwa5j4` predicates require actions.
    - The predicate `gaijkwef ?x ?y` represents some binary relation that needs fixing
      if missing from current state but required in goal state.

    Heuristic Initialization
    ------------------------
    - Extract and store the goal facts, separating those with predicates that have
      arguments.
    - Store the entire set of static facts (though empty here).
    - Prepare quick membership checks for goal facts.
    - No static facts are used because the domain declares none.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state is a goal state, heuristic=0.
    2. Count the number of goal facts that are not true in the current state.
    3. Check if the `(zzar8qlk)` predicate is currently true; this affects how many
       actions may be needed, as `zzar8qlk` toggles in operator effects.
    4. For each object mentioned in any `bclhkf4r`, `mdawkki4`, or `chxwa5j4` fact
       in the state or goal:
       - Identify if that object’s mode/state predicates match the goal.
       - Each mismatch in mode/state predicates contributes to estimated needed actions.
    5. For each `gaijkwef ?x ?y` goal fact not true in current state, add penalty.
    6. Combine these counts with a small weighted cost for toggling `(zzar8qlk)`
       because many actions require `(zzar8qlk)` or its negation.
    7. Sum all these to estimate the number of actions needed to achieve the goal.

    The heuristic is not admissible, but it tries to give a good proxy to distance to goal.
    """

    def __init__(self, task):
        """Initialize heuristic by extracting goal predicates and relevant objects."""
        self.goals = task.goals

        # Separate goal facts by predicate
        self.goal_bclhkf4r = set()
        self.goal_mdawkki4 = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()
        self.goal_zzar8qlk = False
        self.goal_facts = set()

        for fact in self.goals:
            self.goal_facts.add(fact)
            pred, *args = _parse_fact(fact)
            if pred == "bclhkf4r":
                self.goal_bclhkf4r.add(args[0])
            elif pred == "mdawkki4":
                self.goal_mdawkki4.add(args[0])
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(args[0])
            elif pred == "gaijkwef":
                self.goal_gaijkwef.add((args[0], args[1]))
            elif pred == "zzar8qlk":
                self.goal_zzar8qlk = True

        # Gather all objects involved in goal mode predicates and gaijkwef relations
        self.goal_objects = set()
        self.goal_objects.update(self.goal_bclhkf4r)
        self.goal_objects.update(self.goal_mdawkki4)
        self.goal_objects.update(self.goal_chxwa5j4)
        for (x, y) in self.goal_gaijkwef:
            self.goal_objects.add(x)
            self.goal_objects.add(y)

    def __call__(self, node):
        """
        Compute heuristic estimate of number of actions from current state to goal state.
        """
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goal_facts <= state:
            return 0

        # Index current state predicates relevant to mode and relations
        current_bclhkf4r = set()
        current_mdawkki4 = set()
        current_chxwa5j4 = set()
        current_gaijkwef = set()
        current_zzar8qlk = False

        for fact in state:
            pred, *args = _parse_fact(fact)
            if pred == "bclhkf4r":
                current_bclhkf4r.add(args[0])
            elif pred == "mdawkki4":
                current_mdawkki4.add(args[0])
            elif pred == "chxwa5j4":
                current_chxwa5j4.add(args[0])
            elif pred == "gaijkwef":
                current_gaijkwef.add((args[0], args[1]))
            elif pred == "zzar8qlk":
                current_zzar8qlk = True

        heuristic = 0

        # Count missing goal facts: each missing goal fact costs 1 action as baseline
        missing_goal_facts = self.goal_facts - state
        heuristic += len(missing_goal_facts)

        # Penalize differences in (zzar8qlk) flag:
        # Because actions toggle this flag and some require it true/false,
        # we add 1 if current differs from goal.
        if current_zzar8qlk != self.goal_zzar8qlk:
            heuristic += 1

        # For each goal object, compare current and goal modes:
        # Mode transitions usually require at least one action.
        # If mode predicates differ, add penalties.
        # States per object can be bclhkf4r, mdawkki4, chxwa5j4.
        for obj in self.goal_objects:
            have_bcl = obj in current_bclhkf4r
            have_mdaw = obj in current_mdawkki4
            have_chx = obj in current_chxwa5j4

            want_bcl = obj in self.goal_bclhkf4r
            want_mdaw = obj in self.goal_mdawkki4
            want_chx = obj in self.goal_chxwa5j4

            # If current modes exactly match goal, no cost.
            # If not, estimate cost as 1 per mode difference (cautiously).
            # Counting how many mode predicates differ:
            cost_modes = 0
            if have_bcl != want_bcl:
                cost_modes += 1
            if have_mdaw != want_mdaw:
                cost_modes += 1
            if have_chx != want_chx:
                cost_modes += 1

            heuristic += cost_modes

        # For each goal (gaijkwef x y) not present, add 1 to cost.
        missing_gaijkwef = self.goal_gaijkwef - current_gaijkwef
        heuristic += len(missing_gaijkwef)

        return heuristic
