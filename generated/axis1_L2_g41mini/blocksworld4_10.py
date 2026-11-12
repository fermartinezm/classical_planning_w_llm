from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (dnbnd6kg) domain.

    Summary
    -------
    This heuristic estimates the number of actions needed to reach the goal state based on the
    status of key predicates and the relationships between objects encoded by the predicate gaijkwef.

    Assumptions
    -----------
    - The goal conditions mostly involve facts about predicates bclhkf4r, chxwa5j4, and gaijkwef.
    - The predicate zzar8qlk acts like a "token" that must be toggled (true/false) to
      enable certain transitions.
    - Objects can have "ownership" or "attachment" relations represented by gaijkwef(?x ?y).
    - The four actions can be viewed roughly as cover, uncover, switch, and relocate operations
      governed by modeled token presence and predicates.
    - Reaching the goal often requires ensuring the correct assignment of bclhkf4r and chxwa5j4 facts,
      as well as establishing the exact gaijkwef relations stated in the goal.
    - The heuristic is not necessarily admissible, but should minimize node expansions and be efficient.

    Heuristic Initialization
    ------------------------
    - Extract goal sets for the predicates bclhkf4r, chxwa5j4, and gaijkwef for quicker lookups.
    - Store initial presence of zzar8qlk in the current state, since it heavily controls action applicability.
    - As static facts are empty, no static knowledge is used.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state is a goal state (all goal facts hold), return 0 immediately.

    2. Count how many goal facts of predicates bclhkf4r and chxwa5j4 are missing in the current state.
       Each missing fact likely requires at least one relevant action to achieve.

    3. For the gaijkwef goal facts (binary relations):
       - Count how many goal gaijkwef facts are currently false but whose objects exist in the state.
       - These missing relations typically require multiple steps involving token toggling and object swapping.

    4. Count how many unnecessary bclhkf4r or chxwa5j4 facts exist in current state which are not in goal.
       - These might need to be removed or corrected, adding to heuristic cost.

    5. Consider the presence or absence of token predicate zzar8qlk:
       - Many actions require zzar8qlk to be true or false.
       - If zzar8qlk must be toggled to apply certain actions, add a small extra cost for token flips.

    6. Estimate:
       - Each missing bclhkf4r or chxwa5j4 fact contributes about 1 action cost.
       - Each missing gaijkwef relation is at least 2-3 actions due to sequences of toggling and swapping.
       - Each unnecessary predicate to remove adds roughly 1 action.
       - Token toggling costs add 0 or 1 depending if zzar8qlk currently matches what is needed.

    7. Sum the above costs to produce the heuristic estimate.
       This heuristic balances under- and over-estimation to guide greedy best-first search efficiently.
    """

    def __init__(self, task):
        # Extract all goal facts for quick membership checks
        self.goals = task.goals

        # Partition goal facts by predicates for bclhkf4r, chxwa5j4, and gaijkwef
        self.goal_bclhkf4r = set()
        self.goal_chxwa5j4 = set()
        self.goal_gaijkwef = set()
        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "bclhkf4r":
                self.goal_bclhkf4r.add(fact)
            elif pred == "chxwa5j4":
                self.goal_chxwa5j4.add(fact)
            elif pred == "gaijkwef":
                self.goal_gaijkwef.add(fact)

        # As static facts are empty for this domain, none are extracted

    def __call__(self, node):
        state = node.state

        # If already at goal, return 0
        if self.goals <= state:
            return 0

        # Helper sets of current predicates in state for quick lookup
        state_bclhkf4r = {f for f in state if f.startswith("(bclhkf4r ")}
        state_chxwa5j4 = {f for f in state if f.startswith("(chxwa5j4 ")}
        state_gaijkwef = {f for f in state if f.startswith("(gaijkwef ")}

        # Count missing goal predicates
        missing_bclhkf4r = len(self.goal_bclhkf4r - state_bclhkf4r)
        missing_chxwa5j4 = len(self.goal_chxwa5j4 - state_chxwa5j4)
        missing_gaijkwef = len(self.goal_gaijkwef - state_gaijkwef)

        # Count extra predicates present that should not be in goal (possible cleanup cost)
        extra_bclhkf4r = len(state_bclhkf4r - self.goal_bclhkf4r)
        extra_chxwa5j4 = len(state_chxwa5j4 - self.goal_chxwa5j4)
        extra_gaijkwef = len(state_gaijkwef - self.goal_gaijkwef)

        # Check token presence: zzar8qlk predicate is a toggle affecting action applicability
        token_present = "(zzar8qlk)" in state
        # Some actions require token true, some require false; no goal fact for zzar8qlk,
        # so we estimate cost for toggling token if needed:
        # If many gaijkwef facts missing, token toggling may be required multiple times.
        # We approximate this by adding 1 if token currently in "wrong" state given presence of needed changes.
        token_toggle_cost = 0
        if missing_gaijkwef + missing_bclhkf4r + missing_chxwa5j4 > 0:
            # Heuristic guess: often one toggle needed during plan to apply actions.
            token_toggle_cost = 1

        # Weigh missing gaijkwef relations heavier because actions to fix them usually require
        # multiple steps (at least 2-3 actions per missing relation).
        gaijkwef_cost = missing_gaijkwef * 3

        # Missing and extra bclhkf4r and chxwa5j4 costs estimated as 1 action each
        bclhkf4r_cost = missing_bclhkf4r + extra_bclhkf4r
        chxwa5j4_cost = missing_chxwa5j4 + extra_chxwa5j4

        # Combine costs
        heuristic_value = bclhkf4r_cost + chxwa5j4_cost + gaijkwef_cost + token_toggle_cost

        # Minimal heuristic value is 1 for non-goal states to avoid zero except for goals
        if heuristic_value == 0:
            heuristic_value = 1

        return heuristic_value
