from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Helper: split a fact '(pred a b)' into ['pred', 'a', 'b'].
    Assumes the incoming fact is a string surrounded by brackets.
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  The heuristic
    estimates the remaining number of actions by analysing which **goal facts**
    are still unsatisfied in the current state and exploiting the structure of
    the two available operators (`put_top`, `put_bottom`).

    Assumptions
    -----------
    1. Each operator changes **exactly one** structural position on a stick:
       it either places the held ring on the *top* of a stick or on the
       *bottom* of a stick.
    2. While doing so, the operator simultaneously creates or updates **one**
       `on` relationship that links the new ring with its neighbour.
    3. Hence, *one action can satisfy at most*:
         • one `top` **or** one `bot` goal, **and**
         • one `on` goal
       but never more than that.
    4. `holding` never occurs in goal specifications, so it is irrelevant for
       goal achievement.

    Heuristic Initialization
    ------------------------
    The constructor stores the set of goal facts supplied by the task.  There
    are no static facts in this domain, therefore nothing else has to be
    pre-processed.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1. Compute the set `unsat` of all goal facts that are **not** true in the
       current state.
    2. Classify every unsatisfied fact by its predicate:
           • `top`  → increases the counter `top_bot_left`
           • `bot`  → increases the counter `top_bot_left`
           • `on`   → increases the counter `on_left`
       (All other goal predicates are ignored because they do not exist in this
       domain.)
    3. Derive a lower bound on the number of actions that are still necessary:
       • Every missing `on` fact *must* be created explicitly → at least
         `on_left` actions.
       • Every missing `top`/`bot` fact also needs a dedicated action, but some
         of them can be satisfied “for free” while we are already executing the
         actions needed for the `on` facts (see Assumption 3).  Therefore we
         need **at least**
              max(on_left, top_bot_left)
         actions in total.
    4. Return that value.  The heuristic is zero **iff** the state already
       satisfies all goals and strictly positive otherwise, thus fulfilling the
       requirements stated in the checklist.
    """

    def __init__(self, task):
        # Store all goal literals for quick set arithmetic.
        self.goals = task.goals

    def __call__(self, node):
        state = node.state

        # Fast exit for a goal state.
        if self.goals <= state:
            return 0

        # --- classify unsatisfied goal facts --------------------------------
        on_left = 0          # number of missing 'on' relations
        top_bot_left = 0     # number of missing 'top' or 'bot' placements

        for fact in self.goals - state:
            pred = get_parts(fact)[0]
            if pred == "on":
                on_left += 1
            elif pred in ("top", "bot"):
                top_bot_left += 1
            # There are no other goal predicates in this domain.

        # Lower bound derived in the docstring.
        return max(on_left, top_bot_left)