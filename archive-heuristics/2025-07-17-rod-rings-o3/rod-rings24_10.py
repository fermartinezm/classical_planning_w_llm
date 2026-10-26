from heuristics.heuristic_base import Heuristic


def _get_parts(fact):
    """
    Split a PDDL fact into its components.

    Example:
        "(on r2 r1)"   -> ["on", "r2", "r1"]
        "(top r3 s2)"  -> ["top", "r3", "s2"]
    """
    return fact[1:-1].split()


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain–dependent heuristic for the *rod-rings* domain.  
    The heuristic counts how many goal facts of each type
    (`top`, `on`, `bot`, `holding`) are still unsatisfied in the
    current state and converts this to an estimated action cost.

    Assumptions
    -----------
    1.  Every action (`put_top` or `put_bottom`) can make *at most*
        one `on`-fact and one `top` **or** `bot`-fact true.
    2.  Destroying already satisfied goal facts is possible, therefore
        the number of unsatisfied goal facts is *not* admissible, but
        it is a quick, informative proxy for the number of actions that
        are still necessary.
    3.  Goal states never require a specific ring to be held, though the
        heuristic still copes with such goals for completeness.

    Heuristic Initialization
    ------------------------
    The constructor pre-classifies all goal facts so that the expensive
    string handling is done only once:
        • `self.goal_on`   – goal facts of the form `(on r1 r2)`
        • `self.goal_top`  – goal facts of the form `(top r s)`
        • `self.goal_bot`  – goal facts of the form `(bot r s)`
        • `self.goal_hold` – goal facts of the form `(holding r)`

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    1.  Count how many `on` goal facts are still missing.
    2.  Count how many `top` goal facts are still missing.
    3.  Count how many `bot` goal facts are still missing.
    4.  Count how many `holding` goal facts are still missing.
    5.  Because a single action can simultaneously achieve
        – at most one `on` fact and  
        – at most one *end-of-stack* fact (`top` or `bot`),
        the *minimum* number of actions that can close the gap
        in those two categories is  
        `max(missing_on, missing_top + missing_bot)`.
    6.  Add the number of still-unsatisfied `holding` facts, each of
        which can only be achieved by one action at best.
    7.  Return the resulting integer; it is zero **iff** all goal
        conditions are already satisfied and finite for any solvable
        state.
    """

    def __init__(self, task):
        self.goal_on = set()
        self.goal_top = set()
        self.goal_bot = set()
        self.goal_hold = set()

        for fact in task.goals:
            pred = _get_parts(fact)[0]
            if pred == "on":
                self.goal_on.add(fact)
            elif pred == "top":
                self.goal_top.add(fact)
            elif pred == "bot":
                self.goal_bot.add(fact)
            elif pred == "holding":
                self.goal_hold.add(fact)

    def __call__(self, node):
        """
        Calculate the heuristic value for the given search node.
        The value is 0 in a goal state and a positive integer otherwise.
        """
        state = node.state

        missing_on = len([f for f in self.goal_on if f not in state])
        missing_top = len([f for f in self.goal_top if f not in state])
        missing_bot = len([f for f in self.goal_bot if f not in state])
        missing_hold = len([f for f in self.goal_hold if f not in state])

        # actions needed for ordering and end-of-stack conditions
        ordering_cost = max(missing_on, missing_top + missing_bot)

        # every missing holding fact costs at least one extra action
        total_cost = ordering_cost + missing_hold

        return total_cost
