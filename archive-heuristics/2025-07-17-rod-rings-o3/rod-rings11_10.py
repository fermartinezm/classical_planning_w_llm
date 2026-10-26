from math import ceil
from heuristics.heuristic_base import Heuristic


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A very light-weight, domain–dependent heuristic for the *rod–rings* domain.
    The heuristic counts how many goal facts are still false in the current
    state and exploits the structure of the two available actions
    (`put_top` / `put_bottom`) to translate that number into an estimated
    number of actions that are still necessary.

    Assumptions
    -----------
    1. Each planning instance encodes its desired end configuration only with
       `top`, `bot` and `on` predicates.  A fact that does **not** appear in
       the goal does not have to hold in the solution.
    2. Every operator changes at most two of those goal-relevant facts:
       • `put_top` can simultaneously establish the new `top` ring of a stick
         **and** its adjacency to the previous top ring (`on`).
       • `put_bottom` can simultaneously establish the new `bot` ring of a
         stick **and** the adjacency between the old bottom and the new one.
       Therefore **one action can satisfy at most two outstanding goal facts**.
    3. The heuristic does *not* have to be admissible – it should merely be
       informative enough to reduce the number of expanded nodes.

    Heuristic Initialization
    ------------------------
    The constructor stores the (immutable) set of goal facts of the task.
    There are *no* static facts in the rod–rings domain, hence nothing else
    needs to be pre-processed.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    Given a search node:

    1.   Retrieve the current state as a set of strings.
    2.   Count how many goal facts are **not** contained in that state.
         These are the facts that still have to be achieved.
    3.   Because one action can fix at most two such facts, the *minimal*
         number of further actions is bounded below by the ceiling of
         (unsatisfied / 2).
    4.   Return that value (or 0 if we are already in a goal state).  The
         result is always finite for solvable states and equals 0 only for
         goal states.
    """

    def __init__(self, task):
        # Goals are an immutable frozenset of strings, e.g. "(top r2 s1)".
        self.goals = task.goals

    def __call__(self, node):
        state = node.state

        # Fast set subtraction to identify unsatisfied goal facts.
        unsatisfied = self.goals - state
        if not unsatisfied:
            return 0  # Goal reached.

        # One action can satisfy at most two goal facts.
        return ceil(len(unsatisfied) / 2)