from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def _get_parts(fact):
    """
    Split a PDDL fact into its individual components.

    Example
    -------
    "(on r2 r4)"  ->  ["on", "r2", "r4"]
    """
    return fact[1:-1].split()


def _matches(fact, *pattern):
    """
    Lightweight pattern–matching helper (wild-cards allowed).

    >>> _matches("(on r2 r4)", "on", "*", "r4")
    True
    """
    parts = _get_parts(fact)
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))


class RodRingsHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-specific, *informed* but **inadmissible** heuristic for the
    `rod-rings` domain.  It counts how many goal relations of the types
    `top`, `bot` and `on` are *still missing* in the current state and
    returns a weighted sum of those violations.

    Assumptions
    -----------
    1. Every stick always contains the same *set* of rings; only the order
       of these rings can change.  The two operators (`put_top`, `put_bottom`)
       never add or remove rings from a stick, they merely permute them.
    2. Exactly one ring is *always* held – that fact never appears in any
       goal condition, hence it can be ignored for goal distance estimation.
    3. Establishing an `on` relation normally needs **at least** one action,
       whereas moving a ring to the `top` or `bot` position of its stick can
       occasionally be achieved “for free” as a side effect, but more often
       also takes a dedicated action.  Empirically, using a *double weight*
       for unsatisfied `on` relations results in a very good node-reduction
       while staying cheap to compute.

    Heuristic Initialization
    ------------------------
    When the heuristic object is created, all goal atoms are split into three
    easy-to-check categories:

        self.goal_on   ⊆  {(on  x  y)}
        self.goal_top  ⊆  {(top x  s)}
        self.goal_bot  ⊆  {(bot x  s)}

    Static facts are empty in this domain, therefore they are ignored.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a search node, let `S` be its state.

    1. Count how many goal atoms of each category are *missing* in `S`:

           unsat_on   = | { g ∈ goal_on  |  g ∉ S } |
           unsat_top  = | { g ∈ goal_top |  g ∉ S } |
           unsat_bot  = | { g ∈ goal_bot |  g ∉ S } |

    2. Return

           h(S) = 2 · unsat_on  + 1 · unsat_top  + 1 · unsat_bot

       Rationale:
       - One operator can establish **at most one** new `on` relation, hence
         every unsatisfied `on` certainly needs ≥1 action. A factor of *two*
         provides a helpful push in many benchmarks without blowing up the
         estimate.
       - Setting a new `top` or `bot` can usually be piggy-backed onto the
         action that already fixes an `on` relation; therefore they get a
         weight of 1.

    3. If all goal atoms are satisfied, every term in the sum is zero and the
       heuristic correctly returns **0**.

    The computation touches each goal fact exactly once and needs only
    constant-time look-ups in the state representation, thus it is extremely
    efficient.
    """

    def __init__(self, task):
        # Categorise goal atoms for O(1) checks later on.
        self.goal_on = set()
        self.goal_top = set()
        self.goal_bot = set()

        for fact in task.goals:
            predicate, *rest = _get_parts(fact)
            if predicate == "on":
                self.goal_on.add(fact)
            elif predicate == "top":
                self.goal_top.add(fact)
            elif predicate == "bot":
                self.goal_bot.add(fact)

    def __call__(self, node):
        state = node.state

        unsat_on = sum(1 for fact in self.goal_on if fact not in state)
        unsat_top = sum(1 for fact in self.goal_top if fact not in state)
        unsat_bot = sum(1 for fact in self.goal_bot if fact not in state)

        # Weighted sum (see docstring for explanation).
        return 2 * unsat_on + unsat_top + unsat_bot
