from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Remove surrounding parentheses and split fact into parts."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check whether a fact matches a pattern with possible wildcards '*'.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent heuristic for the secret_domain (z1spq3ra).

    The heuristic estimates how many actions are needed to satisfy the goal
    facts from the current state. It treats each conjunctive goal atom
    (yfm3frbn, a4ha3ujl, s0899d5u, ug241uf7) independently and combines
    per-atom estimates into a single score by summation. The estimates
    are informed by the operators in the domain:
      - cb5rcmgi: requires y & a & ug, produces jj and consumes y, a, ug
      - dezigdo8: requires jj, produces y & a & ug
      - ig1fewq3: requires y(q) & jj(sei), produces ug, y(sei), s(sei,q)
      - i57yke4t: requires s(sei,q) & y(sei) & ug, produces jj(sei), y(q), removes s & ug & y(sei)

    Assumptions
    -----------
    - Goals are conjunctive facts (set of atoms). The heuristic is 0 iff
      all goal atoms are satisfied in the current state.
    - Facts in the state are strings of the form "(predicate arg1 arg2)".
    - We ignore complicated interdependencies and cycles between operators;
      the heuristic uses short, pattern-based costs reflecting typical
      small sequences required to achieve each atom.
    - The heuristic is not necessarily admissible (it may over- or
      underestimate), but tries to reflect the domain structure so the
      greedy best-first search expands fewer nodes.

    Heuristic Initialization
    ------------------------
    - Store the set of goal atoms (self.goals) from the task.
    - Static facts are stored but not heavily used here (domain has no static facts),
      kept for completeness.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, we parse the state's facts into convenient sets:
      - has_y: objects with predicate yfm3frbn(...)
      - has_a: objects with predicate a4ha3ujl(...)
      - has_jj: objects with predicate jjtnwk3y(...)
      - s_rel: set of ordered pairs (x, y) for s0899d5u x y
      - ug_present: boolean for global predicate ug241uf7

    For each goal atom, we estimate a small integer cost (0,1,2 or 3)
    according to the following rules:

      1) yfm3frbn(X)
         - If y(X) present: cost=0.
         - Else if jj(X) present: dezigdo8 can produce y(X) in 1 action -> cost=1.
         - Else if exists a structure that i57 could use to produce y(X) in 1 action:
           i.e., exists sei with s(sei,X) and y(sei) and ug present -> cost=1.
         - Else assign cost=2 (assume a short sequence of actions will create it).

      2) a4ha3ujl(X)
         - If a(X) present: cost=0.
         - Else if jj(X) present: dezigdo8 produces a(X) in 1 action -> cost=1.
         - Else assign cost=2 (we assume one additional action sequence to produce jj and then dezigdo8).

      3) s0899d5u(X,Y)
         - If s(X,Y) present: cost=0.
         - Else if (X in has_jj) and (Y in has_y): ig1fewq3 can add s(X,Y) in 1 action -> cost=1.
         - Else if Y in has_y: we assume we can create jj(X) in ~1 action and then ig1 -> cost=2.
         - Else assign cost=3 (need to make Y have y first, and create jj(X), then ig1).

      4) ug241uf7 (global)
         - If ug present: cost=0.
         - Else if there exists an object with jj (dezigdo8 or ig1 can produce ug in 1) -> cost=1.
         - Else assign cost=2.

    The heuristic sums the per-atom costs. This yields a finite,
    efficiently computable estimate for all (reachable) states.

    Notes on design choices
    -----------------------
    - The cost values are deliberately small integers (0..3) to keep the
      heuristic fast to compute and informative. They reflect likely
      short operator sequences required in this domain (1-3 actions).
    - The heuristic is goal-directed (returns 0 only when all goals are
      satisfied) and handles unary (y/a/jj), binary (s) and zero-ary (ug)
      predicates present in the domain.
    """

    def __init__(self, task):
        """Initialize heuristic by storing goals and static facts."""
        self.goals = task.goals  # set of goal fact strings
        self.static = task.static  # might be empty for this domain

    def __call__(self, node):
        """Compute the heuristic value for the node's state."""
        state = node.state  # frozenset of fact strings

        # Parse state into easy-to-query structures.
        has_y = set()  # objects having yfm3frbn
        has_a = set()  # objects having a4ha3ujl
        has_jj = set()  # objects having jjtnwk3y
        s_rel = set()  # pairs (x,y) for s0899d5u x y
        ug_present = "(ug241uf7)" in state

        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "yfm3frbn" and len(parts) >= 2:
                has_y.add(parts[1])
            elif pred == "a4ha3ujl" and len(parts) >= 2:
                has_a.add(parts[1])
            elif pred == "jjtnwk3y" and len(parts) >= 2:
                has_jj.add(parts[1])
            elif pred == "s0899d5u" and len(parts) >= 3:
                s_rel.add((parts[1], parts[2]))
            # ug handled above

        # If all goals already satisfied, return 0 (requirement #2).
        if self.goals <= state:
            return 0

        total_cost = 0

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts:
                continue
            pred = parts[0]

            if pred == "yfm3frbn":
                x = parts[1]
                if x in has_y:
                    cost = 0
                elif x in has_jj:
                    # dezigdo8 produces y from jj
                    cost = 1
                else:
                    # check for i57 producing y(x) as second arg:
                    # i57 adds y(q9qne) when pre s(sei,q9qne) and y(sei) and ug
                    found_fast = False
                    if ug_present:
                        for sei, q in s_rel:
                            if q == x and sei in has_y:
                                found_fast = True
                                break
                    cost = 1 if found_fast else 2

            elif pred == "a4ha3ujl":
                x = parts[1]
                if x in has_a:
                    cost = 0
                elif x in has_jj:
                    # dezigdo8 produces a from jj
                    cost = 1
                else:
                    # assume requires creating jj first then dezigdo8
                    cost = 2

            elif pred == "s0899d5u":
                x, y = parts[1], parts[2]
                if (x, y) in s_rel:
                    cost = 0
                elif (x in has_jj) and (y in has_y):
                    # ig1 can create s(x,y) immediately
                    cost = 1
                elif y in has_y:
                    # need to create jj(x) then ig1
                    cost = 2
                else:
                    # need to make y(y) and jj(x) then ig1
                    cost = 3

            elif pred == "ug241uf7":
                if ug_present:
                    cost = 0
                elif has_jj:
                    # dezigdo8 or ig1 produce ug in 1 action when jj present
                    cost = 1
                else:
                    cost = 2

            else:
                # Unknown predicate: fallback to 1 if not present else 0
                if goal in state:
                    cost = 0
                else:
                    cost = 1

            total_cost += cost

        # Ensure heuristic is finite and integer.
        return int(total_cost)