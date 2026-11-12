from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Return the token list of a fact string by stripping surrounding parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A fast, domain-dependent heuristic for the 'transport' (secret_domain) domain.
    The heuristic estimates how many actions are needed to satisfy the remaining
    goal facts by considering the domain's action patterns (drive, pick-up, drop, drive-2)
    and the typical preconditions required to establish those goal facts.

    Assumptions
    -----------
    - Facts are represented as strings like '(predicate arg1 arg2)'.
    - Goals consist of atomic facts of the predicates used in the domain:
      at, road, capacity-predecessor, capacity, in.
    - We ignore the exact complex interactions of delete effects and assume
      that achieving each missing goal fact requires a small number of representative actions.
    - The heuristic is not admissible; it is intended to be informative and cheaply computable.

    Heuristic Initialization
    ------------------------
    - The constructor stores the set of goal facts and the static facts extracted
      from the task for quick access.
    - Static facts are kept in self.static for potential future use (the domain examples set it empty).

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    For a given state, the heuristic computes an additive estimate for the
    remaining unsatisfied goal facts. Different predicates are assigned
    different base costs reflecting the minimal action(s) needed to produce them:
      - (at X Y): typically requires a drop action (adds the at fact), and drop
        needs both (road Y) and (in X) as preconditions. Thus:
          base cost = 1 (drop) + 1 if road Y is currently missing + 1 if in X is missing
      - (road X): can be established by pick-up or drop; base cost = 1
      - (capacity-predecessor X): created by pick-up which requires (in X);
        cost = 1 + 1 if (in X) is missing
      - (capacity): created by pick-up or drop; base cost = 1
      - (in X): can be created by drive or drive-2; both usually need (road X)
        and the global (capacity) flag. We approximate cost = 1 + penalty(s)
        for missing (road X) and missing (capacity).
    - If a precondition fact is present in the current state or already present
      among the goal facts, its missing-cost is not added (we assume it will
      be achieved or already is true).
    - The heuristic returns the sum of these per-goal estimated costs.
    - If all goal facts are already satisfied (goal subset of state), the
      heuristic returns 0.

    Notes on properties
    -------------------
    - The heuristic is zero exactly for states that satisfy all goal facts.
    - The heuristic returns a finite integer for any state (it only sums small
      finite integers), and thus is well-defined for solvable states.
    - It is intentionally simple and fast to compute to guide greedy best-first search.
    """

    def __init__(self, task):
        # The set of goal facts (strings)
        self.goals = set(task.goals)
        # Static facts (if any) - kept for completeness
        self.static = set(task.static)

    def __call__(self, node):
        """
        Compute the heuristic value for the node's state.

        Parameters
        ----------
        node : search node
            The search node whose state attribute contains a frozenset of fact strings.

        Returns
        -------
        int
            An estimate of the number of actions required to reach a goal state.
        """
        state = node.state

        # If the goal is already reached, return zero.
        if self.goals <= state:
            return 0

        # Remaining (unsatisfied) goal facts.
        remaining = self.goals - state

        cost = 0

        # Helper to check membership in state or goals
        def is_true_or_goal(fact_str):
            return (fact_str in state) or (fact_str in self.goals)

        for fact in remaining:
            parts = get_parts(fact)
            pred = parts[0]

            if pred == "at":
                # To achieve (at X Y) we assume:
                # - one drop action (cost 1)
                # - plus cost for missing preconditions (road Y and in X)
                _, x, y = parts
                c = 1  # drop action
                if not is_true_or_goal(f"(road {y})"):
                    c += 1
                if not is_true_or_goal(f"(in {x})"):
                    c += 1
                cost += c

            elif pred == "road":
                # road X can typically be added by pick-up or drop
                cost += 1

            elif pred == "capacity-predecessor":
                # Created by pick-up which requires (in X)
                _, x = parts
                c = 1
                if not is_true_or_goal(f"(in {x})"):
                    c += 1
                cost += c

            elif pred == "capacity":
                # capacity is a global flag added by pick-up / drop
                cost += 1

            elif pred == "in":
                # (in X) can be added by drive or drive-2. Both usually need (road X)
                # and the global capacity flag. We approximate accordingly.
                _, x = parts
                c = 1  # drive/drive-2 action itself
                if not is_true_or_goal(f"(road {x})"):
                    c += 1
                if not is_true_or_goal("(capacity)"):
                    c += 1
                cost += c

            else:
                # Unknown predicate -- conservative single-step cost
                cost += 1

        return cost