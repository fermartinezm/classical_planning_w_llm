from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The fact string, e.g. "(predicate obj1 obj2)".
    - `args`: Pattern components, wildcards (*) allowed.
    Returns True if fact matches pattern, else False.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (mtfkspuf).

    Summary
    -------
    Estimates the number of actions required to achieve the goal by focusing on differences
    in key predicate assignments that appear critical given the actions:
    - The goal predicates require certain (jft9ynda ?obj-hodp29q0 ?obj-hmhptaxe) facts to hold.
    - Actions manipulate the (jft9ynda ...), (yarbdfgc ...), and (uypgj7u3 ...) predicates.
    The heuristic counts how many goal (jft9ynda ...) facts are currently unsatisfied
    and estimates the minimal number of actions needed to fix these discrepancies,
    reasoning about necessary updates to related predicates.

    Assumptions
    -----------
    - The goal conjunct requires some (jft9ynda ?pk9o1 ?oqsrn) facts to hold.
    - Changing (jft9ynda ?pk9o1 ?old) to (jft9ynda ?pk9o1 ?new) requires an action "k12p0czo" if (z6o4oitz ?old ?new).
    - Other actions "sw2fg9ql" and "vyn92k55" depend on predicates (yarbdfgc, uypgj7u3, jf4qno7u), modeling relations among hodp29q0, r86varlt, and g83opnlj elements.
    - The heuristic treats changes in (jft9ynda ...) as primary costs and adds a small penalty for unsatisfied related predicates, capturing interdependencies.
    - Each necessary (jft9ynda ...) change counts as one estimated action.
    - Additional unsatisfied (yarbdfgc ...) goals add a small penalty proportional to the counts.
    - Static facts (z6o4oitz) are used to verify if direct changes are possible.

    Heuristic Initialization
    ------------------------
    - Extract all goal (jft9ynda ...) facts and store them.
    - Extract all static (z6o4oitz ...) facts and store as pairs for adjacency checks.
    - Extract static (jf4qno7u ...) facts for relevance to costs.
    - Extract static (z6o4oitz A B) directional edges into a dict for quick lookup.
    - Cache all goal (yarbdfgc ...) facts to penalize unsatisfied sub-goals.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the current state satisfies the full goal, heuristic = 0.
    2. Otherwise:
       a. Count how many goal (jft9ynda pk9o1 oqsrn) facts are currently false.
          Each such unsatisfied fact will require at least one 'k12p0czo' action or another action chain.
       b. For each unsatisfied (jft9ynda pk9o1 oqsrn), check current (jft9ynda pk9o1 ?old).
          - If static (z6o4oitz old oqsrn) exists, one 'k12p0czo' action suffices.
          - Otherwise, assume more costly fix, add an extra penalty.
       c. Count how many goal (yarbdfgc pk9o1 fr3fe) facts are unsatisfied.
          Penalize by a fractional cost per unsatisfied fact.
       d. Count how many (uypgj7u3 gpvob pk9o1) facts related to goal (yarbdfgc) are unsatisfied.
          Penalize with small fractional cost.
    3. Sum all these to produce the heuristic value.
    4. Return this estimate of "number of required actions".

    This heuristic focuses on primary predicate changes (jft9ynda), 
    enriched with penalties from other predicates reflecting the domain's interplay.
    """

    def __init__(self, task):
        # Store all goal facts of interest: (jft9ynda ...) and (yarbdfgc ...)
        self.goal_jft9ynda = set()
        self.goal_yarbdfgc = set()
        for g in task.goals:
            parts = get_parts(g)
            if parts[0] == "jft9ynda":
                self.goal_jft9ynda.add(g)
            elif parts[0] == "yarbdfgc":
                self.goal_yarbdfgc.add(g)

        # Store all static z6o4oitz facts as adjacency: from->set(to)
        self.z6o4oitz_adj = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "z6o4oitz":
                src, dst = parts[1], parts[2]
                if src not in self.z6o4oitz_adj:
                    self.z6o4oitz_adj[src] = set()
                self.z6o4oitz_adj[src].add(dst)

        # Store static jf4qno7u predicate pairs for later heuristic use
        self.static_jf4qno7u = set(
            fact for fact in task.static if fact.startswith("(jf4qno7u ")
        )

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is zero
        if all(goal_fact in state for goal_fact in self.goal_jft9ynda.union(self.goal_yarbdfgc)):
            return 0

        cost = 0.0

        # --- Evaluate jft9ynda facts ---
        # For each goal jft9ynda fact, check if true in current state
        for goal_fact in self.goal_jft9ynda:
            if goal_fact not in state:
                # Extract objects: (jft9ynda ?pk9o1 ?oqsrn)
                _, pk9o1, desired_oqsrn = get_parts(goal_fact)

                # Find current jft9ynda fact for pk9o1 (if any)
                current_oqsrn = None
                for fact in state:
                    if fact.startswith("(jft9ynda ") and fact.split()[1] == pk9o1:
                        current_oqsrn = fact.split()[2][:-1]  # remove trailing ')'
                        break

                # Determine if direct change possible via z6o4oitz(current_oqsrn->desired_oqsrn)
                if current_oqsrn is not None and current_oqsrn in self.z6o4oitz_adj and desired_oqsrn in self.z6o4oitz_adj[current_oqsrn]:
                    # One k12p0czo action suffices
                    cost += 1
                else:
                    # No direct static edge - assume a costlier fix requiring at least 2 actions
                    cost += 2

        # --- Evaluate yarbdfgc facts ---
        # Penalize unsatisfied goal yarbdfgc facts with fractional cost
        unsat_yarbdfgc = sum(1 for g in self.goal_yarbdfgc if g not in state)
        cost += 0.3 * unsat_yarbdfgc

        # --- Evaluate related uypgj7u3 facts for domain interaction ---
        # Some actions use uypgj7u3; penalize if any relevant ones false in state that relate to goal yarbdfgc
        # Extract all relevant (uypgj7u3 gpvob pk9o1) where pk9o1 appears in goal yarbdfgc facts
        pk9o1s_in_goal_yarbdfgc = {get_parts(fact)[1] for fact in self.goal_yarbdfgc}
        relevant_uypgj7u3_facts = [
            fact for fact in state if fact.startswith("(uypgj7u3 ")
        ]
        # To estimate needed fixes, count how many pk9o1s have no uypgj7u3 fact in state but do appear in goal_yarbdfgc.
        # But since uypgj7u3 is positive precondition for sw2fg9ql and effect of vyn92k55, missing those facts might need new actions.

        # Gather present keys (gpvob, pk9o1)
        present_pairs = set()
        for fact in state:
            if fact.startswith("(uypgj7u3 "):
                parts = get_parts(fact)
                if len(parts) == 3:
                    present_pairs.add((parts[1], parts[2]))

        # Count how many needed pairs appear missing compared to facts referenced in goal yarbdfgc
        # We approximate here: for each pk9o1 in goal yarbdfgc, if no corresponding uypgj7u3 with any gpvob exists, add penalty
        had_uyp_for_pk9o1 = set(pk for (gpv, pk) in present_pairs)
        missing_count = sum(1 for pk9o1 in pk9o1s_in_goal_yarbdfgc if pk9o1 not in had_uyp_for_pk9o1)

        cost += 0.2 * missing_count

        # The heuristic value is finite and nonzero unless goal reached
        return cost
