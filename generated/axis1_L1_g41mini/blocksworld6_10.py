from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extracts the components of a PDDL fact by removing parentheses and splitting the string.

    For example:
    "(s0899d5u b1 b2)" -> ["s0899d5u", "b1", "b2"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Checks if a PDDL fact matches a given pattern.

    - `fact`: The full fact string, e.g. "(s0899d5u b1 b2)".
    - `args`: The pattern components (supports "*" wildcard).
    Returns True if every part of the fact matches the corresponding pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain (z1spq3ra).

    Summary:
    The domain manipulates objects with predicates yfm3frbn(?obj), a4ha3ujl(?obj),
    jjtnwk3y(?obj), and s0899d5u(?obj1 ?obj2), and a global flag ug241uf7.
    Actions toggle these predicates and create or remove s0899d5u chains between objects.
    The goal is expressed as a conjunction of these predicates.

    This heuristic estimates the number of actions still required to reach a goal state from the current state.
    It does so by:
    - Comparing the current state against each goal fact.
    - Estimating the minimal number of operations needed to fix mismatches.
    - Taking the maximum cost among all goal facts as a heuristic estimate,
      because facts are interconnected and often require dependent chains of actions.

    Assumptions:
    - Only the goal facts define what must hold; other facts are irrelevant.
    - Each action can only affect certain predicates and changes yfm3frbn, a4ha3ujl,
      jjtnwk3y, ug241uf7 predicates, or a relation s0899d5u between pairs.
    - The global predicate ug241uf7 behaves like a toggle needed by some actions.
    - Dependences between objects connected by s0899d5u resemble a linked structure.
    - Efficient approximate estimation (not necessarily admissible) is sufficient.

    Heuristic Initialization:
    - Extract the predicates in the goal state, sorted by predicate.
    - Extract all objects referenced in goals.
    - No static facts are needed since static is empty.
    - Create quick lookup for the goal facts by predicate for faster checks.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state satisfies all goal facts, return 0.

    2. Otherwise, for each goal fact, estimate cost to achieve it if not currently true:
       - For goal (yfm3frbn ?obj): Cost is 1 if yfm3frbn is missing, plus cost to get ug241uf7 and related predicates right.
       - For goal (a4ha3ujl ?obj): Cost similarly estimated.
       - For goal (jjtnwk3y ?obj): ~ 1 action needed to switch between jjtnwk3y and yfm3frbn/a4ha3ujl.
       - For goal (s0899d5u ?obj1 ?obj2): s0899d5u links are created or destroyed by actions ig1fewq3 or i57yke4t;
         estimating cost of creating/removing such links involves checking the predicates yfm3frbn and jjtnwk3y on both objects and the global flag ug241uf7.
         Each s0899d5u fact missing or wrong may require up to 2-3 actions to be fixed.

    3. Because actions toggle ug241uf7 and predicates per object, some costs accumulate.

    4. Return the maximum estimated cost among all goal facts as the heuristic,
       since the plan must achieve all goals which might be done in parallel but only limited by the slowest chain of required actions.

    This heuristic is efficient and reduces node expansions due to its educated cost estimate,
    even though it is not guaranteed admissible.
    """

    def __init__(self, task):
        # Store goal facts by predicate for quick access
        self.goal_facts = task.goals
        self.goal_yfm3frbn = set(f for f in self.goal_facts if match(f, "yfm3frbn", "*"))
        self.goal_a4ha3ujl = set(f for f in self.goal_facts if match(f, "a4ha3ujl", "*"))
        self.goal_jjtnwk3y = set(f for f in self.goal_facts if match(f, "jjtnwk3y", "*"))
        self.goal_s0899d5u = set(f for f in self.goal_facts if match(f, "s0899d5u", "*", "*"))
        self.goal_ug241uf7 = "(ug241uf7)" in self.goal_facts

    def __call__(self, node):
        state = node.state

        # Quick check: if goal already reached, heuristic = 0
        if self.goal_facts <= state:
            return 0

        # Helper functions to check predicate membership for objects
        def has_pred(predicate_name, obj):
            return f"({predicate_name} {obj})" in state

        def has_s0899d5u(obj1, obj2):
            return f"(s0899d5u {obj1} {obj2})" in state

        # Extract all objects referenced in goals per predicate
        objs_yfm3frbn = {get_parts(fact)[1] for fact in self.goal_yfm3frbn}
        objs_a4ha3ujl = {get_parts(fact)[1] for fact in self.goal_a4ha3ujl}
        objs_jjtnwk3y = {get_parts(fact)[1] for fact in self.goal_jjtnwk3y}
        objs_s0899d5u = {(get_parts(f)[1], get_parts(f)[2]) for f in self.goal_s0899d5u}

        # Current global predicate:
        has_ug241uf7 = "(ug241uf7)" in state

        # Estimate costs for yfm3frbn facts
        def cost_for_yfm3frbn(o):
            if has_pred("yfm3frbn", o):
                return 0
            # If jjtnwk3y is present, need 1 action (dezigdo8) to switch back to yfm3frbn+a4ha3ujl+ug241uf7
            # If not, need to perform cb5rcmgi or ig1fewq3/i57yke4t action sequences.
            if has_pred("jjtnwk3y", o):
                # dezigdo8 restores yfm3frbn and a4ha3ujl and ug241uf7
                return 1
            # If neither yfm3frbn nor jjtnwk3y, multiple actions needed, roughly 2 (dezigdo8 + something else).
            # Since only those two states toggle these predicates, assume 2 as upper bound.
            return 2

        # Estimate costs for a4ha3ujl facts (similar to yfm3frbn)
        def cost_for_a4ha3ujl(o):
            if has_pred("a4ha3ujl", o):
                return 0
            # Same reasoning as for yfm3frbn
            if has_pred("jjtnwk3y", o):
                return 1
            return 2

        # Estimate costs for jjtnwk3y facts
        def cost_for_jjtnwk3y(o):
            if has_pred("jjtnwk3y", o):
                return 0
            # To get jjtnwk3y, cb5rcmgi is used from yfm3frbn + a4ha3ujl + ug241uf7
            # So needs the object to have yfm3frbn and a4ha3ujl, and ug241uf7 true at some point.
            cost = 1  # cb5rcmgi
            # If currently missing yfm3frbn or a4ha3ujl, must fix them first: add 1 action (dezigdo8)
            if not has_pred("yfm3frbn", o) or not has_pred("a4ha3ujl", o):
                cost += 1
            # If ug241uf7 missing, another action needed to establish it (dezigdo8 or others)
            if not has_ug241uf7:
                cost += 1
            return cost

        # Estimate costs for s0899d5u facts
        def cost_for_s0899d5u(o1, o2):
            if has_s0899d5u(o1, o2):
                return 0

            # To create s0899d5u(o1,o2), action ig1fewq3 requires:
            # Preconditions:
            #   yfm3frbn o2
            #   jjtnwk3y o1
            # Effects:
            #   s0899d5u(o1,o2), yfm3frbn(o1), ug241uf7
            #   Removes yfm3frbn(o2), jjtnwk3y(o1)
            # So cost depends on states of o1, o2, and ug241uf7

            cost = 0

            # o2 must have yfm3frbn
            if not has_pred("yfm3frbn", o2):
                cost += cost_for_yfm3frbn(o2)  # Make o2 yfm3frbn

            # o1 must have jjtnwk3y
            if not has_pred("jjtnwk3y", o1):
                cost += cost_for_jjtnwk3y(o1)  # Make o1 jjtnwk3y

            # ug241uf7 must be true
            if not has_ug241uf7:
                cost += 1  # Operation to get ug241uf7 (dezigdo8), approximated

            # Performing ig1fewq3 is 1 action
            cost += 1

            return cost

        # Now we gather costs for all goal facts not satisfied
        costs = []

        # yfm3frbn goals
        for o in objs_yfm3frbn:
            if not has_pred("yfm3frbn", o):
                costs.append(cost_for_yfm3frbn(o))

        # a4ha3ujl goals
        for o in objs_a4ha3ujl:
            if not has_pred("a4ha3ujl", o):
                costs.append(cost_for_a4ha3ujl(o))

        # jjtnwk3y goals
        for o in objs_jjtnwk3y:
            if not has_pred("jjtnwk3y", o):
                costs.append(cost_for_jjtnwk3y(o))

        # s0899d5u goals
        for o1, o2 in objs_s0899d5u:
            if not has_s0899d5u(o1, o2):
                costs.append(cost_for_s0899d5u(o1, o2))

        # ug241uf7 goal
        if self.goal_ug241uf7 and not has_ug241uf7:
            # ug241uf7 can be toggled by cb5rcmgi and dezigdo8 actions,
            # approximated at 1 action cost.
            costs.append(1)

        # If no unsatisfied goals found (should not happen due to first check), return 0
        if not costs:
            return 0

        # Return the maximum cost among all goal facts to reflect the longest chain of needed fixes
        return max(costs)
