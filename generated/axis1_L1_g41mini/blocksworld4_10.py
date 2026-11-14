from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a fact matches a pattern with wildcards.

    :param fact: A PDDL fact string, e.g., "(yfm3frbn b1)".
    :param args: Pattern parts, e.g., "yfm3frbn", "*".
    :return: True if each part of fact matches corresponding arg pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal state based on the
    current facts about objects having predicates yfm3frbn, a4ha3ujl, jjtnwk3y, and relations s0899d5u.
    It exploits the domain's structure, which revolves around toggling predicates (yfm3frbn, a4ha3ujl, jjtnwk3y)
    and managing "ug241uf7" (a global on/off predicate), plus establishing links (s0899d5u) between objects.

    Assumptions:
    - The goals require certain objects to have specific predicates (yfm3frbn, a4ha3ujl) and s0899d5u relations between objects.
    - Actions flip these predicates on objects while obeying preconditions involving the global predicate ug241uf7.
    - The manipulation essentially moves "states" among objects, with some actions requiring ug241uf7 to be true, others setting it false.
    - Each change from one status to another on an object can be achieved by a known small number of actions.
    - The s0899d5u predicates represent links between specific pairs of objects; achieving the correct links is crucial.

    Heuristic Initialization:
    - Extract objects from goal predicates involving yfm3frbn, a4ha3ujl, jjtnwk3y, and s0899d5u.
    - Collect the goal facts as sets for quick membership tests.
    - Collect the current global predicate ug241uf7 to guide heuristic adjustments in call.
    - No static facts are used (empty static in domain).

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state is a goal state, return 0.

    2. Extract from the current state:
       - Which objects have yfm3frbn, a4ha3ujl, and jjtnwk3y.
       - Whether ug241uf7 is true.
       - The set of s0899d5u pairs present.

    3. Compare the current configuration of predicates on objects to the goal configuration:
       - Count how many objects must switch their predicate sets to match the goal:
         For each object appearing in goal predicates among yfm3frbn, a4ha3ujl, jjtnwk3y:
           * If its current predicates differ from its predicates in goal, at least one toggle sequence of actions is needed.
       - Recognize that toggling tags on a single object requires at least 1 action (often 2, considering preconditions), so count toggles roughly.

    4. Compare s0899d5u links in the current state to those required by the goal:
       - Count how many s0899d5u links differ (missing or extra).
       - Each differing link must be created or removed by a specific pair of actions (ig1fewq3 and i57yke4t), costing least 1 or 2 actions.

    5. The predicate ug241uf7 acts as a gating condition enabling or preventing some actions:
       - If ug241uf7 differs from the goal state’s ug241uf7 status, count 1 additional step (to toggle it).
       - If ug241uf7 is missing but required soon for any action, include toggling cost.

    6. Combine all these counts with weights approximating minimal actions needed per difference:
       - Each predicate toggle on an object: count as 1 action (heuristically).
       - Each s0899d5u link difference: count as 2 actions (set and unset).
       - Toggling ug241uf7: count as 1 action.

    7. Return the summed estimate as the heuristic value.

    This heuristic is not admissible but aims to provide a realistic estimate that helps focus the search on states progress towards goal configurations.
    """

    def __init__(self, task):
        # Store goal facts for quick membership checks.
        self.goals = task.goals

        # Objects involved in goals by predicate.
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_s0899d5u = set()

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn" and len(parts) == 2:
                self.goal_yfm3frbn.add(parts[1])
            elif pred == "a4ha3ujl" and len(parts) == 2:
                self.goal_a4ha3ujl.add(parts[1])
            elif pred == "jjtnwk3y" and len(parts) == 2:
                self.goal_jjtnwk3y.add(parts[1])
            elif pred == "s0899d5u" and len(parts) == 3:
                self.goal_s0899d5u.add((parts[1], parts[2]))

    def __call__(self, node):
        state = node.state

        # If goal reached, heuristic is 0.
        if self.goals <= state:
            return 0

        # Current predicates sets.
        cur_yfm3frbn = set()
        cur_a4ha3ujl = set()
        cur_jjtnwk3y = set()
        cur_s0899d5u = set()
        ug241uf7_in_state = False

        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn" and len(parts) == 2:
                cur_yfm3frbn.add(parts[1])
            elif pred == "a4ha3ujl" and len(parts) == 2:
                cur_a4ha3ujl.add(parts[1])
            elif pred == "jjtnwk3y" and len(parts) == 2:
                cur_jjtnwk3y.add(parts[1])
            elif pred == "s0899d5u" and len(parts) == 3:
                cur_s0899d5u.add((parts[1], parts[2]))
            elif fact == "(ug241uf7)":
                ug241uf7_in_state = True

        # Collect all objects appearing in any predicate of interest (either current or goal).
        all_objects = (
            self.goal_yfm3frbn
            | self.goal_a4ha3ujl
            | self.goal_jjtnwk3y
            | set(cur_yfm3frbn)
            | set(cur_a4ha3ujl)
            | set(cur_jjtnwk3y)
        )

        # Heuristic components:
        # Count how many objects differ in their predicate state compared to goal.

        toggle_cost = 0
        for obj in all_objects:
            in_goal_y = obj in self.goal_yfm3frbn
            in_goal_a = obj in self.goal_a4ha3ujl
            in_goal_j = obj in self.goal_jjtnwk3y
            in_cur_y = obj in cur_yfm3frbn
            in_cur_a = obj in cur_a4ha3ujl
            in_cur_j = obj in cur_jjtnwk3y

            # For object to match the goal, all three predicates must match exactly.
            # If any predicate differs, count 1 toggle cost (heuristic approximation).
            if (in_goal_y != in_cur_y) or (in_goal_a != in_cur_a) or (in_goal_j != in_cur_j):
                toggle_cost += 1

        # Count differing links: how many s0899d5u links are missing or extra vs goal.
        s0899d5u_missing = self.goal_s0899d5u - cur_s0899d5u
        s0899d5u_extra = cur_s0899d5u - self.goal_s0899d5u
        link_diff = len(s0899d5u_missing) + len(s0899d5u_extra)

        # Each link difference likely costs at least 2 actions (one to create or remove, one to complement).
        link_cost = 2 * link_diff

        # Check if ug241uf7 differs (goal requires ug241uf7 or not).
        ug_goal = any(fact == "(ug241uf7)" for fact in self.goals)
        ug_cost = 0 if ug_goal == ug241uf7_in_state else 1

        # Sum heuristics.
        heuristic_value = toggle_cost + link_cost + ug_cost

        # Return at least 1 if heuristic is zero but not a goal state to help search progress.
        return max(heuristic_value, 1)
