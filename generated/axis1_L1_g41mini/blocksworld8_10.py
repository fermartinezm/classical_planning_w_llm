from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(yfm3frbn b1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (z1spq3ra).

    Summary:
    This heuristic estimates the number of actions required to reach the goal state
    by considering the current configuration of objects with predicates yfm3frbn, a4ha3ujl,
    jjtnwk3y, ug241uf7, and s0899d5u. The domain models a form of state-switching and 
    linking between objects, with one global flag (ug241uf7) controlling applicability.
    The heuristic measures how "far" the current state is from the goal based on:
    1) Matching the yfm3frbn and a4ha3ujl predicates on objects (each needed in the goal).
    2) Matching the presence of jjtnwk3y on the correct objects.
    3) Matching s0899d5u links between objects required by the goal.
    4) Accounting for the global flag ug241uf7 appropriately.

    Assumptions:
    - The global predicate ug241uf7 acts as a toggle, and only one object can have jjtnwk3y at a time.
    - Objects swap their yfm3frbn, a4ha3ujl, and jjtnwk3y predicates through actions.
    - The s0899d5u relation is a directional link from one object to another and must match goal links.
    - The goal requires a set of such predicates and links to be simultaneously true.
    - Each unit of mismatch (missing predicate or wrong link) is roughly a cost of one action.
    - The heuristic does not need to be admissible, it aims for an informative and efficient estimation.

    Heuristic Initialization:
    - Extract the exact set of goal facts.
    - Parse goal facts to identify required objects with:
      - yfm3frbn predicate
      - a4ha3ujl predicate
      - jjtnwk3y predicate (if any, usually from goal).
      - and the s0899d5u pairs of objects indicating required links.
    - No static facts needed as the domain uses few or none.
    
    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state already satisfies all goal facts, heuristic = 0.
    2. Parse current state facts into corresponding sets:
       - Objects with yfm3frbn
       - Objects with a4ha3ujl
       - Objects with jjtnwk3y
       - Whether ug241uf7 is true or not
       - The s0899d5u links as pairs (source, target)
    3. Count mismatches versus goal for each predicate type:
       - Number of objects missing from yfm3frbn in current state compared to goal.
       - Number of objects missing from a4ha3ujl compared to goal.
       - Number of objects missing from jjtnwk3y compared to goal.
       - Count of missing or wrongly set s0899d5u links compared to goal.
    4. Account for the global toggle ug241uf7:
       - If the goal requires it but it's missing, count +1.
       - If the goal does not require it but it is present, count +1.
    5. Combine these counts with weighting towards minimum actions:
       - Usually each predicate mismatch corresponds roughly to one action (e.g. cb5rcmgi or dezigdo8).
       - Each missing or wrong link may require ig1fewq3 or i57yke4t actions.
       - The heuristic sums the mismatches as an estimate of minimal action steps needed.
    6. Return the total estimated cost.

    This heuristic remains efficient by working only with sets and counts, avoiding path or search graph
    expansions, and is zero only at goal states.
    """

    def __init__(self, task):
        # Store the goal facts.
        self.goal_facts = task.goals

        # Parse goal facts into specific sets for quick membership checks.
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_s0899d5u = set()
        self.goal_ug241uf7 = False

        for fact in self.goal_facts:
            if match(fact, "yfm3frbn", "*"):
                obj = get_parts(fact)[1]
                self.goal_yfm3frbn.add(obj)
            elif match(fact, "a4ha3ujl", "*"):
                obj = get_parts(fact)[1]
                self.goal_a4ha3ujl.add(obj)
            elif match(fact, "jjtnwk3y", "*"):
                obj = get_parts(fact)[1]
                self.goal_jjtnwk3y.add(obj)
            elif match(fact, "s0899d5u", "*", "*"):
                parts = get_parts(fact)
                self.goal_s0899d5u.add((parts[1], parts[2]))
            elif fact == "(ug241uf7)":
                self.goal_ug241uf7 = True

    def __call__(self, node):
        state = node.state

        # If goal is reached, return 0.
        if self.goal_facts <= state:
            return 0

        # Extract current state facts sets.
        curr_yfm3frbn = set()
        curr_a4ha3ujl = set()
        curr_jjtnwk3y = set()
        curr_s0899d5u = set()
        curr_ug241uf7 = False

        for fact in state:
            if match(fact, "yfm3frbn", "*"):
                curr_yfm3frbn.add(get_parts(fact)[1])
            elif match(fact, "a4ha3ujl", "*"):
                curr_a4ha3ujl.add(get_parts(fact)[1])
            elif match(fact, "jjtnwk3y", "*"):
                curr_jjtnwk3y.add(get_parts(fact)[1])
            elif match(fact, "s0899d5u", "*", "*"):
                parts = get_parts(fact)
                curr_s0899d5u.add((parts[1], parts[2]))
            elif fact == "(ug241uf7)":
                curr_ug241uf7 = True

        # Compute mismatches:

        # For yfm3frbn: number of objects in goal missing in state + objects in state but not in goal
        yfm3frbn_missing = len(self.goal_yfm3frbn - curr_yfm3frbn)
        yfm3frbn_extra = len(curr_yfm3frbn - self.goal_yfm3frbn)

        # For a4ha3ujl: similarly
        a4ha3ujl_missing = len(self.goal_a4ha3ujl - curr_a4ha3ujl)
        a4ha3ujl_extra = len(curr_a4ha3ujl - self.goal_a4ha3ujl)

        # For jjtnwk3y: similarly
        jjtnwk3y_missing = len(self.goal_jjtnwk3y - curr_jjtnwk3y)
        jjtnwk3y_extra = len(curr_jjtnwk3y - self.goal_jjtnwk3y)

        # For s0899d5u links:
        s0899d5u_missing = len(self.goal_s0899d5u - curr_s0899d5u)
        s0899d5u_extra = len(curr_s0899d5u - self.goal_s0899d5u)

        # For ug241uf7 flag:
        ug241uf7_mismatch = 0 if curr_ug241uf7 == self.goal_ug241uf7 else 1

        # Sum weighted costs (each mismatch roughly equals one action).
        # We weight predicates equally, as domain is symmetric and each flip costs one action.
        # Link mismatches might need 1-2 actions, so we weigh their cost by 1.5.
        heuristic_value = (
            yfm3frbn_missing + yfm3frbn_extra +
            a4ha3ujl_missing + a4ha3ujl_extra +
            jjtnwk3y_missing + jjtnwk3y_extra +
            int(1.5 * (s0899d5u_missing + s0899d5u_extra)) +
            ug241uf7_mismatch
        )

        # Ensure heuristic is at least 1 if not goal to avoid 0 cost for unknown non-goal states.
        if heuristic_value == 0:
            heuristic_value = 1

        return heuristic_value
