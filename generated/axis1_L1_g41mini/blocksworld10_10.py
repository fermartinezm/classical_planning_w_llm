from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - fact: The fact as a string, e.g. "(predicate obj1 obj2)".
    - args: Patterns to match parts of the fact, wildcards '*' supported.
    
    Returns True if all parts match, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (z1spq3ra).

    Summary:
    This heuristic estimates the number of actions required to reach the goal state
    from the current state. The domain’s state primarily tracks three key predicates:
    yfm3frbn(?obj), a4ha3ujl(?obj), jjtnwk3y(?obj), and s0899d5u(?obj ?obj2), plus
    the singleton predicate ug241uf7. The goal typically requires a particular object
    to have predicates yfm3frbn(?), a4ha3ujl(?), and for specific s0899d5u relations to hold.

    Assumptions:
    - Each object can be in one of two "modes": 
      either satisfying yfm3frbn and a4ha3ujl (initial state),
      or satisfying jjtnwk3y (intermediate state).
      Transitions between these states involve toggling ug241uf7 on and off.
    - The s0899d5u predicate encodes relations between pairs of objects, such as a chain or graph edges.
    - The ug241uf7 predicate acts as a global "lock" or "resource" required for some actions.
    - Actions switch the predicates on objects and also update s0899d5u relations and ug241uf7.
    - The goal requires certain yfm3frbn facts, a4ha3ujl facts, and a specific s0899d5u configuration.

    Heuristic Initialization:
    - Parse and store the set of goal facts.
    - Pre-extract from the goal all objects occurring in yfm3frbn, a4ha3ujl, jjtnwk3y, and s0899d5u.
    - There is no static information in the domain to extract here.

    Step-By-Step Thinking for Computing Heuristic:
    1. If current state satisfies the goal, the heuristic returns 0.
    2. Estimate how many objects need to be "converted" from their current state to match goal predicates.
       - For each object required to have yfm3frbn in goal, check if current state has it; if not, count 1.
       - Similarly, count for objects needing a4ha3ujl.
       - For jjtnwk3y predicates, presence or absence counted similarly.
    3. Estimate how many s0899d5u relations need to be added or removed.
       - Compare goal s0899d5u relations with current s0899d5u facts.
       - Each missing s0899d5u relation costs 1 action.
    4. Since the ug241uf7 predicate is a global toggle needed for many actions, 
       factor a small cost if ug241uf7 state differs from goal (or if it's currently unset when needed).
    5. The domain actions suggest object state toggles require at least 1 or 2 steps per object, 
       and establishing s0899d5u facts requires actions that manipulate pairs.
       We conservatively estimate each required state/object toggle as 1 action,
       and each s0899d5u addition as 1 action to keep the heuristic optimistic but effective.
    6. Sum all these costs for a heuristic value.
    7. The heuristic is intentionally non-admissible and aims to minimize expanded nodes by providing an informed estimate.

    This heuristic is efficient since it only inspects facts in the current state and the goal,
    and does not do expensive graph traversals or reasoning.
    """

    def __init__(self, task):
        """
        Initialize heuristic by extracting goal objects and predicates relevant for computing heuristic.
        """
        self.goals = task.goals

        # Extract sets of objects from goal predicates:
        # Objects that must have yfm3frbn(?obj)
        self.goal_yfm3frbn = set()
        # Objects that must have a4ha3ujl(?obj)
        self.goal_a4ha3ujl = set()
        # Objects that must have jjtnwk3y(?obj)
        self.goal_jjtnwk3y = set()
        # Pairs that must be related via s0899d5u(?obj1 ?obj2)
        self.goal_s0899d5u = set()
        # Whether ug241uf7 is in the goal
        self.goal_ug241uf7 = False

        for fact in self.goals:
            parts = get_parts(fact)
            pred = parts[0]
            if pred == "yfm3frbn":
                self.goal_yfm3frbn.add(parts[1])
            elif pred == "a4ha3ujl":
                self.goal_a4ha3ujl.add(parts[1])
            elif pred == "jjtnwk3y":
                self.goal_jjtnwk3y.add(parts[1])
            elif pred == "s0899d5u":
                self.goal_s0899d5u.add((parts[1], parts[2]))
            elif pred == "ug241uf7":
                self.goal_ug241uf7 = True

    def __call__(self, node):
        """
        Compute heuristic estimate for node.state.

        Returns an estimated number of actions to reach the goal.
        """
        state = node.state

        if self.goals <= state:
            # Goal satisfied -> heuristic = 0
            return 0

        # Extract current facts for quick lookup
        facts = state

        # Helper to check if a predicate holds for a given object
        def has_predicate(pred_name, obj):
            return f"({pred_name} {obj})" in facts

        # Helper to check if a relation holds between two objects
        def has_relation(pred_name, obj1, obj2):
            return f"({pred_name} {obj1} {obj2})" in facts

        # Count how many goal yfm3frbn objects are missing in state
        missing_yfm3frbn = sum(1 for obj in self.goal_yfm3frbn if not has_predicate("yfm3frbn", obj))

        # Count how many goal a4ha3ujl objects are missing in state
        missing_a4ha3ujl = sum(1 for obj in self.goal_a4ha3ujl if not has_predicate("a4ha3ujl", obj))

        # Count how many goal jjtnwk3y objects are missing in state
        missing_jjtnwk3y = sum(1 for obj in self.goal_jjtnwk3y if not has_predicate("jjtnwk3y", obj))

        # Count how many s0899d5u relations from goal are missing in state
        missing_s0899d5u = sum(
            1 for (obj1, obj2) in self.goal_s0899d5u if not has_relation("s0899d5u", obj1, obj2)
        )

        # Check ug241uf7: add 1 if current state differs from goal presence
        ug241uf7_in_state = "(ug241uf7)" in facts
        ug241uf7_missing = 1 if ug241uf7_in_state != self.goal_ug241uf7 else 0

        # Since toggle actions affect each object individually and some toggle ug241uf7 state,
        # we weigh object predicate mismatches higher (1 each), relation mismatches as 1 each,
        # and a single cost for ug241uf7 mismatch.
        # We use simple additive costs - non-admissible but domain-informative.

        estimated_cost = (
            missing_yfm3frbn
            + missing_a4ha3ujl
            + missing_jjtnwk3y
            + missing_s0899d5u
            + ug241uf7_missing
        )

        # To not underestimate, let's add a small overhead proportional to number of mismatches,
        # assuming some toggling and action sequencing is necessary.
        if estimated_cost > 0:
            estimated_cost += 1  # overhead to reflect needed toggles/complexity

        return estimated_cost
