from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a given fact matches a pattern with possible wildcards.

    :param fact: A PDDL fact string, e.g. "(yfm3frbn b1)".
    :param args: Pattern parts, e.g. "yfm3frbn", "*".
    :return: True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain.

    Summary:
    This heuristic estimates the minimal number of actions needed to reach the goal state
    by analyzing the predicates (yfm3frbn, a4ha3ujl, jjtnwk3y, s0899d5u and ug241uf7) that
    represent an abstract token state of objects and a global flag (ug241uf7).

    Assumptions:
    - The central global fact ug241uf7 acts like a "mode" flag that must be toggled at specific steps.
    - Objects can be in one of three mutually exclusive states per object:
       * (yfm3frbn ?obj) and (a4ha3ujl ?obj)
       * (jjtnwk3y ?obj)
       * neither (used transiently in chaining)
    - The predicate (s0899d5u ?obj1 ?obj2) encodes a linked relation between two objects.
    - The goal requires specific yfm3frbn, a4ha3ujl, and s0899d5u facts on particular objects.
    - Actions correspond to flipping states of individual objects and changing links (s0899d5u),
      also toggling the global flag ug241uf7.
    - Since ug241uf7 is exclusive and must be toggled for certain actions, switching its state is crucial.
    - Each action changes the state of at most one or two objects and toggles or requires ug241uf7.

    Heuristic Initialization:
    - Extract from the goal all required predicates of types yfm3frbn, a4ha3ujl, jjtnwk3y, s0899d5u.
    - Store goal configurations of objects, including goal yfm3frbn/ a4ha3ujl per object,
      and desired s0899d5u links.
    - No static facts to extract for this domain.

    Step-By-Step Thinking for Computing Heuristic:
    1. If the current state is a goal state, heuristic = 0.

    2. The main differences between current state and goal are:
       - Which objects have yfm3frbn and a4ha3ujl predicates (objects in "yfm3frbn+a4ha3ujl" state).
       - Which objects have jjtnwk3y predicate.
       - The global ug241uf7 flag presence.
       - The current s0899d5u links between objects.

    3. For each object:
       - Count if its "mode" predicates (yfm3frbn + a4ha3ujl) do not match goal.
         Changing this requires 1 or 2 actions (because of cb5rcmgi and dezigdo8 toggles).
       - If the goal requires jjtnwk3y for the object but it is missing, or vice versa,
         fix by toggling states with cb5rcmgi/dezigdo8 or ig1fewq3/i57yke4t sequences.

    4. For all required s0899d5u links from the goal:
       - If missing in current state, count at least one action to establish them (ig1fewq3).
       - If some s0899d5u links exist that contradict goal, might need an action to remove and re-establish.

    5. The global predicate ug241uf7 acts as a gating condition to certain actions and is toggled each action.
       - If the current state's ug241uf7 differs from what is needed for next actions, add 1 action to toggle.

    6. The heuristic sums:
       - Number of mismatched yfm3frbn+a4ha3ujl states per object (count as 1 per object).
       - Number of mismatched jjtnwk3y states per object (count as 1 per object).
       - Number of missing s0899d5u links (count 1 per missing link).
       - +1 if ug241uf7 in state differs from what is presumably needed to fix any of the above.

    7. Finally, return the sum as an estimate.

    This heuristic is not admissible but effectively estimates progress toward goal
    by counting predicate mismatches and required link fixes, while cheaply computable.
    """

    def __init__(self, task):
        """Extract goal predicates and separate them by type for heuristic computations."""
        self.goal_yfm3frbn = set()
        self.goal_a4ha3ujl = set()
        self.goal_jjtnwk3y = set()
        self.goal_s0899d5u = set()
        self.goal_ug241uf7 = None  # True if goal requires ug241uf7

        # Parse goal facts and record by predicate type.
        for fact in task.goals:
            if fact.startswith("(yfm3frbn"):
                obj = get_parts(fact)[1]
                self.goal_yfm3frbn.add(obj)
            elif fact.startswith("(a4ha3ujl"):
                obj = get_parts(fact)[1]
                self.goal_a4ha3ujl.add(obj)
            elif fact.startswith("(jjtnwk3y"):
                obj = get_parts(fact)[1]
                self.goal_jjtnwk3y.add(obj)
            elif fact.startswith("(s0899d5u"):
                parts = get_parts(fact)
                self.goal_s0899d5u.add((parts[1], parts[2]))
            elif fact == "(ug241uf7)":
                self.goal_ug241uf7 = True

        # If ug241uf7 not in goal explicitly, assume False goal for this flag.
        if self.goal_ug241uf7 is None:
            self.goal_ug241uf7 = False

    def __call__(self, node):
        """Compute heuristic estimate of actions to reach goal from current state."""

        state = node.state

        if self.goal_reached(state):
            return 0

        # Extract predicates from current state:
        current_yfm3frbn = set()
        current_a4ha3ujl = set()
        current_jjtnwk3y = set()
        current_s0899d5u = set()
        current_ug241uf7 = False

        for fact in state:
            if fact.startswith("(yfm3frbn"):
                current_yfm3frbn.add(get_parts(fact)[1])
            elif fact.startswith("(a4ha3ujl"):
                current_a4ha3ujl.add(get_parts(fact)[1])
            elif fact.startswith("(jjtnwk3y"):
                current_jjtnwk3y.add(get_parts(fact)[1])
            elif fact.startswith("(s0899d5u"):
                parts = get_parts(fact)
                current_s0899d5u.add((parts[1], parts[2]))
            elif fact == "(ug241uf7)":
                current_ug241uf7 = True

        # Count mismatches in yfm3frbn + a4ha3ujl per object
        # For each object that appears in either current or goal sets, check presence.
        objects_with_mode = self.goal_yfm3frbn | self.goal_a4ha3ujl | current_yfm3frbn | current_a4ha3ujl
        count_mode_mismatch = 0
        for obj in objects_with_mode:
            goal_state = (obj in self.goal_yfm3frbn) and (obj in self.goal_a4ha3ujl)
            current_state = (obj in current_yfm3frbn) and (obj in current_a4ha3ujl)
            if goal_state != current_state:
                count_mode_mismatch += 1

        # Count mismatch in jjtnwk3y predicate per object.
        objects_jjtnwk3y = self.goal_jjtnwk3y | current_jjtnwk3y
        count_jjtnwk3y_mismatch = sum(
            1 for obj in objects_jjtnwk3y if (obj in self.goal_jjtnwk3y) != (obj in current_jjtnwk3y)
        )

        # Count missing s0899d5u links from current to goal (links that should be present but are not)
        missing_links = self.goal_s0899d5u - current_s0899d5u

        # Count extra links (present but not in goal) might also need fixing
        extra_links = current_s0899d5u - self.goal_s0899d5u

        count_links = len(missing_links) + len(extra_links)

        # If any mismatches exist that likely involve ug241uf7 toggling, add 1 action if needed.
        # This is a crude estimate: if ug241uf7 differs from goal ug241uf7 and 
        # there are changes needed, count 1 to toggle flag.
        toggle_cost = 0
        if count_mode_mismatch + count_jjtnwk3y_mismatch + count_links > 0:
            if current_ug241uf7 != self.goal_ug241uf7:
                toggle_cost = 1

        heuristic_value = count_mode_mismatch + count_jjtnwk3y_mismatch + count_links + toggle_cost

        # Ensure heuristic is positive when goal not yet reached.
        return max(1, heuristic_value)

    def goal_reached(self, state):
        """Check if all goal facts are satisfied in this state."""
        # Use the goal facts as a set to test inclusion.
        # The task might not provide goals as a set, but here goal predicates are sets of objects;
        # we reconstruct a goal fact set for check.
        # Instead, we approximate by checking if all goal predicates hold in the state.

        # Check ug241uf7 first:
        if self.goal_ug241uf7:
            if "(ug241uf7)" not in state:
                return False
        else:
            if "(ug241uf7)" in state:
                return False

        # Check yfm3frbn and a4ha3ujl for all goal objects
        for obj in self.goal_yfm3frbn:
            if f"(yfm3frbn {obj})" not in state:
                return False
        for obj in self.goal_a4ha3ujl:
            if f"(a4ha3ujl {obj})" not in state:
                return False

        # Check jjtnwk3y in goal
        for obj in self.goal_jjtnwk3y:
            if f"(jjtnwk3y {obj})" not in state:
                return False

        # Check s0899d5u links in goal
        for src, tgt in self.goal_s0899d5u:
            if f"(s0899d5u {src} {tgt})" not in state:
                return False

        return True
