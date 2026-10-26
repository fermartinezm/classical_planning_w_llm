from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robby rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of children who are currently waiting and not yet served.
    This count serves as a lower bound on the number of actions needed, assuming
    at least one serve action is needed per unserved child.

    # Assumptions:
    - Each waiting child needs to be served exactly once.
    - Making and placing sandwiches on trays are necessary steps before serving.
    - The heuristic simplifies the problem by only counting unserved children,
      ignoring the complexities of sandwich making, tray management, and gluten allergies.
    - It assumes that for each unserved child, a sequence of actions (make sandwich,
      put on tray, serve sandwich) will be required.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task.
    - It also extracts static information about waiting children from the task's static facts.
    - It pre-calculates the set of initially waiting children to efficiently determine
      unserved children in each state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of children who were initially waiting from the static facts.
    2. For a given state, count the number of children from the initial waiting set
       who are NOT yet marked as 'served' in the current state.
    3. The heuristic value is simply the count of these unserved, initially waiting children.
       This is a simplification and assumes that serving each child requires at least one action.
       It does not explicitly account for the actions of making sandwiches or moving trays,
       but provides a basic estimate based on the primary goal of serving children.
    """

    def __init__(self, task):
        """
        Initialize the childsnack heuristic.
        Extracts goal predicates and static facts, and identifies initially waiting children.
        """
        self.goals = task.goals
        static_facts = task.static

        # Identify initially waiting children from static facts.
        self.initial_waiting_children = set()
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                self.initial_waiting_children.add(child)


    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.
        The heuristic value is the number of initially waiting children who are not yet served.
        """
        state = node.state
        served_children_in_state = set()
        for fact in state:
            if match(fact, "served", "*"):
                parts = get_parts(fact)
                child = parts[1]
                served_children_in_state.add(child)

        unserved_children_count = 0
        for child in self.initial_waiting_children:
            if child not in served_children_in_state:
                unserved_children_count += 1

        return unserved_children_count
