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
    It counts the number of children who are currently not served and assumes that for each unserved child,
    a fixed number of actions are needed to make and serve a sandwich. This heuristic is admissible if we consider
    that at least one action is needed to serve each child, but in practice, it is used as a non-admissible heuristic
    to guide greedy best-first search efficiently.

    # Assumptions:
    - For each unserved child, we need to perform a sequence of actions: make a sandwich, put it on a tray, move the tray to the child's location (if necessary), and serve the sandwich.
    - Resources (bread and content) are always available at the kitchen when needed.
    - Trays are available and can be moved to the required locations.
    - The heuristic focuses primarily on serving children and simplifies tray movements and resource management.

    # Heuristic Initialization
    - The heuristic initializes by storing the goal predicates from the task definition.
    - It also extracts static information about children's gluten allergies to differentiate between
      gluten-free and regular sandwiches if needed for more sophisticated versions (currently not used in this simple heuristic).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal predicates: These are the `(served ?c - child)` predicates that must be true in the goal state.
    2. Count the number of children who are listed in the goal predicates but are NOT currently served in the given state.
    3. The heuristic value is simply the count of these unserved children. This is a simplification, assuming that serving each child requires a roughly constant number of actions.
    4. Return this count as the estimated number of actions needed to reach the goal.

    This heuristic is a simplification and does not explicitly consider the actions of making sandwiches,
    putting them on trays, or moving trays. It primarily focuses on the number of children yet to be served,
    providing a basic estimate of the remaining work. More advanced heuristics could incorporate the
    complexity of sandwich making, tray management, and resource availability for better accuracy.
    """

    def __init__(self, task):
        """
        Initialize the childsnack heuristic.

        Extracts goal predicates and static facts (currently only allergy information, though not directly used in this simple version).
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.allergic_children = set()
        self.not_allergic_children = set()

        for fact in self.static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "not_allergic_gluten", "*"):
                self.not_allergic_children.add(get_parts(fact)[1])

        self.goal_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.goal_children.add(get_parts(goal)[1])


    def __call__(self, node):
        """
        Compute the heuristic value for a given state.

        The heuristic value is the number of children who are in the goal but are not yet served in the current state.
        """
        state = node.state
        served_children_in_state = set()
        for fact in state:
            if match(fact, "served", "*"):
                served_children_in_state.add(get_parts(fact)[1])

        unserved_children_count = 0
        for child in self.goal_children:
            if child not in served_children_in_state:
                unserved_children_count += 1

        return unserved_children_count
