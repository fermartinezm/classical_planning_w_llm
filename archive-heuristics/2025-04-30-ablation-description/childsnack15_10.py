from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract components from a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(served child1)" -> ["served", "child1"]
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic counts the number of children who have not yet been served.
    It is based on the observation that the primary goal is to satisfy the
    `(served ?c)` predicate for all children who were initially waiting.

    # Assumptions
    - The goal state is defined by the set of `(served ?c)` facts for all
      children who were initially in a `(waiting ?c ?p)` state.
    - Each unserved child represents at least one final "serve" action, and
      typically several preceding actions (making sandwich, putting on tray,
      moving tray). Counting unserved children provides a simple estimate
      of remaining work.

    # Heuristic Initialization
    - The heuristic is initialized with the planning task object.
    - It extracts the set of children who are required to be served by
      inspecting the goal conditions of the task.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Access the current state of the planning problem.
    2.  Initialize a counter, `unserved_count`, to zero.
    3.  Retrieve the set of children who must be served to reach the goal
        (this set was determined during the heuristic's initialization).
    4.  For each child in this set:
        a. Construct the PDDL fact string representing the child being served (e.g., `"(served child1)"`).
        b. Check if this fact string is present in the current state.
        c. If the fact is *not* present in the state, increment `unserved_count`.
    5.  The final value of `unserved_count` is the heuristic estimate for the current state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by identifying the children who need to be served.

        Args:
            task: The planning task object, containing initial state, goals, etc.
        """
        # The goal facts specify which children must be served.
        # Example goal fact: '(served child1)'
        self.children_to_serve = set()
        for goal_fact in task.goals:
            # Ensure the fact is a 'served' predicate before extracting the child name.
            parts = get_parts(goal_fact)
            if parts and parts[0] == "served":
                self.children_to_serve.add(parts[1])

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        The heuristic value is the number of children who are in the set
        `self.children_to_serve` but for whom the fact `(served <child_name>)`
        is not present in the current state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state
        unserved_count = 0

        # Iterate through the children identified in the goal
        for child in self.children_to_serve:
            # Check if the 'served' fact for this child is in the current state
            served_fact = f"(served {child})"
            if served_fact not in state:
                unserved_count += 1

        return unserved_count
