from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential non-string or malformed inputs gracefully,
        # though in typical planner states, facts are strings.
        # Returning an empty list or raising an error might be options,
        # but splitting the raw string is safer if format isn't guaranteed.
        # However, based on the example state, the format is reliable.
        # Let's assume valid PDDL fact string format.
        pass # Fall through to the split logic

    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the cost to reach the goal by counting the number of children who have not yet been served.

    # Assumptions
    - The goal is to serve a specific set of children, specified by '(served ?c)' predicates in the task goals.
    - Each child needs to be served individually via a 'serve_sandwich' or 'serve_sandwich_no_gluten' action.
    - The cost of each action is 1 (standard STRIPS assumption).
    - The heuristic counts the number of 'served' goal predicates that are not yet true in the current state.

    # Heuristic Initialization
    - Identify the set of children that need to be served by examining the goal conditions of the task. These are the objects '?c' in predicates of the form '(served ?c)' within the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state of the world, which is a frozenset of PDDL fact strings.
    2. Initialize a counter for unserved children to 0.
    3. Iterate through the set of children identified during initialization that need serving (i.e., those appearing in the goal's '(served ?c)' predicates).
    4. For each child in this set, construct the corresponding 'served' fact string, e.g., '(served child1)'.
    5. Check if this 'served' fact string exists in the current state.
    6. If the fact '(served <child_name>)' is *not* found in the current state, it means this child has not yet been served. Increment the unserved children counter.
    7. The heuristic value returned is the final count of unserved children. This count represents the minimum number of 'serve_sandwich' actions (or their gluten-free equivalent) that still need to be performed to satisfy the goal, providing an admissible estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting the set of children that need to be served from the task goals.
        """
        self.children_to_serve = set()
        # The task.goals is a frozenset of fact strings representing the goal state.
        # We look for facts of the form '(served ?c)'.
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            # Check if the fact has at least two parts (predicate and object)
            # and if the predicate is 'served'.
            if len(parts) >= 2 and parts[0] == 'served':
                child_name = parts[1]
                self.children_to_serve.add(child_name)

        # Static facts (task.static) contain information like 'waiting', 'allergic_gluten',
        # 'no_gluten_bread', etc. While useful for more complex heuristics,
        # this simple heuristic only relies on identifying which children need serving
        # from the goal state itself.
        # self.static_facts = task.static # Not needed for this heuristic

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        The heuristic is the number of children that are in the goal set
        (identified during initialization) but are not yet marked as 'served'
        in the current state.
        """
        state = node.state # The current state is a frozenset of fact strings.

        unserved_count = 0
        # Iterate through the set of children we know need to be served based on the goal.
        for child in self.children_to_serve:
            # Construct the specific fact string that indicates the child is served.
            served_fact = f'(served {child})'
            # Check if this fact is present in the current state.
            if served_fact not in state:
                # If the fact is not in the state, the child is not yet served.
                unserved_count += 1

        # The heuristic value is the total count of children who are not yet served.
        # This is an admissible heuristic because each unserved child requires at least
        # one 'serve_sandwich' action (or equivalent) to reach the goal, and each
        # such action serves exactly one child.
        return unserved_count
