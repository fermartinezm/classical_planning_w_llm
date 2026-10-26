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
    This heuristic estimates the number of actions needed to serve all waiting children.
    It counts the number of children who are currently waiting and not yet served,
    and multiplies this count by a constant factor (estimated actions per child).

    # Assumptions:
    - For each unserved child, a fixed number of actions are required to serve them.
    - Resources (bread, content, trays) are always available in sufficient quantities to serve all children.
    - The heuristic focuses on serving each child and does not explicitly consider optimizing resource usage or tray movements.

    # Heuristic Initialization
    - No specific initialization is needed beyond the base heuristic class.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Identify all children from the static facts (though not strictly necessary as children are also in goal).
    3. Iterate through each child.
    4. For each child, check if the predicate `(served child)` is present in the current state.
    5. If `(served child)` is NOT present in the state, it means the child is not yet served.
       Increment the heuristic value by a constant cost estimate for serving one child.
       A reasonable estimate for serving a child is 3 actions (make sandwich, put on tray, serve sandwich).
    6. Return the total accumulated heuristic value.
    """

    def __init__(self, task):
        """Initialize the childsnack heuristic."""
        super().__init__(task)
        self.goals = task.goals
        self.static_facts = task.static
        self.children = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['allergic_gluten', 'not_allergic_gluten', 'waiting']:
                self.children.add(parts[1])
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served':
                self.children.add(parts[1])
        if not self.children:
            # if children are not found in static facts or goals, try to find them in initial state
            for fact in task.initial_state:
                parts = get_parts(fact)
                if parts and parts[0] in ['waiting']:
                    self.children.add(parts[1])


    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
        state = node.state
        heuristic_value = 0
        actions_per_child = 3  # Estimated actions to serve one child

        goal_children = set()
        for goal in self.task.goals:
            if match(goal, "served", "*"):
                goal_children.add(get_parts(goal)[1])

        unserved_children_count = 0
        for child in goal_children:
            served_predicate = f'(served {child})'
            if served_predicate not in state:
                unserved_children_count += 1

        heuristic_value = unserved_children_count * actions_per_child
        return heuristic_value
