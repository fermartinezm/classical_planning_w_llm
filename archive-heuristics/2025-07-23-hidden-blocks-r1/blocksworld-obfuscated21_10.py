from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain domain.

    # Summary
    This heuristic estimates the number of actions required to transform the current state into the goal state.
    The estimation is based on two main components:
      1. The number of objects involved in the goal (either appearing in goal links or having goal-specific properties) multiplied by 2.
         This accounts for the typical cost of setting an object to state C and then transforming it to its required role.
      2. The number of extra links in the current state that are not present in the goal state.
         Each extra link requires at least one action (fpk054nz) to remove.

    # Assumptions
    - Each object involved in the goal (goal_objects) will require at least two actions to be properly configured.
    - Each link present in the current state that does not appear in the goal requires one action to remove.
    - The goal state is a conjunction of facts, including links and property assignments.

    # Heuristic Initialization
    - Extract the set of objects involved in the goal conditions (goal_objects). This includes:
        - Any object appearing in a goal link (x272hpoe).
        - Any object required to have property dhjk1eyy (A) or xk87au5b (B) in the goal.
    - Extract the set of goal links (goal_links) from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. If the current state satisfies the goal conditions, return 0.
    2. Collect all objects from the goal conditions that are part of goal links or have goal-specific properties (A or B).
    3. Count the number of links in the current state that are not present in the goal state (extra_links).
    4. The heuristic value is computed as: 2 * (number of goal_objects) + (number of extra_links).

    Rationale:
    - Transforming an object to its required state typically involves two actions (e.g., setting to C and then to A/B).
    - Removing an extraneous link requires one action (fpk054nz).
    - This heuristic efficiently approximates the action count by focusing on the goal structure and current mismatches.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal objects and goal links."""
        self.task = task
        self.goal_objects = set()
        self.goal_links = set()

        for fact in task.goals:
            parts = get_parts(fact)
            if not parts:
                continue
            predicate = parts[0]
            if predicate == "x272hpoe" and len(parts) == 3:
                obj1, obj2 = parts[1], parts[2]
                self.goal_objects.add(obj1)
                self.goal_objects.add(obj2)
                self.goal_links.add((obj1, obj2))
            elif predicate == "dhjk1eyy" and len(parts) == 2:
                obj = parts[1]
                self.goal_objects.add(obj)
            elif predicate == "xk87au5b" and len(parts) == 2:
                obj = parts[1]
                self.goal_objects.add(obj)

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # Check if the current state is a goal state
        if self.task.goals.issubset(state):
            return 0

        # Count extra links (present in state but not in goal)
        current_links = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == "x272hpoe" and len(parts) == 3:
                obj1, obj2 = parts[1], parts[2]
                current_links.add((obj1, obj2))

        extra_links = 0
        for link in current_links:
            if link not in self.goal_links:
                extra_links += 1

        return 2 * len(self.goal_objects) + extra_links
