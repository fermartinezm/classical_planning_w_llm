import fnmatch
# from heuristics.heuristic_base import Heuristic # Uncomment this line in your environment

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if fact.startswith("(") and fact.endswith(")"):
        content = fact[1:-1]
        if not content:
             return []
        return content.split()
    return [] # Should not happen with valid PDDL facts representation

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class is available
# If running this code standalone for testing, you might need a dummy Heuristic class:
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting the number
    of unsatisfied goal predicates that define the desired stack structure
    (i.e., `(on ?x ?y)` and `(on-table ?x)` goals).

    # Assumptions
    - The goal specifies a set of desired `(on ?x ?y)` and `(on-table ?x)`
      relationships that form one or more stacks.
    - Achieving these stack relationships is the primary difficulty.
    - `(clear ?x)` goals for the top blocks of goal stacks are assumed
      to be satisfied once the stack structure is correct, assuming no
      extraneous blocks are introduced.

    # Heuristic Initialization
    - Extract the set of goal predicates from the task. Store only the
      `(on ?x ?y)` and `(on-table ?x)` predicates as these define the
      target stack configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Iterate through each goal predicate stored during initialization.
    3. For each goal predicate (which is either `(on ?x ?y)` or `(on-table ?x)`):
       - Check if this exact predicate exists in the current state's set of facts.
       - If the predicate is *not* present in the current state, increment the heuristic value by 1.
    4. The final heuristic value is the total count of unsatisfied goal stack predicates.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions related to
        the block stack structure.
        """
        # Filter goal predicates to include only (on ?x ?y) and (on-table ?x)
        self.goal_stack_predicates = frozenset(
            goal for goal in task.goals
            if match(goal, "on", "*", "*") or match(goal, "on-table", "*")
        )
        # Static facts are not used in this heuristic for Blocksworld.
        # self.static = task.static # Not needed

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        The heuristic is the number of unsatisfied goal predicates
        defining the stack structure.
        """
        state = node.state # Current world state as a frozenset of facts

        # The heuristic is the count of goal stack predicates not present in the state.
        unsatisfied_count = sum(1 for goal_fact in self.goal_stack_predicates if goal_fact not in state)

        # This heuristic is 0 if and only if all goal_stack_predicates are satisfied.
        # For standard Blocksworld problems, satisfying these predicates usually
        # implies the state is the goal state (or very close, only missing arm-empty
        # or clear for top blocks which are often side effects of achieving the stack).
        # It's non-admissible but efficiently computable and focuses on the core
        # task of building the stacks.

        return unsatisfied_count
