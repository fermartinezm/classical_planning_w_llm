from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this is available in the target environment

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or malformed fact gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

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

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting the number
    of goal conditions related to block positions (on or on-table) that are not
    satisfied in the current state. It focuses on the structural goals of where
    each block should be placed.

    # Assumptions
    - The primary goal conditions define the desired stack structure using
      `(on ?x ?y)` and `(on-table ?x)` predicates.
    - `(clear ?x)` goal conditions are typically implied by the stack structure
      and are not explicitly counted by this heuristic, as achieving the correct
      stack configuration usually resolves the necessary clear conditions.
    - `(arm-empty)` goal conditions are also not explicitly counted; achieving
      the final block configuration often results in an empty arm or requires
      minimal final actions to clear the arm.
    - Each action that directly places a block in its final desired position
      (stack or putdown) satisfies exactly one `(on ...)` or `(on-table ...)`
      goal predicate. This makes the heuristic admissible.

    # Heuristic Initialization
    - Extract all goal conditions from the task.
    - Filter the goal conditions to keep only `(on ...)` and `(on-table ...)`
      predicates, as these define the target block configuration that this
      heuristic measures progress towards. Store these as the set of relevant goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value (cost estimate) to 0.
    2. Retrieve the set of relevant goal facts (those starting with `(on ` or `(on-table `) that were stored during the heuristic's initialization.
    3. Retrieve the set of facts that are true in the current state from the provided node.
    4. Iterate through each fact in the set of relevant goal facts.
    5. If a relevant goal fact is *not* found in the current state facts, it means this goal condition is not yet satisfied. Increment the heuristic value by 1 for each such unsatisfied goal fact.
    6. The final heuristic value is the total count of unsatisfied relevant goal facts.
    7. A heuristic value of 0 indicates that all relevant goal facts are satisfied. In the blocksworld domain, if all `(on ...)` and `(on-table ...)` goals are met, the block configuration is correct, which implies the goal state is reached (assuming standard blocksworld goals).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting relevant goal conditions.
        Relevant goals are those specifying block positions: (on ?x ?y) and (on-table ?x).
        """
        super().__init__(task) # Call parent constructor to store task info

        # Store only the 'on' and 'on-table' goal predicates
        # Use a set for efficient lookup later
        self.relevant_goals = {
            goal for goal in self.goals
            if match(goal, "on", "*", "*") or match(goal, "on-table", "*")
        }
        # Convert to frozenset for immutability and hashability
        self.relevant_goals = frozenset(self.relevant_goals)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        The heuristic is the number of relevant goal facts (on or on-table)
        that are not present in the current state.
        """
        state = node.state # state is a frozenset of facts (strings)

        # Count how many relevant goal facts are NOT in the current state
        unsatisfied_relevant_goals = 0
        for goal in self.relevant_goals:
            if goal not in state:
                unsatisfied_relevant_goals += 1

        return unsatisfied_relevant_goals
