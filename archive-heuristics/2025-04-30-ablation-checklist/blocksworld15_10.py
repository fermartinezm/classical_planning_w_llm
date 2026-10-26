from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the number of actions needed to reach the goal state
    by considering the blocks that are not in their desired final configuration
    and the blocks that are obstructing the placement of other blocks.

    # Summary
    The heuristic value is the sum of two components:
    1. The number of goal facts (specifically `(on ?x ?y)` and `(on-table ?x)`)
       that are not true in the current state.
    2. The number of blocks that are currently on top of another block
       (`(on ?x ?y)` is true in the state) but are NOT supposed to be on that
       specific block in the goal state. These blocks are considered "in the way"
       and need to be moved.

    # Heuristic Initialization
    The heuristic stores the set of goal facts that define the desired block
    positions (`on` and `on-table` predicates).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Iterate through all goal facts that specify block positions (`on` or `on-table`).
       If a goal fact is not present in the current state, increment `h`. This counts
       the desired positions that haven't been achieved yet.
    3. Iterate through all facts in the current state.
       If a fact is an `(on ?x ?y)` predicate, check if this exact `(on ?x ?y)`
       relationship is also a goal fact. If it is NOT a goal fact, it means block
       `?x` is on block `?y` in the current state, but it shouldn't be there in the
       final configuration. This block `?x` is obstructing the goal state. Increment `h`.
    4. The total value of `h` is the heuristic estimate.

    This heuristic is non-admissible but aims to guide the search towards states
    where more goal conditions are met and fewer blocks are in incorrect,
    obstructing positions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by storing the relevant goal facts.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # Store goal facts that are (on ?x ?y) or (on-table ?x)
        # These represent the desired final positions of the blocks.
        self.goal_facts = {g for g in task.goals if match(g, "on", "*", "*") or match(g, "on-table", "*")}
        
        # Store goal (on ?x ?y) facts separately for efficient lookup
        # when checking for obstructing blocks.
        self.goal_on_facts = {g for g in task.goals if match(g, "on", "*", "*")}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        h = 0

        # Penalty 1: Unsatisfied goal facts ((on X Y) or (on-table X))
        # Count how many desired block positions are not present in the current state.
        for goal_fact in self.goal_facts:
            if goal_fact not in state:
                h += 1

        # Penalty 2: Blocks that are on top of other blocks but shouldn't be there in the goal.
        # These blocks are "in the way" and need to be moved to allow correct stacking.
        # Count how many 'on' facts in the current state represent incorrect placements
        # according to the goal.
        for fact in state:
            if match(fact, "on", "*", "*"):
                # This fact is of the form (on B A), meaning block B is on block A.
                # Check if this specific relationship (on B A) is desired in the goal.
                if fact not in self.goal_on_facts:
                    # If (on B A) is true in the state but not a goal fact,
                    # then block B is on A incorrectly. This placement needs to be undone.
                    h += 1

        return h

