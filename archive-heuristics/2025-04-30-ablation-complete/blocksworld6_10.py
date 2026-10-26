from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(arm-empty)' which have no arguments
    if fact.strip() == "()":
        return []
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to compare parts and args up to the length of the shorter sequence.
    # fnmatch handles the wildcard matching.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting the number of goal position facts ('on' or 'on-table') that
    are not satisfied, the number of incorrect 'on' relationships that exist
    in the current state, and adding a penalty if the arm is holding a block.

    # Assumptions
    - The goal state is defined primarily by 'on' and 'on-table' predicates,
      specifying the desired stack configurations. 'clear' and 'arm-empty'
      goals are often consequences of achieving the main stack goals, but
      '(clear X)' goals are implicitly handled if they require removing
      an incorrect 'on' relationship. '(arm-empty)' goals are handled by
      a direct penalty.
    - Each missing goal position fact or incorrect 'on' relationship generally
      requires at least one or two actions to fix (e.g., unstack/pickup + stack/putdown).
    - Having a block in hand often requires an action to put it down or stack it
      before other blocks can be manipulated.

    # Heuristic Initialization
    - Extract the set of goal predicates that define the desired 'on' and
      'on-table' relationships. These are the target configurations for the blocks.
      Store these in `self.goal_pos_facts`.
    - Store the complete set of goal facts for the h=0 check.
    - Static facts are not used in this heuristic as they are empty in Blocksworld.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state `s`:

    1. Count the number of goal predicates from `self.goal_pos_facts` that are
       *not* present in the current state `s`. Each missing goal position fact
       indicates a block that is not in its required final location relative
       to its support. This contributes to the heuristic, as the block needs
       to be moved into that position. Let this count be `count1`.

    2. Identify all 'on' predicates that are true in the current state `s`.
       Count how many of these state 'on' facts are *not* present in the set
       of goal position facts (`self.goal_pos_facts`). Each such state 'on' fact
       represents an incorrect stacking that must be undone (by unstacking the
       top block). This contributes to the heuristic as it requires actions
       to clear the incorrect stack. Let this count be `count2`.

    3. Check if the robot arm is currently holding any block. This is indicated
       by the presence of any fact matching the pattern `(holding ?)`. If the
       arm is holding a block, add 1 to the heuristic. This penalizes states
       where the arm is occupied, as it might need to free up to perform
       necessary actions. Let this be `count3` (either 0 or 1).

    4. The total heuristic value for state `s` is `count1 + count2 + count3`.
       This sums the estimated effort related to achieving correct positions,
       undoing incorrect stackings, and managing the arm state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal position facts.
        """
        # Store goal predicates related to block positions.
        self.goal_pos_facts = set()
        # Store all goal facts for easy lookup (needed for h=0 check)
        self.all_goal_facts = set(task.goals)

        for goal in task.goals:
            parts = get_parts(goal)
            # Consider 'on' and 'on-table' facts as goal position facts
            if parts and (parts[0] == "on" or parts[0] == "on-table"):
                self.goal_pos_facts.add(goal)

        # Static facts are empty in Blocksworld, so no extraction needed.
        # static_facts = task.static

    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # If the state is the goal state, the heuristic is 0.
        # This is a quick check and ensures h=0 only at the goal.
        if self.all_goal_facts.issubset(state):
             return 0

        # Count 1: Missing goal position facts
        count1 = 0
        for goal_fact in self.goal_pos_facts:
            if goal_fact not in state:
                count1 += 1

        # Count 2: Incorrect 'on' relationships in the current state
        count2 = 0
        # Iterate through state facts to find 'on' predicates
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "on":
                # If a state 'on' fact is not one of the required goal position facts
                if fact not in self.goal_pos_facts:
                    count2 += 1

        # Count 3: Penalty for holding a block
        count3 = 0
        # Check if any fact in the state matches the pattern (holding ?)
        for fact in state:
             if match(fact, "holding", "*"):
                 count3 = 1 # Only one block can be held at a time
                 break # Found the holding fact, no need to check others

        # The heuristic is the sum of the three counts.
        return count1 + count2 + count3
