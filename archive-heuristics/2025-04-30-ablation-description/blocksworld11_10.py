# from heuristics.heuristic_base import Heuristic # Assuming this is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with valid PDDL fact strings
        # but added for robustness.
        return []
    return fact[1:-1].split()

class blocksworldHeuristic: # Inherit from Heuristic if available
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by counting the number
    of goal conditions related to block positions (`on` and `on-table`) that
    are not currently satisfied in the state.

    # Assumptions
    - The goal state specifies the desired position for some blocks, either
      on top of another block (`on ?x ?y`) or on the table (`on-table ?x`).
    - The heuristic focuses only on these positional goal facts.
    - This heuristic is a form of delete-relaxation heuristic where achieving
      one positional goal is assumed not to negatively impact the achievement
      of other positional goals. It is admissible.

    # Heuristic Initialization
    - Extracts all goal facts of the form `(on ?x ?y)` and `(on-table ?x)`.
      These represent the desired final positions of blocks.

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is the goal state using `self.task.goal_reached(state)`.
       If it is, return 0. This ensures the heuristic is 0 only at the goal.
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through the set of pre-extracted goal position facts (`(on ?x ?y)` and `(on-table ?x)`).
    4. For each goal position fact `P`:
       a. Check if `P` is present in the current state.
       b. If `P` is *not* present in the current state, increment the heuristic value `h` by 1.
    5. The final value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal position facts.
        """
        # Store the full task object to use task.goal_reached later
        # If inheriting from a base class with __init__(task), call super().__init__(task)
        self.task = task

        self.goal_position_facts = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on' and len(parts) == 3:
                # Goal is (on block1 block2)
                self.goal_position_facts.add(goal)
            elif parts and parts[0] == 'on-table' and len(parts) == 2:
                # Goal is (on-table block)
                self.goal_position_facts.add(goal)
            # Ignore other goal facts like (clear ?) or (arm-empty)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Requirement: Heuristic is 0 only for goal states.
        # Explicitly check if the state is the goal state.
        if self.task.goal_reached(state):
             return 0

        h = 0

        # Count the number of goal position facts not satisfied.
        for goal_fact in self.goal_position_facts:
            if goal_fact not in state:
                h += 1

        return h
