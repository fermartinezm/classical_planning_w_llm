from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or malformed facts defensively
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
         # Indicate invalid fact
         return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by counting the number of unsatisfied goal conditions and adding penalties
    for state conditions that represent blocks or the arm being in "wrong"
    configurations that need to be undone or moved. The heuristic is designed
    to guide a greedy best-first search and does not need to be admissible.

    # Assumptions
    - Standard Blocksworld actions (pick-up, put-down, stack, unstack) with unit cost.
    - Goal states specify configurations of blocks using `on`, `on-table`,
      and `clear` predicates.
    - Blocks not mentioned as being on top of another block in goal `on` facts
      or explicitly on the table in goal `on-table` facts do not have a
      specific required support, but might need to be clear.
    - The heuristic assigns a cost of 1 for each identified "wrong" item or
      unachieved goal fact.

    # Heuristic Initialization
    - Extract all goal facts and store them in a set for efficient lookup.
    - Identify the set of blocks that appear as the upper block in any goal
      `(on B X)` fact. These are blocks that are required to be on top of
      something else in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is computed as the sum of the following components:

    1.  **Unachieved Goal Facts:** Iterate through all goal facts defined in the
        problem. For each goal fact that is *not* present in the current state,
        add 1 to the heuristic value. This counts how many explicit goal
        conditions are yet to be satisfied.

    2.  **Wrongly Placed Blocks (on):** Iterate through all facts in the current
        state. If a state fact is of the form `(on B A)` (block B is on block A),
        and this exact fact `(on B A)` is *not* one of the goal facts, add 1
        to the heuristic value. This penalizes blocks that are stacked on top
        of another block in a configuration that is not desired in the goal.

    3.  **Wrongly Placed Blocks (on-table):** Continue iterating through state
        facts. If a state fact is of the form `(on-table B)` (block B is on the
        table), and this fact `(on-table B)` is *not* one of the goal facts,
        AND block B is identified as a block that should be on top of *another*
        block in the goal state (i.e., B appears as the first argument in any
        goal `(on B X)` fact), add 1 to the heuristic value. This penalizes
        blocks that are on the table but are required to be part of a stack
        in the goal. Blocks on the table that are *not* required to be on
        something else in the goal are not penalized here, as being on the
        table is a valid final position for them.

    4.  **Busy Arm:** Continue iterating through state facts. If a state fact
        is of the form `(holding B)` (the arm is holding block B), add 1 to the
        heuristic value. This penalizes the state where the arm is busy, as
        the arm must be empty to pick up or unstack other blocks.

    The total sum of these counts provides an estimate of the remaining effort.
    Each component represents a condition that typically requires at least one
    action to resolve (e.g., unstacking a wrongly placed block, picking up a
    block that should be stacked, putting down a held block, or performing
    actions to achieve a missing goal fact).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying
        blocks that are required to be on top of another block in the goal.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        # Store goal facts in a set for efficient O(1) average time lookups.
        self.goal_facts_set = set(task.goals)

        # Identify blocks that appear as the upper block in any goal '(on B X)' fact.
        # These blocks are required to be on top of something else in the goal.
        self.blocks_upper_in_goal_on = set()
        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip invalid facts

            if parts[0] == 'on' and len(parts) == 3:
                # parts[1] is the block B in (on B X)
                self.blocks_upper_in_goal_on.add(parts[1])

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state # Frozenset of state facts (strings)
        state_set = set(state) # Convert to set for faster lookups

        h = 0

        # 1. Count unachieved goal facts
        # For each goal fact, check if it is present in the current state.
        for goal_fact in self.goal_facts_set:
            if goal_fact not in state_set:
                h += 1

        # 2. Count "wrong" state facts
        # Iterate through all facts currently true in the state.
        for state_fact in state_set:
            parts = get_parts(state_fact)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]

            if predicate == 'on' and len(parts) == 3:
                # If (on B A) is true in the state but is not a goal fact.
                if state_fact not in self.goal_facts_set:
                    h += 1
            elif predicate == 'on-table' and len(parts) == 2:
                block = parts[1]
                # If (on-table B) is true in the state but is not a goal fact,
                # AND block B is required to be on top of something else in the goal.
                if state_fact not in self.goal_facts_set and block in self.blocks_upper_in_goal_on:
                    h += 1
            elif predicate == 'holding' and len(parts) == 2:
                 # If the arm is holding a block.
                 h += 1
            # 'clear' and 'arm-empty' state facts are implicitly handled by
            # counting unachieved 'clear' goals and the 'holding' penalty.

        return h

