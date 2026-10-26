# from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

class blocksworldHeuristic: # Inherit from Heuristic in the actual environment
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the difficulty of reaching the goal state by counting
    the number of goal conditions related to block positions and clear states
    that are not met, plus a penalty if the arm is not empty. It aims to guide
    a greedy best-first search efficiently.

    # Assumptions
    - The goal state specifies the desired stack configuration using (on ?x ?y)
      and (on-table ?x) predicates, and potentially (clear ?x) for blocks
      that should be at the top of stacks.
    - Actions have unit cost.

    # Heuristic Initialization
    The heuristic is initialized by parsing the goal facts to identify:
    - `goal_on`: A dictionary mapping a block to the block it should be on in the goal.
    - `goal_on_table`: A set of blocks that should be on the table in the goal.
    - `goal_clear_facts`: A set of blocks that are explicitly required to be clear in the goal.
    - `all_goal_blocks`: A set of all blocks whose final position (on another block or table)
      is specified in the goal.
    Static facts are ignored as they are empty in Blocksworld.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as the sum of three components:

    1.  **Position Cost:** Iterate through each block that is part of the goal configuration (`all_goal_blocks`).
        -   If the block should be on the table according to `goal_on_table`, check if the state contains the fact `(on-table <block>)`. If not, add 1 to the cost.
        -   If the block should be on another block `A` according to `goal_on` (`goal_on[block] == A`), check if the state contains the fact `(on <block> <A>)`. If not, add 1 to the cost.
        This component counts how many blocks are not on their correct base (or table) as specified by the goal.

    2.  **Clear Cost:** Iterate through each block that is explicitly required to be clear in the goal (`goal_clear_facts`).
        -   Check if the state contains the fact `(clear <block>)`. If not, add 1 to the cost.
        This component counts how many blocks that should be clear at the top of goal stacks are currently blocked.

    3.  **Arm Cost:** Check if the state contains the fact `(arm-empty)`.
        -   If the arm is not empty (i.e., `(arm-empty)` is not in the state), add 1 to the cost.
        This component penalizes the state if the arm is holding a block, as the arm must be empty to pick up a block needed for further steps.

    The total heuristic value is the sum of the Position Cost, Clear Cost, and Arm Cost.
    The heuristic is 0 if and only if all goal conditions related to position and clear state are met, and the arm is empty (if arm-empty is a goal, or implicitly required).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.
        """
        # Assuming task object has 'goals' attribute (frozenset of fact strings)
        self.goals = task.goals

        self.goal_on = {}
        self.goal_on_table = set()
        self.goal_clear_facts = set()
        self.all_goal_blocks = set()

        # Parse goal facts
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block, base = parts[1], parts[2]
                self.goal_on[block] = base
                self.all_goal_blocks.add(block)
                # Note: We don't add 'base' to all_goal_blocks here, as its position
                # is defined by the block *above* it in the goal stack. We only
                # care about blocks whose *own* position is specified (on X or on-table).
            elif predicate == "on-table" and len(parts) == 2:
                block = parts[1]
                self.goal_on_table.add(block)
                self.all_goal_blocks.add(block)
            elif predicate == "clear" and len(parts) == 2:
                block = parts[1]
                self.goal_clear_facts.add(block)
            # Ignore (arm-empty) goal fact for structure parsing, handled separately

        # Ensure all_goal_blocks only contains blocks whose position is explicitly set
        # (either on another block or on the table)
        self.all_goal_blocks = set(self.goal_on.keys()) | self.goal_on_table


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # state is a frozenset of fact strings

        # Calculate heuristic components
        position_cost = 0
        clear_cost = 0
        arm_cost = 0

        # 1. Position Cost
        # Count blocks that are not on their correct goal base or table
        for block in self.all_goal_blocks:
            # Check if block should be on table
            if block in self.goal_on_table:
                # Is it currently on table?
                if f"(on-table {block})" not in state:
                    position_cost += 1
            # Check if block should be on another block
            elif block in self.goal_on:
                base = self.goal_on[block]
                # Is it currently on the correct base?
                if f"(on {block} {base})" not in state:
                     position_cost += 1
            # else: block is in all_goal_blocks but not in goal_on_table and not key in goal_on?
            # This case should not happen based on how all_goal_blocks is built.


        # 2. Clear Cost
        # Count blocks that should be clear but aren't
        for block in self.goal_clear_facts:
            # Is it currently clear?
            if f"(clear {block})" not in state:
                clear_cost += 1

        # 3. Arm Cost
        # Penalize if the arm is not empty
        if "(arm-empty)" not in state:
            arm_cost = 1

        # Total heuristic is the sum of costs
        total_cost = position_cost + clear_cost + arm_cost

        return total_cost
