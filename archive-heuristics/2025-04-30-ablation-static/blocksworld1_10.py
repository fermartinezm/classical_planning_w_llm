# Helper function needed by the heuristic class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Assuming Heuristic base class is available in the environment
from heuristics.heuristic_base import Heuristic

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their correct
    goal position relative to their support (another block or the table), plus
    the number of blocks that are stacked on top of another block when they
    should not be in that specific stacking relationship according to the goal,
    plus a penalty if the arm is not empty.

    # Assumptions
    - Standard Blocksworld rules apply.
    - The goal state implies the arm is empty.

    # Heuristic Initialization
    - Extract the goal predicates, specifically focusing on `(on ?x ?y)` and
      `(on-table ?x)` predicates to build the desired goal stack structure.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Parse the Goal:
        - Create a mapping `goal_support` where `goal_support[B] = U` if the goal
          contains `(on B U)`, and `goal_support[B] = 'table'` if the goal
          contains `(on-table B)`. This map defines the desired immediate support
          for each block that is explicitly positioned in the goal.

    2.  Parse the Current State:
        - Create a mapping `current_support` where `current_support[B] = U` if
          the state contains `(on B U)`, and `current_support[B] = 'table'` if
          the state contains `(on-table B)`.
        - Identify the block currently being held, if any, by checking for
          `(holding ?x)`.
        - Collect all `(on A B)` facts present in the current state.

    3.  Calculate Misplaced Blocks Cost:
        - Initialize cost = 0.
        - Iterate through each block `B` that is a key in the `goal_support` map
          (i.e., blocks whose immediate support is specified in the goal).
        - Determine the block's current support (`current_support_B`):
            - If the state contains `(holding B)`, `current_support_B = 'holding'`.
            - Else if the state contains `(on-table B)`, `current_support_B = 'table'`.
            - Else (the block must be on another block), find `U` such that
              the state contains `(on B U)`, and set `current_support_B = U`.
              (A block must be in one of these configurations in a valid state).
        - If `current_support_block` is not equal to `goal_support[B]`, increment the cost.
        - This counts blocks not on their correct immediate support.

    4.  Calculate Wrongly Stacked Above Cost:
        - Initialize a set `wrongly_stacked_blocks`.
        - Iterate through all `(on A B)` facts collected from the current state.
        - Check if the fact `(on A B)` is present in the set of goal predicates.
        - If `(on A B)` is NOT a goal predicate, it means block `A` is stacked
          on block `B` incorrectly according to the goal structure. Add block `A`
          to the `wrongly_stacked_blocks` set.
        - Add the number of distinct blocks in the `wrongly_stacked_blocks` set
          to the total cost.

    5.  Calculate Arm State Cost:
        - If the predicate `(arm-empty)` is not present in the current state
          (meaning the arm is holding something), add 1 to the total cost.
          (Assumes the goal requires an empty arm).

    6.  Return Total Cost:
        - The heuristic value is the sum of the costs from steps 3, 4, and 5.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal predicates.
        """
        # Store goal predicates for quick lookup.
        self.goals = set(task.goals) # Convert frozenset to set for faster lookup

        # Build the goal_support map in the constructor as it's static
        self.goal_support = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'on':
                block, support = parts[1], parts[2]
                self.goal_support[block] = support
            elif parts and parts[0] == 'on-table':
                block = parts[1]
                self.goal_support[block] = 'table'

        # Static facts are not used in this Blocksworld heuristic.
        # static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of strings).

        # 2. Parse the Current State
        current_support = {}
        held_block = None
        current_on_facts = set() # Store (on A B) facts for easy lookup

        # Convert state frozenset to a set for potentially faster lookups
        state_set = set(state)

        for fact in state_set:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_support[block] = support
                current_on_facts.add(fact)
            elif predicate == 'on-table':
                block = parts[1]
                current_support[block] = 'table'
            elif predicate == 'holding':
                held_block = parts[1]

        # 3. Calculate Misplaced Blocks Cost
        misplaced_cost = 0
        # Iterate through all blocks that have a defined goal position
        for block in self.goal_support:
            desired_support = self.goal_support[block]

            # Find the block's current support
            current_support_block = None
            if held_block == block:
                current_support_block = 'holding'
            elif '(on-table ' + block + ')' in state_set:
                 current_support_block = 'table'
            else: # Must be on another block if not held or on table
                 # Find U such that (on block U) is in state
                 # Iterate through current_on_facts which is smaller than state_set
                 found_support = False
                 for fact in current_on_facts:
                     parts = get_parts(fact)
                     if parts[1] == block: # fact is (on block U)
                         current_support_block = parts[2]
                         found_support = True
                         break # Found the support
                 # If not found, the block is likely the held block or there's an issue
                 # with state representation. Assuming valid states.
                 # assert found_support or held_block == block or '(on-table ' + block + ')' in state_set, f"Block {block} not found on anything or held in state {state_set}"


            if current_support_block != desired_support:
                 misplaced_cost += 1

        # 4. Calculate Wrongly Stacked Above Cost
        wrongly_stacked_blocks = set()
        for fact in current_on_facts:
            # Check if this (on A B) fact is NOT in the goal
            if fact not in self.goals:
                parts = get_parts(fact)
                block_on_top = parts[1]
                wrongly_stacked_blocks.add(block_on_top)

        wrongly_stacked_cost = len(wrongly_stacked_blocks)

        # 5. Calculate Arm State Cost
        arm_cost = 0
        if '(arm-empty)' not in state_set:
            arm_cost = 1 # Arm is holding something, assuming goal is arm-empty

        # 6. Return Total Cost
        total_cost = misplaced_cost + wrongly_stacked_cost + arm_cost

        return total_cost
