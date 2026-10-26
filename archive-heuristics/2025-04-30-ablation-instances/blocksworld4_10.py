# Need to import Heuristic base class
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    return fact[1:-1].split()

# Define the heuristic class
class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are not in their
    correct final position within the goal stack structure, plus a penalty
    if the robot's arm is not empty.

    # Assumptions
    - The goal specifies the desired final configuration of blocks on the table
      or stacked on top of each other.
    - Blocks not mentioned as being 'on' another block or 'on-table' in the goal
      are not considered for the 'correctly stacked' count.
    - The final state requires the arm to be empty.
    - Goal stacks are fully specified down to blocks on the table.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired block-on-block or
      block-on-table relationships.
    - Creates a mapping `goal_under` where `goal_under[block]` is the block
      it should be directly on top of in the goal, or the string 'table'
      if it should be on the table. Only blocks whose position is explicitly
      defined in the goal (as being 'on' something or 'on-table') are included
      as keys in `goal_under`.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of blocks whose goal position is explicitly defined (`blocks_with_goal_pos`, which are the keys in `self.goal_under`).
    2. For each block `b` in `blocks_with_goal_pos`, determine if it is "correctly stacked".
       A block `b` is correctly stacked if:
       - Its goal is `(on-table b)` AND `(on-table b)` is true in the current state.
       - OR its goal is `(on b under_b)` AND `(on b under_b)` is true in the current state AND `under_b` is also correctly stacked according to its own goal position.
       This check is performed iteratively until stability:
       - Initialize `is_correctly_stacked[b]` to False for all `b` in `blocks_with_goal_pos`.
       - Repeatedly iterate through `blocks_with_goal_pos`:
         - If `goal_under[b]` is 'table' and `(on-table {block})` is in the state, set `is_correctly_stacked[b]` to True.
         - If `goal_under[b]` is `under_b` (a block) and `(on {block} {under_b})` is in the state AND `is_correctly_stacked[under_b]` is True, set `is_correctly_stacked[b]` to True.
       - Stop when a full iteration completes without any `is_correctly_stacked` value changing from False to True.
    3. Count the number of blocks `b` in `blocks_with_goal_pos` for which `is_correctly_stacked[b]` is still False. This count represents blocks that are not in their correct position relative to the goal stack structure.
    4. Add 1 to the count if the robot's arm is not empty, as the arm typically needs to be empty to perform many necessary actions (like picking up a block).
    5. The total count is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions for blocks.
        """
        self.goals = task.goals  # Goal conditions.

        # Map block to the object it should be directly on top of in the goal.
        # 'table' indicates it should be on the table.
        # Only includes blocks whose position is explicitly defined in the goal.
        self.goal_under = {}

        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "on":
                block, under_block = parts[1], parts[2]
                self.goal_under[block] = under_block
            elif parts[0] == "on-table":
                block = parts[1]
                self.goal_under[block] = 'table'

        # The set of blocks whose goal position is explicitly defined.
        self.blocks_with_goal_pos = set(self.goal_under.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Dictionary to track which goal blocks are correctly stacked.
        # Initialize all to False for blocks whose position is defined in the goal.
        is_correctly_stacked = {b: False for b in self.blocks_with_goal_pos}

        # Iteratively determine which blocks are correctly stacked from the bottom up.
        # This process stabilizes when no more blocks can be marked as correctly stacked.
        changed = True
        while changed:
            changed = False
            for block in self.blocks_with_goal_pos:
                if not is_correctly_stacked[block]: # Only try to update if not already True
                    under_block = self.goal_under[block] # This key is guaranteed to exist

                    if under_block == 'table':
                        if f"(on-table {block})" in state:
                            is_correctly_stacked[block] = True
                            changed = True
                    else: # It should be on another block
                        # Check if the block is on the correct block AND the block below is correctly stacked.
                        # The block below must also have a goal position and be correctly stacked itself.
                        # If under_block is not in blocks_with_goal_pos, it cannot be 'correctly_stacked' by this definition.
                        # The check `is_correctly_stacked.get(under_block, False)` handles this correctly.
                        if f"(on {block} {under_block})" in state and is_correctly_stacked.get(under_block, False):
                             is_correctly_stacked[block] = True
                             changed = True

        # Count the number of blocks that are not correctly stacked.
        misplaced_blocks_count = sum(1 for block in self.blocks_with_goal_pos if not is_correctly_stacked[block])

        # Add penalty if the arm is not empty.
        arm_penalty = 1 if "(arm-empty)" not in state else 0

        return misplaced_blocks_count + arm_penalty
