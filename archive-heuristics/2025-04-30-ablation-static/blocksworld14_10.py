from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact_string):
    """Extract predicate and arguments from a PDDL fact string."""
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    return parts

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the "disorder" of the current state relative to the goal state
    by counting the number of blocks that are either not in their correct goal position
    or have an incorrect block stacked directly on top of them. This count serves as
    an estimate of the number of actions needed to fix these discrepancies.

    # Assumptions
    - The goal state defines a specific configuration of blocks into stacks on the table.
    - Every block mentioned in an 'on' or 'on-table' goal predicate has a unique desired support (either the table or another block).
    - 'clear' goal predicates typically apply only to blocks that are the intended top of a goal stack.
    - The heuristic assumes standard Blocksworld actions (pickup, putdown, stack, unstack).

    # Heuristic Initialization
    The heuristic pre-processes the goal state to build data structures representing the desired configuration:
    - `goal_config`: A dictionary mapping each block to its intended support (the block it should be on, or 'table').
    - `goal_above`: A dictionary mapping each block to the block that should be directly on top of it according to the goal.
    - `goal_is_top`: A set of blocks that should be clear in the goal state (i.e., they are the top block of a goal stack).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Parse the current state to determine the current position of each block (on the table, on another block, or held by the arm) and which block is currently on top of which.
       - `current_pos`: Maps each block to its current support ('table', block name, or 'arm' if held).
       - `current_above`: Maps each block to the block currently stacked directly on top of it.
       - `current_holding`: Stores the block currently held by the arm, or None.
    2. Initialize a counter `h` to 0.
    3. Iterate through each block `B` that has a specified goal position (i.e., is a key in `self.goal_config`).
    4. For each block `B`, compare its current position (`current_pos.get(B)`) with its desired goal position (`self.goal_config[B]`).
       - If the current position is *not* the desired position, increment `h`. This block is in the wrong place.
    5. If the block `B` *is* in its correct goal position, check the block immediately on top of it in the current state (`current_above.get(B)`).
       - If `B` should be clear in the goal (`B` is in `self.goal_is_top`), but there is *any* block on top of it in the current state, increment `h`.
       - If some block `Z` should be on top of `B` in the goal (`self.goal_above[B] == Z`), but the block currently on top of `B` is *not* `Z` (or nothing is on top), increment `h`.
    6. The final value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        self.goal_config = {} # block -> support (block or 'table')
        self.goal_above = {} # support_block -> block_on_top

        all_goal_blocks = set()

        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                self.goal_config[block] = support
                self.goal_above[support] = block
                all_goal_blocks.add(block)
                all_goal_blocks.add(support)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_config[block] = 'table'
                all_goal_blocks.add(block)
            # Ignore 'clear' goals for position mapping, they are derived from goal_above

        # Identify blocks that should be clear in the goal (those in goal_config but not supporting anything in goal_above)
        # Note: A block might be in all_goal_blocks but not in goal_config if it only appears in a (clear B) goal.
        # However, standard blocksworld goals usually specify the position for all blocks.
        # We only care about blocks whose position is specified in the goal.
        self.goal_is_top = set(self.goal_config.keys()) - set(self.goal_above.keys())


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state
        current_pos = {} # block -> support ('table', block, or 'arm')
        current_above = {} # support_block -> block_on_top
        current_holding = None

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_pos[block] = support
                current_above[support] = block
            elif predicate == 'on-table':
                block = parts[1]
                current_pos[block] = 'table'
            elif predicate == 'holding':
                current_holding = parts[1]
                current_pos[current_holding] = 'arm' # Represent holding as being on 'arm'
            # 'clear' and 'arm-empty' facts are not directly used in this heuristic calculation

        h = 0
        # Iterate over blocks that have a specified goal position
        for block, desired_support in self.goal_config.items():
            current_support = current_pos.get(block)

            # Condition 1: Block is not in its correct goal position
            if current_support != desired_support:
                h += 1
            else: # current_support == desired_support (Block is on the right thing/table)
                # Condition 2: Block is in the right place, but something is wrong on top
                # Check what is currently on top of block in the state
                current_block_above = current_above.get(block) # None if block is clear

                if block in self.goal_is_top: # Block should be clear in the goal
                    if current_block_above is not None: # But something is on it in the state
                        h += 1 # Block is blocked by something that shouldn't be there
                else: # Some block Z should be on block in the goal (Z = self.goal_above[block])
                    desired_block_above = self.goal_above[block]
                    if current_block_above != desired_block_above:
                        # The block currently on block is not the one that should be there
                        h += 1

        return h
