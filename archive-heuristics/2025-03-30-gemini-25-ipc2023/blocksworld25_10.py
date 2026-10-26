import os
import sys

# Assume the heuristic_base module is accessible.
# If heuristic_base.py is in a directory named 'heuristics'
# at the same level as this file, the following might be needed:
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Fallback if the above structure doesn't match
    # This allows the class definition even if the base class isn't found immediately
    # Note: Actual execution will fail if Heuristic cannot be imported.
    class Heuristic:
        def __init__(self, task): pass
        def __call__(self, node): raise NotImplementedError


# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """Extracts predicate and arguments from a PDDL fact string.
    Example: "(on b1 b2)" -> ["on", "b1", "b2"]
    Removes parentheses and splits by space.
    Returns an empty list if the format is unexpected.
    """
    # Ensure fact is a string and has the expected format "(...)"
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith("(") and fact.endswith(")"):
        return fact[1:-1].split()
    return []


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from a given state. It is designed for use with Greedy Best-First Search and
    does not need to be admissible. The heuristic counts the number of blocks
    that are part of the goal configuration but are not currently in their final
    correct position relative to the block (or table) below them, considering the
    entire stack down to the table. Each such "incorrect" block contributes 2 to
    the heuristic value (representing one pick-up/unstack and one put-down/stack
    action). The cost is then adjusted based on whether the arm is currently
    holding a block.

    # Assumptions
    - The goal state is defined by a conjunction of `(on A B)` and `(on-table C)` predicates.
    - All blocks mentioned in the goal predicates must be in their specified positions
      in the goal state. `(clear X)` goals are implicitly satisfied if the structure is correct.
    - All actions (`pickup`, `putdown`, `stack`, `unstack`) have a uniform cost of 1.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task's goal description (`task.goals`).
    - It builds data structures representing the target configuration:
        - `goal_on[block] = block_below`: A dictionary mapping each block to the block
          that should be directly below it in the goal configuration.
        - `goal_ontable`: A set containing all blocks that should be on the table
          in the goal configuration.
    - It also determines `goal_blocks`, the set of all unique blocks involved in
      the goal configuration.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Check for Goal State:** If the current state `node.state` already satisfies
        all goal conditions (`self.goals <= state`), the heuristic value is 0, and
        we return immediately. This is crucial for search algorithms.
    2.  **Parse Current State:** Iterate through the facts in the current state `node.state`
        to determine the current arrangement of blocks:
        - `current_on[block] = block_below`: Map each block to the block currently below it.
        - `current_ontable`: Identify the set of blocks currently on the table.
        - `holding`: Identify which block, if any, is currently held by the arm.
    3.  **Identify Correctly Placed Blocks (Bottom-Up):**
        - This step identifies blocks that are currently in their correct final position
          *relative to the blocks below them*, forming correct segments of the goal towers
          starting from the table.
        - Initialize an empty set `correct_blocks`.
        - Use a queue (`processing_queue`) for a breadth-first traversal starting from the table.
        - Initialize the queue with all blocks that are required to be on the table in the goal
          (`goal_ontable`) and are *currently* on the table (`current_ontable`). Add these
          blocks to `correct_blocks`.
        - While the queue is not empty:
            - Dequeue a block `block_below` (which is known to be correctly placed).
            - Find all blocks `B` that should be directly on top of `block_below` in the goal
              (i.e., `self.goal_on[B] == block_below`).
            - For each such block `B`, check if it is *currently* on `block_below`
              (`current_on.get(B) == block_below`).
            - If `B` is correctly positioned on `block_below` and `B` has not already been
              added to `correct_blocks`:
                - Add `B` to `correct_blocks`.
                - Enqueue `B` so that blocks potentially above it can be checked later.
        - After this process, `correct_blocks` contains all blocks that are part of a
          correctly formed stack segment matching the goal, starting from the table upwards.
    4.  **Count Incorrect Goal Blocks:** Determine how many blocks that are part of the
        goal configuration (`self.goal_blocks`) are *not* currently in their correct
        place (i.e., not in `correct_blocks`). Let this count be `num_incorrect_blocks`.
    5.  **Calculate Base Cost:** The initial heuristic estimate `h` is calculated as
        `num_incorrect_blocks * 2`. Each incorrect block needs at least one action to
        be picked up/unstacked and one action to be put down/stacked correctly.
    6.  **Adjust for Holding:** If the arm is currently holding a block `H`:
        - Check if `H` is relevant to the goal (`H in self.goal_blocks`) and if it's
          considered incorrectly placed (`H not in correct_blocks`).
        - If `H` is an incorrect goal block: Subtract 1 from `h`. The `* 2` cost for `H`
          already accounted for pickup + putdown. Since the pickup/unstack is done
          (it's being held), we only need to count the remaining putdown/stack action.
        - Otherwise (if `H` is already correctly placed - implying a possibly suboptimal
          state - or if `H` is not part of the goal configuration): Add 1 to `h`.
          In these cases, one action (putdown/stack) is still required to clear the
          arm or place the held block.
    7.  **Return Heuristic Value:** Return the final calculated heuristic value `h`, ensuring
        it is non-negative.
    """

    def __init__(self, task):
        """Initializes the heuristic by parsing goal conditions."""
        super().__init__(task) # Call parent constructor if needed
        self.goals = task.goals
        # Static facts are typically empty or unused in standard Blocksworld PDDL.
        # self.static = task.static

        self.goal_on = {} # Maps block -> block below it in goal
        self.goal_ontable = set() # Blocks on table in goal
        self.goal_blocks = set() # All blocks mentioned in the goal

        # Parse goal predicates to build the goal configuration representation
        for goal in self.goals:
            parts = get_parts(goal)
            # Ensure parsing was successful and format is recognized
            if not parts:
                continue # Skip malformed or irrelevant goal facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                # Example: (on b1 b2) -> goal_on[b1] = b2
                block_on_top = parts[1]
                block_below = parts[2]
                self.goal_on[block_on_top] = block_below
                self.goal_blocks.add(block_on_top)
                self.goal_blocks.add(block_below)
            elif predicate == "on-table" and len(parts) == 2:
                # Example: (on-table b3) -> b3 in goal_ontable
                block = parts[1]
                self.goal_ontable.add(block)
                self.goal_blocks.add(block)
            # We ignore 'clear' goals for heuristic calculation, as they are implicitly
            # handled by achieving the 'on'/'on-table' structure.

        # Cache the total number of unique blocks involved in the goal
        self.num_goal_blocks = len(self.goal_blocks)


    def __call__(self, node):
        """Computes the heuristic value for the given state node."""
        state = node.state

        # 1. Check for Goal State: If the state satisfies all goal predicates, h=0.
        if self.goals <= state:
             return 0

        # 2. Parse Current State to understand the current block arrangement
        current_on = {} # Maps block -> block below it currently
        current_ontable = set() # Blocks currently on the table
        holding = None # Block currently held by the arm, if any

        for fact in state:
            parts = get_parts(fact)
            # Ensure parsing was successful
            if not parts:
                continue

            predicate = parts[0]
            # Extract relevant information based on predicate type
            if predicate == "on" and len(parts) == 3:
                current_on[parts[1]] = parts[2]
            elif predicate == "on-table" and len(parts) == 2:
                current_ontable.add(parts[1])
            elif predicate == "holding" and len(parts) == 2:
                holding = parts[1]
            # Ignore 'clear' and 'arm-empty' predicates for this heuristic calculation

        # 3. Identify Correctly Placed Blocks (Bottom-Up using a queue)
        correct_blocks = set()
        # Use a list as a queue (efficient for typical Blocksworld problem sizes)
        processing_queue = []

        # Initialize queue with blocks that are correctly on the table according to the goal
        for block in self.goal_ontable:
            if block in current_ontable:
                correct_blocks.add(block)
                processing_queue.append(block)

        # Perform a breadth-first traversal upwards from the table
        head = 0 # Index for dequeuing from the list
        while head < len(processing_queue):
            block_below = processing_queue[head]
            head += 1

            # Find all blocks B that should be directly on 'block_below' in the goal
            for block_on_top, goal_support in self.goal_on.items():
                 if goal_support == block_below:
                    # Check if B is currently on 'block_below' and not already marked correct
                    if current_on.get(block_on_top) == block_below and block_on_top not in correct_blocks:
                        # B is correctly placed relative to block_below
                        correct_blocks.add(block_on_top)
                        # Add B to queue to check blocks that might be placed on it
                        processing_queue.append(block_on_top)

        # 4. Count Incorrect Goal Blocks
        # Iterate through all blocks required by the goal and count those not in 'correct_blocks'
        num_incorrect_blocks = 0
        for block in self.goal_blocks:
            if block not in correct_blocks:
                num_incorrect_blocks += 1

        # 5. Calculate Base Cost (2 actions per incorrect block)
        h = num_incorrect_blocks * 2

        # 6. Adjust Cost for Holding a Block
        if holding is not None:
            # Check if the held block H is a goal block AND is currently considered incorrect
            if holding in self.goal_blocks and holding not in correct_blocks:
                # H is an incorrect goal block. The base cost 'h' includes 2 actions
                # for H (pickup/unstack + putdown/stack). Since pickup/unstack is
                # already done (it's being held), subtract 1 from the cost.
                h = h - 1
            else:
                # Held block H is either:
                # a) Already correctly placed (implies a suboptimal state).
                # b) Not part of the goal configuration.
                # In either case, one action (putdown/stack) is needed.
                h += 1

        # 7. Return Heuristic Value (ensure non-negativity)
        return max(0, h)
