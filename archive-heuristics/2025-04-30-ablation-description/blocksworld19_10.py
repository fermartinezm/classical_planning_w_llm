from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    fact = fact.strip()
    if not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential malformed facts, though unlikely with planner output
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic function for the Blocksworld domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by counting
    the number of blocks that are not in their correct position within the
    goal stacks. A block is considered correctly placed in its goal stack
    if it is on the correct block (or the table) according to the goal,
    AND the block below it is also correctly placed in its goal stack.
    This heuristic counts the number of blocks that *fail* this condition.

    # Assumptions
    - The heuristic focuses on achieving the correct stack configurations
      defined by the goal state.
    - It assumes that achieving the correct relative position for a block
      is a step towards the goal.
    - It does not explicitly count actions like clearing blocks or using
      the arm, but the number of blocks out of place in the goal stacks
      correlates with the minimum number of blocks that need to be moved.

    # Heuristic Initialization
    - The heuristic parses the goal predicates (`self.goals`) to build a
      representation of the desired stack structure. This includes a mapping
      `goal_on_map` where `goal_on_map[Y] = B` if `(on B Y)` is a goal,
      and a set `goal_on_table_set` for blocks that should be on the table.
    - It also identifies all unique blocks mentioned in the goal `on` or
      `on-table` predicates.
    - Static facts are not used as the domain has none.

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the goal predicates (`self.goals`) to create the `goal_on_map`,
       `goal_on_table_set`, and `goal_blocks` set as described in Initialization.
       Ignore `(clear ?x)` and `(arm-empty)` goals for this heuristic as they
       are often consequences of stack configurations rather than primary
       positional goals for blocks.
    2. Initialize `correctly_placed_count = 0`.
    3. Initialize a set `checked_blocks` to keep track of blocks confirmed
       to be in their correct goal stack position relative to the base.
    4. Initialize a queue `propagation_queue` with blocks from `goal_on_table_set`
       that are currently on the table in the state (`(on-table B)` is true).
       For each such block, increment `correctly_placed_count` and add it to
       `checked_blocks`. These are the base blocks of the correctly built stacks.
    5. While the `propagation_queue` is not empty:
       - Dequeue a block `Y` (which is correctly placed relative to its base).
       - Find the block `B` that is supposed to be directly on top of `Y`
         in the goal stack (`B = self.goal_on_map.get(Y)`).
       - If such a block `B` exists AND `B` has not already been counted
         as correctly placed (`B` not in `checked_blocks`):
         - Check if `(on B Y)` is true in the current state (`node.state`).
         - If `(on B Y)` is true, it means `B` is correctly placed relative
           to `Y`, and since `Y` is correctly placed, `B` is also correctly
           placed in the goal stack. Increment `correctly_placed_count`,
           add `B` to `checked_blocks`, and enqueue `B` into the
           `propagation_queue` for checking blocks above it.
    6. The heuristic value is the total number of unique blocks involved
       in the goal (`len(self.goal_blocks)`) minus the `correctly_placed_count`.
       This represents the number of blocks that are not in their correct
       position within the goal stack structure, considering dependencies.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by parsing goal predicates to build the
        goal stack structure and identify all goal blocks.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        super().__init__(task)

        self.goal_on_map = {} # Maps block_below -> block_above in goal stacks
        self.goal_on_table_set = set() # Blocks that should be on the table in goal
        self.goal_blocks = set() # All blocks mentioned in goal on/on-table predicates

        # Parse goal predicates to build the goal stack structure
        for goal_predicate in self.goals:
            parts = get_parts(goal_predicate)
            if not parts: continue # Skip malformed facts

            predicate_name = parts[0]
            if predicate_name == 'on' and len(parts) == 3:
                block_above, block_below = parts[1], parts[2]
                self.goal_on_map[block_below] = block_above
                self.goal_blocks.add(block_above)
                self.goal_blocks.add(block_below)
            elif predicate_name == 'on-table' and len(parts) == 2:
                block = parts[1]
                self.goal_on_table_set.add(block)
                self.goal_blocks.add(block)
            # Ignore (clear ?) and (arm-empty) goals for this stack-based heuristic

    def __call__(self, node):
        """
        Compute the heuristic value based on the number of blocks not
        correctly placed in the goal stack structure.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer representing the estimated cost to reach the goal.
        """
        state = node.state

        correctly_placed_count = 0
        checked_blocks = set()
        propagation_queue = [] # Use a list as a simple queue

        # Step 4: Identify and queue blocks that are correctly placed on the table base
        for block_on_table_goal in self.goal_on_table_set:
            if f'(on-table {block_on_table_goal})' in state:
                correctly_placed_count += 1
                checked_blocks.add(block_on_table_goal)
                propagation_queue.append(block_on_table_goal) # Add to queue for propagation

        # Step 5: Propagate correctness upwards through the stacks
        while propagation_queue:
            block_below = propagation_queue.pop(0) # Dequeue block

            # Find the block that should be on top of block_below in the goal
            block_above_goal = self.goal_on_map.get(block_below)

            # If there is a block that should be above block_below in the goal,
            # and we haven't already counted it as correctly placed:
            if block_above_goal and block_above_goal not in checked_blocks:
                # Check if block_above_goal is currently on block_below in the state
                if f'(on {block_above_goal} {block_below})' in state:
                    # block_above_goal is correctly placed relative to block_below,
                    # and block_below is correctly placed, so block_above_goal is
                    # correctly placed in the goal stack.
                    correctly_placed_count += 1
                    checked_blocks.add(block_above_goal)
                    propagation_queue.append(block_above_goal) # Add to queue for further propagation

        # Step 6: Heuristic is total goal blocks minus correctly placed blocks
        total_goal_blocks = len(self.goal_blocks)

        # Ensure heuristic is non-negative (should be guaranteed by logic, but defensive)
        return max(0, total_goal_blocks - correctly_placed_count)

