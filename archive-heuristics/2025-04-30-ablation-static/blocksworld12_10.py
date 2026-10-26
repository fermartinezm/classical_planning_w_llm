from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the distance to the goal by combining two components:
    1. The number of blocks that are not in their correct position within the goal stacks,
       considering recursive support (a block is correctly placed on another only if the
       support block is also correctly placed).
    2. The number of blocks that are required to be clear in the goal state but are
       not clear in the current state.

    # Assumptions
    - The goal specifies the desired position (on another block or on the table) for
      all blocks relevant to the goal configuration.
    - The heuristic considers both block positions ('on', 'on-table') and 'clear'
      predicates specified in the goal.

    # Heuristic Initialization
    - Extracts the desired position for each block involved in an 'on' or 'on-table' goal
      predicate, storing it in a dictionary `goal_pos_map` (mapping block to its support,
      or 'table').
    - Identifies the set of all blocks relevant to these position goals (`goal_blocks`).
    - Identifies the set of blocks that must be clear in the goal state (`goal_clear_blocks`).

    # Step-By-Step Thinking for Computing Heuristic
    1. **Parse State:** Determine the current position for each block in the given state
       (on another block, on the table, or held) and which blocks are currently clear.
       Store this in `current_pos_map` and `current_clear_blocks`. Also identify the block being held.
    2. **Compute Position Heuristic (h1):**
       a. Initialize a set `correctly_placed` to track blocks in their correct goal stack position.
       b. Seed `correctly_placed` with blocks that are supposed to be on the table
          according to the goal (`goal_pos_map[block] == 'table'`) and are currently
          on the table (`current_pos_map.get(block) == 'table'`). Use `.get()` for safety as a block might be held.
       c. Iteratively expand `correctly_placed`: Add any block X to the set if the goal
          is `(on X Y)` (`goal_pos_map.get(X) == Y`), the current state is `(on X Y)`
          (`current_pos_map.get(X) == Y`), and block Y is already in the `correctly_placed` set.
          Repeat until no new blocks are added. This propagates the "correctly placed"
          status up the goal stacks.
       d. `h1` is the total number of blocks involved in the goal positions (`len(self.goal_blocks)`)
          minus the number of blocks in the `correctly_placed` set (`len(correctly_placed)`).
    3. **Compute Clear Heuristic (h2):**
       a. Initialize `h2` to 0.
       b. For each block X that must be clear in the goal state (`X in self.goal_clear_blocks`),
          if X is not currently clear (`X not in current_clear_blocks`), increment `h2`.
    4. **Total Heuristic:** The final heuristic value is the sum of `h1` and `h2`.
       This value is 0 if and only if all blocks are in their correct goal positions
       *and* all blocks required to be clear are clear, which corresponds to the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and clear requirements.
        """
        self.task = task # Store task if needed, though not explicitly used in __call__ logic below
        self.goals = task.goals

        # Build goal_pos_map: block -> support_block or 'table'
        self.goal_pos_map = {}
        # Collect all blocks that appear in 'on' or 'on-table' goals
        self.goal_blocks = set()

        # Collect blocks that must be clear in the goal
        self.goal_clear_blocks = set()

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                self.goal_pos_map[block] = support
                self.goal_blocks.add(block)
                self.goal_blocks.add(support)
            elif predicate == 'on-table':
                block = parts[1]
                self.goal_pos_map[block] = 'table'
                self.goal_blocks.add(block)
            elif predicate == 'clear':
                 self.goal_clear_blocks.add(parts[1])


        # Remove 'table' from goal_blocks if it was added (it's not a block object)
        self.goal_blocks = frozenset(b for b in self.goal_blocks if b != 'table')
        self.goal_clear_blocks = frozenset(self.goal_clear_blocks)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Parse current state to find block positions and clear status
        current_pos_map = {} # block -> support_block or 'table'
        currently_held = None
        current_clear_blocks = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                block, support = parts[1], parts[2]
                current_pos_map[block] = support
            elif predicate == 'on-table':
                block = parts[1]
                current_pos_map[block] = 'table'
            elif predicate == 'holding':
                currently_held = parts[1]
            elif predicate == 'clear':
                current_clear_blocks.add(parts[1])

        # --- Compute Position Heuristic (h1) ---
        correctly_placed = set()

        # Seed with blocks correctly placed on the table according to the goal
        for block in self.goal_blocks:
            if self.goal_pos_map.get(block) == 'table' and current_pos_map.get(block) == 'table':
                 correctly_placed.add(block)

        # Iteratively add blocks correctly placed on other correctly placed blocks
        changed = True
        while changed:
            changed = False
            # Iterate over blocks that are part of the goal stacks but not yet marked correctly placed
            for block in self.goal_blocks:
                if block in correctly_placed:
                    continue

                goal_support = self.goal_pos_map.get(block)

                # Check if the block is supposed to be on another block (not table)
                if goal_support and goal_support != 'table':
                    # Check if it's currently on the correct support AND that support is correctly placed
                    # Use .get() for current_pos_map as a block might be held or not in state facts (though unlikely)
                    if current_pos_map.get(block) == goal_support and goal_support in correctly_placed:
                        correctly_placed.add(block)
                        changed = True

        # h1: Number of blocks in goal stacks not correctly placed
        # Blocks that are held are implicitly not correctly placed in their goal position.
        # The correctly_placed set only includes blocks on table or on other blocks.
        # Blocks that are held are not in current_pos_map, so current_pos_map.get(block) will be None.
        # The loop for correctly_placed correctly excludes held blocks from being added.
        # So, h1 counts blocks in goal_blocks that are NOT in correctly_placed.
        # This includes blocks that are misplaced, or on a wrong support, or held.
        h1 = len(self.goal_blocks) - len(correctly_placed)

        # --- Compute Clear Heuristic (h2) ---
        h2 = 0
        # For each block that needs to be clear in the goal
        for block in self.goal_clear_blocks:
            # If it's not currently clear
            if block not in current_clear_blocks:
                h2 += 1

        # Total heuristic value
        h_value = h1 + h2

        return h_value
