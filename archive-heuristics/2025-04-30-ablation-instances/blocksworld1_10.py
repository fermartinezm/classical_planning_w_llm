from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe log a warning or raise an error
        # For robustness, return empty list or handle based on expected input
        # Assuming valid PDDL fact strings as input based on problem description
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are either not in their
    correct goal location or are in their correct goal location but have an
    incorrect block stacked immediately on top of them. It counts each such
    block as contributing 1 to the heuristic value.

    # Assumptions
    - The goal state consists of one or more stacks of blocks on the table.
    - Every block in the problem instance is mentioned in the goal state,
      specifying its final position (either on another block or on the table).
    - States are valid: each block is either on another block, on the table,
      or held by the arm, and no two blocks occupy the same space.

    # Heuristic Initialization
    - The heuristic constructor parses the goal conditions (`task.goals`) to
      determine the target position for each block (what it should be on, or
      if it should be on the table) and what block should be immediately on
      top of it in the goal state.
    - It also identifies the set of all blocks involved in the problem instance
      by examining the initial state and goal state facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:

    1. **Parse the current state:** Determine the current position of each block
       (what it is on, if it's on the table, or if it's being held) and, for
       blocks that are supporting others, identify the block immediately on top.
       Also, note if the arm is holding a block.

    2. **Initialize heuristic value:** Start with a heuristic value `h = 0`.

    3. **Iterate through all blocks:** For each block `B` identified during
       initialization:

       a. **Check the block's position:** Compare the current position of `B`
          (on block `U`, on the table, or held) with its target goal position.
          - If the current position does *not* match the goal position, increment `h`.
          (Note: If a block is being held, its current position is "holding",
           which will never match its goal position, so held blocks are counted here).

       b. **If the block is in its goal position (and not held):** Check the
          block immediately on top of `B`.
          - Compare the block currently on top of `B` in the state with the
            block that should be on top of `B` in the goal state (this includes
            the case where `B` should be clear in the goal).
          - If the block currently on top does *not* match the block that should
            be on top in the goal, increment `h`.

    4. **Return the total count:** The final value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration and all blocks.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state # Needed to find all blocks

        # Store goal configuration: block -> block_below or "table"
        self.goal_config = {}
        # Store goal stack order: block_below -> block_above or None (if block_below should be clear)
        self.goal_above = {}
        # Set of all blocks in the problem
        self.all_blocks = set()

        # Helper to add objects from facts
        def add_objects_from_fact(fact_str):
            parts = get_parts(fact_str)
            if not parts: return # Skip invalid facts
            predicate = parts[0]
            if predicate in ["on", "on-table", "holding", "clear"]:
                # Objects are the arguments after the predicate
                for obj in parts[1:]:
                    # Avoid adding "table" as a block object if it appears in args
                    if obj != "table":
                        self.all_blocks.add(obj)

        # Extract all blocks from initial state and goal state
        for fact in self.initial_state:
            add_objects_from_fact(fact)
        for fact in self.goals:
            add_objects_from_fact(fact)

        # Parse goal facts to build goal configuration mappings
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == "on":
                block_above, block_below = parts[1], parts[2]
                self.goal_config[block_above] = block_below
                self.goal_above[block_below] = block_above
            elif predicate == "on-table":
                block = parts[1]
                self.goal_config[block] = "table"
                # Blocks on the table in the goal should not have anything below them in goal_above
                # (though 'table' isn't a block, this map is block_below -> block_above)
                # We don't need to explicitly set goal_above["table"] = None as get() handles None default

        # Ensure all blocks found are in the goal_config (assuming all blocks have a goal position)
        # If a block is not in goal_config, it means it wasn't in any (on X Y) or (on-table X) goal fact.
        # This might indicate an issue with the problem definition or an implicit goal (e.g., clear).
        # For standard blocksworld, all blocks have a specified location.
        # We'll assume any block not in goal_config should be on the table.
        for block in list(self.all_blocks): # Iterate over a copy if modifying set
             if block not in self.goal_config:
                 # print(f"Warning: Block {block} not found in goal config. Assuming goal is (on-table {block}).")
                 self.goal_config[block] = "table"


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # Parse current state to build configuration mappings
        current_config = {} # block -> block_below or "table" or "holding"
        current_above = {}  # block_below -> block_above or None
        current_holding = None # block or None

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == "on":
                block_above, block_below = parts[1], parts[2]
                current_config[block_above] = block_below
                current_above[block_below] = block_above
            elif predicate == "on-table":
                block = parts[1]
                current_config[block] = "table"
            elif predicate == "holding":
                block = parts[1]
                current_config[block] = "holding"
                current_holding = block
            # We don't need 'clear' or 'arm-empty' facts directly for this heuristic logic

        # Calculate heuristic value
        h = 0

        for block in self.all_blocks:
            goal_pos = self.goal_config.get(block)
            current_pos = current_config.get(block) # .get() returns None if block not found in state config

            # If a block is not found in the current state configuration facts,
            # it implies an invalid state representation for this heuristic.
            # For robustness, we could assume it's misplaced, but standard states
            # should have a location fact for every block.
            if current_pos is None and block != current_holding:
                 # This case should ideally not happen in valid Blocksworld states
                 # where every block is either on/on-table/holding.
                 # print(f"Warning: Block {block} has no location fact in state.")
                 # Treat as misplaced
                 h += 1
                 continue # Cannot check block above if location is unknown

            # Condition 1: Block is not in its goal location
            if current_pos != goal_pos:
                h += 1
            # Condition 2: Block is in its goal location (and not held), but has wrong block above
            # Only check this if the block is in its goal position and not being held
            elif current_pos == goal_pos and current_pos != "holding":
                 # Get the block that should be above in the goal (None if it should be clear)
                 goal_block_above = self.goal_above.get(block)
                 # Get the block that is currently above in the state (None if it is clear)
                 current_block_above = current_above.get(block)

                 # If the block above in the state is different from the block above in the goal
                 # (This correctly handles cases like None vs block, block vs None, block1 vs block2)
                 if current_block_above != goal_block_above:
                     h += 1

        return h

