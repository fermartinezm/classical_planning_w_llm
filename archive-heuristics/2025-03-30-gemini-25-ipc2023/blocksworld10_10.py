import sys
import os
from typing import Set, Dict, Optional, FrozenSet

# Ensure the path includes the directory containing heuristic_base if necessary
# This might be needed depending on the execution environment.
# Example: sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import the base class Heuristic.
# If the module or class isn't found, define a dummy class to allow the code to load.
try:
    # This assumes 'heuristics.heuristic_base' contains the base class 'Heuristic'.
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: heuristics.heuristic_base.Heuristic not found. Using a dummy base class.", file=sys.stderr)
    # Define a dummy base class if the import fails, so the script can still be parsed.
    class Heuristic:
        def __init__(self, task):
            """Dummy constructor."""
            self.task = task # Store task for potential reference
        def __call__(self, node):
            """Dummy heuristic calculation method."""
            # A real heuristic should return an estimated cost to the goal.
            # Returning 0 ensures it doesn't crash but provides no guidance.
            # Returning NotImplementedError would be clearer if execution reaches here.
            print("Error: Heuristic logic cannot execute because base class was not found.", file=sys.stderr)
            return 0 # Or raise NotImplementedError("Heuristic base class not found.")


# Helper function to parse PDDL fact strings
def get_parts(fact: str) -> list[str]:
    """Extracts predicate and arguments from a PDDL fact string.

    Removes surrounding parentheses and splits the content by spaces.
    Handles potential leading/trailing whitespace around the fact.

    Args:
        fact: The PDDL fact string, e.g., "(on b1 b2)".

    Returns:
        A list of strings representing the predicate and its arguments,
        e.g., ["on", "b1", "b2"]. Returns an empty list if the fact
        is malformed (e.g., doesn't start/end with parentheses) or empty.
    """
    stripped_fact = fact.strip()
    if not stripped_fact.startswith("(") or not stripped_fact.endswith(")"):
        return []
    # Remove parentheses and split by whitespace
    content = stripped_fact[1:-1].strip()
    if not content: # Handle "()" case or "( )"
        return []
    return content.split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state
    from a given state. It is designed for use with Greedy Best-First Search and
    prioritizes informativeness over admissibility (it is likely non-admissible).
    The heuristic value is primarily based on counting the number of blocks that
    are not in their final target position (i.e., not resting on the correct
    block below them or on the table as specified in the goal) and whether the
    robot arm is currently holding a block.

    # Assumptions
    - The goal state description (`task.goals`) primarily specifies the final
      configuration using `(on blockA blockB)` and `(on-table blockA)` facts.
      Other goal predicates like `(clear x)` or `(arm-empty)` are assumed to be
      implicitly satisfied when the positional goals are met, or they are ignored
      by this heuristic's calculation.
    - All blocks mentioned in the goal's positional facts (`on`, `on-table`)
      exist within the problem instance's objects.
    - The heuristic estimates costs as follows:
      - Each block that is currently placed (on the table or another block) but
        is not resting on its correct goal support contributes 2 to the heuristic
        value. This estimates the cost of one pickup/unstack action and one
        putdown/stack action needed to correct its position.
      - If the robot arm is currently holding a block (`(holding ...)` fact is true),
        this contributes 1 to the heuristic value. This represents the single
        action (stack or putdown) needed to place the held block.

    # Heuristic Initialization
    - The constructor (`__init__`) is called once when the heuristic is created.
    - It processes the task's goal description (`task.goals`).
    - It builds a dictionary `goal_support`: `block_name -> support_element`.
      The `support_element` is either the name of the block that should be
      directly below `block_name` in the goal, or the special string 'table'
      if `block_name` should be on the table in the goal. This map stores the
      target position for the bottom face of each relevant block.
    - It also compiles a set `goal_blocks` containing all unique block names
      that appear in these goal positional facts (either as the block being
      placed or as the supporting block).

    # Step-By-Step Thinking for Computing Heuristic
    1. **Check Goal Reached:** The calculation (`__call__`) first checks if the
       current state (`node.state`) already satisfies all goal conditions defined
       in `self.goals`. If `self.goals <= node.state` is true, the goal is reached,
       and the heuristic value is 0, indicating no more actions are needed.
    2. **Parse Current State:**
       - Iterate through all facts present in the current state (`node.state`).
       - Identify which block, if any, is being held by the arm by looking for
         an `(holding block_name)` fact. Store the name in `held_block`.
       - Build the `current_support` dictionary. For each fact like `(on block_a block_b)`
         or `(on-table block_a)`, map `block_a` to its current support (`block_b` or 'table').
       - Collect the set `blocks_in_current_config` containing all blocks that are
         currently placed (i.e., appear as the first argument in `on` or `on-table` facts).
    3. **Calculate Cost - Held Block:**
       - Initialize the total `heuristic_value` to 0.
       - If `held_block` is not `None` (meaning the arm is holding a block),
         increment `heuristic_value` by 1.
    4. **Calculate Cost - Misplaced Blocks:**
       - Determine the set `all_relevant_blocks` by taking the union of the blocks
         currently placed (`blocks_in_current_config`) and the blocks involved in
         the goal configuration (`self.goal_blocks`). This ensures that all blocks
         that might contribute to the heuristic value are considered.
       - Iterate through each `block` in this `all_relevant_blocks` set:
         - If the current `block` is the same as `held_block`, skip it because its
           cost contribution has already been accounted for.
         - Retrieve the block's target support `goal_pos` from the precomputed
           `self.goal_support` map (this will be `None` if the block's final position
           isn't specified by an `on` or `on-table` goal).
         - Retrieve the block's current support `current_pos` from the `current_support`
           map derived from the current state (this will be `None` if the block is not
           currently placed on the table or another block).
         - **Condition Check:** If the block *has* a defined goal position
           (`goal_pos is not None`) **and** it *is* currently placed somewhere
           (`current_pos is not None`):
           - **Misplacement Check:** Compare the current support with the goal support.
             If they are different (`current_pos != goal_pos`):
             - Increment `heuristic_value` by 2.
    5. **Return Value:** The final computed `heuristic_value` is returned. This integer
       represents the heuristic's estimate of the minimum number of actions required
       to reach the goal state from the current state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing the goal state to determine
        the target position (support) for each relevant block.

        Args:
            task: The planning task object, containing task.goals, task.static, etc.
                  We primarily use task.goals here.
        """
        super().__init__(task) # Initialize the base class
        self.goals: FrozenSet[str] = task.goals
        # Map: block_name -> name_of_block_below_or_'table' in the goal state
        self.goal_support: Dict[str, str] = {}
        # Set of all block names involved in the goal's on/on-table predicates
        goal_blocks_temp: Set[str] = set()

        # Parse the goal facts to build the goal_support map and identify goal blocks
        for fact in self.goals:
            parts = get_parts(fact)
            # Ensure the fact is not malformed and has a predicate
            if not parts:
                continue
            predicate = parts[0]

            # Check for (on block_a block_b) goal predicate
            if predicate == "on" and len(parts) == 3:
                block_a, block_b = parts[1], parts[2]
                self.goal_support[block_a] = block_b
                goal_blocks_temp.add(block_a)
                goal_blocks_temp.add(block_b)
            # Check for (on-table block_a) goal predicate
            elif predicate == "on-table" and len(parts) == 2:
                block_a = parts[1]
                self.goal_support[block_a] = 'table'
                goal_blocks_temp.add(block_a)
            # Other goal predicates (like 'clear', 'arm-empty') are ignored for this heuristic logic

        # Store the set of goal blocks as a frozenset for efficiency
        self.goal_blocks: FrozenSet[str] = frozenset(goal_blocks_temp)


    def __call__(self, node) -> int:
        """
        Calculates the heuristic value for the state represented by the search node.

        Args:
            node: A search node object containing the state (`node.state`) for which
                  to compute the heuristic value. The state is expected to be a
                  frozenset of strings, where each string is a PDDL fact.

        Returns:
            An integer estimate of the number of actions required to reach the goal.
            Returns 0 if the state in the node already satisfies all goal conditions.
        """
        state: FrozenSet[str] = node.state

        # --- Pre-computation: Check if goal is already reached ---
        # If all goal facts are present in the current state, the cost-to-go is 0.
        if self.goals <= state:
           return 0

        # --- State Parsing ---
        heuristic_value: int = 0
        held_block: Optional[str] = None
        # Map: block_name -> name_of_block_below_or_'table' in the current state
        current_support: Dict[str, str] = {}
        # Set of blocks currently placed on the table or on other blocks
        blocks_in_current_config: Set[str] = set()

        # Iterate through the facts in the current state to understand the configuration
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            # Check for (on block_a block_b) state fact
            if predicate == "on" and len(parts) == 3:
                block_a, block_b = parts[1], parts[2]
                current_support[block_a] = block_b
                blocks_in_current_config.add(block_a)
            # Check for (on-table block_a) state fact
            elif predicate == "on-table" and len(parts) == 2:
                block_a = parts[1]
                current_support[block_a] = 'table'
                blocks_in_current_config.add(block_a)
            # Check for (holding block_a) state fact
            elif predicate == "holding" and len(parts) == 2:
                held_block = parts[1]
            # 'clear' and 'arm-empty' facts are ignored by this heuristic calculation

        # --- Heuristic Calculation ---

        # 1. Add cost if the arm is holding a block.
        # This represents the 1 action needed to put it down or stack it.
        if held_block is not None:
            heuristic_value += 1

        # 2. Add cost for blocks that are placed but are not in their goal position.
        # We need to consider all blocks that are currently placed OR are part of the goal configuration.
        all_relevant_blocks: Set[str] = blocks_in_current_config.union(self.goal_blocks)

        for block in all_relevant_blocks:
            # Skip the held block - its cost contribution (1) is already added.
            if block == held_block:
                continue

            # Get the target support for this block from the precomputed goal map.
            goal_pos: Optional[str] = self.goal_support.get(block)
            # Get the current support for this block from the parsed state map.
            # It will be None if the block is not currently on the table or another block.
            current_pos: Optional[str] = current_support.get(block)

            # Check if this block has a defined goal position AND is currently placed somewhere.
            if goal_pos is not None and current_pos is not None:
                # Check if the block's current support differs from its goal support.
                if current_pos != goal_pos:
                    # If misplaced, add 2 to the heuristic value (estimate for move).
                    heuristic_value += 2

        # Return the final calculated heuristic value.
        return heuristic_value

