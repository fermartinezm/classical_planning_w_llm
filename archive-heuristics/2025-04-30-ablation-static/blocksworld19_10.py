# from heuristics.heuristic_base import Heuristic
# Assuming Heuristic base class is available as described in the problem context.
# If running standalone, you might need a dummy Heuristic class definition.

from fnmatch import fnmatch

# Define a dummy Heuristic base class if the actual one is not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy base class for heuristic."""
        def __init__(self, task):
            self.task = task
            self._process_static_facts(task.static)
            self._process_goals(task.goals)

        def _process_static_facts(self, static_facts):
            pass # To be implemented by domain heuristic

        def _process_goals(self, goals):
            pass # To be implemented by domain heuristic

        def __call__(self, node):
            raise NotImplementedError("Heuristic __call__ method not implemented.")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or invalid fact format gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# The match function from examples is useful but not strictly needed
# if we parse facts manually as done in the heuristic. Keeping it for completeness
# or potential future use, but the heuristic logic relies on get_parts directly.
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#     """
#     parts = get_parts(fact)
#     if len(parts) != len(args):
#         return False
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting structural mismatches in the block stacks compared to the goal
    configuration. It penalizes blocks that are not on their correct base,
    blocks that have the wrong block on top, goal-specified top blocks
    that are not clear, and the arm not being empty if required by the goal.

    # Assumptions
    - The goal specifies the desired stack configuration using `on` and `on-table`
      predicates for all blocks that are part of a goal stack.
    - `clear` goal predicates are typically only specified for blocks that are the intended
      top of a goal stack.
    - Standard Blocksworld actions (pickup, putdown, stack, unstack) are used.
    - Only one block can be held at a time.
    - Only one block can be directly on top of another block.
    - Multiple blocks can be on the table.
    - The state representation is consistent (e.g., a block is either on another block,
      on the table, or held, but not more than one at a time).

    # Heuristic Initialization
    - Parses the goal facts to determine the desired base for each block
      (`goal_stack_map`) and identify blocks that should be clear
      (`goal_clear_blocks`).
    - Identifies blocks that are the intended top of a goal stack (`goal_top_blocks`).
    - Collects all unique objects (blocks) mentioned in the initial state and goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Build the current stack configuration: Determine which block is on which base
       (`current_stack_map`) and which block is directly on top of another block
       or the arm (`current_block_on_top_map`).
    2. Initialize heuristic value `h = 0`.
    3. Iterate through all known objects (blocks):
       a. If the block `B` is part of a goal stack (i.e., is a key in `goal_stack_map`):
          i. Check if `B` is on its correct goal base (`goal_stack_map[B]`). If not, increment `h`.
          ii. If `B` *is* on its correct goal base, check if the block currently on top of `B`
              (`current_block_on_top_map.get(B)`) is the block that should be on top of `B`
              according to the goal stack (`GoalTop`, found by searching `goal_stack_map`).
              If the block on top is incorrect (wrong block, or any block when it should be clear),
              increment `h`.
    4. Iterate through blocks `B` that are the intended top of a goal stack (`goal_top_blocks`):
       a. If `(clear B)` is false in the current state, increment `h`. This penalizes
          extra blocks on top of goal stacks.
    5. If `(arm-empty)` is a goal fact and the arm is not empty in the current state,
       increment `h`.

    6. Return the total heuristic value `h`.
    """

    def _process_static_facts(self, static_facts):
        """
        Process static facts from the task.
        Blocksworld domain provided has no static facts defined with :static.
        """
        # No static facts to process in this domain based on the provided file.
        pass

    def _process_goals(self, goals):
        """
        Process goal facts to build goal_stack_map, goal_clear_blocks,
        and identify goal_top_blocks.
        """
        self.goals = goals
        self.goal_stack_map = {} # block -> base ('table' or block name)
        self.goal_clear_blocks = set() # set of blocks that must be clear

        # Collect all objects mentioned in the goal
        goal_objects = set()

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block, base = args
                self.goal_stack_map[block] = base
                goal_objects.add(block)
                goal_objects.add(base)
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_stack_map[block] = 'table'
                goal_objects.add(block)
            elif predicate == 'clear' and len(args) == 1:
                block = args[0]
                self.goal_clear_blocks.add(block)
                goal_objects.add(block)
            # Ignore 'arm-empty' in goals for building stack maps

        # Identify goal_top_blocks (blocks X in goal_stack_map where no Y has goal_stack_map[Y] == X)
        blocks_that_are_bases_in_goal = set(self.goal_stack_map.values())
        # 'table' is a base but not a block
        blocks_that_are_bases_in_goal.discard('table')

        self.goal_top_blocks = set()
        for block in self.goal_stack_map:
            if block not in blocks_that_are_bases_in_goal:
                 self.goal_top_blocks.add(block)

        # Collect all unique objects from initial state and goal
        all_objects = set(goal_objects)
        for fact in self.task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             # Assume any argument that is not a predicate name is an object
             all_objects.update(parts[1:]) # Add all arguments except the predicate name

        # Remove known non-object terms that might appear as arguments
        all_objects.discard('table')
        all_objects.discard('held') # 'held' is a state, not an object or location
        all_objects.discard('arm-empty') # 'arm-empty' is a state, not an object

        self.all_objects = frozenset(all_objects)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings)

        # 1. Build current stack configuration
        current_stack_map = {} # block -> base ('table' or block name or 'held')
        # Initialize all potential bases (blocks + 'table' + 'held') to have nothing on top
        all_possible_bases = set(self.all_objects) | {'table', 'held'}
        current_block_on_top_map = {base: None for base in all_possible_bases}

        # First pass to find immediate base for each block
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block, base = args
                current_stack_map[block] = base
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                current_stack_map[block] = 'table'
            elif predicate == 'holding' and len(args) == 1:
                block = args[0]
                current_stack_map[block] = 'held'

        # Second pass to build the inverse map (what's on top of what)
        # For each block, the block it's on has this block on top
        for block, base in current_stack_map.items():
             # Ensure the base is a valid key in the map (it should be if it's an object, 'table', or 'held')
             if base in current_block_on_top_map:
                 current_block_on_top_map[base] = block
             # Note: This correctly handles only one block being on another block or held.
             # It doesn't explicitly track multiple blocks on the table, but the heuristic
             # only needs to know *if* something is on a base (a block or 'held'), and if so, *which* block.

        # Get current clear blocks
        current_clear_blocks = {get_parts(fact)[1] for fact in state if get_parts(fact) and get_parts(fact)[0] == "clear"}


        total_cost = 0

        # 2. Iterate through all objects and check goal stack positions
        for block in self.all_objects:
            # Only consider blocks that have a specified goal position
            if block in self.goal_stack_map:
                goal_base = self.goal_stack_map[block]
                current_base = current_stack_map.get(block) # None if block is not on/on-table/held in state

                # If block is not found in current_stack_map, it's likely an inconsistency
                # or the block is not in the initial state. Assume it's misplaced.
                # In a valid state, every block is either on-table, on another block, or held.
                # So current_stack_map should contain all blocks from self.all_objects.
                # If current_base is None, it implies the block is not in a standard location/state.
                # This should ideally not happen in valid Blocksworld states.
                # We'll treat it as misplaced relative to any goal base.
                if current_base is None:
                     total_cost += 1 # Penalize block in unknown state
                     continue # Cannot check block on top if base is unknown


                # Check 1: Is the block on the correct base?
                if current_base != goal_base:
                    total_cost += 1

                # Check 2: If the block is on the correct base, is the block on top correct?
                # This check is only relevant if the block is on its correct base.
                if current_base == goal_base:
                    current_top = current_block_on_top_map.get(block) # What is currently on top of 'block'

                    # Find the block that *should* be on top of 'block' according to the goal stack
                    goal_top = None
                    # Iterate through goal_stack_map to find the block whose goal base is 'block'
                    for potential_goal_top, base in self.goal_stack_map.items():
                        if base == block:
                            goal_top = potential_goal_top
                            break # Assuming only one block can be on another in the goal stack

                    # If the block on top is not the goal_top (either wrong block or something when it should be clear)
                    if current_top != goal_top:
                        total_cost += 1

        # 3. Penalize goal-top blocks that are not clear
        for block in self.goal_top_blocks:
             # Check if the block is in the state and is not clear
             # A block is not clear if it's held or has something on it.
             # We can check if it's clear by looking at current_clear_blocks.
             if block in self.all_objects and block not in current_clear_blocks:
                 total_cost += 1

        # 4. Penalize if arm-empty is a goal and arm is not empty
        if "(arm-empty)" in self.goals and "(arm-empty)" not in state:
             total_cost += 1

        # The heuristic is 0 iff total_cost is 0.
        # total_cost is 0 iff:
        # - For every block in goal_stack_map:
        #   - It's on the correct base.
        #   - It has the correct block on top (or nothing if it's a goal_top_block).
        # - For every goal_top_block, it is clear.
        # - If (arm-empty) is a goal, then (arm-empty) is true in the state.
        # This set of conditions precisely matches the typical conjunction of goal facts
        # in Blocksworld problems ((on X Y), (on-table Z), (clear W), (arm-empty)).
        # Therefore, h=0 iff the state is the goal state.

        return total_cost
