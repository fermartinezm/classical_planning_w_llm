from fnmatch import fnmatch
# Assuming Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

# The match function from the example heuristics
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # The original example match function doesn't strictly check length.
    # We follow the example's implementation.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal
    by counting how many blocks are not in their correct position relative
    to their base (the block below them or the table), and how many blocks
    are on their correct base but have the wrong block directly on top.

    # Assumptions
    - The goal specifies the desired configuration of blocks in stacks or on the table.
    - The goal predicates primarily consist of `(on ?x ?y)`, `(on-table ?x)`, and `(clear ?x)`.
    - The heuristic counts two types of "misplacements" for each block:
        1. Being on the wrong base (block, table, or arm).
        2. Being on the correct base, but having the wrong block directly on top (or should be clear but isn't, or shouldn't be clear but is).
    - Each such misplacement is estimated to cost 1 action. This is a simplification;
      moving a block typically costs 2 actions (unstack/pickup + stack/putdown),
      and clearing blocks on top adds more cost. However, this simplified count
      often correlates well with the distance to the goal in Blocksworld.

    # Heuristic Initialization
    - Extract the goal configuration to determine the desired base and the desired
      block directly on top for every block. This information is stored in
      `self.goal_base` and `self.goal_on` dictionaries.
    - Identify all blocks involved in the problem from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For the current state, determine the current base and the current block
       directly on top for every block. Store this in `current_base` and
       `current_on` dictionaries. The table is treated as a special base "Table",
       and the arm as a special base "Arm".
    2. Initialize the heuristic value `h` to 0.
    3. Iterate through each block in the problem:
       a. Get the block's current base (`current_base[block]`) and goal base (`self.goal_base[block]`).
       b. If the current base is different from the goal base, increment `h` by 1.
       c. If the current base is the same as the goal base:
          i. Get the block's current top (`current_on[block]`) and goal top (`self.goal_on[block]`).
          ii. If the current top is different from the goal top, increment `h` by 1.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal configuration.

        @param task: The planning task object.
        """
        # Call the base class constructor (if needed, depending on Heuristic definition)
        # super().__init__(task)

        self.goals = task.goals # Access goals from the task object

        # Determine all objects (blocks) in the domain.
        # Collect all unique arguments from all facts in initial state and goals.
        all_args = set()
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             if parts:
                 all_args.update(parts[1:]) # Add all arguments

        # Filter out arguments that are predicate names.
        predicate_names = {'clear', 'on-table', 'arm-empty', 'holding', 'on'}
        self.blocks = sorted(list(arg for arg in all_args if arg not in predicate_names))

        # Map each block to its goal base (the block it should be on, or "Table").
        # Initialize all blocks to be on the table in the goal by default,
        # then update based on (on ?x ?y) goals.
        self.goal_base = {block: "Table" for block in self.blocks}
        # Map each block to the block that should be directly on top of it in the goal.
        # Initialize all blocks to have nothing on top (None) in the goal,
        # then update based on (on ?x ?y) goals.
        self.goal_on = {block: None for block in self.blocks}

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            if parts[0] == "on" and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                # Ensure they are recognized blocks
                if block_on_top in self.blocks and block_below in self.blocks:
                    self.goal_base[block_on_top] = block_below
                    self.goal_on[block_below] = block_on_top
            # (on-table ?x) goals are handled by the default initialization
            # (clear ?x) goals are handled by the default initialization of goal_on (None)

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.

        @param node: The search node containing the current state.
        @return: The estimated cost (heuristic value).
        """
        state = node.state

        # Determine the current base and block on top for each block in the current state.
        # Initialize all blocks to be on the table by default, and nothing on top.
        current_base = {block: "Table" for block in self.blocks}
        current_on = {block: None for block in self.blocks}
        # The arm's state is implicitly handled by tracking which block is held.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                block_on_top, block_below = parts[1], parts[2]
                if block_on_top in self.blocks and block_below in self.blocks:
                     current_base[block_on_top] = block_below
                     current_on[block_below] = block_on_top
            elif predicate == "on-table" and len(parts) == 2:
                 block = parts[1]
                 if block in self.blocks:
                     current_base[block] = "Table" # Explicitly set, though default is Table
            elif predicate == "holding" and len(parts) == 2:
                 block = parts[1]
                 if block in self.blocks:
                     # If a block is held, its base is the "Arm".
                     current_base[block] = "Arm"
            # (clear ?x) and (arm-empty) are state properties not directly defining base/top relations for blocks.

        # Calculate the heuristic value.
        h = 0
        for block in self.blocks:
            current_b = current_base.get(block)
            goal_b = self.goal_base.get(block)

            # If the block is held, its current base is "Arm".
            # The goal base is never "Arm" in standard Blocksworld.
            # So, if current_b is "Arm", it will always be different from goal_b,
            # contributing +1 for being on the wrong base.
            # The second condition (checking current_on vs goal_on) is skipped
            # because current_b != goal_b. This is the desired behavior.
            if current_b != goal_b:
                 h += 1
            else: # current_b == goal_b (implies current_b is not "Arm")
                 # Block is on the correct base (block or table), check the block on top
                 current_t = current_on.get(block)
                 goal_t = self.goal_on.get(block)
                 if current_t != goal_t:
                     # Block has the wrong block directly on top (or should be clear but isn't, etc.)
                     h += 1

        return h
