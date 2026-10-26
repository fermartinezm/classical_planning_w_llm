from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings defensively
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of blocks that are part of the goal
    configuration but are not in their correct position relative to the goal
    stack structure below them, or are supposed to be clear but are not.
    It counts how many blocks are in a "broken" segment of a goal stack,
    considering both the base below and the clear status if it's a goal top.

    # Assumptions
    - The goal state defines specific stacks of blocks on the table or on other blocks.
    - Every block mentioned as the first argument of an 'on' or 'on-table'
      predicate in the goal has a defined goal position.
    - Goal stacks are well-formed (no cycles).
    - Objects are defined using `(object obj)` facts in the static part, or
      can be inferred from initial state and goals.

    # Heuristic Initialization
    - Parses the goal conditions to determine the desired base (block or table)
      for each block that appears as the first argument of an 'on' or 'on-table' goal.
    - Identifies the set of all blocks that have an explicit goal position defined.
    - Identifies the set of blocks that are the top of a goal stack based on the goal definition.
    - Identifies all objects in the domain from the static facts (assuming `(object obj)` facts exist)
      or by inferring from initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. In `__init__`, process the task goals to create:
       - `self.goal_base`: mapping `block` to its desired base (`block` or 'table').
       - `self.blocks_with_goal_pos`: set of blocks that are keys in `self.goal_base`.
       - `self.goal_tops`: set of blocks in `self.blocks_with_goal_pos` that are not values in `self.goal_base`.
       Identify `self.all_objects` from static facts or initial state/goals.

    2. In `__call__`, for the given state:
       a. Determine the current base for every object in `self.all_objects`. Iterate through
          objects and check state facts for `(on obj X)`, `(on-table obj)`, or `(holding obj)`.
          Store this in a dictionary `current_base`.
       b. Define a recursive helper function `is_correctly_stacked(block)` with memoization:
          - Base case 1: If `block` is not in `self.blocks_with_goal_pos`, return `True`
            (blocks not in the goal configuration don't need a specific position).
          - If the result for `block` is already in the memoization dictionary, return it.
          - Get `goal_b = self.goal_base.get(block)`.
          - Get `current_b = current_base.get(block, 'unknown')`.
          - Check if `current_b` matches `goal_b`. If not, store `False` in memo and return `False`.
          - If `current_b` matches `goal_b`:
            - Initialize `recursive_ok = True`. If `goal_b` is a block (not 'table'), set `recursive_ok = is_correctly_stacked(goal_b)`.
            - Initialize `clear_ok = True`. If `block` is in `self.goal_tops` and `(clear block)` is not in the state, set `clear_ok = False`.
            - The block is correctly stacked if `recursive_ok` is `True` AND `clear_ok` is `True`.
            - Store the result in memo and return it.
       c. Initialize `total_cost = 0`.
       d. Iterate through each `block` in `self.blocks_with_goal_pos`.
       e. If `is_correctly_stacked(block)` returns `False`, increment `total_cost` by 1.
       f. Return `total_cost`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal stack structure and objects.
        """
        self.goals = task.goals
        self.static_facts = task.static

        self.goal_base = {}
        self.blocks_with_goal_pos = set()

        # Extract goal base for each relevant block
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'on':
                if len(parts) == 3:
                    block, base = parts[1], parts[2]
                    self.goal_base[block] = base
                    self.blocks_with_goal_pos.add(block)
            elif predicate == 'on-table':
                 if len(parts) == 2:
                    block = parts[1]
                    self.goal_base[block] = 'table'
                    self.blocks_with_goal_pos.add(block)
            # Ignore other goal predicates like 'clear' and 'arm-empty' here;
            # 'clear' is handled in is_correctly_stacked based on goal structure.

        # Identify blocks that are the top of a goal stack
        goal_bases_used = set(self.goal_base.values())
        # Remove 'table' from the set of bases used by blocks
        goal_bases_used.discard('table')
        self.goal_tops = {b for b in self.blocks_with_goal_pos if b not in goal_bases_used}


        # Identify all objects in the domain
        self.all_objects = set()
        # Objects are typically defined with the 'object' predicate in static facts
        for fact in self.static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'object' and len(parts) == 2:
                 self.all_objects.add(parts[1])

        # Fallback: If 'object' facts are not present, infer objects from init/goals
        if not self.all_objects:
             # Collect all arguments from initial state and goals
             all_args = set()
             for fact in task.initial_state | task.goals:
                 parts = get_parts(fact)
                 if not parts: continue
                 all_args.update(parts[1:])

             # Remove known predicates and constants
             # Add all predicate names from the domain file
             known_non_objects = {'arm-empty', 'table', 'clear', 'on-table', 'on', 'holding', 'object',
                                  'pickup', 'putdown', 'stack', 'unstack'} # Add action names too just in case
             self.all_objects = {arg for arg in all_args if arg not in known_non_objects}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        Counts blocks that are part of the goal configuration but are not
        correctly stacked relative to their goal base and the stack below,
        or are supposed to be clear but are not.
        """
        state = node.state

        # 2a. Determine the current base for every object
        current_base = {}
        for obj in self.all_objects:
            found_base = False
            # Check state facts to find the object's current location
            if f'(on-table {obj})' in state:
                current_base[obj] = 'table'
                found_base = True
            elif f'(holding {obj})' in state:
                current_base[obj] = 'arm'
                found_base = True
            else:
                 # Check if it's on another block
                 for fact in state:
                     parts = get_parts(fact)
                     if not parts: continue
                     if parts[0] == 'on' and len(parts) == 3 and parts[1] == obj:
                         current_base[obj] = parts[2]
                         found_base = True
                         break # Found the base, move to next object

            # If not found, something is wrong with the state representation or my understanding.
            # In a valid blocksworld state, every object must be on something, on the table, or held.
            # If found_base is False here, it indicates an issue with the state representation.
            # For robustness, we could default, but asserting might be better for debugging.
            # assert found_base, f"Block {obj} has no defined base/location in state {state}"
            if not found_base:
                 # This case should ideally not happen in valid states.
                 # Assigning 'unknown' might prevent crashes but could lead to incorrect heuristic values.
                 # Let's assume valid states where every object has one of the three base predicates true.
                 # If it's not found, it implies the object is not in the state in a way we understand.
                 # It won't have a current_base entry, and current_base.get(block, 'unknown') will handle it.
                 pass # No base found for this object in the state


        # 2b. Define recursive helper with memoization
        memo = {}
        def is_correctly_stacked(block):
            # Base case 1: Blocks not in the goal configuration are considered 'correct'
            # from the perspective of needing to reach a specific goal position.
            if block not in self.blocks_with_goal_pos:
                return True

            # Check memoization
            if block in memo:
                return memo[block]

            # Get goal and current base
            goal_b = self.goal_base.get(block) # Should exist if block is in blocks_with_goal_pos
            current_b = current_base.get(block, 'unknown') # Use get for safety

            # Base case 2: Current base does not match goal base
            if current_b != goal_b:
                memo[block] = False
                return False

            # Current base matches goal base. Check recursive condition and clear condition.

            # Check recursive condition (base below)
            recursive_ok = True
            if goal_b != 'table': # If base is a block A
                # Check if A is correctly stacked.
                # If A is not in blocks_with_goal_pos, is_correctly_stacked(A) will return True by Base case 1.
                recursive_ok = is_correctly_stacked(goal_b)

            # Check clear condition if this block is a goal top
            clear_ok = True
            if block in self.goal_tops:
                 # Check if the block is currently clear
                 if f'(clear {block})' not in state:
                     clear_ok = False # It's not clear when it should be

            # The block is correctly stacked if both conditions are met
            result = recursive_ok and clear_ok
            memo[block] = result
            return result

        # 2c, d, e, f. Calculate total cost
        total_cost = 0
        # Only consider blocks that have a specific goal position
        for block in self.blocks_with_goal_pos:
            if not is_correctly_stacked(block):
                total_cost += 1 # Count each block that is part of a broken goal stack segment

        return total_cost
