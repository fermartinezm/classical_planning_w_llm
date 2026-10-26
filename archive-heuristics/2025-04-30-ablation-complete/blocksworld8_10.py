from heuristics.heuristic_base import Heuristic

# Helper function to extract components of a PDDL fact
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Helper function to get all objects from a set of facts
def get_objects_from_facts(facts):
    """Collect all unique arguments from a set of PDDL facts."""
    objects = set()
    for fact in facts:
        parts = get_parts(fact)
        # Add all arguments (skip predicate name)
        for part in parts[1:]:
            objects.add(part)
    return objects


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by counting two types of
    misplacements for each block:
    1. The block is not on its correct goal base (the block or table it should be directly on).
    2. The block has something on top of it that should not be there according to the goal
       (either a different block or it should be clear).

    # Assumptions
    - All blocks mentioned in the goal or initial state are the relevant objects.
    - Every block must be either on another block, on the table, or held by the arm
      in any valid state.
    - The goal specifies the desired base for blocks in goal stacks and on the table,
      and implies which block (if any) should be directly on top of another block
      in a goal stack, or if a block should be clear.

    # Heuristic Initialization
    - Collect all objects involved in the problem from the initial state and goal facts.
    - Build a map (`goal_base`) storing the desired base (block or 'table') for each block
      that appears as the first argument of an `(on ...)` goal or in an `(on-table ...)` goal.
    - Build a map (`goal_top`) storing the block that should be directly on top of another block
      according to `(on ...)` goals. If no block should be on top (i.e., the block should be clear
      or is the top of a goal stack), the value is 'clear'. This map is initialized to 'clear'
      for all objects and then updated based on `(on X Y)` goals.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Determine the current base for every block (the block or table it is on, or 'hand' if held).
       Iterate through state facts: `(on X Y)` means X is on Y, `(on-table X)` means X is on table,
       `(holding X)` means X is held. Store this in `current_base_map`.
    2. Determine the current top for every block (the block directly on top, or 'clear' if nothing is on top).
       Initialize `current_top_map` to 'clear' for all objects. Iterate through state facts:
       `(on X Y)` means X is on top of Y. Update `current_top_map[Y] = X`.
    3. Initialize heuristic value `h = 0`.
    4. For each object `obj` identified during initialization:
       a. Check Part 1 (Wrong Base): If `obj` has a specified goal base (i.e., `obj` is a key in `self.goal_base`)
          and its current base (`current_base_map.get(obj)`) is different from its goal base (`self.goal_base[obj]`),
          increment `h`. Note that a block not found in `current_base_map` implies an invalid state
          in Blocksworld (not on anything, table, or held), but the `.get(obj)` handles this safely.
       b. Check Part 2 (Wrong Top): If the current top of `obj` (`current_top_map.get(obj, 'clear')`)
          is different from its goal top (`self.goal_top[obj]`), increment `h`. The `.get(obj, 'clear')`
          handles blocks that are currently clear.
    5. Return `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal positions and relationships.
        """
        # Collect all objects from initial state and goal facts
        all_objects_set = set()
        all_objects_set.update(get_objects_from_facts(task.initial_state))
        all_objects_set.update(get_objects_from_facts(task.goals))
        self.all_objects = list(all_objects_set) # Store as list for consistent iteration order if needed

        # Build goal maps
        self.goal_base = {}
        # Initialize goal_top for all objects to 'clear' by default
        self.goal_top = {obj: 'clear' for obj in self.all_objects}

        for goal in task.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == 'on':
                obj, base = parts[1], parts[2]
                self.goal_base[obj] = base
                # If (on obj base) is a goal, then 'obj' should be on top of 'base'
                self.goal_top[base] = obj
            elif predicate == 'on-table':
                obj = parts[1]
                self.goal_base[obj] = 'table'
            # 'clear' goals are implicitly handled by the default 'clear' in goal_top.
            # If a block Y has a (clear Y) goal and no (on X Y) goal, goal_top[Y] remains 'clear'.
            # If it has both, the (on X Y) goal overrides the 'clear' requirement for goal_top[Y].

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Determine current base and top for all objects in the state
        current_base_map = {} # Map object -> its base ('table', 'hand', or another object)
        current_top_map = {obj: 'clear' for obj in self.all_objects} # Map object -> object on top, or 'clear'

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'on':
                obj, base = parts[1], parts[2]
                current_base_map[obj] = base
                current_top_map[base] = obj # obj is on top of base
            elif predicate == 'on-table':
                obj = parts[1]
                current_base_map[obj] = 'table'
            elif predicate == 'holding':
                obj = parts[1]
                current_base_map[obj] = 'hand'
            # 'clear' facts are not needed to build current_top_map, as 'clear' is the default.

        h = 0

        # Part 1: Count blocks not on their goal base
        for obj in self.all_objects:
            if obj in self.goal_base: # Check if goal base is specified for this object
                # Get current base. In a valid state, every object should have a base.
                current_base = current_base_map.get(obj)
                # If current_base is None, the object's location is not specified in the state,
                # which is an invalid Blocksworld state. Assuming valid states, current_base is never None here.
                if current_base != self.goal_base[obj]:
                     h += 1

        # Part 2: Count blocks with wrong block on top (or should be clear but aren't)
        for obj in self.all_objects:
            # goal_top is defined for all objects, defaulting to 'clear'.
            # current_top_map defaults to 'clear' if no block is on top.
            if current_top_map.get(obj, 'clear') != self.goal_top[obj]:
                 h += 1

        return h
