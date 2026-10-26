from fnmatch import fnmatch
import collections # Used for list as a queue, though deque is more efficient for large problems

# Assuming heuristic_base.py exists and defines a Heuristic base class
# If running standalone for testing or demonstration, you might need a dummy class:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         raise NotImplementedError

# Assuming task.py exists and defines Task and Operator classes
# If running standalone for testing or demonstration, you might need dummy classes:
# class Task:
#     def __init__(self, name, facts, initial_state, goals, operators, static):
#         self.name = name
#         self.facts = facts
#         self.initial_state = initial_state
#         self.goals = goals
#         self.operators = operators
#         self.static = static
#     def goal_reached(self, state):
#         return self.goals.issubset(state)
#
# class Operator:
#      def __init__(self, name, preconditions, add_effects, del_effects):
#          self.name = name
#          self.preconditions = frozenset(preconditions)
#          self.add_effects = frozenset(add_effects)
#          self.del_effects = frozenset(del_effects)
#      def applicable(self, state):
#          return self.preconditions <= state
#      def apply(self, state):
#          return (state - self.del_effects) | self.add_effects


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact string or invalid format gracefully
    if not fact or not isinstance(fact, str) or fact.strip() == "" or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Estimates the number of actions needed to achieve the goal state.
    The heuristic counts blocks that are misplaced relative to their goal base,
    blocks currently stacked on top of those, and blocks blocking goal-clear
    conditions, estimating 2 actions per block movement needed.

    Heuristic value is 0 iff the state is a goal state.
    Heuristic value is finite for solvable states.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals

        # Parse goal facts to build goal structure
        self.goal_below = {} # block -> block_below or 'table'
        # self.goal_above = {} # block_below -> block (not strictly needed for this heuristic logic)
        self.goal_blocks = set() # All blocks mentioned in goal 'on' or 'on-table'
        self.goal_clear = set() # Blocks that must be clear in the goal

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue # Skip invalid facts
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                obj, underob = parts[1], parts[2]
                self.goal_below[obj] = underob
                # self.goal_above[underob] = obj
                self.goal_blocks.add(obj)
                self.goal_blocks.add(underob)
            elif predicate == 'on-table' and len(parts) == 2:
                obj = parts[1]
                self.goal_below[obj] = 'table'
                self.goal_blocks.add(obj)
            elif predicate == 'clear' and len(parts) == 2:
                obj = parts[1]
                self.goal_clear.add(obj)
            # Ignore 'arm-empty' goal if present

        # Identify all objects in the domain (from initial state and goals)
        # This is useful for building current_above map correctly
        self.all_objects = set(self.goal_blocks)
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             # Add objects mentioned in state facts
             if predicate in ['on', 'on-table', 'holding', 'clear'] and len(parts) > 1:
                 self.all_objects.add(parts[1])
             if predicate == 'on' and len(parts) == 3:
                 self.all_objects.add(parts[2])


    def __call__(self, node):
        """Compute the domain-dependent heuristic value for the given state."""
        state = node.state

        # Check if goal is reached
        if self.goals.issubset(state):
             return 0

        # Parse current state facts
        current_below = {} # block -> block_below or 'table' or 'hand'
        current_above = {obj: None for obj in self.all_objects} # block -> block_above (None if clear)
        current_clear = set() # Blocks that are clear
        # current_holding = None # Block currently held

        # Build current_below and current_above
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'on' and len(parts) == 3:
                obj, underob = parts[1], parts[2]
                current_below[obj] = underob
                current_above[underob] = obj # This block is on top of underob
            elif predicate == 'on-table' and len(parts) == 2:
                obj = parts[1]
                current_below[obj] = 'table'
                # Nothing is above the table in this representation
            elif predicate == 'holding' and len(parts) == 2:
                obj = parts[1]
                current_below[obj] = 'hand' # Represent being held
                # current_holding = obj
                # Nothing is above a held object
            elif predicate == 'clear' and len(parts) == 2:
                 obj = parts[1]
                 current_clear.add(obj) # Store clear facts explicitly


        # Step 1: Identify blocks in the goal stack whose required base position is not met in the state.
        # These are blocks 'b' where the goal requires (on b a) or (on-table b), but that fact is not in the state.
        misplaced_base_blocks = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: continue
            predicate = parts[0]
            if (predicate == 'on' and len(parts) == 3):
                obj, underob = parts[1], parts[2]
                # Check if (on obj underob) is true in state
                if goal not in state:
                     misplaced_base_blocks.add(obj)
            elif (predicate == 'on-table' and len(parts) == 2):
                 obj = parts[1]
                 # Check if (on-table obj) is true in state
                 if goal not in state:
                     misplaced_base_blocks.add(obj)


        # Step 2: Identify all blocks that need moving. This includes blocks with a misplaced base
        # and any blocks currently stacked on top of them.
        # Use a queue for breadth-first traversal upwards from misplaced blocks.
        needs_moving = set(misplaced_base_blocks)
        processing_queue = list(misplaced_base_blocks) # Use list as a queue

        added_to_needs_moving = set(misplaced_base_blocks) # Track blocks already added

        while processing_queue:
            current_block = processing_queue.pop(0)
            # Find the block directly on top of current_block in the state
            block_on_top = current_above.get(current_block)

            if block_on_top is not None and block_on_top not in added_to_needs_moving:
                 needs_moving.add(block_on_top)
                 added_to_needs_moving.add(block_on_top)
                 processing_queue.append(block_on_top)


        # Step 3: Identify blocks that are blocking goal-clear conditions but are not
        # already included in the 'needs_moving' set.
        blocks_blocking_goal_clear = set()
        for b in self.goal_clear:
            # If b is a goal-clear block but is not clear in the state
            # A block is clear if nothing is on top of it.
            block_on_top = current_above.get(b)

            if block_on_top is not None: # b is not clear
                # If the block on top is not already marked for moving
                if block_on_top not in needs_moving:
                    blocks_blocking_goal_clear.add(block_on_top)


        # Step 4: Calculate the heuristic cost.
        # Each block in 'needs_moving' needs at least 2 actions (pickup/unstack + stack/putdown).
        # Each block in 'blocks_blocking_goal_clear' needs at least 2 actions (unstack + putdown).
        # This is a non-admissible estimate.
        heuristic_cost = 2 * len(needs_moving) + 2 * len(blocks_blocking_goal_clear)

        return heuristic_cost

