from heuristics.heuristic_base import Heuristic # Assuming this exists

def get_parts(fact):
    """Helper to parse a PDDL fact string into a list of parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
        This heuristic estimates the cost to reach the goal by counting the
        number of blocks that are not in their "correct" position relative
        to the block (or table) immediately below them in the goal state,
        plus penalties for unsatisfied goal `clear` conditions for goal
        stack top blocks and for an unsatisfied `arm-empty` goal.
        A block is considered "correctly placed" if it is on the table and
        should be on the table according to the goal, OR if it is on another
        block Y, should be on Y according to the goal, and Y is itself
        correctly placed. The core heuristic value is the total number of blocks
        minus the number of correctly placed blocks.

    Assumptions:
        - The PDDL domain is Blocksworld as provided.
        - Goal states consist of `(on X Y)` and `(on-table Z)` predicates
          forming one or more stacks, and potentially `(clear W)` predicates
          for the top blocks of goal stacks, and optionally `(arm-empty)`.
        - The heuristic is used with a greedy best-first search and does not
          need to be admissible.
        - The internal state representation is a frozenset of strings.
        - The task object provides initial_state, goals, and static facts.

    Heuristic Initialization:
        The constructor processes the goal facts to build data structures
        representing the desired stack configuration:
        - `goal_on_map`: A dictionary mapping a block `Y` to the block `X`
                         that should be directly on top of it according to
                         the goal (`(on X Y)` is a goal).
        - `goal_on_table_set`: A set of blocks that should be directly on
                               the table according to the goal (`(on-table Z)`
                               is a goal).
        - `goal_clear_set`: A set of blocks that should be clear according
                            to the goal (`(clear W)` is a goal).
        It also determines the total number of blocks in the problem instance
        by examining the initial state.

    Step-By-Step Thinking for Computing Heuristic:
        1. Parse the current state facts to build data structures representing
           the current configuration:
           - `current_on_map`: A dictionary mapping a block `Y` to the block `X`
                               currently directly on top of it (`(on X Y)` is
                               in the state).
           - `current_on_table_set`: A set of blocks currently directly on the
                                     table (`(on-table Z)` is in the state).
           - `current_clear_set`: A set of blocks currently clear (`(clear W)`
                                  is in the state).
           - `arm_is_empty`: Boolean indicating if the arm is empty.
        2. Identify the set of "correctly placed" blocks based on the goal
           stack structure from the table up. This is computed iteratively:
           - Initialize an empty set `correctly_placed`.
           - Initialize a queue with all blocks `Z` that are in the `goal_on_table_set`
             AND are also currently on the table (`Z` in `current_on_table_set`).
             These blocks form the base of correctly placed stack segments.
           - Add these initial blocks to the `correctly_placed` set.
           - Use a set `queued_set` to keep track of blocks already added to the queue
             to avoid redundant processing. Initialize `queued_set` with blocks
             initially added to `correctly_placed`.
           - While the queue is not empty:
             - Dequeue a block `Y` that is known to be correctly placed.
             - Check if there is a block `X` that should be on `Y` according
               to the goal (`goal_on_map.get(Y)`).
             - If such a block `X` exists, check if `X` is currently directly
               on `Y` in the state (`current_on_map.get(Y) == X`).
             - If both conditions are true, then `X` is correctly placed on `Y`.
               Add `X` to the `correctly_placed` set. If `X` has not been queued
               before, enqueue `X` and add it to `queued_set`.
        3. The core heuristic value is the total number of blocks minus the number
           of blocks in the `correctly_placed` set. This counts the blocks
           that are not part of any correctly built stack segment from the
           table up according to the goal.
        4. Add a penalty for unsatisfied `(clear X)` goals where `X` is a block
           that is the top of a goal stack (i.e., `(clear X)` is a goal and no
           block should be on `X` in the goal). Count how many such `X` are
           not in the `current_clear_set`.
        5. Add a penalty of 1 if `(arm-empty)` is a goal and the arm is not empty
           in the current state.
        6. The final heuristic value is the sum of the core value and the penalties.
           This value is 0 if and only if the state is a goal state (assuming
           typical blocksworld goals).
    """
    def __init__(self, task):
        self.goals = task.goals
        # static_facts = task.static # Blocksworld has no static facts

        # Build goal stack structure and identify goal clear facts
        self.goal_on_map = {}
        self.goal_on_table_set = set()
        self.goal_clear_set = set()
        self.goal_has_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                block_on = parts[1]
                block_under = parts[2]
                self.goal_on_map[block_under] = block_on
            elif predicate == "on-table":
                block = parts[1]
                self.goal_on_table_set.add(block)
            elif predicate == "clear":
                 block = parts[1]
                 self.goal_clear_set.add(block)
            elif predicate == "arm-empty":
                 self.goal_has_arm_empty = True

        # Determine total number of blocks from the initial state
        self.total_blocks = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == "on":
                self.total_blocks.add(parts[1])
                self.total_blocks.add(parts[2])
            elif parts[0] == "on-table":
                self.total_blocks.add(parts[1])
            elif parts[0] == "holding":
                 self.total_blocks.add(parts[1])
            # clear and arm-empty don't introduce new objects

        self.num_total_blocks = len(self.total_blocks)

        # Identify goal top blocks (blocks that should be clear and nothing should be on them in the goal)
        self.goal_top_blocks = self.goal_clear_set - set(self.goal_on_map.keys())


    def __call__(self, node):
        state = node.state

        # Build current state configuration
        current_on_map = {}
        current_on_table_set = set()
        current_clear_set = set()
        arm_is_empty = False

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                block_on = parts[1]
                block_under = parts[2]
                current_on_map[block_under] = block_on
            elif predicate == "on-table":
                block = parts[1]
                current_on_table_set.add(block)
            elif predicate == "clear":
                 block = parts[1]
                 current_clear_set.add(block)
            elif predicate == "arm-empty":
                 arm_is_empty = True
            # Ignore holding, it's implicitly not arm-empty

        # Find correctly placed blocks using the recursive definition
        correctly_placed = set()
        queue = []

        # Base cases: blocks that should be on the table and are on the table
        for block in self.goal_on_table_set:
            if block in current_on_table_set:
                queue.append(block)
                correctly_placed.add(block)

        # Recursive step: blocks correctly placed on top of correctly placed blocks
        # Use a set for visited nodes in the queue to avoid redundant processing.
        queued_set = set(correctly_placed)

        while queue:
            under_block = queue.pop(0) # Use pop(0) for queue behavior

            # Check if there's a block X that should be on under_block in the goal
            block_on_goal = self.goal_on_map.get(under_block)

            if block_on_goal:
                # Check if that block X is currently on under_block in the state
                block_on_current = current_on_map.get(under_block)

                if block_on_current == block_on_goal:
                    # This block is correctly placed on the correctly placed block below it
                    if block_on_goal not in correctly_placed:
                         correctly_placed.add(block_on_goal)
                         if block_on_goal not in queued_set: # Avoid adding to queue multiple times
                            queue.append(block_on_goal)
                            queued_set.add(block_on_goal)


        # Core heuristic: number of blocks not correctly placed
        h_value = self.num_total_blocks - len(correctly_placed)

        # Penalty for unsatisfied clear goals for goal top blocks
        unsatisfied_clear_penalty = 0
        for block in self.goal_top_blocks:
             if block not in current_clear_set:
                  unsatisfied_clear_penalty += 1

        # Penalty for unsatisfied arm-empty goal
        arm_empty_penalty = 0
        if self.goal_has_arm_empty and not arm_is_empty:
             arm_empty_penalty = 1

        # Total heuristic value
        h_value += unsatisfied_clear_penalty + arm_empty_penalty

        # Ensure heuristic is 0 only for goal states.
        # If the state is the goal state, all goal facts are true.
        # This means all goal (on X Y) and (on-table Z) are true, so all blocks are correctly placed.
        # All goal (clear W) for top blocks are true, so unsatisfied_clear_penalty is 0.
        # If (arm-empty) is a goal, it's true in the goal state, so arm_empty_penalty is 0.
        # Thus, h_value is 0 at the goal.
        # If h_value is 0, it implies all components are 0, which means all goal facts (on, on-table, clear for top, arm-empty if goal) are true.
        # This covers typical Blocksworld goals.

        return h_value
