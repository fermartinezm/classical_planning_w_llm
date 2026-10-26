from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with valid PDDL fact strings,
        # but adding a safeguard or handling malformed input might be necessary
        # depending on the robustness required. Assuming valid format for now.
        pass

    # Remove parentheses and split by whitespace
    return fact[1:-1].split()

class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal state
    by summing three components:
    1. The number of blocks that are not in their final goal position (recursively defined).
    2. The number of blocks that are currently stacked on top of a block that must be clear in the goal state.
    3. A cost of 1 if the arm is not empty and the goal requires it to be empty.

    # Assumptions
    - The goal state is defined by a set of `(on ?x ?y)`, `(on-table ?x)`, `(clear ?x)`, and `(arm-empty)` facts.
    - The heuristic assumes standard Blocksworld actions and constraints.
    - It does not guarantee admissibility but aims to be informative for greedy best-first search.

    # Heuristic Initialization
    The heuristic is initialized by parsing the goal conditions from the task:
    - `goal_pos_map`: A dictionary mapping each block whose final position is specified in the goal to its required base ('table' or another block).
    - `blocks_that_must_be_clear_in_goal`: A set of blocks that must have nothing on top of them in the goal state (derived from `(clear ?x)` goal facts).
    - `goal_arm_empty`: A boolean indicating whether the arm must be empty in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the state is the goal state using `task.goal_reached(state)`. If it is, the heuristic value is 0.
    2. Initialize the heuristic value `h` to 0.
    3. Calculate the number of blocks that are not in their final goal position:
       - Define a recursive helper function `is_correctly_placed(block)` that uses memoization.
       - A block is correctly placed if its final position is not specified in the goal (i.e., it's not a key in `goal_pos_map`), OR if it is currently on its required goal base/table AND its goal base is also correctly placed (recursively checked).
       - Iterate through all blocks whose final position is specified in `goal_pos_map` (i.e., the keys of `goal_pos_map`).
       - For each such block, if `is_correctly_placed(block)` returns False, increment a counter `misplaced_goal_blocks_count`.
       - Add `misplaced_goal_blocks_count` to `h`.
    4. Calculate the cost associated with clearing blocks that must be clear in the goal:
       - Build a map `current_on_map` representing the current `(on X Y)` relationships (mapping Y to X).
       - Define a helper function `count_blocks_above(block)` that uses `current_on_map` to count the number of blocks stacked directly or indirectly on `block`.
       - Iterate through the set `blocks_that_must_be_clear_in_goal`.
       - For each block Y in this set, calculate `count_blocks_above(Y)` and add this count to a `clearing_cost` counter.
       - Add `clearing_cost` to `h`.
    5. Check the arm state:
       - If `goal_arm_empty` is True and `(arm-empty)` is not present in the current state, add 1 to `h`.
    6. Return the total calculated value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions.

        @param task: The planning task object containing initial state, goals, etc.
        """
        self.task = task # Store task to use goal_reached method

        self.goal_pos_map = {} # block -> base_block or 'table'
        self.blocks_that_must_be_clear_in_goal = set() # Only from (clear ?x) goal facts
        self.goal_arm_empty = False

        for goal_fact in task.goals:
            parts = get_parts(goal_fact)
            if not parts: # Skip empty facts if any
                continue
            predicate = parts[0]
            args = parts[1:]

            if predicate == 'on' and len(args) == 2:
                block, base = args
                self.goal_pos_map[block] = base
            elif predicate == 'on-table' and len(args) == 1:
                block = args[0]
                self.goal_pos_map[block] = 'table'
            elif predicate == 'clear' and len(args) == 1:
                self.blocks_that_must_be_clear_in_goal.add(args[0])
            elif predicate == 'arm-empty':
                self.goal_arm_empty = True

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        @param node: The search node containing the current state.
        @return: The estimated heuristic cost.
        """
        state = node.state

        # 1. Check if the state is the goal state.
        if self.task.goal_reached(state):
             return 0

        h = 0

        # Memoization dictionary for the recursive is_correctly_placed function
        correctly_placed_status = {}

        def is_correctly_placed(block):
            """
            Checks if a block is in its final goal position recursively.
            Uses the memoization dict `correctly_placed_status` from the outer scope.
            Returns True if the block's final position is not specified in the goal.
            """
            if block in correctly_placed_status:
                return correctly_placed_status[block]

            # If the block's final position is not specified in the goal, it's considered correctly placed
            # with respect to the goal structure defined by goal_pos_map.
            if block not in self.goal_pos_map:
                 correctly_placed_status[block] = True
                 return True

            goal_base = self.goal_pos_map[block]

            if goal_base == 'table':
                is_current_pos_goal = (f'(on-table {block})' in state)
                result = is_current_pos_goal
            else: # goal_base is another block
                is_current_pos_goal = (f'(on {block} {goal_base})' in state)
                # Block is correctly placed only if it's on the correct base AND the base is correctly placed.
                base_is_correctly_placed = is_correctly_placed(goal_base) # Recursive call handles base not in goal_pos_map

                result = is_current_pos_goal and base_is_correctly_placed

            correctly_placed_status[block] = result
            return result

        # 3. Calculate the number of blocks that are not in their final goal position.
        misplaced_goal_blocks_count = 0
        # Iterate only over blocks that are keys in goal_pos_map (i.e., blocks whose final position is specified).
        for block in self.goal_pos_map.keys():
             if not is_correctly_placed(block):
                 misplaced_goal_blocks_count += 1
        h += misplaced_goal_blocks_count

        # 4. Calculate the cost associated with clearing blocks that must be clear in the goal.
        clearing_cost = 0
        # Find current `on` relationships efficiently.
        current_on_map = {} # base -> block_on_top
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'on' and len(parts) == 3:
                block_on_top, base = parts[1], parts[2]
                current_on_map[base] = block_on_top # Assumes only one block can be on top

        def count_blocks_above(block):
            """Counts the number of blocks stacked directly or indirectly on 'block' in the current state."""
            count = 0
            block_on_top = current_on_map.get(block)
            while block_on_top:
                 count += 1
                 block_on_top = current_on_map.get(block_on_top)
            return count

        for block_to_be_clear in self.blocks_that_must_be_clear_in_goal:
            clearing_cost += count_blocks_above(block_to_be_clear)

        h += clearing_cost

        # 5. Check the arm state.
        if self.goal_arm_empty and "(arm-empty)" not in state:
             h += 1

        return h
