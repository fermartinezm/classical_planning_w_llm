from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(on b1 b2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class BlocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the Blocksworld domain.
    It counts the number of blocks that are not in their goal positions, considering 'on', 'on-table', and 'clear' goal predicates.
    It also penalizes blocks that are stacked incorrectly or have blocks on top of them when they should be clear in the goal.

    # Assumptions:
    - The heuristic assumes that each unsatisfied goal predicate will require at least one action to be satisfied.
    - It does not explicitly consider the sequence of actions needed, focusing on the current state's deviation from the goal.
    - It is a domain-dependent heuristic tailored for the blocksworld domain and its typical goal configurations.

    # Heuristic Initialization
    - The heuristic is initialized with the goal predicates from the task definition.
    - No static facts are explicitly used in this heuristic, as the block positions are the primary concern.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value to 0.
    2. Process 'on' goal predicates:
       - For each goal `(on block1 block2)`:
         - Check if `(on block1 block2)` is present in the current state. If not, increment the heuristic value.
         - Check if in the current state `block1` is on top of a block that is not `block2`. If so, increment the heuristic value (penalty for being in the wrong stack).
    3. Process 'on-table' goal predicates:
       - For each goal `(on-table block)`:
         - Check if `(on-table block)` is present in the current state. If not, increment the heuristic value.
         - Check if in the current state `block` is on top of any other block. If so, increment the heuristic value (penalty for not being on the table).
    4. Process 'clear' goal predicates:
       - For each goal `(clear block)`:
         - Check if `(clear block)` is present in the current state. If not, increment the heuristic value.
         - Identify if there is any block on top of `block` in the current state. If so, increment the heuristic value (penalty for not being clear).
    5. Return the accumulated heuristic value.

    This heuristic provides a simple estimate based on counting unsatisfied goal conditions and penalizing blocks that are misplaced or have obstructions, aiming to guide the search towards goal states efficiently.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal predicates."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute the heuristic value for a given state."""
        state = node.state
        h_value = 0

        goal_on_predicates = [goal for goal in self.goals if match(goal, "on", "*", "*")]
        goal_on_table_predicates = [goal for goal in self.goals if match(goal, "on-table", "*")]
        goal_clear_predicates = [goal for goal in self.goals if match(goal, "clear", "*")]

        current_on_relations = [fact for fact in state if match(fact, "on", "*", "*")]
        current_on_table_blocks = [get_parts(fact)[1] for fact in state if match(fact, "on-table", "*")]
        current_clear_blocks = [get_parts(fact)[1] for fact in state if match(fact, "clear", "*")]

        for goal_on in goal_on_predicates:
            if goal_on not in state:
                h_value += 1
                block1_goal, block2_goal = get_parts(goal_on)[1], get_parts(goal_on)[2]
                for current_on in current_on_relations:
                    block1_current, block2_current = get_parts(current_on)[1], get_parts(current_on)[2]
                    if block1_current == block1_goal and block2_current != block2_goal:
                        h_value += 1 # penalty for being on wrong block

        for goal_on_table in goal_on_table_predicates:
            if goal_on_table not in state:
                h_value += 1
                block_goal = get_parts(goal_on_table)[1]
                for current_on in current_on_relations:
                    block1_current, block2_current = get_parts(current_on)[1], get_parts(current_on)[2]
                    if block1_current == block_goal:
                        h_value += 1 # penalty for not being on table

        for goal_clear in goal_clear_predicates:
            goal_block = get_parts(goal_clear)[1]
            if goal_clear not in state:
                h_value += 1
            is_clear_in_state = False
            if goal_block in current_clear_blocks:
                is_clear_in_state = True
            if not is_clear_in_state:
                for current_on in current_on_relations:
                    block1_current, block2_current = get_parts(current_on)[1], get_parts(current_on)[2]
                    if block2_current == goal_block:
                        h_value += 1 # penalty for not being clear

        return h_value
