# from fnmatch import fnmatch # Not strictly needed for this heuristic
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Remove leading/trailing parentheses and split by spaces
    parts = fact[1:-1].split()
    return parts


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed by summing three components:
    1. The number of blocks that are not currently on their correct immediate base
       (either another block or the table) as defined by the goal state.
    2. The number of goal 'clear' predicates that are not satisfied in the current state.
    3. A cost of 1 if the '(arm-empty)' predicate is a goal and is not satisfied
       in the current state.

    This heuristic is non-admissible but aims to guide a greedy best-first search
    by prioritizing states where more blocks are on their correct base, more blocks
    that should be clear are clear, and the arm is empty if required.

    # Assumptions
    - The goal state defines a specific configuration of blocks on top of
      other blocks or the table, and which blocks should be clear, and if the
      arm should be empty.
    - Each block that appears as the first argument of an 'on' predicate
      or in an 'on-table' predicate in the goal has a unique desired
      immediate base.
    - The heuristic counts different types of discrepancies between the current
      state and the goal state, where each discrepancy is estimated to cost
      at least one action to resolve (though the actual cost might be higher
      due to dependencies).

    # Heuristic Initialization
    - Extract the goal 'on' and 'on-table' facts from the task's goals to
      determine the desired immediate base for each block involved in the
      goal configuration. Store this mapping (`self.goal_base_map`).
    - Store the full set of goal facts (`self.goals`) for checking satisfaction
      of 'clear' and 'arm-empty' goals later.

    # Step-By-Step Thinking for Computing Heuristic
    1. In the constructor (`__init__`), parse the goal facts provided in the `task`
       object. Create a dictionary `self.goal_base_map` where keys are blocks
       and values are their desired immediate base (either the name of the block
       they should be on, or the string 'table'). Store the complete set of
       goal facts in `self.goals`.
    2. In the `__call__` method, which takes a `node` object representing the
       current state, access the state facts via `node.state`.
    3. Parse the current state facts to create a dictionary `current_base_map`.
       For each `(on ?x ?y)` fact in the state, record that `?x` is currently
       on `?y` (`current_base_map[?x] = ?y`). For each `(on-table ?x)` fact,
       record that `?x` is on the table (`current_base_map[?x] = 'table'`).
       If `(holding ?x)` is true, record that `?x` is held (`current_base_map[?x] = 'arm'`).
    4. Initialize the heuristic value (`heuristic_value`) to 0.
    5. **Component 1 (Misplaced Base):** Iterate through each block that is present
       as a key in `self.goal_base_map` (i.e., each block whose goal immediate base
       is defined). Look up its current immediate base in `current_base_map`.
       If the block's current immediate base is different from its goal immediate base,
       increment `heuristic_value` by 1. (Note: A block must have a current base
       if it exists in the state; if it's in `self.goal_base_map`, it must exist).
    6. **Component 2 (Unsatisfied Clear Goals):** Iterate through the goal facts
       stored in `self.goals`. If a goal fact is of the form `(clear ?x)` and this
       exact fact `(clear ?x)` is not present in the current state facts (`node.state`),
       increment `heuristic_value` by 1.
    7. **Component 3 (Unsatisfied Arm-Empty Goal):** Check if the fact `(arm-empty)`
       is present in the goal facts (`self.goals`). If it is, and the fact
       `(arm-empty)` is not present in the current state facts (`node.state`),
       increment `heuristic_value` by 1.
    8. Return the final calculated `heuristic_value`. This value will be 0 if and
       only if the current state is the goal state (assuming standard Blocksworld
       goals consist only of `on`, `on-table`, `clear`, and `arm-empty` predicates).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal base positions and goal facts.

        Args:
            task: The planning task object containing initial state, goals, etc.
        """
        self.goals = task.goals # Store all goal facts

        # Map block to its desired immediate base (block or 'table')
        self.goal_base_map = {}
        for goal in self.goals:
            parts = get_parts(goal)
            predicate = parts[0]
            if predicate == "on":
                # Goal is (on block base)
                block, base = parts[1], parts[2]
                self.goal_base_map[block] = base
            elif predicate == "on-table":
                # Goal is (on-table block)
                block = parts[1]
                self.goal_base_map[block] = 'table'
            # Ignore 'clear' and 'arm-empty' goals for the base map

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.

        Args:
            node: The search node containing the current state.

        Returns:
            An integer heuristic value >= 0.
        """
        state = node.state

        heuristic_value = 0

        # Build current base map: where is each block currently?
        current_base_map = {}
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "on":
                # Fact is (on block base)
                block, base = parts[1], parts[2]
                current_base_map[block] = base
            elif predicate == "on-table":
                # Fact is (on-table block)
                block = parts[1]
                current_base_map[block] = 'table'
            elif predicate == "holding":
                # Fact is (holding block)
                holding_block = parts[1]
                current_base_map[holding_block] = 'arm'
            # 'clear' and 'arm-empty' facts don't define a block's base

        # Component 1: Count blocks whose current base is not their goal base
        # We only care about blocks that have a defined goal base position.
        for block, goal_base in self.goal_base_map.items():
             # Find the current base for this block.
             # A block that is part of the goal configuration must exist in the state.
             # It will be either on a block, on the table, or held.
             # So it should have an entry in current_base_map.
             current_base = current_base_map.get(block)

             # If current_base is None, it implies the block is not in the state
             # in a way that defines its base (not on/on-table/holding). This
             # shouldn't happen in a valid Blocksworld state. We treat it as misplaced.
             if current_base != goal_base:
                 heuristic_value += 1 # Count block that is not on its correct base

        # Component 2: Count unsatisfied clear goals
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "clear":
                # Goal is (clear block)
                if goal not in state:
                    heuristic_value += 1

        # Component 3: Count unsatisfied arm-empty goal
        arm_empty_goal_fact = "(arm-empty)"
        if arm_empty_goal_fact in self.goals and arm_empty_goal_fact not in state:
             heuristic_value += 1

        # This heuristic is 0 if and only if the state is the goal state.
        # If state is goal, all goal facts are in state:
        # - (on X Y) goal in state => current_base(X)=Y, goal_base(X)=Y => Comp 1 += 0
        # - (on-table X) goal in state => current_base(X)=table, goal_base(X)=table => Comp 1 += 0
        # - (clear X) goal in state => Comp 2 += 0
        # - (arm-empty) goal in state => Comp 3 += 0
        # Total h = 0.
        # If state is not goal, at least one goal fact is missing.
        # - If missing (on X Y) or (on-table X), and X is in goal_base_map, current_base(X) != goal_base(X) => Comp 1 >= 1.
        # - If missing (clear X) => Comp 2 >= 1.
        # - If missing (arm-empty) => Comp 3 >= 1.
        # Since standard BW goals are of these types, h > 0 for non-goal states.

        return heuristic_value
