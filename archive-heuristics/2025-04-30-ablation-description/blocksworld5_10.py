# from heuristics.heuristic_base import Heuristic # Assuming this base class is available in the planner environment

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle facts like '(arm-empty)' which have no arguments
    if fact.strip() == '(arm-empty)':
        return ['arm-empty']
    # Handle facts like '(clear b1)' or '(on-table b1)' or '(on b1 b2)'
    if fact.startswith('(') and fact.endswith(')'):
         return fact[1:-1].split()
    # Should not happen with valid PDDL facts represented as strings
    # Return empty list or raise error for unexpected format? Empty list is safer.
    return []


class blocksworldHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of goal conditions that are not met
    in the current state, focusing primarily on the desired positions of blocks
    (`on` or `on-table`) and the state of the robot's arm (`arm-empty`).
    It counts the number of blocks that are not in their correct goal location
    plus a penalty if the arm is not empty when it should be.

    # Assumptions
    - The goal state is defined by a conjunction of `on`, `on-table`, and
      potentially `clear` and `arm-empty` predicates.
    - The most critical goal predicates for block arrangement are `on` and
      `on-table`. `clear` predicates are often implicitly satisfied if the
      `on`/`on-table` goals are met.
    - Action costs are uniform (e.g., 1).

    # Heuristic Initialization
    - Parses the goal state to identify the desired position (`on` or `on-table`)
      for each block that has an explicit position goal. Stores this in `goal_pos`.
    - Determines if `(arm-empty)` is a goal condition.
    - Collects the set of all objects (blocks) present in the problem instance
      by examining all possible ground facts (`task.facts`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize the heuristic value `h` to 0.
    2. Check the arm state goal: If `(arm-empty)` is a goal and the current state
       does not contain `(arm-empty)`, increment `h` by 1.
    3. Count misplaced blocks: Iterate through all blocks identified in the problem.
       For each block, check if it has a defined goal position (`on` or `on-table`)
       in the `goal_pos` mapping created during initialization.
       - If the block has a goal position: Check if the corresponding goal predicate
         (e.g., `(on B Y)` or `(on-table B)`) is present in the current state.
         If the required goal predicate is *not* in the current state, increment `h` by 1.
       - If the block does *not* have a defined goal position (i.e., it's not mentioned
         in any `on` or `on-table` goal): This block's final position is not constrained
         by the primary position goals. It is not counted as "misplaced" by this part
         of the heuristic.
    4. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and objects."""
        self.goals = task.goals

        # 1. Parse goal predicates to get goal_pos mapping and check for arm-empty goal.
        self.goal_pos = {}
        self.goal_arm_empty = False

        for goal in self.goals:
            parts = get_parts(goal)
            if not parts: # Skip empty or malformed facts
                continue
            predicate = parts[0]
            if predicate == "on" and len(parts) == 3:
                # Goal is (on obj underob)
                obj, underob = parts[1], parts[2]
                self.goal_pos[obj] = underob
            elif predicate == "on-table" and len(parts) == 2:
                # Goal is (on-table obj)
                obj = parts[1]
                self.goal_pos[obj] = 'table'
            elif predicate == "arm-empty" and len(parts) == 1:
                # Goal is (arm-empty)
                self.goal_arm_empty = True
            # Ignore 'clear' goals and other predicates

        # 2. Collect the set of all objects (blocks) in the problem instance.
        self.all_objects = set()
        # Relevant predicates whose arguments are objects
        object_predicates = {'on', 'on-table', 'clear', 'holding'}

        # Iterate through all possible ground facts in the task definition
        for fact_str in task.facts:
            parts = get_parts(fact_str)
            if not parts: # Skip empty or malformed facts
                continue
            predicate = parts[0]
            if predicate in object_predicates:
                # Arguments are objects
                for obj in parts[1:]:
                    self.all_objects.add(obj)
            # Note: 'arm-empty' has no object arguments.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        state_set = set(state) # Convert to set for efficient lookup

        h = 0 # Initialize heuristic value

        # 2. Check the arm state goal.
        if self.goal_arm_empty and "(arm-empty)" not in state_set:
            h += 1

        # 3. Count misplaced blocks.
        for obj in self.all_objects:
            # Check if this object has a defined goal position
            if obj in self.goal_pos:
                goal_target = self.goal_pos[obj]

                if goal_target == 'table':
                    # Goal is (on-table obj)
                    required_fact = f"(on-table {obj})"
                    if required_fact not in state_set:
                        h += 1
                else:
                    # Goal is (on obj goal_target)
                    required_fact = f"(on {obj} {goal_target})"
                    if required_fact not in state_set:
                        h += 1
            # Else: The block does not have a specific 'on' or 'on-table' goal.
            # It is not counted as "misplaced" by this heuristic component.

        # 4. Return the total heuristic value.
        return h
