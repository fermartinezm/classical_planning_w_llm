from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal state by:
      1. Accounting for the robot's current state (holding a block or free).
      2. Identifying blocks that are not in their goal positions.
      3. Considering the availability of goal supports for held blocks.

    # Assumptions
    - The goal specifies the desired support (base or another block) for each block.
    - The robot can hold only one block at a time.
    - A block on the base has "base" as support, while a block on another block has that block as support.
    - A held block has "held" as support.
    - The goal does not require any block to be held.

    # Heuristic Initialization
    - Extracts the goal conditions to determine the desired support for each block.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract State Information**:
        - Identify if the robot arm is free (flag set).
        - Identify held blocks, blocks on the base, blocks on other blocks, and clear blocks.
    2. **Determine Current and Goal Supports**:
        - For each block, determine its current support (base, another block, or held).
        - For each block, determine its goal support from the goal conditions.
    3. **Count Misplaced Blocks**:
        - **Held Blocks**: If the goal support is a block and it is not clear, count as misplaced. If the goal support is the base, do not count as misplaced (since the base is always available).
        - **Non-Held Blocks**: Count as misplaced if their current support does not match the goal support.
    4. **Compute Heuristic**:
        - If the arm is busy (flag not set), add 1 action for the next put.
        - Add 2 actions for each misplaced block (pick and put).
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the goal conditions."""
        self.goals = task.goals

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        flag_set = '(hxtpdpff)' in state

        held = set()
        current_base = set()
        current_on_map = {}
        clear_blocks = set()

        for fact in state:
            if fact == '(hxtpdpff)':
                continue
            parts = fact.strip('()').split()
            if not parts:
                continue
            pred = parts[0]
            if pred == 'q71n8mtz' and len(parts) == 2:
                held.add(parts[1])
            elif pred == 'xk87au5b' and len(parts) == 2:
                current_base.add(parts[1])
            elif pred == 'x272hpoe' and len(parts) == 3:
                block = parts[1]
                supporter = parts[2]
                current_on_map[block] = supporter
            elif pred == 'dhjk1eyy' and len(parts) == 2:
                clear_blocks.add(parts[1])
        
        all_objects = set()
        all_objects.update(held)
        all_objects.update(current_base)
        all_objects.update(current_on_map.keys())
        all_objects.update(current_on_map.values())
        all_objects.update(clear_blocks)
        for fact in self.goals:
            parts = fact.strip('()').split()
            if len(parts) > 1:
                all_objects.update(parts[1:])
        
        current_support_map = {}
        for block in all_objects:
            if block in held:
                current_support_map[block] = "held"
            elif block in current_base:
                current_support_map[block] = "base"
            elif block in current_on_map:
                current_support_map[block] = current_on_map[block]
            else:
                current_support_map[block] = "base"
        
        goal_support_map = {}
        for fact in self.goals:
            parts = fact.strip('()').split()
            if not parts:
                continue
            pred = parts[0]
            if pred == 'xk87au5b' and len(parts) == 2:
                goal_support_map[parts[1]] = "base"
            elif pred == 'x272hpoe' and len(parts) == 3:
                goal_support_map[parts[1]] = parts[2]
        
        for block in all_objects:
            if block not in goal_support_map:
                goal_support_map[block] = current_support_map[block]
        
        n_misplaced = 0
        for block in all_objects:
            if block in held:
                goal_support = goal_support_map.get(block, "base")
                if goal_support == "base":
                    continue
                else:
                    if goal_support in clear_blocks:
                        continue
                    else:
                        n_misplaced += 1
            else:
                current_support = current_support_map.get(block, "base")
                goal_support = goal_support_map.get(block, current_support)
                if current_support != goal_support:
                    n_misplaced += 1
        
        heuristic_val = 0
        if not flag_set:
            heuristic_val += 1
        heuristic_val += 2 * n_misplaced
        
        return heuristic_val
