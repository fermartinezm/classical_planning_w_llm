# Required imports (none needed for this heuristic)

def blocksworldHeuristic(task, state, static):
    """
    Domain-dependent heuristic for Blocksworld.

    Summary:
    The heuristic estimates the cost by counting the number of structural facts
    (on X Y) and (on-table X) that are mismatched between the current state
    and the goal state. It counts facts that are in the goal but not the state,
    and facts that are in the state but not the goal. It also adds a penalty
    if the arm's status (empty or holding a specific block) does not match
    the preferred arm status in the goal.

    Assumptions:
    - The goal state primarily consists of (on X Y), (on-table X), (clear X), and optionally (arm-empty) or (holding X) facts.
    - If neither (arm-empty) nor any (holding ?x) fact is present in the goal, (arm-empty) is implicitly preferred.
    - The input state and goal are frozensets of PDDL fact strings.
    - Static facts are ignored as they don't change and are unlikely to be structural facts relevant to this heuristic.
    - A valid state always contains exactly one of (arm-empty) or (holding ?x).

    Heuristic Initialization:
    The heuristic value starts at 0.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize heuristic value `h = 0`.
    2. Identify the set of goal facts that define the block structure (on X Y or on-table X):
       Iterate through `task.goals`. If a fact string starts with '(on ' or '(on-table ', add it to `goal_struct`.
    3. Identify the set of current state facts that define the block structure (on X Y or on-table X):
       Iterate through `state`. If a fact string starts with '(on ' or '(on-table ', add it to `state_struct`.
    4. Calculate the set difference `goal_struct - state_struct`. The size of this set represents
       the number of desired structural facts that are missing in the current state. Add this size to `h`.
    5. Calculate the set difference `state_struct - goal_struct`. The size of this set represents
       the number of current structural facts that are incorrect (not part of the goal). Add this size to `h`.
    6. Add penalty if the arm status does not match the preferred goal arm status:
       a. Determine the current arm status fact: Check if `state` contains '(arm-empty)'. If yes, the fact is '(arm-empty)'. Otherwise, find the fact starting with '(holding ' in `state`. A valid blocksworld state always has exactly one of these.
       b. Determine the preferred goal arm status fact: Iterate through `task.goals` to find a fact starting with '(holding '. If found, this is the preferred fact. If not found, check if '(arm-empty)' is in `task.goals`. If yes, '(arm-empty)' is the preferred fact. If neither is found, '(arm-empty)' is implicitly preferred.
       c. If the current state arm status fact is different from the preferred goal arm status fact, add 1 to `h`.

    7. Return the calculated heuristic value `h`.
    """
    # 1. Initialize heuristic value
    h = 0

    # 2. Identify goal structural facts
    goal_struct = {f for f in task.goals if f.startswith('(on ') or f.startswith('(on-table ')}

    # 3. Identify current state structural facts
    state_struct = {f for f in state if f.startswith('(on ') or f.startswith('(on-table ')}

    # 4. Add penalty for missing goal structural facts
    h += len(goal_struct - state_struct)

    # 5. Add penalty for incorrect state structural facts
    h += len(state_struct - goal_struct)

    # 6. Add penalty if the arm status does not match the preferred goal arm status
    # Determine current state arm fact
    current_state_arm_fact = '(arm-empty)' if '(arm-empty)' in state else next(f for f in state if f.startswith('(holding '))

    # Determine preferred goal arm fact
    preferred_goal_arm_fact = None
    goal_holding_fact = next((f for f in task.goals if f.startswith('(holding ')), None)

    if goal_holding_fact:
        preferred_goal_arm_fact = goal_holding_fact
    elif '(arm-empty)' in task.goals:
        preferred_goal_arm_fact = '(arm-empty)'
    else:
        # Neither explicit arm status in goal, assume arm-empty is preferred
        preferred_goal_arm_fact = '(arm-empty)'

    # Add penalty if current arm status does not match preferred goal arm status
    if current_state_arm_fact != preferred_goal_arm_fact:
         h += 1

    # 7. Return the heuristic value
    return h
