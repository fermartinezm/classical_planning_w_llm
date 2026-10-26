# Need to import Heuristic base class
from heuristics.heuristic_base import Heuristic

from fnmatch import fnmatch

# Helper functions from examples
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
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    # A simple zip and all check is sufficient if fnmatch handles mismatched lengths gracefully
    # or if we assume valid PDDL structure. Let's stick to the example's simple zip/all.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class blocksworldHeuristic(Heuristic): # Inherit from Heuristic
    """
    A domain-dependent heuristic for the Blocksworld domain.

    # Summary
    This heuristic estimates the number of actions needed to reach the goal
    by counting structural mismatches between the current state and the goal state.
    It counts blocks that are not on their correct base, blocks that are
    on top of others where they shouldn't be, and whether the arm needs to be empty.
    It aims to guide the search towards states where blocks are in their
    correct relative positions and the arm is free when needed.

    # Assumptions
    - The goal specifies the desired 'on' and 'on-table' relationships for blocks,
      defining the target stack configuration.
    - The goal implicitly requires '(arm-empty)' unless a specific block is
      required to be held (which is uncommon in standard Blocksworld goals).
    - The goal implicitly requires '(clear ?x)' for blocks that are at the top
      of goal stacks; this is partially addressed by counting misplaced blocks
      on top.

    # Heuristic Initialization
    - Extract the set of goal 'on' and 'on-table' predicates for quick lookup.
    - Identify all objects (blocks) involved in the problem instance from
      the initial state and goal conditions.
    - Determine if '(arm-empty)' is a required goal condition.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Initialize the heuristic value `h` to 0.
    2. Determine the current state of the robot arm (is it holding a block?).
    3. Identify the current base (the object it's 'on' or 'table') for every block that is NOT currently held by the arm.
    4. Count "incorrectly based" blocks: Iterate through all identified objects (blocks). If a block is NOT currently held by the arm, find its required goal base predicate ('on' or 'on-table'). If the block has a required goal base predicate and its current base predicate does NOT match it, increment `h`.
    5. Count "misplaced on top" blocks: Iterate through all facts in the current state. For each fact that is an 'on' predicate, say '(on B A)', check if this exact predicate '(on B A)' is present in the set of goal 'on' predicates. If '(on B A)' is true in the current state but is NOT a goal 'on' predicate, it means block B is on top of block A incorrectly. Increment `h`.
    6. Check the arm state: If the robot arm is currently holding a block, AND the goal requires the arm to be empty (which is the default unless a block needs to be held in the goal), increment `h` by 1.
    7. The total heuristic value is the final value of `h`.

    This heuristic is non-admissible as it doesn't guarantee the shortest path,
    but it provides a direct measure of how far the current state's structure
    is from the goal structure, guiding the search towards states that look
    more like the goal.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and identifying objects.
        """
        self.goals = task.goals

        # Extract goal 'on' and 'on-table' predicates for quick lookup
        self.goal_on_table = set()
        self.goal_on = set()
        # Assume arm-empty is a goal unless 'holding' is explicitly a goal
        self.goal_arm_empty = True

        # Also check if any block is required to be held in the goal (unlikely in blocksworld)
        # If the goal contains (holding ?x), then arm-empty is NOT a goal.
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "on-table":
                 self.goal_on_table.add(goal)
             elif parts[0] == "on":
                 self.goal_on.add(goal)
             elif parts[0] == "holding":
                 # If goal requires holding something, arm-empty is not a goal
                 self.goal_arm_empty = False

        # Extract all object names from initial state and goal state
        self.objects = set()
        # Objects are typically arguments to predicates
        for fact in task.initial_state:
            parts = get_parts(fact)
            # Add all arguments except the predicate name itself
            for part in parts[1:]:
                 # Basic check to avoid adding predicate names or numbers if any exist
                 if isinstance(part, str) and part and part[0].isalpha(): # Added check for empty string
                     self.objects.add(part)

        for goal in self.goals:
             parts = get_parts(goal)
             for part in parts[1:]:
                 if isinstance(part, str) and part and part[0].isalpha(): # Added check for empty string
                     self.objects.add(part)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Step 2: Determine the current state of the robot arm ---
        is_holding = None # Store the block being held, if any
        if "(arm-empty)" not in state:
             # If arm is not empty, it must be holding something
             for fact in state:
                 if match(fact, "holding", "*"):
                     is_holding = get_parts(fact)[1]
                     break # Found the held block

        # --- Step 3: Identify current position for every block not held ---
        # Map block -> its current base fact (e.g., "(on b1 b2)", "(on-table b3)")
        current_base_facts = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "on":
                # parts = ['on', block, base_block]
                current_base_facts[parts[1]] = fact
            elif parts[0] == "on-table":
                # parts = ['on-table', block]
                current_base_facts[parts[1]] = fact

        total_cost = 0

        # --- Step 4: Count "incorrectly based" blocks ---
        # A block is incorrectly based if it's not held AND its current base (on/on-table)
        # does not match its goal base.
        for obj in self.objects:
            if is_holding == obj:
                # Block is held, its position is not fixed on a base yet.
                # Its cost is partially captured by the holding state check later.
                continue

            current_base_fact = current_base_facts.get(obj)

            # Find the goal base predicate for this object
            goal_base_fact = None
            # Check goal 'on' predicates where this object is the top block
            for goal_on_fact in self.goal_on:
                parts = get_parts(goal_on_fact)
                if parts[1] == obj:
                    goal_base_fact = goal_on_fact
                    break # Found the 'on' goal for this block
            # If not found in 'on' goals, check 'on-table' goals
            if goal_base_fact is None:
                 for goal_ontable_fact in self.goal_on_table:
                     parts = get_parts(goal_ontable_fact)
                     if parts[1] == obj:
                         goal_base_fact = goal_ontable_fact
                         break # Found the 'on-table' goal for this block

            # If the block has no goal base specified (e.g., only clear is a goal),
            # it doesn't contribute to this part of the heuristic.
            # This case is unlikely in standard Blocksworld goals defining stacks.
            if goal_base_fact is None:
                 continue

            # Compare current base fact with goal base fact
            if current_base_fact != goal_base_fact:
                 total_cost += 1


        # --- Step 5: Count "misplaced on top" blocks ---
        # Count 'on' facts in the current state that are NOT goal 'on' facts.
        for fact in state:
            if match(fact, "on", "*", "*"):
                if fact not in self.goal_on:
                    total_cost += 1

        # --- Step 6: Check the arm state ---
        # Add cost if robot is holding something and arm-empty is a goal
        if is_holding is not None and self.goal_arm_empty:
             total_cost += 1

        return total_cost
