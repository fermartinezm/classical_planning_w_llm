class blocksworldHeuristic:
    """
    Domain-dependent heuristic for the Blocksworld domain.

    Summary:
        This heuristic estimates the number of actions required to reach the
        goal state by combining two components:
        1. A structural component: Counts the number of blocks that are not
           part of a "correct stack prefix". A block is part of a correct
           stack prefix if it is either correctly placed on the table according
           to the goal, or it is correctly placed on another block according
           to the goal, and the block below it is also part of a correct stack
           prefix. This captures the effort needed to build the goal stacks
           from the bottom up.
        2. An "other goals" component: Counts the number of unsatisfied goal
           facts that are not related to the `on` or `on-table` predicates
           (typically `clear` and `arm-empty`). This accounts for the effort
           needed to achieve these additional conditions.
        The total heuristic value is the sum of these two components.

    Assumptions:
        - The input `task` object conforms to the structure defined in the
          provided `code-file-task`.
        - PDDL facts in states and goals are represented as strings like
          '(predicate arg1 arg2)' or '(predicate arg1)'.
        - The goal state consists of facts using predicates defined in the
          blocksworld domain (`clear`, `on-table`, `arm-empty`, `holding`, `on`).
          `holding` facts are typically not goal facts in standard blocksworld.

    Heuristic Initialization:
        During initialization, the heuristic preprocesses the goal facts to
        identify the desired position for each block (what block it should be
        on, or if it should be on the table). It builds a map `goal_below`
        where `goal_below[X] = Y` if the goal is `(on X Y)`, and a set
        `goal_on_table` for blocks that should be on the table. It also
        collects the set of all objects involved in the problem from the
        initial and goal states. All goal facts are stored for checking
        unsatisfied "other" goals.

    Step-By-Step Thinking for Computing Heuristic:
        1. Check if the current state is the goal state. If yes, the heuristic
           value is 0.
        2. Parse the current state facts to determine the current position of
           each block. Create a map `current_on` where `current_on[X] = Y` if
           `(on X Y)` is true, and a set `current_on_table` for blocks on the
           table.
        3. Compute the structural component (`h_structural`):
           a. Initialize a set `correctly_placed_blocks` to store blocks that are
              currently in a correct stack prefix.
           b. Add blocks to `correctly_placed_blocks` that are correctly placed
              at the bottom of a goal stack: For each block X in `self.goal_on_table`,
              if `(on-table X)` is true in the current state, add X to
              `correctly_placed_blocks`.
           c. Iteratively add blocks that are correctly placed on top of an
              already correctly placed block: Repeat the following loop until no
              new blocks are added to `correctly_placed_blocks`:
              i. Create a temporary set `newly_placed` for blocks found in this iteration.
              ii. For each block X in the set of all objects:
                 - If X is not already in `correctly_placed_blocks`:
                 - Check if X has a desired block Y below it in the goal (i.e.,
                   `(on X Y)` is a goal fact, found in `self.goal_below`).
                 - If yes, check if `(on X Y)` is true in the current state AND
                   if Y is already in `correctly_placed_blocks`.
                 - If both conditions are met, add X to `newly_placed`.
              d. Add all blocks from `newly_placed` to `correctly_placed_blocks`.
           e. `h_structural` is the total number of objects minus the number
              of blocks in `correctly_placed_blocks`.
        4. Compute the "other goals" component (`h_other`):
           a. Initialize `h_other = 0`.
           b. Iterate through all goal facts (`self.goal_facts`).
           c. For each goal fact `g`:
              i. If `g` is not in the current state:
              ii. Check if `g` is an `(on ...)` or `(on-table ...)` fact. If yes,
                  these are primarily accounted for by `h_structural`, so do not
                  increment `h_other` for these.
              iii. If `g` is a `(clear ...)` fact or `(arm-empty)` fact, increment
                   `h_other` by 1. (Assuming `holding` is not a goal).
        5. The total heuristic value is `h_structural + h_other`.
    """

    def __init__(self, task):
        self.task = task
        self.goal_facts = task.goals
        self.initial_state = task.initial_state

        self.goal_below = {}
        self.goal_on_table = set()
        self.all_objects = set()

        # Preprocess goal facts to build goal structure and collect objects
        for fact in self.goal_facts:
            self.all_objects.update(self._get_objects_from_fact(fact))
            if fact.startswith('(on '):
                # Assuming format is strictly '(on obj1 obj2)'
                parts = fact[1:-1].split() # Remove brackets first
                obj1 = parts[1]
                obj2 = parts[2]
                self.goal_below[obj1] = obj2
            elif fact.startswith('(on-table '):
                # Assuming format is strictly '(on-table obj1)'
                parts = fact[1:-1].split() # Remove brackets first
                obj1 = parts[1]
                self.goal_on_table.add(obj1)

        # Collect objects from initial state as well
        for fact in self.initial_state:
             self.all_objects.update(self._get_objects_from_fact(fact))

    def _get_objects_from_fact(self, fact):
        """Helper to extract objects from a fact string."""
        # Remove surrounding brackets and split by space
        parts = fact[1:-1].split()
        objects = set()
        # Skip the predicate name (parts[0])
        for part in parts[1:]:
            # Objects are clean after [1:-1].split()
            objects.add(part)
        return objects

    def __call__(self, state):
        """
        Compute the heuristic value for the given state.
        """
        # 1. Check if the state is a goal state.
        if self.task.goal_reached(state):
            return 0

        # 2. Parse current state facts
        current_on = {}
        current_on_table = set()

        for fact in state:
            if fact.startswith('(on '):
                parts = fact[1:-1].split() # Remove brackets first
                obj1 = parts[1]
                obj2 = parts[2]
                current_on[obj1] = obj2
            elif fact.startswith('(on-table '):
                parts = fact[1:-1].split() # Remove brackets first
                obj1 = parts[1]
                current_on_table.add(obj1)

        # 3. Compute the structural component (h_structural)
        correctly_placed_blocks = set()

        # Add blocks correctly placed on the table according to the goal
        for obj in self.goal_on_table:
            if obj in current_on_table:
                correctly_placed_blocks.add(obj)

        # Iteratively add blocks correctly placed on top of correctly placed blocks
        while True:
            newly_placed = set()
            for obj in self.all_objects:
                if obj not in correctly_placed_blocks:
                    # Check if this block has a goal position on another block
                    if obj in self.goal_below:
                        obj_below_goal = self.goal_below[obj]
                        # Check if it's currently on the correct block AND that block is correctly placed
                        # Need to check if obj is even in current_on before accessing current_on[obj]
                        if obj in current_on and current_on[obj] == obj_below_goal and obj_below_goal in correctly_placed_blocks:
                            newly_placed.add(obj)

            if not newly_placed:
                break # No new blocks were added in this iteration

            correctly_placed_blocks.update(newly_placed)

        # h_structural = number of blocks not in a correct stack prefix
        h_structural = len(self.all_objects) - len(correctly_placed_blocks)

        # 4. Compute the "other goals" component (h_other)
        h_other = 0
        for goal_fact in self.goal_facts:
            if goal_fact not in state:
                # Check if the goal fact is NOT an 'on' or 'on-table' fact
                if not (goal_fact.startswith('(on ') or goal_fact.startswith('(on-table ')) :
                     # Assuming typical blocksworld goals only include clear and arm-empty besides on/on-table
                     h_other += 1

        # 5. Total heuristic
        return h_structural + h_other
