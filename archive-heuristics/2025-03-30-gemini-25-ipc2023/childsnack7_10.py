import itertools
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this path is correct

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Removes parentheses and splits fact string."""
    # Handles potential extra spaces, removes leading/trailing whitespace
    return fact.strip()[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain childsnacks.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    as specified in the goal. It counts the necessary 'make_sandwich',
    'put_on_tray', 'move_tray', and 'serve' actions based on the current state
    and the children yet to be served. It focuses on the lifecycle of a sandwich
    needed for each unserved child: Make -> Put on Tray -> Move Tray -> Serve.

    # Assumptions
    - Sufficient bread and content portions (regular and gluten-free) are
      available or can be made available implicitly when estimating 'make' costs.
      The heuristic does not check for specific ingredient availability.
    - Sufficient unused sandwich objects (`notexist`) are available to be made.
    - Trays can be moved between any two locations in one 'move_tray' action.
    - Each sandwich that needs to be made ('make' action) will subsequently require
      a 'put_on_tray' action.
    - Each sandwich currently existing at the kitchen also requires a 'put_on_tray' action.
    - Each unserved child requires a 'serve' action eventually.
    - Each unserved child requires a 'move_tray' action to deliver their sandwich.
      This is an approximation, as one tray movement can potentially serve multiple
      children at the same location, or deliver multiple sandwiches. However, it
      captures the necessity of transport for each child's need.
    - Regular (non-allergic) children can be served gluten-free sandwiches if
      such sandwiches are available and not required by allergic children.

    # Heuristic Initialization
    - Stores the goal conditions (`served ?c`).
    - Parses static facts (`allergic_gluten`, `waiting`) to create a mapping
      (`child_info`) for each child specified in the goal, storing their waiting
      location and allergy status. This focuses the heuristic on relevant children.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Determine which children listed in the goal
        are not yet served (`(served c)`) in the current state. Filter this list
        to include only those children for whom static information (waiting location,
        allergy status) was available during initialization.
    2.  **Categorize Unserved Children:** Count the total number of relevant unserved
        children (`N_unserved`), the number among them who are allergic to gluten
        (`N_unserved_gf`), and the number who are not (`N_unserved_reg`).
    3.  **Inventory Existing Sandwiches:**
        - Find all sandwiches marked as gluten-free (`no_gluten_sandwich s`).
        - Find all sandwiches currently at the kitchen (`at_kitchen_sandwich s`).
        - Find all sandwiches currently on any tray (`ontray s t`).
        - Count the total number of existing gluten-free sandwiches (`N_existing_gf`)
          and regular sandwiches (`N_existing_reg`) across both kitchen and trays.
        - Count how many sandwiches are currently specifically at the kitchen
          (`N_kitchen_sandwiches`).
    4.  **Calculate Needed 'Make' Actions:**
        - Estimate how many new gluten-free sandwiches must be made to satisfy
          the unserved allergic children, considering existing GF sandwiches:
          `make_gf_needed = max(0, N_unserved_gf - N_existing_gf)`.
        - Estimate how many new regular sandwiches must be made for non-allergic
          children, considering existing regular sandwiches AND any surplus
          gluten-free sandwiches (those existing beyond the needs of allergic children):
          `surplus_gf = max(0, N_existing_gf - N_unserved_gf)`
          `make_reg_needed = max(0, N_unserved_reg - (N_existing_reg + surplus_gf))`.
    5.  **Estimate Total Actions:** The heuristic value is the sum of estimated costs
        for the key action types involved in serving all children:
        - Cost for making sandwiches: `make_gf_needed + make_reg_needed`.
        - Cost for putting sandwiches on trays: This includes sandwiches just made
          (`make_gf_needed + make_reg_needed`) and those already existing at the
          kitchen (`N_kitchen_sandwiches`). Total put actions =
          `make_gf_needed + make_reg_needed + N_kitchen_sandwiches`.
        - Cost for moving trays: Approximated as one move per unserved child: `N_unserved`.
        - Cost for serving: One serve action per unserved child: `N_unserved`.
    6.  **Combine Costs:** The total heuristic estimate `H` is:
        `H = (make_gf_needed + make_reg_needed)  # make actions`
        `  + (make_gf_needed + make_reg_needed + N_kitchen_sandwiches) # put actions`
        `  + N_unserved  # move actions (approximation)`
        `  + N_unserved  # serve actions`
        Simplified:
        `H = 2 * (make_gf_needed + make_reg_needed) + N_kitchen_sandwiches + 2 * N_unserved`
    7.  **Goal State:** If `N_unserved` (the count of relevant, waiting, unserved goal children)
        is 0, the heuristic value is 0.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information and goals.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.child_info = {} # Map child -> (place, is_allergic)

        allergic = set()
        waiting_loc = {}

        # Parse static facts efficiently
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts
            predicate = parts[0]
            # Store allergy info
            if predicate == "allergic_gluten" and len(parts) == 2:
                allergic.add(parts[1])
            # Store waiting locations
            elif predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                waiting_loc[child] = place
            # Note: not_allergic_gluten is implicitly handled (child not in allergic set)

        # Identify children mentioned in the goal state
        self.goal_children = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served" and len(parts) == 2:
                 self.goal_children.add(parts[1])

        # Build child_info only for goal children using the parsed static info
        for child in self.goal_children:
             is_allergic = child in allergic
             place = waiting_loc.get(child)
             # We only compute heuristic contributions for children who are
             # part of the goal AND are currently waiting somewhere.
             if place:
                 self.child_info[child] = (place, is_allergic)
             # else: If a goal child isn't found in waiting_loc, they might be
             # already served in init, or the problem is ill-defined.
             # The heuristic will ignore them if they aren't 'waiting'.


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # 1. Identify Unserved Children (among those relevant: in goal and waiting)
        served_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == "served" and len(parts) == 2:
                served_in_state.add(parts[1])

        # Consider only goal children who are also defined in child_info (i.e., were waiting)
        # These are the children the heuristic tracks progress for.
        unserved_children = {c for c in self.goal_children if c not in served_in_state and c in self.child_info}

        N_unserved = len(unserved_children)

        # Goal reached condition: If there are no relevant unserved children left.
        if N_unserved == 0:
            # We should technically check if *all* goal children are served,
            # but this heuristic focuses on the process for initially waiting children.
            # If all initially waiting goal children are served, estimate is 0.
            return 0

        # 2. Categorize Unserved Children
        N_unserved_gf = 0
        N_unserved_reg = 0
        for child in unserved_children:
            # We already filtered for child in self.child_info
            _, is_allergic = self.child_info[child]
            if is_allergic:
                N_unserved_gf += 1
            else:
                N_unserved_reg += 1

        # 3. Inventory Existing Sandwiches
        sandwiches_at_kitchen = set()
        sandwiches_on_tray = set()
        gf_sandwiches_state = set() # Stores names of all GF sandwiches in the state

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]

            # Identify all gluten-free sandwiches
            if predicate == "no_gluten_sandwich" and len(parts) == 2:
                gf_sandwiches_state.add(parts[1])
            # Identify sandwiches at the kitchen
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                sandwiches_at_kitchen.add(parts[1])
            # Identify sandwiches on trays
            elif predicate == "ontray" and len(parts) == 3:
                sandwiches_on_tray.add(parts[1])

        # Count sandwiches currently at the kitchen
        N_kitchen_sandwiches = len(sandwiches_at_kitchen)

        # Count existing GF/Regular sandwiches available anywhere (kitchen or tray)
        N_existing_gf = 0
        N_existing_reg = 0
        # Combine sandwiches from both locations
        all_existing_sandwiches = sandwiches_at_kitchen | sandwiches_on_tray
        for sandwich in all_existing_sandwiches:
            if sandwich in gf_sandwiches_state:
                N_existing_gf += 1
            else:
                N_existing_reg += 1

        # 4. Calculate Needed 'Make' Actions
        make_gf_needed = max(0, N_unserved_gf - N_existing_gf)
        # Calculate surplus GF sandwiches that could potentially feed regular children
        surplus_gf = max(0, N_existing_gf - N_unserved_gf)
        # Calculate needed regular sandwiches, considering existing ones and surplus GF ones
        make_reg_needed = max(0, N_unserved_reg - (N_existing_reg + surplus_gf))

        # 5. & 6. Combine Costs using the derived formula
        # H = 2 * (make_actions) + N_kitchen_sandwiches + 2 * (serve/move actions)
        h_val = 2 * (make_gf_needed + make_reg_needed) \
              + N_kitchen_sandwiches \
              + 2 * N_unserved

        # Ensure heuristic is non-negative (the formula should guarantee this, but safe practice)
        h_val = max(0, h_val)

        return h_val
