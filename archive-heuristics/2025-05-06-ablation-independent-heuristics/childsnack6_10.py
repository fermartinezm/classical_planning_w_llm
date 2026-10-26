# Add the necessary import based on the provided code structure
from heuristics.heuristic_base import Heuristic
# Task class definition is provided in code-file-task, but we only need the type hint and access to task.goals and task.static
# from task import Task

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to reach the goal state
    (all children served) by summing up the estimated costs for the
    main steps: serving children, making necessary sandwiches, putting
    sandwiches on trays, and moving trays to children's locations.

    Assumptions:
    - Sufficient ingredients (bread and content) are available in the kitchen
      to make any required sandwiches. The heuristic counts the 'make' action
      cost based on the number of sandwiches needed, but does not check
      ingredient availability.
    - Trays have infinite capacity for sandwiches.
    - A tray is available at the kitchen when needed for the 'put_on_tray' action.
    - A tray is available somewhere to be moved when a location needs one.
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    The constructor processes the static facts from the task description.
    It identifies which children are allergic, and stores the waiting
    location for each child. It also extracts the set of all children that
    need to be served from the goal facts.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic calculates the sum of four components:
    1. h_serve: The number of children who are not yet served. Each unserved
       child requires at least one 'serve' action.
    2. h_make: The number of sandwiches that need to be made. This is calculated
       by determining the total number of suitable sandwiches required for all
       unserved children that are not already on trays or in the kitchen,
       considering gluten constraints. This counts the minimum number of
       'make_sandwich' actions needed.
    3. h_put_on_tray: The number of sandwiches that need to be put on trays.
       This is calculated as the total number of suitable sandwiches required
       for all unserved children that are not already on trays. These sandwiches
       must transition from 'at_kitchen_sandwich' (either existing or newly made)
       to 'ontray'. This counts the minimum number of 'put_on_tray' actions needed.
    4. h_move_tray: The number of locations where unserved children are waiting
       but no tray is currently present. At least one 'move_tray' action is
       required for each such location to bring a tray there so serving can occur.

    The total heuristic value is the sum of h_serve + h_make + h_put_on_tray + h_move_tray.
    The heuristic is 0 if and only if all children are served (goal state).
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task for access to goals and static facts

        # Process static facts in __init__
        self.allergic_children_static = set()
        self.child_location_static = {} # child_name -> place_name

        for fact_string in task.static:
            # Assumes fact strings are simple like '(predicate arg1 arg2)'
            parts = fact_string[1:-1].split()
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                self.allergic_children_static.add(parts[1])
            elif predicate == 'waiting':
                # parts[1] is child, parts[2] is place
                self.child_location_static[parts[1]] = parts[2]
            # Other static facts like no_gluten_bread/content, not_allergic_gluten
            # are not explicitly needed for this heuristic's current logic

        # Get all children who need serving from the goal
        # The goal is a conjunction of (served child) facts
        self.all_children_in_goal = set()
        for goal_fact_string in task.goals:
             # Goal facts are like '(served child1)'
             parts = goal_fact_string[1:-1].split()
             if parts[0] == 'served':
                 self.all_children_in_goal.add(parts[1])


    def __call__(self, node):
        state = node.state

        # --- Step 1: Calculate h_serve ---
        # Count unserved children
        served_children_in_state = set()
        for fact_string in state:
            if fact_string.startswith('(served '):
                # Extract child name from '(served child_name)'
                served_children_in_state.add(fact_string[8:-1]) # Slice removes '(served ' and ')'

        unserved_children = self.all_children_in_goal - served_children_in_state
        h_serve = len(unserved_children)

        # If no children are unserved, goal is reached, heuristic is 0
        if h_serve == 0:
             return 0

        # --- Step 2 & 3: Calculate h_make and h_put_on_tray ---
        # Count suitable sandwiches needed by unserved children
        needed_gf = 0
        needed_reg = 0
        for child in unserved_children:
            if child in self.allergic_children_static:
                needed_gf += 1
            else: # not_allergic_gluten (assuming all children are either allergic or not)
                needed_reg += 1 # Non-allergic can take regular or GF, but we count needed_reg as the demand for non-GF sandwiches first

        # Count available suitable sandwiches on trays and in kitchen
        N_ontray_gf = 0
        N_ontray_reg = 0
        N_kitchen_gf = 0
        N_kitchen_reg = 0
        no_gluten_sandwich_set = set()
        ontray_sandwiches = set()
        at_kitchen_sandwiches = set()

        for fact_string in state:
            parts = fact_string[1:-1].split()
            predicate = parts[0]
            if predicate == 'no_gluten_sandwich':
                no_gluten_sandwich_set.add(parts[1])
            elif predicate == 'ontray':
                ontray_sandwiches.add(parts[1])
            elif predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwiches.add(parts[1])

        for s in ontray_sandwiches:
            if s in no_gluten_sandwich_set:
                N_ontray_gf += 1
            else:
                N_ontray_reg += 1

        for s in at_kitchen_sandwiches:
             if s in no_gluten_sandwich_set:
                 N_kitchen_gf += 1
             else:
                 N_kitchen_reg += 1

        # Calculate how many sandwiches of each type are still needed on trays
        # These are the ones that are not already on trays
        sandwiches_needed_ontray_gf = max(0, needed_gf - N_ontray_gf)
        sandwiches_needed_ontray_reg = max(0, needed_reg - N_ontray_reg)

        # Calculate how many of the needed-ontray sandwiches must be made
        # These are the ones not already on trays AND not already in the kitchen
        sandwiches_to_make_gf = max(0, sandwiches_needed_ontray_gf - N_kitchen_gf)
        sandwiches_to_make_reg = max(0, sandwiches_needed_ontray_reg - N_kitchen_reg)
        h_make = sandwiches_to_make_gf + sandwiches_to_make_reg

        # Calculate how many sandwiches need to be put on trays
        # These are the ones needed on trays that are not already on trays.
        # They must come from the kitchen (either existing or newly made).
        h_put_on_tray = sandwiches_needed_ontray_gf + sandwiches_needed_ontray_reg


        # --- Step 4: Calculate h_move_tray ---
        places_with_waiting_children = set()
        for child in unserved_children:
            # Child's location is static, stored in child_location_static
            if child in self.child_location_static: # Should always be true if child is in goal and waiting
                 places_with_waiting_children.add(self.child_location_static[child])

        places_with_trays = set()
        for fact_string in state:
            # Fact format is like '(at tray1 table1)' or '(at tray2 kitchen)'
            # Need to exclude facts like '(at_kitchen_bread bread1)'
            if fact_string.startswith('(at '):
                 parts = fact_string[1:-1].split()
                 # Check if it's a tray location fact: starts with 'at', has 3 parts, second part starts with 'tray'
                 if len(parts) == 3 and parts[1].startswith('tray'):
                      places_with_trays.add(parts[2])

        places_needing_tray = places_with_waiting_children - places_with_trays
        h_move_tray = len(places_needing_tray)

        # --- Total heuristic ---
        total_heuristic = h_serve + h_make + h_put_on_tray + h_move_tray

        return total_heuristic

# Note: The base class Heuristic is assumed to be provided in the environment.
# The Task class definition is provided but not explicitly imported as its
# structure is known from the problem description and code-file-task.
