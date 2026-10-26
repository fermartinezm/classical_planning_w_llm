from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved children.
    It counts the number of missing prerequisite conditions for serving each child,
    plus the final 'serve' action itself. The missing conditions are:
    1. A suitable sandwich is made.
    2. That suitable sandwich is on a tray.
    3. That tray is at the child's waiting table.

    # Assumptions
    - The goal is to serve all children specified in the task goals.
    - Children's waiting locations and allergy statuses are static.
    - A sandwich is "suitable" for a child if the child is not allergic to gluten, or if the sandwich is gluten-free.
    - Ingredients (bread, content) are assumed to be available in the kitchen if needed to make a sandwich, provided there are unused sandwich object placeholders (`notexist`).
    - Each tray can hold at least one sandwich.
    - The primary steps involve making a sandwich, putting it on a tray, moving the tray to the child's table, and serving.

    # Heuristic Initialization
    The heuristic initializes by extracting static information from the task:
    - Identifies all children, trays, sandwiches, bread, content, and place objects.
    - Stores the waiting table and allergy status for each child.
    - Stores the set of gluten-free bread and content items.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:
    1. Identify all children that have not yet been served based on the goal conditions and the current state.
    2. Parse the current state to build useful lookup structures:
       - Locations of all trays.
       - Contents of all trays (which sandwich is on which tray).
       - Gluten-free status of all made sandwiches.
       - Made status of all sandwich objects (whether they exist or are `notexist`).
       - Available ingredients (bread, content) in the kitchen, including gluten-free ones.
    3. Initialize the total heuristic value to 0.
    4. For each unserved child:
       a. Determine the child's waiting table and allergy status using the pre-calculated static information.
       b. Check if there is already a suitable sandwich on a tray located at this child's waiting table in the current state.
          - A sandwich is suitable if the child is not allergic OR the sandwich is marked as gluten-free.
       c. If such a ready sandwich on a tray at the table is found:
          - Add 1 to the child's heuristic contribution (representing the final 'serve' action).
       d. If no such ready sandwich on a tray at the table is found:
          - This means the combined condition (suitable sandwich on tray at table) is not met.
          - Add 1 to the child's heuristic contribution for the final 'serve' action.
          - Count how many of the following prerequisite conditions are missing:
            i.  A suitable sandwich exists (is made).
            ii. That suitable sandwich is on a tray.
            iii. A tray is located at the child's waiting table.
          - Add the count of missing conditions to the child's heuristic contribution.
            - To check condition (i): Is there any sandwich object that is marked as "made" (not `notexist`) and is suitable for this child (based on allergy and sandwich gluten status)?
            - To check condition (ii): Is there any suitable made sandwich (from condition i) that is currently on *any* tray?
            - To check condition (iii): Is there any tray currently located at the child's waiting table?
    5. The total heuristic value is the sum of the contributions for all unserved children.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and object lists.
        """
        self.goals = task.goals  # Goal conditions
        self.static = task.static  # Static facts

        # Extract all object names from initial state, goals, and static facts
        all_objects = set()
        for fact in task.initial_state | task.goals | task.static:
             all_objects.update(get_parts(fact)[1:])

        # Categorize objects based on naming conventions
        self.children = {obj for obj in all_objects if obj.startswith('child')}
        self.trays = {obj for obj in all_objects if obj.startswith('tray')}
        self.sandwiches = {obj for obj in all_objects if obj.startswith('sandw')}
        self.breads = {obj for obj in all_objects if obj.startswith('bread')}
        self.contents = {obj for obj in all_objects if obj.startswith('content')}
        self.places = {obj for obj in all_objects if obj.startswith('table') or obj == 'kitchen'} # Assuming 'kitchen' is a place

        # Extract waiting children info (table, allergy) from static facts
        self.waiting_children_info = {} # child -> {'table': table, 'allergic': bool}
        for fact in self.static:
            if match(fact, "waiting", "?c", "?t"):
                child = get_parts(fact)[1]
                table = get_parts(fact)[2]
                is_allergic = f"(allergic_gluten {child})" in self.static
                self.waiting_children_info[child] = {'table': table, 'allergic': is_allergic}

        # Extract gluten-free ingredient lists from static facts
        self.gf_breads = {get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")}
        self.gf_contents = {get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state

        # Parse current state into useful lookup structures
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        tray_locations = {} # tray -> location
        tray_contents = {} # tray -> sandwich
        sandwich_gluten_status = {} # sandwich -> bool (is_gluten_free)
        sandwich_made_status = {} # sandwich -> bool (is_made)
        ingredient_locations = {} # ingredient -> location (e.g., kitchen)
        notexist_sandwiches = set()

        for fact in state:
            if match(fact, "at", "?obj", "?loc"):
                 if get_parts(fact)[1] in self.trays:
                     tray_locations[get_parts(fact)[1]] = get_parts(fact)[2]
                 # Add other object locations if needed, e.g., ingredients
                 elif get_parts(fact)[1] in self.breads or get_parts(fact)[1] in self.contents:
                     ingredient_locations[get_parts(fact)[1]] = get_parts(fact)[2]
            elif match(fact, "ontray", "?s", "?t"):
                 if get_parts(fact)[2] in self.trays and get_parts(fact)[1] in self.sandwiches:
                    tray_contents[get_parts(fact)[2]] = get_parts(fact)[1]
                    sandwich_made_status[get_parts(fact)[1]] = True
            elif match(fact, "no_gluten_sandwich", "?s"):
                 if get_parts(fact)[1] in self.sandwiches:
                    sandwich_gluten_status[get_parts(fact)[1]] = True
            elif match(fact, "at_kitchen_sandwich", "?s"):
                 if get_parts(fact)[1] in self.sandwiches:
                    sandwich_made_status[get_parts(fact)[1]] = True
            elif match(fact, "at_kitchen_bread", "?b"):
                 if get_parts(fact)[1] in self.breads:
                    ingredient_locations[get_parts(fact)[1]] = 'kitchen'
            elif match(fact, "at_kitchen_content", "?c"):
                 if get_parts(fact)[1] in self.contents:
                    ingredient_locations[get_parts(fact)[1]] = 'kitchen'
            elif match(fact, "notexist", "?s"):
                 if get_parts(fact)[1] in self.sandwiches:
                    notexist_sandwiches.add(get_parts(fact)[1])


        total_cost = 0

        # Heuristic calculation per unserved child
        for child, info in self.waiting_children_info.items():
            if child in served_children:
                continue # Child is already served

            table = info['table']
            is_allergic = info['allergic']
            h_c = 0

            # Check if the combined condition (suitable sandwich on tray at table) is met
            found_ready_sandwich_at_table = False
            for tray, sandwich in tray_contents.items():
                if tray_locations.get(tray) == table:
                    # Check if the sandwich is suitable for this child
                    is_gluten_free_sandwich = sandwich_gluten_status.get(sandwich, False)
                    if not is_allergic or is_gluten_free_sandwich:
                        found_ready_sandwich_at_table = True
                        break # Found one suitable sandwich on a tray at this table

            if found_ready_sandwich_at_table:
                # Only the final 'serve' action is needed
                h_c += 1
            else:
                # The combined condition is not met. Need to achieve it.
                # Count missing prerequisite conditions + the final serve action.
                missing_conditions = 0

                # Condition 1: A suitable sandwich S exists (is made).
                # Check if there is any sandwich object that is made (not notexist) and is suitable.
                suitable_made_sandwich_exists_anywhere = False
                for s in self.sandwiches:
                    if s not in notexist_sandwiches: # Check if made
                        is_gluten_free_sandwich = sandwich_gluten_status.get(s, False)
                        if not is_allergic or is_gluten_free_sandwich:
                            suitable_made_sandwich_exists_anywhere = True
                            break # Found a suitable made sandwich

                if not suitable_made_sandwich_exists_anywhere:
                    missing_conditions += 1 # Need to make one

                # Condition 2: That suitable sandwich is on a tray T.
                # Check if any suitable made sandwich is currently on *any* tray.
                suitable_sandwich_on_any_tray = False
                # Re-evaluate suitable made sandwiches based on current state info
                suitable_made_sandwiches_anywhere = {s for s in self.sandwiches if s not in notexist_sandwiches and ((not is_allergic) or sandwich_gluten_status.get(s, False))}

                for s in suitable_made_sandwiches_anywhere:
                     if s in tray_contents.values(): # Check if this sandwich is a value in tray_contents
                        suitable_sandwich_on_any_tray = True
                        break # Found a suitable sandwich on a tray

                if not suitable_sandwich_on_any_tray:
                    missing_conditions += 1 # Need to put one on a tray

                # Condition 3: Tray T is at Table.
                # Check if any tray is currently located at the child's waiting table.
                tray_at_this_table = any(loc == table for loc in tray_locations.values())
                if not tray_at_this_table:
                    missing_conditions += 1 # Need to move a tray

                # Total cost for this child is the number of missing conditions + the final serve action
                h_c = missing_conditions + 1

            total_cost += h_c

        return total_cost

