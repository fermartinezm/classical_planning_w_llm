from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the necessary 'serve', 'move-tray', 'put-on-tray', and
    'make-sandwich' actions based on the current state of children and sandwiches,
    considering gluten allergies and sandwich locations.

    # Assumptions
    - Each waiting child needs one suitable sandwich.
    - A suitable sandwich matches the child's gluten needs (gluten-free for allergic children).
    - Sandwiches must be made in the kitchen, put on a tray in the kitchen, the tray moved to the child's table, and then the child served from the tray.
    - A single 'move-tray' action can transport multiple sandwiches on one tray to a single table.
    - Sufficient bread and content ingredients are available in the kitchen to make any required sandwich type (gluten-free or regular), up to the total number of sandwich objects defined in the problem.
    - Tray capacity is sufficient to hold all sandwiches needed for a single table delivery.
    - The heuristic counts actions needed to get sandwiches to the *point of serving* (on a tray at the correct table), plus the final 'serve' action for each child.

    # Heuristic Initialization
    The heuristic extracts static information about gluten allergies for children
    and gluten status of bread and content portions. It also identifies all
    objects of relevant types (children, tables, trays, sandwiches, bread, content)
    from the initial state and static facts to correctly interpret state predicates.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Identify all children who are currently 'waiting' but not 'served'. For each such child, determine the table they are waiting at and whether they require a gluten-free sandwich (based on static allergy facts). Store this as a list of (child, table, needs_gf) tuples. If this list is empty, the state is a goal state, and the heuristic is 0.

    2.  Count the number of children needing service at each table, separated by gluten requirement ('gf' for allergic, 'reg' for non-allergic). Store this in a dictionary `children_needed_at_table = { table: {gf_status: count} }`.

    3.  Count available sandwiches in the current state, categorized by their location/state ('at_table', 'on_kitchen_tray', 'kitchen_made') and gluten status ('gf' or 'reg'). Gluten status for existing sandwiches is determined by the `(no_gluten_sandwich S)` predicate in the state. Also, count the available gluten-free and regular bread/content portions in the kitchen to determine the maximum number of gluten-free and regular sandwiches that *can* be made ('makeable').

    4.  Calculate how many sandwiches of each gluten type ('gf', 'reg') are still needed *from the kitchen* for each table. This is done by subtracting the suitable sandwiches already available *at that table* from the total number of children needing that type of sandwich at that table. Prioritize using gluten-free sandwiches at the table to satisfy both 'gf' and 'reg' needs. Store this in `needed_from_kitchen = { table: {gf_status: count} }`.

    5.  Calculate the total heuristic cost by summing up estimated actions:
        -   **Serve actions**: Add 1 for each unserved waiting child (total `len(unserved_children)`).
        -   **Move-tray actions**: Add 1 for each table that needs at least one sandwich delivered from the kitchen (i.e., where the total count of 'gf' or 'reg' needed from kitchen for that table is greater than 0).
        -   **Put-on-tray actions**: Count the total number of sandwiches needed from the kitchen pool (sum of counts in `needed_from_kitchen`). Distribute this total need across the kitchen pool states ('on_kitchen_tray', 'kitchen_made', 'makeable'), prioritizing those already on trays, then those made, then those not yet made, while respecting gluten requirements. The number of sandwiches taken from the 'kitchen_made' state represents the number of 'put-on-tray' actions needed.
        -   **Make-sandwich actions**: Using the same distribution process as above, the number of sandwiches taken from the 'makeable' state represents the number of 'make-sandwich' actions needed.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and object lists.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract object lists from initial state and static facts
        # This is necessary to iterate over all possible objects of a type
        self.children = set()
        self.tables = set()
        self.trays = set()
        self.sandwiches = set()
        self.bread_portions = set()
        self.content_portions = set()

        all_facts = set(task.initial_state) | set(task.static)

        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            args = parts[1:]

            if predicate in ['waiting', 'served', 'allergic_gluten', 'not_allergic_gluten', 'has_sandwich']:
                if len(args) > 0: self.children.add(args[0])
                # Note: table is a place, not a 'table' object type in PDDL, but we collect them as tables here
                if len(args) > 1: self.tables.add(args[1]) # waiting ?child ?table
            elif predicate == 'at':
                 if len(args) > 0: self.trays.add(args[0]) # at ?tray ?loc
                 if len(args) > 1 and args[1] != 'kitchen': self.tables.add(args[1]) # at ?tray ?table
            elif predicate == 'ontray':
                 if len(args) > 0: self.sandwiches.add(args[0]) # ontray ?sandw ?tray
                 if len(args) > 1: self.trays.add(args[1])
            elif predicate in ['notexist', 'made', 'at_kitchen_sandwich', 'at_table_sandwich', 'no_gluten_sandwich', 'has_sandwich']:
                 if len(args) > 0: self.sandwiches.add(args[0]) # ?sandw
            elif predicate in ['at_kitchen_bread', 'no_gluten_bread']:
                 if len(args) > 0: self.bread_portions.add(args[0]) # ?b
            elif predicate in ['at_kitchen_content', 'no_gluten_content']:
                 if len(args) > 0: self.content_portions.add(args[0]) # ?c
            elif predicate == 'made': # (made ?sandw ?b ?c)
                 if len(args) > 1: self.bread_portions.add(args[1])
                 if len(args) > 2: self.content_portions.add(args[2])

        # Precompute gluten status of ingredients from static facts
        self.gf_bread_static = {get_parts(f)[1] for f in self.static if match(f, "no_gluten_bread", "*")}
        self.gf_content_static = {get_parts(f)[1] for f in self.static if match(f, "no_gluten_content", "*")}
        self.allergic_children = {get_parts(f)[1] for f in self.static if match(f, "allergic_gluten", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved waiting children and their needs
        unserved_children = []
        waiting_children_facts = {f for f in state if match(f, "waiting", "*", "*")}
        served_children_facts = {f for f in state if match(f, "served", "*")}

        for fact in waiting_children_facts:
            parts = get_parts(fact)
            if len(parts) < 3: continue # Malformed fact
            _, child, table = parts
            if f"(served {child})" not in served_children_facts:
                needs_gf = child in self.allergic_children
                unserved_children.append((child, table, needs_gf))

        # If no children are unserved, the goal is reached
        if not unserved_children:
            return 0

        # 2. Count children needing service at each table by gluten type
        children_needed_at_table = defaultdict(lambda: defaultdict(int)) # {table: {gf_status: count}}
        for _, table, needs_gf in unserved_children:
            gf_status = 'gf' if needs_gf else 'reg' # 'reg' means regular or GF is fine
            children_needed_at_table[table][gf_status] += 1

        # 3. Count available sandwiches by state and gluten status
        sandwich_counts = {
            'at_table': defaultdict(lambda: defaultdict(int)), # {table: {gf_status: count}}
            'on_kitchen_tray': defaultdict(int),             # {gf_status: count}
            'kitchen_made': defaultdict(int),                # {gf_status: count}
            'makeable': defaultdict(int)                     # {gf_status: count}
        }

        # Count ingredients in kitchen
        kitchen_bread = {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        kitchen_content = {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}

        num_gf_bread_kitchen = len(kitchen_bread.intersection(self.gf_bread_static))
        num_reg_bread_kitchen = len(kitchen_bread) - num_gf_bread_kitchen
        num_gf_content_kitchen = len(kitchen_content.intersection(self.gf_content_static))
        num_reg_content_kitchen = len(kitchen_content) - num_gf_content_kitchen

        # Max makeable sandwiches based on ingredients
        sandwich_counts['makeable']['gf'] = min(num_gf_bread_kitchen, num_gf_content_kitchen)
        sandwich_counts['makeable']['reg'] = min(num_reg_bread_kitchen, num_reg_content_kitchen)

        # Count existing sandwiches
        ontray_facts = {f for f in state if match(f, "ontray", "*", "*")}
        at_facts = {f for f in state if match(f, "at", "*", "*")} # For tray location
        kitchen_sandwich_facts = {f for f in state if match(f, "at_kitchen_sandwich", "*")}
        no_gluten_sandwich_facts = {f for f in state if match(f, "no_gluten_sandwich", "*")}

        # Map tray to location
        tray_locations = {}
        for fact in at_facts:
             parts = get_parts(fact)
             if len(parts) >= 3:
                _, tray, loc = parts
                tray_locations[tray] = loc

        # Sandwiches on trays
        for fact in ontray_facts:
            parts = get_parts(fact)
            if len(parts) < 3: continue
            _, sandw, tray = parts
            s_gf_status = 'gf' if f"(no_gluten_sandwich {sandw})" in no_gluten_sandwich_facts else 'reg'
            loc = tray_locations.get(tray)
            if loc and loc != 'kitchen': # Assume any non-kitchen location is a table
                 sandwich_counts['at_table'][loc][s_gf_status] += 1
            elif loc == 'kitchen':
                 sandwich_counts['on_kitchen_tray'][s_gf_status] += 1

        # Sandwiches made but not on trays
        for fact in kitchen_sandwich_facts:
            parts = get_parts(fact)
            if len(parts) < 2: continue
            _, sandw = parts
            s_gf_status = 'gf' if f"(no_gluten_sandwich {sandw})" in no_gluten_sandwich_facts else 'reg'
            sandwich_counts['kitchen_made'][s_gf_status] += 1

        # 4. Calculate needed sandwiches from kitchen per table/gluten type
        needed_from_kitchen = defaultdict(lambda: defaultdict(int)) # {table: {gf_status: count}}

        for table, needs in children_needed_at_table.items():
            gf_needed_at_table = needs.get('gf', 0)
            reg_needed_at_table = needs.get('reg', 0)

            available_gf_at_table = sandwich_counts['at_table'][table].get('gf', 0)
            available_reg_at_table = sandwich_counts['at_table'][table].get('reg', 0)

            # Satisfy GF needs first with available GF sandwiches at the table
            served_gf_by_gf = min(gf_needed_at_table, available_gf_at_table)
            remaining_gf_needed = gf_needed_at_table - served_gf_by_gf
            remaining_available_gf = available_gf_at_table - served_gf_by_gf

            # Satisfy Reg needs with available Reg sandwiches at the table
            served_reg_by_reg = min(reg_needed_at_table, available_reg_at_table)
            remaining_reg_needed = reg_needed_at_table - served_reg_by_reg
            remaining_available_reg = available_reg_at_table - served_reg_by_reg

            # Satisfy remaining Reg needs with remaining available GF sandwiches at the table
            served_reg_by_gf = min(remaining_reg_needed, remaining_available_gf)
            remaining_reg_needed -= served_reg_by_gf

            # The remaining needed sandwiches must come from the kitchen
            needed_from_kitchen[table]['gf'] = remaining_gf_needed
            needed_from_kitchen[table]['reg'] = remaining_reg_needed # These can be satisfied by Reg or GF from kitchen

        # 5. Calculate total heuristic cost
        total_cost = 0

        # Cost for serving
        total_cost += len(unserved_children)

        # Cost for tray moves
        tables_needing_move = {table for table, needs in needed_from_kitchen.items() if needs.get('gf', 0) > 0 or needs.get('reg', 0) > 0}
        total_cost += len(tables_needing_move)

        # Cost for put-on-tray and make-sandwich
        total_needed_gf_from_kitchen = sum(needs.get('gf', 0) for needs in needed_from_kitchen.values())
        total_needed_reg_from_kitchen = sum(needs.get('reg', 0) for needs in needed_from_kitchen.values())

        # Available in kitchen pool by state and gluten type
        pool_ontray_gf = sandwich_counts['on_kitchen_tray'].get('gf', 0)
        pool_ontray_reg = sandwich_counts['on_kitchen_tray'].get('reg', 0)
        pool_made_gf = sandwich_counts['kitchen_made'].get('gf', 0)
        pool_made_reg = sandwich_counts['kitchen_made'].get('reg', 0)
        pool_makeable_gf = sandwich_counts['makeable'].get('gf', 0)
        pool_makeable_reg = sandwich_counts['makeable'].get('reg', 0)

        # Track how many sandwiches are taken from each state/type in the kitchen pool
        take_ontray_gf = 0
        take_ontray_reg = 0
        take_made_gf = 0
        take_made_reg = 0
        take_makeable_gf = 0
        take_makeable_reg = 0

        # Satisfy GF needs first from GF pool (ontray, made, makeable)
        take_ontray_gf = min(total_needed_gf_from_kitchen, pool_ontray_gf)
        total_needed_gf_from_kitchen -= take_ontray_gf
        pool_ontray_gf -= take_ontray_gf

        take_made_gf = min(total_needed_gf_from_kitchen, pool_made_gf)
        total_needed_gf_from_kitchen -= take_made_gf
        pool_made_gf -= take_made_gf

        take_makeable_gf = min(total_needed_gf_from_kitchen, pool_makeable_gf)
        total_needed_gf_from_kitchen -= take_makeable_gf
        pool_makeable_gf -= take_makeable_gf

        # Satisfy Reg needs from remaining pool (ontray_gf, ontray_reg, made_gf, made_reg, makeable_gf, makeable_reg)
        # Prioritize ontray, then made, then makeable
        available_ontray_for_reg = pool_ontray_gf + pool_ontray_reg
        take_ontray_reg = min(total_needed_reg_from_kitchen, available_ontray_for_reg)
        total_needed_reg_from_kitchen -= take_ontray_reg
        # available_ontray_for_reg -= take_ontray_reg # No need to update pool counts further for heuristic

        available_made_for_reg = pool_made_gf + pool_made_reg
        take_made_reg = min(total_needed_reg_from_kitchen, available_made_for_reg)
        total_needed_reg_from_kitchen -= take_made_reg
        # available_made_for_reg -= take_made_reg

        available_makeable_for_reg = pool_makeable_gf + pool_makeable_reg
        take_makeable_reg = min(total_needed_reg_from_kitchen, available_makeable_for_reg)
        total_needed_reg_from_kitchen -= take_makeable_reg
        # available_makeable_for_reg -= take_makeable_reg

        # Add costs for put-on-tray and make-sandwich actions
        total_cost += take_made_gf + take_made_reg   # put-on-tray cost
        total_cost += take_makeable_gf + take_makeable_reg # make-sandwich cost

        return total_cost
