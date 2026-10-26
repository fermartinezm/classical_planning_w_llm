from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Splits a PDDL fact string into predicate and arguments."""
    # Remove surrounding parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact string matches a pattern of predicate and arguments."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated actions needed for each unserved child. It
    considers the current status and location of suitable sandwiches (needs
    making, at kitchen, on tray elsewhere) and the resources available
    (ingredients, notexist sandwiches) to make new ones. It counts the number
    of make, put_on_tray, move_tray, and serve actions needed based on the
    deficit of suitable sandwiches in different stages of delivery.

    Assumptions:
    - Used with greedy best-first search, so admissibility is not required.
    - Assumes sufficient trays are available in total, although it partially
      accounts for trays needed at the kitchen for the put_on_tray action.
    - Assumes children remain waiting at their initial places.
    - Assumes the 'kitchen' is the only place where sandwiches can be made
      and put on trays initially.

    Heuristic Initialization:
    The constructor parses the static facts and the initial state/goals from
    the task to identify:
    - `allergic_children`: Set of children who are allergic to gluten.
    - `non_allergic_children`: Set of children who are not allergic to gluten.
    - `gf_bread_types`: Set of bread types that are gluten-free.
    - `gf_content_types`: Set of content types that are gluten-free.
    - Lists of all objects by type (`all_children`, `all_bread`, etc.)
      by inspecting facts in the initial state and goals. This is used
      to iterate over all possible objects when counting facts in the state.
    - `kitchen_place`: The name of the kitchen place constant.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify all children that are not yet served based on the goal state
        and the current state. Count the total number of unserved children
        (`N_unserved`) and the number of allergic and non-allergic unserved
        children (`N_allergic_unserved`, `N_non_allergic_unserved`). If
        `N_unserved` is 0, the goal is reached, and the heuristic is 0.
    2.  Count the number of available sandwiches in the current state by type
        (gluten-free or regular) and location/status:
        - At the kitchen (`gf_at_kitchen`, `reg_at_kitchen`).
        - On trays (anywhere, including kitchen if applicable, though 'ontray'
          implies not at kitchen) (`gf_ontray_total`, `reg_ontray_total`).
          Also count those specifically not at the kitchen
          (`gf_ontray_not_kitchen`, `reg_ontray_not_kitchen`).
    3.  Count the available ingredients (gluten-free and regular bread/content)
        and `notexist` sandwich objects at the kitchen.
    4.  Calculate the deficit of suitable sandwiches required to serve all
        unserved children, prioritizing GF sandwiches for allergic children.
        Determine how many GF and Regular sandwiches (`GF_to_make_needed`,
        `Reg_to_make_needed`) need to be made to cover this deficit.
    5.  Determine how many of the needed sandwiches can actually be made
        (`GF_actually_made`, `Reg_actually_made`) based on the available
        ingredients and `notexist` sandwich objects. Update remaining ingredients
        and `notexist` counts after hypothetically making GF sandwiches before
        calculating makeable regular sandwiches.
    6.  Calculate the heuristic value as the sum of estimated actions:
        - `N_unserved`: Each unserved child requires a `serve` action.
        - `GF_actually_made + Reg_actually_made`: Each sandwich that needs to be
          made requires a `make` action.
        - `(gf_at_kitchen + reg_at_kitchen) + (GF_actually_made + Reg_actually_made)`:
          Each sandwich that is at the kitchen (either initially or newly made)
          requires a `put_on_tray` action.
        - `(gf_ontray_not_kitchen + reg_ontray_not_kitchen) + (gf_at_kitchen + reg_at_kitchen + GF_actually_made + Reg_actually_made)`:
          Each sandwich that is on a tray (either already on a tray not at the
          kitchen, or newly put on a tray at the kitchen) requires a `move_tray`
          action to reach the child's location.
    7.  Optionally, add a cost for moving trays to the kitchen if the number
        of sandwiches needing `put_on_tray` exceeds the trays currently at the
        kitchen. This adds `max(0, sandwiches_need_put_on_tray - trays_at_kitchen)`
        to the heuristic. (This optional step is included in the final code).
    """
    def __init__(self, task):
        self.goals = task.goals
        self.kitchen_place = 'kitchen' # Constant defined in domain

        # Parse static facts
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.gf_bread_types = set()
        self.gf_content_types = set()
        # Waiting facts are static for unserved children in the initial state
        # but dynamic in the state representation. We need the initial mapping
        # to know where children are supposed to be served.
        # However, the heuristic only needs to know *that* a child is waiting
        # and *if* they are served. The location is implicitly handled by
        # needing sandwiches on trays at places. Let's rely on the state
        # for waiting status and allergy from static.
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.non_allergic_children.add(parts[1])
            elif parts[0] == 'no_gluten_bread':
                self.gf_bread_types.add(parts[1])
            elif parts[0] == 'no_gluten_content':
                self.gf_content_types.add(parts[1])

        # Collect all object names by type from initial state and goals
        self.all_children = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = {self.kitchen_place} # Start with kitchen

        # Helper to add object to the correct set based on predicate
        def add_objects_from_fact(fact_str):
            parts = get_parts(fact_str)
            pred = parts[0]
            args = parts[1:]
            if pred in ['served', 'allergic_gluten', 'not_allergic_gluten', 'waiting']:
                if len(args) > 0: self.all_children.add(args[0])
                if pred == 'waiting' and len(args) > 1: self.all_places.add(args[1])
            elif pred in ['at_kitchen_bread', 'no_gluten_bread']:
                 if len(args) > 0: self.all_bread.add(args[0])
            elif pred in ['at_kitchen_content', 'no_gluten_content']:
                 if len(args) > 0: self.all_content.add(args[0])
            elif pred in ['at_kitchen_sandwich', 'ontray', 'notexist', 'no_gluten_sandwich']:
                 if len(args) > 0: self.all_sandwiches.add(args[0])
                 if pred == 'ontray' and len(args) > 1: self.all_trays.add(args[1])
            elif pred == 'at':
                 if len(args) > 0: self.all_trays.add(args[0])
                 if len(args) > 1: self.all_places.add(args[1])

        for fact_str in task.initial_state:
            add_objects_from_fact(fact_str)
        for goal_fact_str in task.goals:
             add_objects_from_fact(goal_fact_str)


    def __call__(self, node):
        state = node.state

        def is_gf_sandwich(s, current_state):
            """Checks if a sandwich is gluten-free in the current state."""
            return f'(no_gluten_sandwich {s})' in current_state

        def is_reg_sandwich(s, current_state):
            """Checks if a sandwich is regular (not gluten-free) and exists."""
            # A sandwich exists if it's not (notexist s)
            # We assume all sandwiches are in self.all_sandwiches
            return s in self.all_sandwiches and f'(notexist {s})' not in current_state and not is_gf_sandwich(s, current_state)

        def is_gf_bread(b):
            """Checks if a bread portion is gluten-free."""
            return b in self.gf_bread_types

        def is_reg_bread(b):
            """Checks if a bread portion is regular (not gluten-free)."""
            return b in self.all_bread and b not in self.gf_bread_types

        def is_gf_content(c):
            """Checks if a content portion is gluten-free."""
            return c in self.gf_content_types

        def is_reg_content(c):
            """Checks if a content portion is regular (not gluten-free)."""
            return c in self.all_content and c not in self.gf_content_types

        def is_tray_at_place(current_state, t, p):
            """Checks if a tray is at a specific place in the current state."""
            return f'(at {t} {p})' in current_state

        def count_facts(current_state, pattern, condition=None):
            """Counts facts in the state matching pattern and condition."""
            count = 0
            pattern_parts = get_parts(pattern)
            for fact_str in current_state:
                fact_parts = get_parts(fact_str)
                if len(fact_parts) != len(pattern_parts):
                    continue
                if all(fnmatch(f_part, p_part) for f_part, p_part in zip(fact_parts, pattern_parts)):
                    if condition is None:
                        count += 1
                    else:
                        # Pass arguments to the condition function
                        if condition(*fact_parts[1:]):
                            count += 1
            return count

        # 1. Identify unserved children and their needs
        unserved_children = {c for goal in self.goals if match(goal, "served", "?c") for c in [get_parts(goal)[1]] if goal not in state}
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal reached

        N_allergic_unserved = sum(1 for c in unserved_children if c in self.allergic_children)
        N_non_allergic_unserved = N_unserved - N_allergic_unserved

        # 2. Count available sandwiches by type and location
        gf_at_kitchen = count_facts(state, '(at_kitchen_sandwich ?s)', lambda s: is_gf_sandwich(s, state))
        reg_at_kitchen = count_facts(state, '(at_kitchen_sandwich ?s)', lambda s: is_reg_sandwich(s, state))

        gf_ontray_at_place = {}
        reg_ontray_at_place = {}
        trays_at_place = {}

        for p in self.all_places:
             trays_at_place[p] = count_facts(state, f'(at ?t {p})', lambda t: t in self.all_trays)

        trays_at_kitchen = trays_at_place.get(self.kitchen_place, 0)

        for p in self.all_places:
            # Count sandwiches on trays at place p
            gf_ontray_at_place[p] = count_facts(state, '(ontray ?s ?t)', lambda s, t: is_gf_sandwich(s, state) and is_tray_at_place(state, t, p))
            reg_ontray_at_place[p] = count_facts(state, '(ontray ?s ?t)', lambda s, t: is_reg_sandwich(s, state) and is_tray_at_place(state, t, p))

        gf_ontray_total = sum(gf_ontray_at_place.values())
        reg_ontray_total = sum(reg_ontray_at_place.values())

        # Sandwiches on trays NOT at the kitchen
        gf_ontray_not_kitchen = gf_ontray_total - gf_ontray_at_place.get(self.kitchen_place, 0)
        reg_ontray_not_kitchen = reg_ontray_total - reg_ontray_at_place.get(self.kitchen_place, 0)


        # 3. Count ingredients and notexist
        N_gf_bread = count_facts(state, '(at_kitchen_bread ?b)', is_gf_bread)
        N_reg_bread = count_facts(state, '(at_kitchen_bread ?b)', is_reg_bread)
        N_gf_content = count_facts(state, '(at_kitchen_content ?c)', is_gf_content)
        N_reg_content = count_facts(state, '(at_kitchen_content ?c)', is_reg_content)
        N_notexist_sandwiches = count_facts(state, '(notexist ?s)')

        # 4. Calculate sandwiches to make based on deficit and ingredient availability
        Total_gf_available_made = gf_at_kitchen + gf_ontray_total
        Total_reg_available_made = reg_at_kitchen + reg_ontray_total

        # Calculate deficit prioritizing GF for allergic children
        GF_deficit = max(0, N_allergic_unserved - Total_gf_available_made)
        Remaining_available_gf = max(0, Total_gf_available_made - N_allergic_unserved)
        Reg_deficit = max(0, N_non_allergic_unserved - (Total_reg_available_made + Remaining_available_gf))

        GF_to_make_needed = GF_deficit
        Reg_to_make_needed = Reg_deficit

        # Calculate how many can actually be made
        Max_gf_makeable = min(N_gf_bread, N_gf_content, N_notexist_sandwiches)
        GF_actually_made = min(GF_to_make_needed, Max_gf_makeable)

        # Update resources after hypothetically making GF sandwiches
        Remaining_gf_bread = N_gf_bread - GF_actually_made
        Remaining_gf_content = N_gf_content - GF_actually_made
        Remaining_notexist_sandwiches = N_notexist_sandwiches - GF_actually_made

        Max_reg_makeable = min(Remaining_gf_bread + N_reg_bread, Remaining_gf_content + N_reg_content, Remaining_notexist_sandwiches)
        Reg_actually_made = min(Reg_to_make_needed, Max_reg_makeable)

        # 5. Calculate heuristic cost components
        h = 0

        # Cost for serving: Each unserved child needs one serve action.
        h += N_unserved

        # Cost for making sandwiches: Each sandwich made needs one make action.
        h += GF_actually_made + Reg_actually_made

        # Cost for putting on tray: Each sandwich made or at kitchen needs one put_on_tray action.
        sandwiches_need_put_on_tray = gf_at_kitchen + reg_at_kitchen + GF_actually_made + Reg_actually_made
        h += sandwiches_need_put_on_tray

        # Cost for moving trays:
        # Each sandwich on a tray (either already or newly put) needs a move action.
        # Sandwiches already on trays (not at kitchen) need moving.
        h += gf_ontray_not_kitchen + reg_ontray_not_kitchen

        # Sandwiches put on trays at kitchen need moving.
        h += sandwiches_need_put_on_tray

        # Cost for moving trays to kitchen for put_on_tray:
        # Number of put_on_tray actions needed at kitchen = sandwiches_need_put_on_tray
        # Trays available at kitchen = trays_at_kitchen
        # Tray moves needed to bring trays to kitchen = max(0, sandwiches_need_put_on_tray - trays_at_kitchen)
        h += max(0, sandwiches_need_put_on_tray - trays_at_kitchen)


        return h

