from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle negative literals like (not (predicate ...))
    if fact.startswith('(not ('):
        parts = ['not'] + fact[6:-1].split()
    else:
        parts = fact[1:-1].split()
    return parts

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Handle 'not' prefix in pattern matching
    if args and args[0] == 'not':
        if parts and parts[0] == 'not':
            # Match the predicate and arguments after 'not'
            return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts[1:], args[1:]))
        else:
            return False # Fact is not negative, pattern is negative
    elif parts and parts[0] == 'not':
         return False # Fact is negative, pattern is not negative

    # Ensure lengths match for non-negative patterns
    if len(parts) != len(args):
        return False

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the total number of actions required to serve all
    unserved children. It counts the number of sandwiches of each type (gluten-free
    and regular) that are needed and are not yet in a state ready for serving
    (on a tray at the child's location). It then estimates the number of
    make_sandwich, put_on_tray, move_tray, and serve actions required to get
    these sandwiches to the children, considering available resources at each stage.

    # Assumptions
    - Each unserved child requires exactly one sandwich of the appropriate type.
    - Gluten-free sandwiches can satisfy both gluten-allergic and non-allergic children.
    - Regular sandwiches can only satisfy non-allergic children.
    - Trays have unlimited capacity.
    - The cost of each action is 1.
    - Solvable problems have sufficient resources (ingredients, notexist sandwiches, trays) eventually reachable. If not, the heuristic returns a large value.

    # Heuristic Initialization
    - Extracts static information: which children are allergic, which children are not allergic,
      and where each child is waiting.
    - Infers all objects of each type (child, bread-portion, content-portion, sandwich, tray, place)
      from the arguments of predicates found in the initial state and static facts.
    - Identifies which children are goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the total number of actions by summing the estimated
    number of `make_sandwich`, `put_on_tray`, `move_tray`, and `serve` actions
    needed. It does this by counting the number of sandwiches (corresponding to
    unserved children) that are *not* yet in a state ready for the next action
    in the serving pipeline, starting from the final `serve` action and working
    backwards to `make_sandwich`.

    1. Identify Unserved Children: Count the number of children who are goal
       conditions but are not yet in the `(served ?c)` state, separating them
       into gluten-allergic (needing GF) and non-allergic (needing Regular or GF).

    2. Count Available Resources at Each Stage:
       - Sandwiches on trays at the correct place (`(ontray s t)` and `(at t p)` where `p` is a child's waiting place): Count GF and Regular sandwiches available at the specific locations where unserved children are waiting.
       - Sandwiches on trays anywhere (`(ontray s t)`): Count total GF and Regular sandwiches on any tray.
       - Sandwiches in the kitchen (`(at_kitchen_sandwich s)`): Count total GF and Regular sandwiches in the kitchen.
       - Ingredients and `notexist` sandwich objects in the kitchen: Count available bread (GF/Reg), content (GF/Reg), and `notexist` sandwich objects to estimate how many GF and Regular sandwiches can be made.

    3. Calculate Needed Sandwiches at Each Stage (Working Backwards):
       This is done by calculating the deficit of suitable sandwiches at each stage.
       - `needed_gf_0`, `needed_reg_0`: Initial number of unserved GF and Regular children. This is the total number of `serve` actions needed.
       - `needed_gf_1`, `needed_reg_1`: Number of GF and Regular sandwiches needed that are *not* yet on a tray at the correct place. These need `move_tray` + `serve` actions. Calculated by subtracting sandwiches available at the correct place (prioritizing GF for GF needs, then remaining GF for Reg needs, then Reg for Reg needs) from `needed_gf_0` and `needed_reg_0`.
       - `needed_gf_2`, `needed_reg_2`: Number of GF and Regular sandwiches needed that are *not* yet on any tray. These need `put_on_tray` + `move_tray` + `serve` actions. Calculated by subtracting sandwiches available on trays anywhere (prioritizing GF for GF needs, then remaining GF for Reg needs, then Reg for Reg needs) from `needed_gf_1` and `needed_reg_1`.
       - `needed_gf_3`, `needed_reg_3`: Number of GF and Regular sandwiches needed that are *not* yet in the kitchen. These need `make_sandwich` + `put_on_tray` + `move_tray` + `serve` actions. Calculated by subtracting sandwiches available in the kitchen (prioritizing GF for GF needs, then remaining GF for Reg needs, then Reg for Reg needs) from `needed_gf_2` and `needed_reg_2`.
       - `needed_gf_4`, `needed_reg_4`: Number of GF and Regular sandwiches needed that *cannot* be made with available ingredients and `notexist` objects. If this is greater than 0, the problem might be unsolvable or the heuristic is limited; a large value is returned.

    4. Sum Action Counts:
       The total heuristic value is the sum of the number of sandwiches that need
       to pass through each action stage:
       - Total `serve` actions needed (`needed_gf_0 + needed_reg_0`).
       - Total `move_tray` actions needed (`needed_gf_1 + needed_reg_1`).
       - Total `put_on_tray` actions needed (`needed_gf_2 + needed_reg_2`).
       - Total `make_sandwich` actions needed (`needed_gf_3 + needed_reg_3`).
       This additive structure counts how many items *must* pass through each stage of the pipeline.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children,
        allergies, waiting places, and objects.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Infer objects and types from predicate arguments in initial/static facts
        self._infer_objects_and_types(self.initial_state | self.static_facts)

        # Populate allergy and place maps using the identified children
        self.child_allergy = {}
        self.child_place = {}
        for fact in self.static_facts | self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten' and len(parts) > 1 and parts[1] in self.children:
                self.child_allergy[parts[1]] = 'gluten'
            elif parts[0] == 'not_allergic_gluten' and len(parts) > 1 and parts[1] in self.children:
                 self.child_allergy[parts[1]] = 'none'
            elif parts[0] == 'waiting' and len(parts) > 2 and parts[1] in self.children and parts[2] in self.places:
                 self.child_place[parts[1]] = parts[2]

        # Identify children who are goals
        self.goal_children = set()
        for goal in self.goals:
             parts = get_parts(goal)
             # Handle goal literals like (not (served ?c)) if they existed, but domain only has (served ?c)
             if parts[0] == 'served' and len(parts) > 1 and parts[1] in self.children:
                 self.goal_children.add(parts[1])

    def _infer_objects_and_types(self, facts):
        """Helper to infer objects and their types from predicate arguments."""
        inferred_objects = {}
        type_map = {
            'child': ['allergic_gluten', 'not_allergic_gluten', 'served', 'waiting'],
            'bread-portion': ['at_kitchen_bread', 'no_gluten_bread'],
            'content-portion': ['at_kitchen_content', 'no_gluten_content'],
            'sandwich': ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich', 'notexist'],
            'tray': ['ontray', 'at'],
            'place': ['waiting', 'at']
        }

        for fact in facts:
            parts = get_parts(fact)
            pred = parts[0]
            args = parts[1:]
            if pred == 'not' and len(args) > 0: # Handle (not (predicate ...))
                 pred = args[0]
                 args = args[1:]

            if pred in type_map['child']:
                 if len(args) > 0: inferred_objects[args[0]] = 'child'
                 if pred == 'waiting' and len(args) > 1: inferred_objects[args[1]] = 'place'
            elif pred in type_map['bread-portion']:
                 if len(args) > 0: inferred_objects[args[0]] = 'bread-portion'
            elif pred in type_map['content-portion']:
                 if len(args) > 0: inferred_objects[args[0]] = 'content-portion'
            elif pred in type_map['sandwich']:
                 if len(args) > 0: inferred_objects[args[0]] = 'sandwich'
                 if pred == 'ontray' and len(args) > 1: inferred_objects[args[1]] = 'tray'
            elif pred in type_map['tray']:
                 if len(args) > 0: inferred_objects[args[0]] = 'tray'
                 if pred == 'at' and len(args) > 1: inferred_objects[args[1]] = 'place'
            elif pred in type_map['place']:
                 # Places are inferred from waiting/at predicates
                 pass # Already handled above

        # Store inferred objects by type
        self.children = {obj for obj, obj_type in inferred_objects.items() if obj_type == 'child'}
        self.places = {obj for obj, obj_type in inferred_objects.items() if obj_type == 'place'}
        self.sandwiches = {obj for obj, obj_type in inferred_objects.items() if obj_type == 'sandwich'}
        self.bread_portions = {obj for obj, obj_type in inferred_objects.items() if obj_type == 'bread-portion'}
        self.content_portions = {obj for obj, obj_type in inferred_objects.items() if obj_type == 'content-portion'}
        self.trays = {obj for obj, obj_type in inferred_objects.items() if obj_type == 'tray'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children
        unserved_children = {c for c in self.goal_children if f'(served {c})' not in state}

        if not unserved_children:
            return 0 # Goal reached

        unserved_gf_children = {c for c in unserved_children if self.child_allergy.get(c) == 'gluten'}
        unserved_reg_children = unserved_children - unserved_gf_children

        N_unserved_gf = len(unserved_gf_children)
        N_unserved_reg = len(unserved_reg_children)

        needed_gf_0 = N_unserved_gf
        needed_reg_0 = N_unserved_reg

        # 2. Count Available Resources at Each Stage

        # Sandwiches on trays at the correct place
        # Count distinct sandwiches suitable for unserved children at their specific locations
        avail_gf_at_place_set = set() # GF sandwiches at place of GF child
        avail_reg_at_place_set = set() # Reg sandwiches at place of Reg child
        avail_gf_at_place_for_reg_set = set() # GF sandwiches at place of Reg child

        ontray_facts = {fact for fact in state if match(fact, "ontray", "*", "*")}
        at_facts = {fact for fact in state if match(fact, "at", "*", "*")}
        no_gluten_sandwich_facts = {fact for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Map trays to their current location
        tray_location = {get_parts(fact)[1]: get_parts(fact)[2] for fact in at_facts if len(get_parts(fact)) == 3}
        # Map sandwiches on trays to their tray
        sandwich_on_tray = {get_parts(fact)[1]: get_parts(fact)[2] for fact in ontray_facts if len(get_parts(fact)) == 3}
        # Map sandwiches to their gluten status
        sandwich_is_gf = {get_parts(fact)[1]: True for fact in no_gluten_sandwich_facts if len(get_parts(fact)) == 2}


        for child in unserved_children:
            place = self.child_place.get(child)
            if not place: continue # Cannot serve child without a waiting place

            is_gf_child = self.child_allergy.get(child) == 'gluten'

            # Find sandwiches on trays at this child's place
            sandwiches_at_child_place = {
                s for s, t in sandwich_on_tray.items()
                if tray_location.get(t) == place
            }

            for s in sandwiches_at_child_place:
                is_gf_sandwich = sandwich_is_gf.get(s, False) # Assume not GF if status unknown

                if is_gf_child:
                    if is_gf_sandwich:
                        avail_gf_at_place_set.add(s)
                else: # Regular child
                    if is_gf_sandwich:
                        avail_gf_at_place_for_reg_set.add(s)
                    else:
                        avail_reg_at_place_set.add(s)

        Avail_GF_at_place_count = len(avail_gf_at_place_set)
        Avail_Reg_at_place_count = len(avail_reg_at_place_set)
        Avail_GF_at_place_for_Reg_count = len(avail_gf_at_place_for_reg_set)


        # Sandwiches on trays anywhere
        Avail_GF_ontray_total_count = len({s for s in sandwich_on_tray if sandwich_is_gf.get(s, False)})
        Avail_Reg_ontray_total_count = len({s for s in sandwich_on_tray if not sandwich_is_gf.get(s, False)})

        # Sandwiches in the kitchen
        at_kitchen_sandwich_facts = {fact for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        Avail_GF_kitchen_count = len({s for fact in at_kitchen_sandwich_facts if match(fact, "at_kitchen_sandwich", s) and sandwich_is_gf.get(s, False)})
        Avail_Reg_kitchen_count = len({s for fact in at_kitchen_sandwich_facts if match(fact, "at_kitchen_sandwich", s) and not sandwich_is_gf.get(s, False)})

        # Ingredients and notexist objects
        at_kitchen_bread_facts = {fact for fact in state if match(fact, "at_kitchen_bread", "*")}
        no_gluten_bread_facts = {fact for fact in state if match(fact, "no_gluten_bread", "*")}
        at_kitchen_content_facts = {fact for fact in state if match(fact, "at_kitchen_content", "*")}
        no_gluten_content_facts = {fact for fact in state if match(fact, "no_gluten_content", "*")}
        notexist_facts = {fact for fact in state if match(fact, "notexist", "*")}

        Avail_GF_bread_count = len({b for fact in at_kitchen_bread_facts if match(fact, "at_kitchen_bread", b) and f'(no_gluten_bread {b})' in no_gluten_bread_facts})
        Avail_GF_content_count = len({c for fact in at_kitchen_content_facts if match(fact, "at_kitchen_content", c) and f'(no_gluten_content {c})' in no_gluten_content_facts})
        Avail_Reg_bread_count = len({b for fact in at_kitchen_bread_facts if match(fact, "at_kitchen_bread", b) and f'(no_gluten_bread {b})' not in no_gluten_bread_facts})
        Avail_Reg_content_count = len({c for fact in at_kitchen_content_facts if match(fact, "at_kitchen_content", c) and f'(no_gluten_content {c})' not in no_gluten_content_facts})
        Avail_Notexist_count = len(notexist_facts)

        # Estimate make capacity
        Avail_GF_make_count = min(Avail_GF_bread_count, Avail_GF_content_count, Avail_Notexist_count)
        # Reg make uses remaining notexist and regular ingredients
        Avail_Reg_make_count = min(Avail_Reg_bread_count, Avail_Reg_content_count, max(0, Avail_Notexist_count - Avail_GF_make_count))


        # 3. Calculate Needed Sandwiches at Each Stage (Working Backwards)

        needed_gf_0 = N_unserved_gf
        needed_reg_0 = N_unserved_reg

        # Stage 1: Served by sandwiches at correct place (Cost 1: serve)
        # Prioritize GF needs with GF sandwiches at place
        met_gf_1 = min(needed_gf_0, Avail_GF_at_place_count)
        # Then meet Reg needs with Reg sandwiches at place
        met_reg_1 = min(needed_reg_0, Avail_Reg_at_place_count)
        # Then meet remaining Reg needs with remaining GF sandwiches at place
        met_reg_by_gf_1 = min(max(0, needed_reg_0 - met_reg_1), Avail_GF_at_place_for_Reg_count)


        needed_gf_1 = needed_gf_0 - met_gf_1
        needed_reg_1 = needed_reg_0 - met_reg_1 - met_reg_by_gf_1

        # Stage 2: Served by sandwiches on trays anywhere (Cost 2: move + serve)
        # These sandwiches need a move action.
        # Prioritize GF needs with GF sandwiches on trays anywhere
        met_gf_ontray = min(needed_gf_1, Avail_GF_ontray_total_count)
        # Then meet Reg needs with Reg sandwiches on trays anywhere
        met_reg_ontray = min(needed_reg_1, Avail_Reg_ontray_total_count)
        # Then meet remaining Reg needs with remaining GF sandwiches on trays anywhere
        met_reg_by_gf_ontray = min(max(0, needed_reg_1 - met_reg_ontray), max(0, Avail_GF_ontray_total_count - met_gf_ontray))

        needed_gf_2 = needed_gf_1 - met_gf_ontray
        needed_reg_2 = needed_reg_1 - met_reg_ontray - met_reg_by_gf_ontray

        # Stage 3: Served by sandwiches in kitchen (Cost 3: put + move + serve)
        # These sandwiches need a put_on_tray action.
        # Prioritize GF needs with GF sandwiches in kitchen
        met_gf_kitchen = min(needed_gf_2, Avail_GF_kitchen_count)
        # Then meet Reg needs with Reg sandwiches in kitchen
        met_reg_kitchen = min(needed_reg_2, Avail_Reg_kitchen_count)
        # Then meet remaining Reg needs with remaining GF sandwiches in kitchen
        met_reg_by_gf_kitchen = min(max(0, needed_reg_2 - met_reg_kitchen), max(0, Avail_GF_kitchen_count - met_gf_kitchen))

        needed_gf_3 = needed_gf_2 - met_gf_kitchen
        needed_reg_3 = needed_reg_2 - met_reg_kitchen - met_reg_by_gf_kitchen

        # Stage 4: Served by making new sandwiches (Cost 4: make + put + move + serve)
        # These sandwiches need a make action.
        # Prioritize GF needs with makeable GF sandwiches
        met_gf_make = min(needed_gf_3, Avail_GF_make_count)
        # Then meet Reg needs with makeable Reg sandwiches
        met_reg_make = min(needed_reg_3, Avail_Reg_make_count)
        # Then meet remaining Reg needs with remaining makeable GF sandwiches
        met_reg_by_gf_make = min(max(0, needed_reg_3 - met_reg_make), max(0, Avail_GF_make_count - met_gf_make))

        needed_gf_4 = needed_gf_3 - met_gf_make
        needed_reg_4 = needed_reg_3 - met_reg_make - met_reg_by_gf_make

        # If needed_gf_4 > 0 or needed_reg_4 > 0, it means we don't have enough
        # ingredients/notexist objects to make all needed sandwiches.
        # Return a large value.
        if needed_gf_4 > 0 or needed_reg_4 > 0:
             # A large number indicating unsolvability or severe resource shortage
             # Add the remaining deficit to a base large number
             return 1000 + needed_gf_4 + needed_reg_4


        # 4. Sum Action Counts
        # Total serve actions = N_unserved_gf + N_unserved_reg
        # Total move actions = needed_gf_1 + needed_reg_1
        # Total put_on_tray actions = needed_gf_2 + needed_reg_2
        # Total make_sandwich actions = needed_gf_3 + needed_reg_3

        h = (needed_gf_0 + needed_reg_0) + \
            (needed_gf_1 + needed_reg_1) + \
            (needed_gf_2 + needed_reg_2) + \
            (needed_gf_3 + needed_reg_3)

        return h
