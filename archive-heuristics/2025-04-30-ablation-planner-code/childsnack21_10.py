from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Simple check: if the number of parts doesn't match args and no wildcard is used, it's not a match.
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    waiting children. It counts the number of sandwiches that need to be
    made, put on trays, moved to the correct locations, and finally served.
    It breaks down the problem into stages: making needed sandwiches,
    getting them onto trays from the kitchen, moving trays to the correct
    locations, and serving the children.

    # Assumptions
    - All children initially marked as 'waiting' must be 'served' to reach the goal.
    - Resource availability (bread, content) in the kitchen is sufficient to make all needed sandwiches (the heuristic counts 'make' actions up to the number needed, but doesn't explicitly check if resources are exhausted beyond the total needed count).
    - Trays can be used to transport multiple sandwiches to a location.
    - The heuristic counts actions needed for distinct stages (make, put, move, serve)
      for the required items (sandwiches, trays) relative to the unserved children.
    - The cost of moving a tray is approximated by the number of distinct places
      with unserved children that do not currently have a tray with a suitable sandwich.

    # Heuristic Initialization
    - Identify all children who are initially waiting and their waiting places from static facts.
    - Identify which children are allergic to gluten from static facts.
    - Store the gluten-free status of bread and content portions from static facts.
    - Get lists of all sandwich, tray, and place objects from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify unserved children: Count children who were initially waiting but are not yet 'served' in the current state. The number of unserved children is a direct estimate of the minimum 'serve' actions needed. Add this count to the heuristic.

    2. Determine sandwich needs: For the unserved children, count how many require gluten-free sandwiches (allergic) and how many require regular sandwiches (not allergic).

    3. Count existing sandwiches: Check the current state for sandwiches that exist (are not 'notexist'), their gluten-free status, and their current location (kitchen or on a tray). Also, determine the current location of all trays.

    4. Estimate sandwiches to make: Calculate how many gluten-free and regular sandwiches still need to be made based on the number needed and the number existing. Add this count to the heuristic (cost of 'make_sandwich' actions). This count is capped by the total available bread and content in the kitchen, assuming any bread can be combined with any content to make a sandwich if needed.

    5. Count sandwiches in kitchen needing tray placement: Identify existing sandwiches in the kitchen ('at_kitchen_sandwich') that are suitable for unserved children. Count the number of such sandwiches. Add this count to the heuristic (cost of 'put_on_tray' actions). This counts any suitable sandwich in the kitchen if at least one child needs that type.

    6. Count places needing tray visits: Identify the distinct places where unserved children are waiting. For each such place, check if there is already a tray located there that contains a suitable sandwich for at least one child waiting at that place. Count the number of places that *do not* have such a ready tray. Add this count to the heuristic (proxy for 'move_tray' actions needed to bring trays to locations).

    7. Sum the components: The total heuristic value is the sum of the counts from steps 1, 4, 5, and 6.
    """

    def __init__(self, task):
        """Initialize the heuristic."""
        self.goals = task.goals
        self.static_facts = task.static
        self.objects = task.objects # Access objects defined in the instance

        # Identify children who are initially waiting and their places
        self.initial_waiting_children = {} # child -> place
        for fact in self.static_facts:
            if match(fact, "waiting", "*", "*"):
                _, child, place = get_parts(fact)
                self.initial_waiting_children[child] = place

        # Identify allergic children
        self.allergic_children_set = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")
        }

        # Identify gluten-free status of bread and content from static facts
        self.obj_is_gf = {}
        for fact in self.static_facts:
            if match(fact, "no_gluten_bread", "*"):
                self.obj_is_gf[get_parts(fact)[1]] = True
            elif match(fact, "no_gluten_content", "*"):
                self.obj_is_gf[get_parts(fact)[1]] = True

        # Get all object names by type
        self.all_sandwich_objects = self.objects.get('sandwich', [])
        self.all_tray_objects = self.objects.get('tray', [])
        # Places include those defined in objects plus the constant 'kitchen'
        self.all_place_objects = self.objects.get('place', []) + ['kitchen']


    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {
            c: p for c, p in self.initial_waiting_children.items() if c not in served_children
        }
        num_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached.
        if num_unserved == 0:
            return 0

        # Heuristic starts with the minimum number of serve actions
        h = num_unserved

        # 2. Determine sandwich needs
        needed_gf_count = sum(1 for c in unserved_children if c in self.allergic_children_set)
        needed_reg_count = num_unserved - needed_gf_count

        # 3. Count existing sandwiches and their state/type
        s_state = {} # sandwich -> ('kitchen' | ('ontray', tray) | 'notexist')
        s_gluten_free = {} # sandwich -> bool (True if GF)
        tray_loc = {} # tray -> place

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "at_kitchen_sandwich":
                s_state[parts[1]] = 'kitchen'
            elif predicate == "ontray":
                s_state[parts[1]] = ('ontray', parts[2])
            elif predicate == "notexist":
                s_state[parts[1]] = 'notexist'
            elif predicate == "no_gluten_sandwich":
                s_gluten_free[parts[1]] = True
            elif predicate == "at":
                tray_loc[parts[1]] = parts[2]

        # Identify existing GF and regular sandwiches
        existing_gf_sandwiches = {
            s for s in self.all_sandwich_objects
            if s_state.get(s) != 'notexist' and s_gluten_free.get(s, False)
        }
        existing_reg_sandwiches = {
            s for s in self.all_sandwich_objects
            if s_state.get(s) != 'notexist' and not s_gluten_free.get(s, False)
        }

        # 4. Estimate sandwiches to make
        num_gf_exist = len(existing_gf_sandwiches)
        num_reg_exist = len(existing_reg_sandwiches)

        # Count available resources in the kitchen
        B_gf_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and self.obj_is_gf.get(get_parts(fact)[1], False))
        C_gf_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and self.obj_is_gf.get(get_parts(fact)[1], False))
        B_reg_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and not self.obj_is_gf.get(get_parts(fact)[1], False))
        C_reg_kitchen = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and not self.obj_is_gf.get(get_parts(fact)[1], False))

        total_bread_kitchen = B_gf_kitchen + B_reg_kitchen
        total_content_kitchen = C_gf_kitchen + C_reg_kitchen
        max_can_make = min(total_bread_kitchen, total_content_kitchen)

        # Sandwiches needed from making stage
        num_to_make_needed = max(0, needed_gf_count - num_gf_exist) + max(0, needed_reg_count - num_reg_exist)
        num_to_make = min(num_to_make_needed, max_can_make) # Cannot make more than resources allow

        h += num_to_make # Cost for make_sandwich actions

        # 5. Count sandwiches in kitchen needing tray placement
        # Count any existing sandwich in the kitchen that is suitable for *any* unserved child.
        num_gf_kitchen = sum(1 for s in existing_gf_sandwiches if s_state.get(s) == 'kitchen')
        num_reg_kitchen = sum(1 for s in existing_reg_sandwiches if s_state.get(s) == 'kitchen')

        needed_kitchen_sandwiches_count = 0
        if needed_gf_count > 0:
            needed_kitchen_sandwiches_count += num_gf_kitchen
        if needed_reg_count > 0:
            needed_kitchen_sandwiches_count += num_reg_kitchen

        h += needed_kitchen_sandwiches_count # Cost for put_on_tray actions

        # 6. Count places needing tray visits
        places_with_unserved_children = set(unserved_children.values())
        places_needing_tray_visit = set()

        for place in places_with_unserved_children:
            has_ready_tray_at_place = False
            # Find unserved children at this place
            children_at_place = {c for c, p in unserved_children.items() if p == place}
            allergic_at_place = {c for c in children_at_place if c in self.allergic_children_set}
            not_allergic_at_place = children_at_place - allergic_at_place

            # Check trays at this place
            trays_at_this_place = {t for t, p in tray_loc.items() if p == place}

            for tray in trays_at_this_place:
                # Check sandwiches on this tray
                sandwiches_on_this_tray = {s for s, state_val in s_state.items() if state_val == ('ontray', tray)}

                # Check if any sandwich on this tray is suitable for any child at this place
                for sandwich in sandwiches_on_this_tray:
                    is_gf_sandwich = s_gluten_free.get(sandwich, False)

                    # Is it suitable for an allergic child at this place?
                    if is_gf_sandwich and len(allergic_at_place) > 0:
                        has_ready_tray_at_place = True
                        break # Found a suitable sandwich on a tray at the right place

                    # Is it suitable for a non-allergic child at this place?
                    if not is_gf_sandwich and len(not_allergic_at_place) > 0:
                        has_ready_tray_at_place = True
                        break # Found a suitable sandwich on a tray at the right place

                if has_ready_tray_at_place:
                    break # Found a ready tray for this place

            if not has_ready_tray_at_place:
                places_needing_tray_visit.add(place)

        h += len(places_needing_tray_visit) # Proxy cost for move_tray actions

        return h
