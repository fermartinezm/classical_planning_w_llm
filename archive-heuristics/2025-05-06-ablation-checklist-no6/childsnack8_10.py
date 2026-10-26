from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or fact[0] != '(' or fact[-1] != ')':
         return [] # Handle potential malformed input gracefully
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

    Estimates the number of actions needed to serve all waiting children.
    The heuristic is a sum of estimated costs for:
    1. Making necessary sandwiches.
    2. Putting necessary kitchen sandwiches onto trays.
    3. Moving trays to locations where children are waiting and need a delivery.
    4. Serving the children.

    It assumes resources (bread, content, trays, notexist sandwiches) are available
    when needed for heuristic calculation purposes.
    It counts the number of children needing serving, the number of sandwiches
    that need to be made, the number of sandwiches at the kitchen that need
    to be put on trays, and the number of locations needing a tray delivery.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and objects.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Extract all objects of relevant types from initial state, goals, and static facts
        # A more robust parser would provide this, but we can infer from facts
        all_relevant_facts = set(task.initial_state) | set(task.goals) | set(self.static_facts)

        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_bread = set()
        self.all_content = set()


        # Infer objects by looking at argument positions in known predicates
        for fact in all_relevant_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]

            # Children
            if predicate in ['waiting', 'served', 'allergic_gluten', 'not_allergic_gluten']:
                 if len(parts) > 1: self.all_children.add(parts[1])
            # Sandwiches
            elif predicate in ['make_sandwich_no_gluten', 'make_sandwich', 'at_kitchen_sandwich', 'ontray', 'serve_sandwich_no_gluten', 'serve_sandwich', 'notexist', 'no_gluten_sandwich', 'put_on_tray']:
                 if len(parts) > 1: self.all_sandwiches.add(parts[1])
            # Trays
            elif predicate in ['ontray', 'put_on_tray']:
                 if len(parts) > 2: self.all_trays.add(parts[2]) # tray is 3rd arg
            elif predicate in ['serve_sandwich_no_gluten', 'serve_sandwich']:
                 if len(parts) > 3: self.all_trays.add(parts[3]) # tray is 4th arg
            elif predicate in ['move_tray', 'at']:
                 if len(parts) > 1: self.all_trays.add(parts[1]) # tray is 2nd arg
            # Places
            if predicate in ['waiting', 'at']:
                 if len(parts) > 2: self.all_places.add(parts[2]) # place is 3rd arg
            elif predicate in ['move_tray']:
                 if len(parts) > 2: self.all_places.add(parts[2]) # place is 3rd arg
                 if len(parts) > 3: self.all_places.add(parts[3]) # place is 4th arg
            elif predicate in ['serve_sandwich_no_gluten', 'serve_sandwich']:
                 if len(parts) > 4: self.all_places.add(parts[4]) # place is 5th arg
            # Bread
            elif predicate in ['at_kitchen_bread', 'no_gluten_bread', 'make_sandwich_no_gluten', 'make_sandwich']:
                 if len(parts) > 2: self.all_bread.add(parts[2]) # bread is 3rd arg
            # Content
            elif predicate in ['at_kitchen_content', 'no_gluten_content', 'make_sandwich_no_gluten', 'make_sandwich']:
                 if len(parts) > 3: self.all_content.add(parts[3]) # content is 4th arg

        self.all_places.add('kitchen') # kitchen is a constant place


        # Map children to their allergy status (static)
        self.is_allergic_child = {c: False for c in self.all_children}
        for fact in self.static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.is_allergic_child[child] = True

        # Identify children who are goals
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        h = 0

        # 1. Identify waiting, unserved children and their locations
        waiting_children_loc = {}
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, loc = get_parts(fact)[1:3]
                if child in self.goal_children and child not in served_children:
                     waiting_children_loc[child] = loc

        U = set(waiting_children_loc.keys())
        N_U = len(U)

        # If all goal children are served, heuristic is 0
        if N_U == 0:
            return 0

        # Add cost for serving each child
        h += N_U # Each unserved child needs a 'serve' action

        # 2. Count suitable sandwiches available (at kitchen or on tray)
        # A sandwich is suitable if it matches the allergy status of *any* waiting unserved child
        suitable_sandwiches_available_at_kitchen = set()
        suitable_sandwiches_available_ontray = set()

        # Get gluten status of existing sandwiches from the current state
        is_gluten_free_sandwich_state = {get_parts(fact)[1]: True for fact in state if match(fact, "no_gluten_sandwich", "*")}

        for s in self.all_sandwiches:
            # Does this sandwich object exist (i.e., not in notexist state)?
            if f'(notexist {s})' in state:
                continue # This sandwich object is not yet created

            is_gf_s = is_gluten_free_sandwich_state.get(s, False) # Default to False if not explicitly marked GF

            # Check if this sandwich is suitable for *any* waiting unserved child
            is_suitable_for_any_waiting = False
            for child in U:
                child_is_allergic = self.is_allergic_child.get(child, False) # Default to False if status not found
                # A sandwich is suitable if its gluten status matches the child's allergy status
                if child_is_allergic == is_gf_s:
                    is_suitable_for_any_waiting = True
                    break

            if is_suitable_for_any_waiting:
                if f'(at_kitchen_sandwich {s})' in state:
                    suitable_sandwiches_available_at_kitchen.add(s)
                # Check if on any tray
                for t in self.all_trays:
                    if f'(ontray {s} {t})' in state:
                        suitable_sandwiches_available_ontray.add(s)
                        break # A sandwich is only on one tray at a time

        N_available_at_kitchen = len(suitable_sandwiches_available_at_kitchen)
        N_available_ontray = len(suitable_sandwiches_available_ontray)
        N_available_suitable = N_available_at_kitchen + N_available_ontray

        # 3. Count make_sandwich actions needed
        # We need N_U suitable sandwiches in total. N_available_suitable exist.
        N_make = max(0, N_U - N_available_suitable)
        h += N_make # Each sandwich to be made costs 1 action

        # 4. Count put_on_tray actions needed
        # These are needed for sandwiches that are at the kitchen (initially or just made)
        # and are required for the N_U children.
        # The number of sandwiches that will be at the kitchen and needed is N_available_at_kitchen + N_make.
        # Each needs a put_on_tray action.
        h += (N_available_at_kitchen + N_make) # Each needs 1 put_on_tray action (assuming tray at kitchen)

        # 5. Count move_tray actions needed
        # Identify locations where children in U are waiting but which do not have a suitable
        # sandwich already on a tray at that location.
        locations_waiting_children = set(waiting_children_loc.values())
        locations_with_suitable_ontray = set()

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:3]
                # Find tray location
                tray_location = None
                for loc_fact in state:
                    if match(loc_fact, "at", t, "*"):
                        tray_location = get_parts(loc_fact)[2]
                        break

                if tray_location and tray_location in locations_waiting_children:
                    is_gf_s = is_gluten_free_sandwich_state.get(s, False)
                    for child in U:
                        if waiting_children_loc[child] == tray_location:
                            child_is_allergic = self.is_allergic_child.get(child, False)
                            if child_is_allergic == is_gf_s:
                                locations_with_suitable_ontray.add(tray_location)
                                break # This location has at least one suitable sandwich on a tray

        locations_needing_delivery = locations_waiting_children - locations_with_suitable_ontray
        h += len(locations_needing_delivery) # Each location needing delivery needs at least one tray move

        return h
