# Need to import Task if it's not in the same file, but the prompt implies
# this code will be used within a framework that provides Task.
# Assuming Task and Operator classes are available in the environment.
# from task import Task, Operator # Assuming this import is needed

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal
    state by summing up estimated costs for different stages of the process:
    serving children, making necessary sandwiches, putting sandwiches onto
    trays, and moving trays to the correct locations. It is non-admissible
    and designed for greedy best-first search.

    Assumptions:
    - The heuristic is domain-dependent for the 'childsnacks' PDDL domain.
    - It is non-admissible and intended for greedy best-first search.
    - The cost of each action is 1.
    - The heuristic calculation assumes a simplified view of resource
      availability (ingredients, trays) for estimating make and put_on_tray
      actions, and counts individual child needs for tray movement.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition.
    It identifies:
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - The waiting place for each child.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    - Which sandwich objects, once made with specific ingredients, will be
      gluten-free.
    This static information is stored in sets and dictionaries for efficient
    lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    The heuristic value for a given state is computed as the sum of four components:
    h = h_serve + h_make + h_put_on_tray + h_move_tray

    1.  h_serve:
        This component estimates the number of 'serve_sandwich' actions needed.
        It is simply the count of children who have not yet been served.
        Each unserved child requires one final 'serve' action.

    2.  h_make:
        This component estimates the number of 'make_sandwich' actions needed.
        It calculates the total number of sandwiches required (one for each
        unserved child) and subtracts the number of sandwiches that have
        already been made (either in the kitchen or on trays). The difference
        is the number of sandwiches that still need to be made. This count
        is then limited by the available ingredients (any bread and any content
        in the kitchen) and the available 'notexist' sandwich objects, as
        making a sandwich requires one of each.

    3.  h_put_on_tray:
        This component estimates the number of 'put_on_tray' actions needed.
        It counts the number of sandwiches that are currently in the kitchen
        ('at_kitchen_sandwich'). These sandwiches need to be put onto a tray
        before they can be moved or served. This count is limited by the
        number of trays currently available in the kitchen, as a sandwich can
        only be put on a tray that is in the kitchen.

    4.  h_move_tray:
        This component estimates the number of 'move_tray' actions needed
        to get sandwiches to the children's locations. It counts, for each
        unserved child, whether a suitable sandwich is already available on
        a tray at their waiting place. A sandwich is suitable if it's on a
        tray at the child's location and, if the child is allergic, the
        sandwich is gluten-free. If no such suitable sandwich is found for
        a child at their location, it is assumed that a tray containing a
        suitable sandwich will eventually need to be moved to that child's
        location. This counts the number of children who individually need
        a sandwich delivered to them.

    The total heuristic value is the sum of these four components. If the state
    is a goal state (all children served), the heuristic returns 0.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Args:
            task: The planning task object.
        """
        self.task = task
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.waiting_places = {} # child -> place
        self.no_gluten_bread_static = set()
        self.no_gluten_content_static = set()
        self.no_gluten_sandwich_static = set()
        self.all_children = set() # Keep track of all children objects mentioned in static

        for fact in task.static:
            fact_parts = fact.strip('()').split()
            predicate = fact_parts[0]
            objects = fact_parts[1:]

            if predicate == 'allergic_gluten':
                child = objects[0]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten':
                child = objects[0]
                self.non_allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == 'waiting':
                child, place = objects
                self.waiting_places[child] = place
                self.all_children.add(child) # Ensure all children from waiting are included
            elif predicate == 'no_gluten_bread':
                self.no_gluten_bread_static.add(objects[0])
            elif predicate == 'no_gluten_content':
                self.no_gluten_content_static.add(objects[0])
            elif predicate == 'no_gluten_sandwich':
                self.no_gluten_sandwich_static.add(objects[0])

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            An integer representing the estimated number of actions to reach the goal.
        """
        # Check if goal is reached (heuristic is 0 only at goal)
        if self.task.goal_reached(state):
            return 0

        # --- Parse Dynamic State Information ---
        served_children = set()
        at_kitchen_bread_state = set()
        at_kitchen_content_state = set()
        at_kitchen_sandwich_state = set()
        ontray_state = set() # set of (sandwich, tray) tuples
        at_state = set() # set of (tray, place) tuples
        notexist_state = set() # set of sandwich objects

        sandwiches_made = set() # set of sandwich objects that exist (not notexist)

        for fact in state:
            fact_parts = fact.strip('()').split()
            predicate = fact_parts[0]
            objects = fact_parts[1:]

            if predicate == 'served':
                served_children.add(objects[0])
            elif predicate == 'at_kitchen_bread':
                at_kitchen_bread_state.add(objects[0])
            elif predicate == 'at_kitchen_content':
                at_kitchen_content_state.add(objects[0])
            elif predicate == 'at_kitchen_sandwich':
                sandwiches_made.add(objects[0])
                at_kitchen_sandwich_state.add(objects[0])
            elif predicate == 'ontray':
                sandwich, tray = objects
                sandwiches_made.add(sandwich)
                ontray_state.add((sandwich, tray))
            elif predicate == 'at':
                tray, place = objects
                at_state.add((tray, place))
            elif predicate == 'notexist':
                notexist_state.add(objects[0])

        # --- Calculate Derived Counts ---
        unserved_children = self.all_children - served_children
        num_unserved = len(unserved_children)

        allergic_unserved = unserved_children.intersection(self.allergic_children)
        non_allergic_unserved = unserved_children.intersection(self.non_allergic_children)
        num_allergic_unserved = len(allergic_unserved)
        num_non_allergic_unserved = len(non_allergic_unserved)

        num_gf_sandwiches_made = len([s for s in sandwiches_made if s in self.no_gluten_sandwich_static])
        num_nongf_sandwiches_made = len([s for s in sandwiches_made if s not in self.no_gluten_sandwich_static])

        num_gf_sandwiches_kitchen = len([s for s in at_kitchen_sandwich_state if s in self.no_gluten_sandwich_static])
        num_nongf_sandwiches_kitchen = len([s for s in at_kitchen_sandwich_state if s not in self.no_gluten_sandwich_static])

        num_any_bread_kitchen = len(at_kitchen_bread_state)
        num_any_content_kitchen = len(at_kitchen_content_state)
        # num_gf_bread_kitchen = len(at_kitchen_bread_state.intersection(self.no_gluten_bread_static)) # Not directly used in h_make calculation
        # num_gf_content_kitchen = len(at_kitchen_content_state.intersection(self.no_gluten_content_static)) # Not directly used in h_make calculation
        num_notexist_sandwiches = len(notexist_state)

        trays_at_place_map = {} # place -> set of trays
        for tray, place in at_state:
            trays_at_place_map.setdefault(place, set()).add(tray)

        sandwiches_on_tray_map = {} # tray -> set of sandwiches
        for sandwich, tray in ontray_state:
            sandwiches_on_tray_map.setdefault(tray, set()).add(sandwich)

        trays_kitchen = trays_at_place_map.get('kitchen', set())
        num_trays_kitchen = len(trays_kitchen)

        # --- Calculate Heuristic Components ---

        # h_serve: Number of unserved children
        # Each unserved child needs one 'serve' action.
        h_serve = num_unserved

        # h_make: Number of sandwiches that need to be made
        # Estimate the number of 'make_sandwich' actions required.
        # This is the deficit of made sandwiches compared to needed sandwiches,
        # limited by available ingredients and sandwich objects.
        total_sandwiches_needed = num_allergic_unserved + num_non_allergic_unserved
        total_sandwiches_made = num_gf_sandwiches_made + num_nongf_sandwiches_made
        sandwiches_to_make_needed = max(0, total_sandwiches_needed - total_sandwiches_made)
        available_any_ingredients = min(num_any_bread_kitchen, num_any_content_kitchen)
        h_make = min(sandwiches_to_make_needed, available_any_ingredients, num_notexist_sandwiches)

        # h_put_on_tray: Number of sandwiches in kitchen needing to be put on tray
        # Estimate the number of 'put_on_tray' actions required.
        # This is the number of sandwiches currently in the kitchen, limited
        # by the number of trays available in the kitchen.
        num_kitchen_sandwiches = num_gf_sandwiches_kitchen + num_nongf_sandwiches_kitchen
        h_put_on_tray = min(num_kitchen_sandwiches, num_trays_kitchen)

        # h_move_tray: Number of unserved children needing sandwich delivery to their location
        # Estimate the number of 'move_tray' actions related to delivery.
        # Count how many unserved children do not currently have a suitable
        # sandwich available on a tray at their waiting location.
        num_children_need_delivery = 0
        for child in unserved_children:
            place = self.waiting_places.get(child) # Get waiting place for the child
            if place is None:
                 # This child is unserved but not waiting anywhere? Problematic state.
                 # Or perhaps child was waiting but location predicate was deleted?
                 # Assuming valid states where unserved children are waiting somewhere.
                 continue

            suitable_sandwich_found_at_place = False
            trays_at_this_place = trays_at_place_map.get(place, set())

            for tray in trays_at_this_place:
                sandwiches_on_this_tray = sandwiches_on_tray_map.get(tray, set())
                for sandwich in sandwiches_on_this_tray:
                    is_gf_sandwich = sandwich in self.no_gluten_sandwich_static
                    is_allergic_child = child in self.allergic_children

                    # Check if the sandwich is suitable for the child
                    if (is_allergic_child and is_gf_sandwich) or (not is_allergic_child):
                        suitable_sandwich_found_at_place = True
                        break # Found a suitable sandwich for this child at this location
                if suitable_sandwich_found_at_place:
                    break # Found a tray with a suitable sandwich at this location

            if not suitable_sandwich_found_at_place:
                num_children_need_delivery += 1

        h_move_tray = num_children_need_delivery

        # Total heuristic value is the sum of the components
        heuristic_value = h_serve + h_make + h_put_on_tray + h_move_tray

        return heuristic_value
