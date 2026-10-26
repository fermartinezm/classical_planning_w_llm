from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
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

    Estimates the number of actions needed to serve all children.
    The heuristic is based on the steps required for each unserved child:
    Make Sandwich -> Put on Tray -> Move Tray -> Serve.
    It counts the number of actions needed in each stage, considering available
    resources (sandwiches, components, trays) and locations.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.non_allergic_children = set()
        self.gluten_free_bread = set()
        self.gluten_free_content = set()
        self.child_wait_place = {} # child -> place
        self.all_children = set()
        self.all_places = {'kitchen'} # kitchen is a constant place

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == "allergic_gluten":
                child = parts[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                self.non_allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == "no_gluten_bread":
                self.gluten_free_bread.add(parts[1])
            elif predicate == "no_gluten_content":
                self.gluten_free_content.add(parts[1])
            elif predicate == "waiting":
                child, place = parts[1], parts[2]
                self.child_wait_place[child] = place
                self.all_children.add(child)
                self.all_places.add(place)

        # Ensure all children mentioned in goals are included, even if not in static (e.g., no allergy info)
        # If a child is in goals but not in static allergy info, assume non-allergic.
        for goal in self.goals:
             parts = get_parts(goal)
             if parts and parts[0] == "served":
                 child = parts[1]
                 self.all_children.add(child)
                 if child not in self.allergic_children and child not in self.non_allergic_children:
                      self.non_allergic_children.add(child)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # --- Extract Dynamic State Information ---
        served_children = set()
        at_kitchen_bread_list = []
        at_kitchen_content_list = []
        at_kitchen_sandwich_list = []
        ontray_sandwich_tray = {} # sandwich -> tray
        tray_at_place = {} # tray -> place
        notexist_sandwich_list = []
        no_gluten_sandwich_list = []

        # Collect all places mentioned in the state (from tray locations)
        current_places_in_state = set()
        current_places_in_state.add('kitchen') # Kitchen is always a place

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "served":
                served_children.add(parts[1])
            elif predicate == "at_kitchen_bread":
                at_kitchen_bread_list.append(parts[1])
            elif predicate == "at_kitchen_content":
                at_kitchen_content_list.append(parts[1])
            elif predicate == "at_kitchen_sandwich":
                at_kitchen_sandwich_list.append(parts[1])
            elif predicate == "ontray":
                s, t = parts[1], parts[2]
                ontray_sandwich_tray[s] = t
            elif predicate == "at":
                 # In childsnacks, this is always (at tray place)
                 if len(parts) == 3 and parts[1].startswith('tray'):
                    t, p = parts[1], parts[2]
                    tray_at_place[t] = p
                    current_places_in_state.add(p)
            elif predicate == "notexist":
                notexist_sandwich_list.append(parts[1])
            elif predicate == "no_gluten_sandwich":
                no_gluten_sandwich_list.append(parts[1])

        # Combine all known places
        all_relevant_places = self.all_places.union(current_places_in_state)


        # --- Heuristic Calculation ---
        h = 0

        # 1. Count unserved children
        unserved_children = self.all_children - served_children
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal state

        # Map unserved children to their waiting places
        unserved_children_by_place = {p: [] for p in all_relevant_places}
        unserved_allergic_by_place = {p: [] for p in all_relevant_places}
        unserved_non_allergic_by_place = {p: [] for p in all_relevant_places}

        for child in unserved_children:
            place = self.child_wait_place.get(child)
            if place: # Child must have a waiting place
                 unserved_children_by_place[place].append(child)
                 if child in self.allergic_children:
                     unserved_allergic_by_place[place].append(child)
                 else:
                     unserved_non_allergic_by_place[place].append(child)

        # Identify destination places (places with unserved children)
        dest_places = {p for p, children in unserved_children_by_place.items() if children}

        # 2. Count children ready to be served (suitable sandwich on tray at their location)
        # Count available suitable sandwiches on trays at each destination place
        ontray_gf_by_place = {p: [] for p in all_relevant_places}
        ontray_reg_by_place = {p: [] for p in all_relevant_places}

        for s, t in ontray_sandwich_tray.items():
            place = tray_at_place.get(t)
            if place in dest_places: # Only consider trays at destination places
                if s in no_gluten_sandwich_list:
                    ontray_gf_by_place[place].append(s)
                else:
                    ontray_reg_by_place[place].append(s)

        N_ready_to_serve = 0

        for p in dest_places:
            allergic_at_p = unserved_allergic_by_place[p]
            non_allergic_at_p = unserved_non_allergic_by_place[p]
            gf_ontray_at_p = ontray_gf_by_place[p]
            reg_ontray_at_p = ontray_reg_by_place[p]

            # Match allergic children with GF sandwiches at p
            num_match_allergic = min(len(allergic_at_p), len(gf_ontray_at_p))
            N_ready_to_serve += num_match_allergic

            # Match non-allergic children with remaining GF and regular sandwiches at p
            remaining_gf_at_p = len(gf_ontray_at_p) - num_match_allergic
            available_for_non_allergic_at_p = remaining_gf_at_p + len(reg_ontray_at_p)
            num_match_non_allergic = min(len(non_allergic_at_p), available_for_non_allergic_at_p)
            N_ready_to_serve += num_match_non_allergic


        # 3. Number of children needing the full pipeline (Make -> Put -> Move)
        N_pipeline = N_unserved - N_ready_to_serve

        # --- Add Costs ---

        # Cost Component 1: Serving (all unserved children need this eventually)
        h += N_unserved

        # Cost Component 2: Making Sandwiches for the pipeline
        N_kitchen_sandwich = len(at_kitchen_sandwich_list)
        N_ontray_sandwich = len(ontray_sandwich_tray)
        N_made_total = N_kitchen_sandwich + N_ontray_sandwich

        # Sandwiches needed for the pipeline that are not already made
        # These must come from making actions.
        N_to_make = max(0, N_pipeline - N_kitchen_sandwich) # Only count those not already at kitchen

        h += N_to_make

        # Check makability (infinity check)
        N_bread_kitchen = len(at_kitchen_bread_list)
        N_content_kitchen = len(at_kitchen_content_list)
        N_notexist_sandwich = len(notexist_sandwich_list)

        N_gluten_free_bread_kitchen = len([b for b in at_kitchen_bread_list if b in self.gluten_free_bread])
        N_gluten_free_content_kitchen = len([c for c in at_kitchen_content_list if c in self.gluten_free_content])
        N_gluten_free_kitchen_sandwich = len([s for s in at_kitchen_sandwich_list if s in no_gluten_sandwich_list])
        N_gluten_free_ontray_sandwich = len([s for s in ontray_sandwich_tray if s in no_gluten_sandwich_list])

        N_gf_made_total = N_gluten_free_kitchen_sandwich + N_gluten_free_ontray_sandwich
        N_gf_makable_potential = min(N_gluten_free_bread_kitchen, N_gluten_free_content_kitchen, N_notexist_sandwich)

        N_allergic_unserved_total = len(self.allergic_children - served_children)

        if N_allergic_unserved_total > N_gf_made_total + N_gf_makable_potential:
             return float('inf') # Cannot make enough GF sandwiches

        # Total sandwiches needed (including regular)
        # Total sandwiches needed across all children is N_unserved.
        # Total sandwiches available (made + makable)
        N_any_made_total = N_kitchen_sandwich + N_ontray_sandwich
        N_any_makable_potential = min(N_bread_kitchen, N_content_kitchen, N_notexist_sandwich)

        if N_unserved > N_any_made_total + N_any_makable_potential:
             return float('inf') # Cannot make enough total sandwiches


        # Cost Component 3: Putting pipeline sandwiches on trays
        # All N_pipeline sandwiches need to be put on trays (they are either just made or were at kitchen)
        h += N_pipeline

        # Cost Component 4: Moving Trays to Kitchen (to pick up pipeline sandwiches)
        # N_pipeline sandwiches need trays. These trays must be at the kitchen for the 'put_on_tray' action.
        # Number of trays needed at kitchen for this step is N_pipeline.
        N_trays_kitchen = len([t for t, p in tray_at_place.items() if p == 'kitchen'])
        Trays_to_move_to_kitchen = max(0, N_pipeline - N_trays_kitchen)
        h += Trays_to_move_to_kitchen

        # Cost Component 5: Moving Trays with pipeline sandwiches to destinations
        # The N_pipeline sandwiches need to go to the destinations of the N_pipeline children.
        # Identify the distinct destinations for the N_pipeline children.
        # This is the set of places 'p' where there is at least one unserved child
        # who is NOT covered by the N_ready_to_serve count.
        pipeline_dest_places = set()
        for p in dest_places:
             num_unserved_at_p = len(unserved_children_by_place[p])
             # Recalculate how many children at 'p' are ready_to_serve
             allergic_at_p = unserved_allergic_by_place[p]
             non_allergic_at_p = unserved_non_allergic_by_place[p]
             gf_ontray_at_p = ontray_gf_by_place[p]
             reg_ontray_at_p = ontray_reg_by_place[p]

             served_allergic_at_p = min(len(allergic_at_p), len(gf_ontray_at_p))
             served_non_allergic_at_p = min(len(non_allergic_at_p), len(gf_ontray_at_p) - served_allergic_at_p + len(reg_ontray_at_p))
             num_served_at_p = served_allergic_at_p + served_non_allergic_at_p

             if num_unserved_at_p > num_served_at_p:
                 pipeline_dest_places.add(p)

        N_pipeline_dest_places = len(pipeline_dest_places)

        # Number of trays currently at these pipeline destination places
        N_trays_at_pipeline_dest = sum(len([t for t, place in tray_at_place.items() if place == p]) for p in pipeline_dest_places)

        # Number of trays to move to destinations for the pipeline
        Trays_to_move_to_dest = max(0, N_pipeline_dest_places - N_trays_at_pipeline_dest)
        h += Trays_to_move_to_dest

        # Total heuristic is the sum of costs for each stage.
        # The serving cost (N_unserved) was added first.

        return h
