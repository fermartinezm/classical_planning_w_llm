# Import necessary modules (none needed beyond built-ins for this logic)

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the remaining effort to reach the goal (all children served)
    by summing up four components, each representing a necessary stage or bottleneck
    in the process of serving children:
    1. The number of unserved children: Each requires a final 'serve' action.
    2. The estimated number of *suitable* sandwiches that still need to be made:
       Represents the 'make_sandwich' actions required.
    3. The estimated number of *suitable* sandwiches currently in the kitchen that
       need to be put onto trays: Represents the 'put_on_tray' actions required.
    4. The number of distinct locations with unserved children that do not yet have
       a tray carrying a suitable sandwich: Represents the 'move_tray' actions required
       to bring sandwiches to children's locations.

    Assumptions:
    - The problem instance is solvable. This implies that necessary resources
      (ingredients to make suitable sandwiches, sandwich objects, trays) are
      available throughout the planning process to eventually satisfy all needs.
    - A tray can carry multiple sandwiches simultaneously. The heuristic counts
      the *presence* of at least one suitable sandwich on a tray at a location
      as sufficient for that location's needs in the 'move_tray' component,
      implicitly assuming additional suitable sandwiches can be added if needed
      or that one tray can serve multiple children at the same spot.
    - The heuristic does not explicitly model limited quantities of ingredients
      or the total number of available sandwich objects beyond counting how many
      sandwiches are needed relative to those already existing.

    Heuristic Initialization:
    In the constructor, static facts from the task definition (`task.static`) are
    parsed to extract information that remains constant throughout the planning
    process. This includes:
    - Identifying all children and their allergy status (allergic_gluten).
    - Mapping each child to their waiting place.
    - Identifying static gluten-free ingredients (though this is not directly used
      in the current heuristic calculation, it's good practice to extract).
    - Collecting the names of all objects (children, trays, places, sandwiches,
      bread, content) mentioned in static facts, categorized by type.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Parse the state facts to gather dynamic information:
       - Which children have been served.
       - The current location of each tray.
       - Which sandwiches are currently in the kitchen.
       - Which sandwiches are currently on which trays.
       - Which sandwich objects have not yet been created (`notexist`).
       - Which sandwiches are currently marked as gluten-free (`no_gluten_sandwich`).
    2. Identify the set of unserved children by comparing the full list of children
       (from static facts) with the set of served children in the current state.
    3. If there are no unserved children, the goal is reached, and the heuristic value is 0.
    4. Calculate `num_unserved`: This is simply the count of children in the unserved set. This is the base cost, as each unserved child requires a final 'serve' action.
    5. Determine which sandwiches are considered *suitable* in the current state. A sandwich `s` is suitable if there is at least one unserved non-allergic child OR there is at least one unserved allergic child AND `s` is currently marked as gluten-free (`no_gluten_sandwich`). This check is performed once based on the overall set of unserved children.
    6. Count the number of suitable sandwiches that are currently on trays (`num_suitable_on_trays`) and the number of suitable sandwiches currently in the kitchen (`num_suitable_kitchen`).
    7. Calculate `num_sandwiches_to_make`: Estimate the number of *new suitable* sandwiches that need to be created. This is calculated as the total number of suitable sandwiches required (assumed to be equal to the number of unserved children) minus the number of suitable sandwiches already available (on trays or in the kitchen). The result is capped at a minimum of 0. `max(0, num_unserved - num_suitable_on_trays - num_suitable_kitchen)`.
    8. Calculate `num_sandwiches_kitchen_to_put`: Estimate the number of *suitable* sandwiches currently in the kitchen that need to be moved onto trays. This is the number of suitable sandwiches found in the kitchen, capped by the number of additional suitable sandwiches still needed on trays (total needed minus those already on trays). The result is capped at a minimum of 0. `min(num_suitable_kitchen, max(0, num_unserved - num_suitable_on_trays))`.
    9. Calculate `num_places_missing_tray_with_sandwich`: Identify all distinct places where unserved children are waiting. For each such place, check if there is at least one tray currently located at that place (`at ?t ?p`) that is carrying at least one sandwich (`ontray ?s ?t`) which is suitable for *any* unserved child waiting specifically at that place (using the refined suitability check based on allergy status and the sandwich's GF status). Count the number of places that do *not* satisfy this condition. This estimates the number of 'move_tray' actions needed to bring required sandwiches to the correct locations.
    10. The total heuristic value is the sum of these four calculated components: `num_unserved + num_sandwiches_to_make + num_sandwiches_kitchen_to_put + num_places_missing_tray_with_sandwich`. This sum represents a relaxed plan cost, counting necessary steps in different stages of the process.
    """
    def __init__(self, task):
        """
        Initializes the heuristic by parsing static task information and collecting object names.
        """
        # Initialize sets for object types (populated from static and state facts)
        self.children = set()
        self.trays = set()
        self.places = set()
        self.sandwiches = set()
        self.bread_portions = set()
        self.content_portions = set()

        # Store static predicate information
        self.allergic_children = set()
        self.waiting_places = {} # child -> place
        self.no_gluten_bread_static = set() # Not directly used in current heuristic logic, but extracted
        self.no_gluten_content_static = set() # Not directly used in current heuristic logic, but extracted

        # Parse static facts and collect objects and static predicates
        for fact_string in task.static:
            pred, objs = self._parse_fact(fact_string)
            if pred == 'allergic_gluten':
                self.children.add(objs[0])
                self.allergic_children.add(objs[0])
            elif pred == 'not_allergic_gluten':
                 self.children.add(objs[0])
            elif pred == 'waiting':
                self.children.add(objs[0])
                self.places.add(objs[1])
                self.waiting_places[objs[0]] = objs[1]
            elif pred == 'no_gluten_bread':
                self.bread_portions.add(objs[0])
                self.no_gluten_bread_static.add(objs[0])
            elif pred == 'no_gluten_content':
                self.content_portions.add(objs[0])
                self.no_gluten_content_static.add(objs[0])
            # Collect objects from other potential static predicates found in examples
            elif pred == 'at':
                 self.trays.add(objs[0])
                 self.places.add(objs[1])
            elif pred == 'at_kitchen_sandwich':
                 self.sandwiches.add(objs[0])
            elif pred == 'ontray':
                 self.sandwiches.add(objs[0])
                 self.trays.add(objs[1])
            elif pred == 'notexist':
                 self.sandwiches.add(objs[0])
            elif pred == 'no_gluten_sandwich':
                 self.sandwiches.add(objs[0])

        # Note: Objects might also be defined in the problem file's :objects section
        # which is not directly available here. We assume parsing static and state
        # facts will eventually reveal all relevant objects for solvable problems.


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.
        """
        # Parse state facts to get dynamic information
        served_children_state = set()
        at_tray_state = {} # tray -> place
        at_kitchen_sandwich_state = set()
        ontray_sandwich_state = {} # sandwich -> tray
        notexist_sandwiches_state = set()
        no_gluten_sandwiches_state = set()

        # Parse state facts and collect objects (in case new objects appear, like made sandwiches)
        for fact_string in state:
            pred, objs = self._parse_fact(fact_string)
            if pred == 'served':
                served_children_state.add(objs[0])
            elif pred == 'at':
                self.trays.add(objs[0])
                self.places.add(objs[1])
                at_tray_state[objs[0]] = objs[1]
            elif pred == 'at_kitchen_sandwich':
                self.sandwiches.add(objs[0])
                at_kitchen_sandwich_state.add(objs[0])
            elif pred == 'ontray':
                self.sandwiches.add(objs[0])
                self.trays.add(objs[1])
                ontray_sandwich_state[objs[0]] = objs[1]
            elif pred == 'notexist':
                self.sandwiches.add(objs[0])
                notexist_sandwiches_state.add(objs[0])
            elif pred == 'no_gluten_sandwich':
                self.sandwiches.add(objs[0])
                no_gluten_sandwiches_state.add(objs[0])
            # Collect other object types if they appear in state facts not covered above
            elif pred == 'at_kitchen_bread':
                 self.bread_portions.add(objs[0])
            elif pred == 'at_kitchen_content':
                 self.content_portions.add(objs[0])


        # Identify unserved children
        unserved_children = {c for c in self.children if c not in served_children_state}

        # Heuristic is 0 if all children are served (goal state)
        if not unserved_children:
            return 0

        # --- Calculate Heuristic Components ---

        # Component 1: Number of unserved children (cost for final 'serve' actions)
        num_unserved = len(unserved_children)

        # Determine if there are any unserved allergic or non-allergic children
        has_unserved_any = any(c not in self.allergic_children for c in unserved_children)
        has_unserved_gf = any(c in self.allergic_children for c in unserved_children)

        # Helper to check if a sandwich is suitable for *any* unserved child
        def is_sandwich_suitable_for_any_unserved(s):
             return has_unserved_any or (has_unserved_gf and s in no_gluten_sandwiches_state)

        # Count suitable sandwiches in useful locations
        suitable_sandwiches_on_trays = {s for s, t in ontray_sandwich_state.items() if is_sandwich_suitable_for_any_unserved(s)}
        suitable_sandwiches_kitchen = {s for s in at_kitchen_sandwich_state if is_sandwich_suitable_for_any_unserved(s)}

        num_suitable_on_trays = len(suitable_sandwiches_on_trays)
        num_suitable_kitchen = len(suitable_sandwiches_kitchen)

        # Total suitable sandwiches needed is at least the number of unserved children
        # This assumes one suitable sandwich per child is eventually required.
        num_total_suitable_needed = num_unserved

        # Component 2: Estimate suitable sandwiches that need to be made (cost for 'make' actions)
        # These are the needed suitable sandwiches that are not already on trays or in the kitchen.
        num_sandwiches_to_make = max(0, num_total_suitable_needed - num_suitable_on_trays - num_suitable_kitchen)

        # Component 3: Estimate suitable sandwiches in the kitchen that need to be put on trays (cost for 'put' actions)
        # These are suitable sandwiches currently in the kitchen, up to the number needed beyond those already on trays.
        num_sandwiches_kitchen_to_put = min(num_suitable_kitchen, max(0, num_total_suitable_needed - num_suitable_on_trays))

        # Component 4: Estimate places with unserved children that need a tray with a suitable sandwich (cost for 'move' actions)
        places_with_unserved_children = set(self.waiting_places[c] for c in unserved_children)
        num_places_missing_tray_with_sandwich = 0

        for p in places_with_unserved_children:
            tray_with_suitable_sandwich_found_at_p = False
            trays_at_p = {t for t, loc in at_tray_state.items() if loc == p}

            for t in trays_at_p:
                sandwiches_on_t = {s for s, tray in ontray_sandwich_state.items() if tray == t}
                for s in sandwiches_on_t:
                    # Check if this sandwich 's' is suitable for ANY unserved child waiting at place 'p'
                    is_suitable_for_any_child_at_p = False
                    for c in unserved_children:
                        if self.waiting_places[c] == p: # Child c is waiting at p
                            if c in self.allergic_children: # Child is allergic
                                if s in no_gluten_sandwiches_state:
                                    is_suitable_for_any_child_at_p = True
                                    break # Found a suitable child for this sandwich at this place
                            else: # Child is not allergic
                                is_suitable_for_any_child_at_p = True
                                break # Any sandwich is suitable for this child at this place
                    if is_suitable_for_any_child_at_p:
                        tray_with_suitable_sandwich_found_at_p = True
                        break # Found a suitable sandwich on this tray for someone at this place
                if tray_with_suitable_sandwich_found_at_p:
                    break # Found a tray at p with a suitable sandwich

            if not tray_with_suitable_sandwich_found_at_p:
                num_places_missing_tray_with_sandwich += 1

        # Sum the components to get the total heuristic value
        # This sum represents a relaxed plan cost, counting necessary steps in different stages of the process.
        heuristic_value = num_unserved + num_sandwiches_to_make + num_sandwiches_kitchen_to_put + num_places_missing_tray_with_sandwich

        return heuristic_value

    def _parse_fact(self, fact_string):
        """Helper function to parse a fact string."""
        # Remove surrounding brackets and split by space
        parts = fact_string.strip("()").split()
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects
