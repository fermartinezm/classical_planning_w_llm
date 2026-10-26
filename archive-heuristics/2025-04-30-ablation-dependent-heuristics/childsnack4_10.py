# Import necessary modules (none specific needed beyond standard types)

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for each unserved child. It considers the
    need to make suitable sandwiches, put them on trays, move trays to the
    children's locations, and finally serve the children. It is designed for
    greedy best-first search and is not admissible.

    Assumptions:
    - The heuristic assumes the problem is solvable, meaning necessary resources
      (bread, content, sandwich objects, trays) are eventually available, even
      if not immediately present in sufficient quantities for all parallel needs.
      It primarily counts the actions (make, put_on_tray, move_tray, serve)
      required based on the state of children and available/needed sandwiches/trays,
      without complex resource conflict analysis.
    - The heuristic is not admissible; it may overestimate the cost.
    - The heuristic is 0 if and only if the goal state is reached.
    - The heuristic value is finite for all solvable states.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task definition.
    It extracts and stores:
    - The set of children that need to be served (from the goal).
    - The allergy status (allergic/not_allergic) for each child.
    - The waiting place for each child.
    - The gluten-free status of bread and content portions (though these static GF facts for ingredients are not directly used in the current heuristic calculation, they are part of the domain's static info and could be useful for more complex heuristics or validation).
    This information is stored in class attributes for efficient lookup during
    heuristic computation for different states.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic is computed as follows:
    1.  Identify the set of children that are in the goal but are not yet marked as 'served' in the current state. If this set is empty, the goal is reached, and the heuristic is 0.
    2.  For each unserved child, determine their waiting place and allergy status using the pre-processed static information.
    3.  Count the total number of gluten-free sandwiches and regular sandwiches required to serve all unserved children based on their allergy status.
    4.  Count the number of gluten-free and regular sandwiches currently available in the kitchen or on any tray in the current state.
    5.  Estimate the number of 'make_sandwich' (or 'make_sandwich_no_gluten') actions needed: This is the number of required sandwiches of each type minus the number of available sandwiches of that type, taking the maximum with 0. Sum these counts for gluten-free and regular sandwiches. Add this sum to the heuristic value.
    6.  Count the number of sandwiches currently located 'at_kitchen_sandwich'. These sandwiches need a 'put_on_tray' action. Add this count to the heuristic value.
    7.  Identify all unique places where unserved children are waiting. For each such place, check if there is at least one tray currently located there. If a place with unserved children has no tray, estimate that one 'move_tray' action is needed to bring a tray there. Add the count of such places to the heuristic value.
    8.  Add the total number of unserved children to the heuristic value. This represents the final 'serve_sandwich' action needed for each child.
    9.  The total heuristic value is the sum of the counts from steps 5, 6, 7, and 8.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by pre-processing static task information.

        Args:
            task: The planning task object (an instance of the Task class).
        """
        self.goal_children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {} # {child: place}
        self.no_gluten_bread_set = set() # Static GF bread
        self.no_gluten_content_set = set() # Static GF content

        # Process static facts
        for fact in task.static:
            if fact.startswith('(allergic_gluten '):
                child = self._extract_objects(fact)[0]
                self.allergic_children.add(child)
            elif fact.startswith('(not_allergic_gluten '):
                child = self._extract_objects(fact)[0]
                self.not_allergic_children.add(child)
            elif fact.startswith('(waiting '):
                child, place = self._extract_objects(fact)
                self.child_waiting_place[child] = place
            elif fact.startswith('(no_gluten_bread '):
                bread = self._extract_objects(fact)[0]
                self.no_gluten_bread_set.add(bread)
            elif fact.startswith('(no_gluten_content '):
                content = self._extract_objects(fact)[0]
                self.no_gluten_content_set.add(content)

        # Process goal facts
        for fact in task.goals:
            if fact.startswith('(served '):
                child = self._extract_objects(fact)[0]
                self.goal_children.add(child)

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (a frozenset of facts).

        Returns:
            The estimated number of actions to reach the goal state.
        """
        # 1. Identify unserved children
        served_children_in_state = set()
        kitchen_sandwiches = set()
        ontray_sandwiches_map = {} # {tray: set(sandwiches)}
        tray_locations_map = {} # {tray: place}
        no_gluten_sandwich_set_state = set() # Sandwiches known to be GF *in this state*

        for fact in state:
            if fact.startswith('(served '):
                served_children_in_state.add(self._extract_objects(fact)[0])
            elif fact.startswith('(at_kitchen_sandwich '):
                kitchen_sandwiches.add(self._extract_objects(fact)[0])
            elif fact.startswith('(ontray '):
                s, t = self._extract_objects(fact)
                ontray_sandwiches_map.setdefault(t, set()).add(s)
            elif fact.startswith('(at '):
                t, p = self._extract_objects(fact)
                tray_locations_map[t] = p
            elif fact.startswith('(no_gluten_sandwich '):
                no_gluten_sandwich_set_state.add(self._extract_objects(fact)[0])

        unserved_children = self.goal_children - served_children_in_state

        if not unserved_children:
            return 0 # Goal reached

        h = 0

        # 8. Add cost for the final 'serve' action for each unserved child
        h += len(unserved_children)

        # 2. Map unserved children to their needs and location
        places_with_unserved = set()
        gf_needed_total = 0
        reg_needed_total = 0

        for child in unserved_children:
            # Place should exist based on problem definition, but use .get for robustness
            place = self.child_waiting_place.get(child)
            places_with_unserved.add(place)

            if child in self.allergic_children:
                gf_needed_total += 1
            elif child in self.not_allergic_children:
                 reg_needed_total += 1
            # else: child allergy status is unknown, assume not allergic? Or problem is ill-defined.
            # Assuming problem is well-defined and child is either allergic or not.


        # 3. Count available sandwiches by type and location
        N_gf_kitchen = len([s for s in kitchen_sandwiches if s in no_gluten_sandwich_set_state])
        N_reg_kitchen = len([s for s in kitchen_sandwiches if s not in no_gluten_sandwich_set_state])

        N_gf_ontray = 0
        N_reg_ontray = 0
        for sandwiches_on_t in ontray_sandwiches_map.values():
             N_gf_ontray += len([s for s in sandwiches_on_t if s in no_gluten_sandwich_set_state])
             N_reg_ontray += len([s for s in sandwiches_on_t if s not in no_gluten_sandwich_set_state])

        N_gf_available_total = N_gf_kitchen + N_gf_ontray
        N_reg_available_total = N_reg_kitchen + N_reg_ontray

        # 5. Estimate 'make' actions needed
        make_gf_count = max(0, gf_needed_total - N_gf_available_total)
        make_reg_count = max(0, reg_needed_total - N_reg_available_total)
        h += make_gf_count
        h += make_reg_count

        # 6. Estimate 'put_on_tray' actions needed
        put_on_tray_count = len(kitchen_sandwiches)
        h += put_on_tray_count

        # 7. Estimate 'move_tray' actions needed
        places_with_trays = set(tray_locations_map.values())
        places_needing_tray = places_with_unserved - places_with_trays
        h += len(places_needing_tray)

        return h

    @staticmethod
    def _extract_objects(fact_string):
        """
        Helper to extract object names from fact strings like '(predicate obj1 obj2)'.
        Assumes fact_string is a valid PDDL fact representation string.
        """
        # Remove leading/trailing brackets and split by space
        parts = fact_string[1:-1].split()
        # Return all parts except the first one (the predicate name)
        return parts[1:]
