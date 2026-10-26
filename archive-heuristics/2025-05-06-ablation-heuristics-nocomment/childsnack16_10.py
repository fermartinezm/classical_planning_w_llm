from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Remove leading/trailing parentheses and split by space
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
        Estimates the cost to reach the goal by summing up the minimum
        estimated actions required for each unserved child independently.
        The cost for a child depends on the current state of a suitable
        sandwich for them, representing the estimated number of actions
        needed to get a suitable sandwich to the child and serve it.
        The cost levels are:
        - 1 action: Serve (suitable sandwich already on a tray at the child's location).
        - 2 actions: Move tray + Serve (suitable sandwich on a tray elsewhere).
        - 3 actions: Put on tray + Move tray + Serve (suitable sandwich made, available at kitchen).
        - 4 actions: Make sandwich + Put on tray + Move tray + Serve (ingredients available to make suitable sandwich).
        - 1000 actions: Cannot make a suitable sandwich (likely unsolvable state for this child).

    Assumptions:
        - The goal is always a conjunction of (served ?c) for some children.
        - The heuristic assumes unit cost for all actions.
        - For cost levels 3 and 4, it is implicitly assumed that a tray can be made
          available at the kitchen to put the sandwich on. The cost of moving a
          tray to the kitchen, if necessary, is not explicitly added per child,
          as this is a shared action. This simplification contributes to the
          heuristic being non-admissible but aims to provide better guidance
          for greedy search.
        - It is assumed that if ingredients are available and an unused sandwich
          object exists, a suitable sandwich can be made.

    Heuristic Initialization:
        - Parses static facts from the task to identify:
            - Children who are allergic to gluten.
            - Children who are not allergic to gluten.
            - The waiting place for each child.
            - Gluten-free bread portions.
            - Gluten-free content portions.
        - This information is stored in sets and a dictionary for efficient lookup
          during heuristic computation.
        - Identifies the set of children that need to be served based on the task goals.

    Step-By-Step Thinking for Computing Heuristic:
        1. Initialize the total heuristic cost `h_value` to 0.
        2. Identify the set of children that are listed in the task goals but are not
           currently marked as `served` in the state. These are the `unserved_children`.
        3. If there are no unserved children, the goal is reached, so return `h_value = 0`.
        4. Pre-process the current state facts into a dictionary mapping predicate names
           to lists of their argument parts. This allows for faster lookup of facts
           by predicate.
        5. For each `child` in the `unserved_children` set:
            a. Retrieve the child's waiting place (`child_place`) using the pre-calculated
               `waiting_places` dictionary. If a child is in the goal but not waiting,
               add a large penalty (1000) as this indicates an unexpected state.
            b. Determine if the child is allergic or not using the pre-calculated
               `allergic_children` and `not_allergic_children` sets. This determines
               the required sandwich type (gluten-free or any).
            c. Check for the "closest" available suitable sandwich for this child,
               following a prioritized order:
                i.  **Cost 1 (Serve):** Is there a sandwich `S` on a tray `T` (`(ontray S T)` in state)
                    such that `T` is at the child's place (`(at T child_place)` in state)
                    and `S` is suitable for the child (`_is_suitable_sandwich(S, child, state_predicates)` is true)?
                    If yes, add 1 to `h_value` for this child and move to the next unserved child.
                ii. **Cost 2 (Move + Serve):** If not found at the child's place, is there a
                    sandwich `S` on a tray `T` (`(ontray S T)` in state) such that `T` is
                    at *any other* place `P'` (`(at T P')` in state where `P' != child_place`)
                    and `S` is suitable for the child?
                    If yes, add 2 to `h_value` for this child and move to the next unserved child.
                iii. **Cost 3 (Put on Tray + Move + Serve):** If not found on any tray, is there a
                     suitable sandwich `S` available at the kitchen (`(at_kitchen_sandwich S)` in state
                     and `_is_suitable_sandwich(S, child, state_predicates)` is true)?
                     If yes, add 3 to `h_value` for this child and move to the next unserved child.
                     (This assumes a tray can be used at the kitchen).
                iv. **Cost 4 (Make + Put on Tray + Move + Serve):** If not found at the kitchen,
                    can a suitable sandwich be made (`_can_make_suitable_sandwich(child, state_predicates)` is true)?
                    This requires available ingredients at the kitchen and an unused sandwich object.
                    If yes, add 4 to `h_value` for this child and move to the next unserved child.
                    (This assumes a tray can be used at the kitchen).
                v.  **Cost 1000 (Unsolvable):** If none of the above conditions are met, it is
                    likely impossible to get a suitable sandwich for this child. Add 1000
                    to `h_value` as a large penalty.
        6. Return the total `h_value`.
    """
    def __init__(self, task):
        super().__init__(task) # Call the base class constructor
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_places = {}
        self.no_gluten_bread = set()
        self.no_gluten_content = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif parts[0] == "waiting":
                # Static fact is (waiting child place)
                self.waiting_places[parts[1]] = parts[2]
            elif parts[0] == "no_gluten_bread":
                self.no_gluten_bread.add(parts[1])
            elif parts[0] == "no_gluten_content":
                self.no_gluten_content.add(parts[1])

        # Identify all children who are goals
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "served"}


    def _is_suitable_sandwich(self, sandwich, child, state_predicates):
        """Checks if a sandwich is suitable for a child based on allergy."""
        is_allergic = child in self.allergic_children
        # Check if '(no_gluten_sandwich sandwich)' is in the state_predicates
        is_gluten_free_sandwich = any(p[1] == sandwich for p in state_predicates.get('no_gluten_sandwich', []))

        if is_allergic:
            return is_gluten_free_sandwich
        else: # not_allergic_gluten
            return True # Any sandwich is suitable

    def _can_make_suitable_sandwich(self, child, state_predicates):
        """Checks if ingredients and unused sandwich object exist to make a suitable sandwich."""
        # Need an unused sandwich object
        has_unused_sandwich_object = any(True for _ in state_predicates.get('notexist', []))
        if not has_unused_sandwich_object:
            return False

        is_allergic = child in self.allergic_children

        available_bread = {p[1] for p in state_predicates.get('at_kitchen_bread', [])}
        available_content = {p[1] for p in state_predicates.get('at_kitchen_content', [])}

        if is_allergic:
            # Need gluten-free bread and content
            has_gf_bread = any(b in self.no_gluten_bread for b in available_bread)
            has_gf_content = any(c in self.no_gluten_content for c in available_content)
            return has_gf_bread and has_gf_content
        else: # not_allergic_gluten
            # Need any bread and any content
            return len(available_bread) > 0 and len(available_content) > 0


    def __call__(self, node):
        state = node.state
        h_value = 0

        # Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if get_parts(fact)[0] == 'served'}
        unserved_children = self.goal_children - served_children

        if not unserved_children:
            return 0 # Goal state

        # Pre-process state facts for faster lookup
        state_predicates = {}
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate not in state_predicates:
                state_predicates[predicate] = []
            state_predicates[predicate].append(parts)

        # Calculate cost for each unserved child
        for child in unserved_children:
            child_place = self.waiting_places.get(child)
            if child_place is None:
                 # Child is a goal but not waiting? Problem definition issue or dead end.
                 h_value += 1000
                 continue

            # Cost 1: Suitable sandwich on tray at child's location?
            found_ready_sandwich = False
            for ontray_parts in state_predicates.get('ontray', []):
                sandwich = ontray_parts[1]
                tray = ontray_parts[2]
                # Check if '(at tray child_place)' is in state
                if any(p[1] == tray and p[2] == child_place for p in state_predicates.get('at', [])):
                    if self._is_suitable_sandwich(sandwich, child, state_predicates):
                        h_value += 1
                        found_ready_sandwich = True
                        break # Found the best case for this child

            if found_ready_sandwich:
                continue # Move to the next child

            # Cost 2: Suitable sandwich on tray elsewhere?
            found_ontray_elsewhere = False
            for ontray_parts in state_predicates.get('ontray', []):
                sandwich = ontray_parts[1]
                tray = ontray_parts[2]
                # Check if tray is anywhere *not* at child_place
                tray_at_other_place = any(p[1] == tray and p[2] != child_place for p in state_predicates.get('at', []))
                if tray_at_other_place:
                     if self._is_suitable_sandwich(sandwich, child, state_predicates):
                        h_value += 2
                        found_ontray_elsewhere = True
                        break # Found the next best case for this child

            if found_ontray_elsewhere:
                continue # Move to the next child

            # Cost 3: Suitable sandwich at kitchen?
            found_kitchen_sandwich = False
            for kitchen_sandwich_parts in state_predicates.get('at_kitchen_sandwich', []):
                sandwich = kitchen_sandwich_parts[1]
                if self._is_suitable_sandwich(sandwich, child, state_predicates):
                    h_value += 3
                    found_kitchen_sandwich = True
                    break # Found the next best case for this child

            if found_kitchen_sandwich:
                 continue # Move to the next child

            # Cost 4: Can make a suitable sandwich?
            if self._can_make_suitable_sandwich(child, state_predicates):
                 h_value += 4
            else:
                # Cannot make a suitable sandwich
                h_value += 1000 # Large penalty

        return h_value
