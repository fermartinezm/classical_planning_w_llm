from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at_kitchen_bread bread1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnack domain.

    Summary:
    ----------
    Estimates the number of actions required to serve all children by preparing
    appropriate sandwiches (gluten or no-gluten based on each child’s allergy),
    placing them on trays, moving trays if needed, and serving those sandwiches.

    Assumptions:
    -----------
    - Every child waiting to be served must get exactly one sandwich matching their allergy
      (no gluten sandwich for allergic children, any sandwich for non-allergic).
    - Bread portions and content portions are used up when making sandwiches.
    - Each sandwich is made from exactly one bread portion and one content portion.
    - Sandwiches must be placed on trays that may need to be moved to the child's waiting place.
    - Trays start in the kitchen and can be moved between places.
    - No partial orders or re-use of sandwich ingredients.
    - Heuristic does not need to be admissible but should guide search efficiently.

    Heuristic Initialization:
    -------------------------
    - Extract from static facts which children are allergic or not.
    - Extract which breads and contents are gluten-free.
    - Extract which sandwiches currently exist and if they are gluten-free.
    - Extract facts about trays and their current places.
    - Store sets of children that still need to be served (goal is served ?child).
    - Store all bread and content portions initially available in the kitchen.
    - Extract the initial count of trays available.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Determine the set of children not yet served.
    2. For each child, identify allergy status to determine sandwich requirement:
       - Allergic children must receive a no-gluten sandwich.
       - Non-allergic children may have any sandwich.
    3. Count how many sandwiches of appropriate type currently exist and are available in the kitchen or on trays.
    4. Compute the number of sandwiches that still need to be made for each category.
    5. Estimate how many bread and content portions remain in the kitchen for sandwich-making.
       Ensure enough no-gluten bread/content for allergic children sandwiches.
    6. Estimate the minimal number of sandwiches needed to serve all remaining children.
    7. Estimate the minimal number of sandwiches not yet placed on trays and need to be put on trays.
    8. Estimate tray moves needed to bring trays with sandwiches to children’s places.
       - Since trays start in kitchen or somewhere else, at least one move per distinct target place that has waiting children.
    9. Estimate the number of serve actions needed (one per child).
    10. Sum actions for making sandwiches, putting sandwiches on trays, moving trays, and serving children.

    Action cost assumptions:
    -------------------------
    - Making one sandwich: 1 action
    - Putting sandwich on tray: 1 action
    - Moving one tray between places: 1 action per move
      (Assume at least one move per distinct place that has waiting children,
      ignoring actual multipath complexity)
    - Serving one child: 1 action

    This heuristic aims to be a quick yet informative approximation of how many steps remain.
    """

    def __init__(self, task):
        """
        Extract static domain information such as:
        - Children and their gluten allergy status (allergic_gluten or not_allergic_gluten).
        - Bread portions with and without gluten.
        - Content portions with and without gluten.
        Store these for quick reference during heuristic evaluation.
        """
        self.goals = task.goals
        self.static = task.static

        self.allergic_children = set()
        self.non_allergic_children = set()

        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        # Track children from static info
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                self.non_allergic_children.add(parts[1])
            elif parts[0] == "no_gluten_bread":
                self.no_gluten_breads.add(parts[1])
            elif parts[0] == "no_gluten_content":
                self.no_gluten_contents.add(parts[1])

    def __call__(self, node):
        """
        Calculate the heuristic value estimating the needed number of actions 
        from the current state to reach the goal.

        Major reasoning steps:
         - Identify children still waiting to be served.
         - Identify sandwiches currently existing:
            * which sandwiches are no-gluten
            * which sandwiches are gluten-containing
         - Count breads and contents available in the kitchen.
         - Count how many sandwiches are already put on trays.
         - Count trays and their locations.
         - Count children grouped by place.
         - Compute how many sandwiches must yet be made for allergic and non-allergic children.
         - Compute actions for making sandwiches, placing on trays, moving trays if needed, and serving.
        """

        state = node.state

        # Helpers to collect facts by predicate
        def facts_by_predicate(pred):
            return {get_parts(fact)[1:] for fact in state if fact.startswith(f'({pred} ')}

        # Children who still need to be served (waiting but not served)
        served_children = {get_parts(goal)[1] for goal in self.goals}  # from goals "served ?child"
        served_now = {get_parts(fact)[1] for fact in state if fact.startswith("(served ")}
        waiting_facts = [get_parts(fact) for fact in state if fact[0] == "waiting"]
        waiting_children = {child for child, place in waiting_facts if child in served_children and child not in served_now}

        # Children allergic and non-allergic
        allergic_children_waiting = self.allergic_children.intersection(waiting_children)
        non_allergic_children_waiting = self.non_allergic_children.intersection(waiting_children)

        # Count breads in kitchen
        breads_in_kitchen = {parts[1] for fact in state if fact.startswith("(at_kitchen_bread ")}
        # Count contents in kitchen
        contents_in_kitchen = {parts[1] for fact in state if fact.startswith("(at_kitchen_content ")}

        # Count sandwiches in kitchen
        sandwiches_in_kitchen = {parts[1] for fact in state if fact.startswith("(at_kitchen_sandwich ")}

        # Sandwiches that are no-gluten and exist (not notexist)
        no_gluten_sandwiches = {parts[1] for fact in state if fact.startswith("(no_gluten_sandwich ")}

        # Sandwiches on trays (placed)
        sandwiches_on_trays = {get_parts(fact)[1] for fact in state if fact.startswith("(ontray ")}

        # Trays and their current places
        trays_places = {}  # tray -> place
        for fact in state:
            if fact.startswith("(at "):
                parts = get_parts(fact)
                # Only trays have the predicate (at ?t ?p)
                # but places also appear in other predicates - consider as tray if ?t is tray
                tray_candidate = parts[1]
                if tray_candidate.startswith("tray"):
                    trays_places[tray_candidate] = parts[2]

        # Wait places of children still waiting to be served
        waiting_places = {place for child, place in waiting_facts if child in waiting_children}

        # Existing sandwiches that can be used to serve
        # Categorize sandwiches by gluten status
        gluten_sandwiches = set()
        for s in sandwiches_in_kitchen.union(sandwiches_on_trays):
            if s in no_gluten_sandwiches:
                continue
            gluten_sandwiches.add(s)
        no_gluten_sandwiches_available = no_gluten_sandwiches.intersection(sandwiches_in_kitchen.union(sandwiches_on_trays))

        # Count sandwiches of each type available (in kitchen or on trays)
        num_no_gluten_sandwiches = len(no_gluten_sandwiches_available)
        num_gluten_sandwiches = len(gluten_sandwiches)

        # How many sandwiches are needed for each category
        needed_no_gluten = max(0, len(allergic_children_waiting) - num_no_gluten_sandwiches)
        needed_gluten = max(0, len(non_allergic_children_waiting) - num_gluten_sandwiches)

        # Check if enough bread and content portions for sandwiches to be made
        # no_gluten breads and contents needed for no_gluten sandwiches
        num_no_gluten_breads_available = len(self.no_gluten_breads.intersection(breads_in_kitchen))
        num_no_gluten_contents_available = len(self.no_gluten_contents.intersection(contents_in_kitchen))

        # gluten breads and contents for gluten sandwiches
        gluten_breads_available = len(breads_in_kitchen) - num_no_gluten_breads_available
        gluten_contents_available = len(contents_in_kitchen) - num_no_gluten_contents_available

        # For heuristic, just check minimal availability as an estimate (not exact matching)

        # Estimate sandwiches that can be made limited by ingredients
        can_make_no_gluten = min(needed_no_gluten, num_no_gluten_breads_available, num_no_gluten_contents_available)
        can_make_gluten = min(needed_gluten, gluten_breads_available, gluten_contents_available)

        # The heuristic assumes we can make at most these; if not enough ingredients, add a penalty of missing sandwiches

        # Current sandwiches already on trays for each category
        # For simplicity, sandwiches on trays count as already 'placed', unplaced sandwiches are counts of needed sandwiches
        sandwiches_on_trays_no_gluten = no_gluten_sandwiches_available.intersection(sandwiches_on_trays)
        sandwiches_on_trays_gluten = gluten_sandwiches.intersection(sandwiches_on_trays)

        # How many sandwiches still need to be put on trays
        not_on_tray_no_gluten = max(0, len(allergic_children_waiting) - len(sandwiches_on_trays_no_gluten))
        not_on_tray_gluten = max(0, len(non_allergic_children_waiting) - len(sandwiches_on_trays_gluten))

        # Total sandwiches to serve = children waiting
        total_sandwiches_to_serve = len(waiting_children)

        # Number of trays present
        total_trays = len(trays_places)

        # Estimate tray moves: at least one move per distinct waiting place (except kitchen if trays there)
        # If tray already at the child's place, no move needed for that place
        places_needing_tray_move = 0
        for place in waiting_places:
            trays_at_place = [t for t, p in trays_places.items() if p == place]
            if len(trays_at_place) == 0:
                # No tray currently at child’s place, one move assumed
                places_needing_tray_move += 1

        # Actions estimation:
        # Make sandwiches (needed sandwich count) -> 1 action each
        make_actions = can_make_no_gluten + can_make_gluten
        # For missing ingredients/sandwiches, add penalty as missing count (to keep heuristic finite)
        missing_no_gluten = needed_no_gluten - can_make_no_gluten
        missing_gluten = needed_gluten - can_make_gluten
        missing_penalty = missing_no_gluten + missing_gluten

        # Put sandwiches on trays -> 1 action per sandwich not on tray
        put_on_tray_actions = not_on_tray_no_gluten + not_on_tray_gluten

        # Move trays actions as estimated
        move_tray_actions = places_needing_tray_move

        # Serve actions: one per child waiting
        serve_actions = total_sandwiches_to_serve

        heuristic_value = (
            make_actions
            + missing_penalty  # penalty for missing ingredients or sandwiches
            + put_on_tray_actions
            + move_tray_actions
            + serve_actions
        )

        # The heuristic value is zero only if all children are served (goal reached)
        if len(waiting_children) == 0:
            heuristic_value = 0

        # Ensure heuristic is at least zero
        return max(heuristic_value, 0)
