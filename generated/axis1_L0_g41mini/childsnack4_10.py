from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(served child1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnack domain.

    Summary
    -------
    This heuristic estimates the number of actions required to serve all children by
    making sandwiches (with gluten or gluten-free bread and contents), placing
    sandwiches on trays, moving trays if necessary, and serving sandwiches to children.

    Assumptions
    -----------
    - Each child needs exactly one sandwich served to achieve the goal.
    - Gluten-allergic children must be served no-gluten sandwiches.
    - Sandwiches are made by combining bread and content portions (gluten-free or not).
    - Sandwiches must be placed on trays located in the kitchen before serving.
    - Trays may need to be moved from kitchen to the child's waiting place.
    - Each serving action serves a single child and consumes one sandwich from a tray.
    - Bread and content portions are limited resources; each used once to make one sandwich.
    - There are enough tray(s) to place sandwiches (no explicit constraint on trays assumed).
    - The heuristic does not need to be admissible but should guide search efficiently.

    Heuristic Initialization
    ------------------------
    From static facts, extract:
    - The set of children who are allergic or not allergic to gluten.
    - The mapping of children to their waiting places.
    
    From the task goals, extract the children that must be served.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all children not yet served (goal children without a served fact).
    
    2. For each unserved child, determine if gluten-free sandwich is required.
    
    3. Count the sandwiches currently available (at_kitchen_sandwich) and on trays (ontray),
       distinguishing gluten-free sandwiches.
    
    4. Count number of sandwiches already served (to remove from "to serve").
    
    5. Compute how many sandwiches still need to be made:
       - For gluten-allergic children, sandwiches must be gluten-free.
       - For others, any sandwich suffices.
    
    6. Count available bread and content portions:
       - gluten-free bread & content for gluten-free sandwiches,
       - any bread & content for regular sandwiches.
    
    7. Estimate actions to make missing sandwiches:
       - One action per sandwich making.
       - Check if enough bread and content exist for required sandwich types.
       - If insufficient resources are detected, heuristic returns high cost.
    
    8. Estimate the actions to put sandwiches onto trays:
       - Sandwiches not already on trays need one put_on_tray action each.
    
    9. Estimate the tray movements:
       - For each tray that holds sandwiches but is not at the child's waiting place,
         at least one move_tray action is needed (assumed one move per tray).
    
    10. Estimate the serving actions:
        - One serve action per unserved child.
    
    11. Sum all estimated action counts.
    
    This heuristic focuses on counting the minimum required actions to:
    - make sandwiches,
    - place sandwiches on trays,
    - move trays to children,
    - serve children.
    It assumes optimal grouping of sandwiches on trays and trays moved efficiently,
    but counts as if each tray requires one move when not at the goal place.

    The heuristic returns 0 if all children are served (goal state).
    """

    def __init__(self, task):
        """Extract static knowledge about children, allergies, and waiting places."""
        self.goals = task.goals

        # Extract children to serve from goal conditions: (served child)
        self.goal_children = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "served" and len(parts) == 2:
                self.goal_children.add(parts[1])

        # Extract static facts about allergy and waiting places
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {}

        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif len(parts) == 2 and parts[0] == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif len(parts) == 3 and parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.child_waiting_place[child] = place

        # Extract sets of gluten-free bread and content from static facts
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        for fact in task.static:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == "no_gluten_bread":
                self.no_gluten_bread.add(parts[1])
            elif len(parts) == 2 and parts[0] == "no_gluten_content":
                self.no_gluten_content.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to reach the goal from the current state."""
        state = node.state

        # Helper to check if a fact is in the state efficiently
        def has_fact(pattern):
            # pattern is a tuple of parts, e.g. ("served", child)
            fact_str = "(" + " ".join(pattern) + ")"
            return fact_str in state

        # 1. Identify unserved children (goal children that are not served)
        unserved_children = set(
            child for child in self.goal_children if not has_fact(("served", child))
        )
        if not unserved_children:
            # Goal achieved, no cost needed
            return 0

        # 2. Partition unserved children by allergy
        gluten_allergic_children = unserved_children & self.allergic_children
        non_allergic_children = unserved_children - gluten_allergic_children

        # 3. Count sandwiches at kitchen and on trays, gluten-free and regular
        # Note: sandwiches can be distinguished by presence of no_gluten_sandwich predicate
        sandwiches_at_kitchen = set()
        gluten_free_sandwiches_at_kitchen = set()
        for fact in state:
            if fnmatch(fact, "(at_kitchen_sandwich *)"):
                parts = get_parts(fact)
                s = parts[1]
                sandwiches_at_kitchen.add(s)
        gluten_free_sandwiches = set()
        for fact in state:
            if fnmatch(fact, "(no_gluten_sandwich *)"):
                parts = get_parts(fact)
                gluten_free_sandwiches.add(parts[1])

        gluten_free_sandwiches_at_kitchen = sandwiches_at_kitchen & gluten_free_sandwiches
        regular_sandwiches_at_kitchen = sandwiches_at_kitchen - gluten_free_sandwiches_at_kitchen

        # Sandwiches on trays: ontray ?s ?t
        sandwiches_on_tray = set()
        gluten_free_sandwiches_on_tray = set()
        trays_with_sandwiches = set()  # trays that currently hold any sandwich
        tray_location = {}
        for fact in state:
            if fnmatch(fact, "(ontray * *)"):
                parts = get_parts(fact)
                s, t = parts[1], parts[2]
                sandwiches_on_tray.add(s)
                trays_with_sandwiches.add(t)
                # We record tray location in the next step inside tray_location dict (see next step)
                # Also determine if sandwich is gluten free
                if s in gluten_free_sandwiches:
                    gluten_free_sandwiches_on_tray.add(s)
        # Trays locations: (at ?t ?p)
        tray_places = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "at":
                t, p = parts[1], parts[2]
                tray_places[t] = p

        # 4. Count how many sandwiches are ready to be served per allergy type:
        # Sandwiches counted are those on trays (ready to serve).
        gluten_free_sandwiches_ready = gluten_free_sandwiches_on_tray
        regular_sandwiches_ready = sandwiches_on_tray - gluten_free_sandwiches_ready

        # 5. Count served children to exclude them
        # Already excluded in unserved_children, so no extra action here

        # 6. Compute how many sandwiches must be made
        gluten_free_needed = max(0, len(gluten_allergic_children) - len(gluten_free_sandwiches_ready) - len(gluten_free_sandwiches_at_kitchen))
        regular_needed = max(0, len(non_allergic_children) - len(regular_sandwiches_ready) - len(regular_sandwiches_at_kitchen))

        # 7. Count available bread and content portions, gluten-free and regular
        bread_at_kitchen = set()
        for fact in state:
            if fnmatch(fact, "(at_kitchen_bread *)"):
                parts = get_parts(fact)
                bread_at_kitchen.add(parts[1])
        content_at_kitchen = set()
        for fact in state:
            if fnmatch(fact, "(at_kitchen_content *)"):
                parts = get_parts(fact)
                content_at_kitchen.add(parts[1])

        gluten_free_bread_available = bread_at_kitchen & self.no_gluten_bread
        gluten_free_content_available = content_at_kitchen & self.no_gluten_content
        regular_bread_available = bread_at_kitchen - gluten_free_bread_available
        regular_content_available = content_at_kitchen - gluten_free_content_available

        # Check if resources are enough, else return a large heuristic value to discourage deadends
        # For gluten-free sandwiches: need one gluten-free bread & content per sandwich made
        if len(gluten_free_bread_available) < gluten_free_needed or len(gluten_free_content_available) < gluten_free_needed:
            return 10000  # large heuristic cost representing impossible resource shortage

        # For regular sandwiches similarly
        if len(regular_bread_available) < regular_needed or len(regular_content_available) < regular_needed:
            return 10000

        # 8. Estimate number of actions for making sandwiches
        # Each sandwich requires exactly one make_sandwich[_no_gluten] action
        make_actions = gluten_free_needed + regular_needed

        # 9. Estimate number of put_on_tray actions
        # Sandwiches not already on trays must be put on trays.
        # Count sandwiches at kitchen (excluding those already on trays)
        gluten_free_sandwiches_not_on_tray = gluten_free_sandwiches_at_kitchen
        regular_sandwiches_not_on_tray = regular_sandwiches_at_kitchen
        # Actually sandwiches on trays are not at kitchen, so these are correctly the sets of sandwiches not on trays.
        put_on_tray_actions = len(gluten_free_sandwiches_not_on_tray) + len(regular_sandwiches_not_on_tray)

        # 10. Estimate number of move_tray actions
        # For each tray holding sandwiches, if tray is not at the waiting place of all children it needs to serve,
        # then at least one move action is needed.
        # Since trays can serve multiple children, assume that for each tray at a different place than child's waiting place,
        # one move_tray action is required per tray moving from kitchen to child's place or between places.
        # To estimate effectively, for each tray:
        #   - Check if tray location equals the waiting place of any unserved child whom sandwiches on that tray could serve.
        # We assume sandwiches on a tray can serve children of matching allergy (optimistically).
        # For simplicity, conservatively count needed moves as one per tray not at the waiting places of *any* unserved children.

        # First, map sandwiches to allergy requirement for serving:
        sandwich_allergy_map = {}
        for s in gluten_free_sandwiches_at_kitchen | gluten_free_sandwiches_on_tray:
            sandwich_allergy_map[s] = "gf"
        for s in regular_sandwiches_at_kitchen | (sandwiches_on_tray - gluten_free_sandwiches_on_tray):
            sandwich_allergy_map[s] = "reg"

        # For each tray, find the allergy types of sandwiches on it
        tray_allergy_types = {}
        for fact in state:
            if fnmatch(fact, "(ontray * *)"):
                parts = get_parts(fact)
                s, t = parts[1], parts[2]
                if t not in tray_allergy_types:
                    tray_allergy_types[t] = set()
                tray_allergy_types[t].add(sandwich_allergy_map.get(s, None))

        move_tray_actions = 0
        for t in trays_with_sandwiches:
            tray_pos = tray_places.get(t, None)
            if tray_pos is None:
                # unknown tray position should count as needing a move
                move_tray_actions += 1
                continue
            allergies = tray_allergy_types.get(t, set())
            # Find waiting places of children that can be served by this tray's sandwiches
            served_children_places = set()
            for child in unserved_children:
                # Check if child's allergy matches tray sandwiches allergy
                if child in gluten_allergic_children and "gf" in allergies:
                    served_children_places.add(self.child_waiting_place.get(child, None))
                elif child in non_allergic_children and "reg" in allergies:
                    served_children_places.add(self.child_waiting_place.get(child, None))
                elif child in non_allergic_children and "gf" in allergies:
                    # gluten-free sandwiches can also serve non-allergic children if needed?
                    # According to domain, this is not explicitly stated but safe to assume yes.
                    served_children_places.add(self.child_waiting_place.get(child, None))
            # If the tray position is not any of those waiting places, one move needed
            if tray_pos not in served_children_places:
                move_tray_actions += 1

        # 11. Serving actions: One per unserved child
        serve_actions = len(unserved_children)

        # Summing all actions
        total_value = make_actions + put_on_tray_actions + move_tray_actions + serve_actions

        return total_value
