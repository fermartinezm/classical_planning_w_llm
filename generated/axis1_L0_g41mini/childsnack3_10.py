from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """
    Extract the components of a PDDL fact by removing parentheses and splitting the string.
    Example: "(at_kitchen_bread bread1)" -> ["at_kitchen_bread", "bread1"]
    """
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards '*'.

    :param fact: The PDDL fact string, e.g. "(at_kitchen_bread bread1)".
    :param args: The pattern parts to match, e.g. ("at_kitchen_bread", "*").
    :return: True if fact matches the pattern, False otherwise.
    """
    parts = get_parts(fact)
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnack domain.

    Summary
    -------
    This heuristic estimates how many actions are required to serve all waiting children
    in the current state. It does so by reasoning about the sandwiches needed per child,
    the preparation of sandwiches (no gluten or regular), placing them on trays,
    tray movements, and serving sandwiches to children. The heuristic is designed
    to guide a greedy best-first search effectively and efficiently.

    Assumptions
    -----------
    - Each child needs exactly one served sandwich.
    - Gluten-allergic children require a no-gluten sandwich.
    - Bread and content portions are consumed when making sandwiches.
    - Sandwiches must be placed on trays and trays must be moved to the child's place before serving.
    - It is possible to distinguish no-gluten sandwiches by the predicate `no_gluten_sandwich`.
    - Each action costs 1 unit.
    - Trays exist and start at the kitchen or elsewhere.
    - Multiple sandwiches can be placed on a tray before moving and serving.

    Heuristic Initialization
    ------------------------
    - Extract from static facts which children are allergic and which are not.
    - Extract from current goals which children must be served.
    - Extract sets of available breads and contents with no gluten for making no-gluten sandwiches.
    - Extract sets of breads and contents for regular sandwiches.
    - Extract tray identities.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify which children still need to be served (goal served fact missing in current state).
    2. For allergic children waiting unserved, count how many no-gluten sandwiches are available or can be made.
    3. For non-allergic children waiting unserved, count how many regular sandwiches are available or can be made.
    4. Compute the number of sandwiches still needed (subtract already available sandwiches on trays or in kitchen).
    5. Count the number of sandwiches currently on any tray and how many trays are at kitchen or at children places.
    6. For sandwiches not made yet, count the estimated needed "make_sandwich" actions:
       - For allergic children: need "make_sandwich_no_gluten" action.
       - For others: "make_sandwich".
    7. For sandwiches made but not on trays, add "put_on_tray" actions.
    8. For trays not at the child's place, add "move_tray" actions.
    9. For each waiting child not served yet, add a "serve_sandwich" or "serve_sandwich_no_gluten" action.
    10. Sum these actions for the heuristic estimate.
    11. Use minimal estimates, assuming sandwiches can be bundled on trays and trays can serve multiple children if possible.
    """

    def __init__(self, task):
        """
        Initialize heuristic by extracting static information and goal info.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract child allergy info
        self.allergic_children = set()
        self.non_allergic_children = set()
        for fact in task.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.non_allergic_children.add(child)

        # Extract trays
        # Trays will be identified dynamically as any object of type tray from initial facts or static (better done dynamically at call time, but domain only defines trays)
        # Given no explicit static trays, we'll find all trays from facts using pattern (at ?t place) or (ontray ?s ?t)
        # We do minimal static, trays will be identified in the call method.

    def __call__(self, node):
        """
        Compute heuristic estimate of the number of actions required to serve all waiting children.

        :param node: The search node containing the current state.
        :return: Estimated number of actions to reach goal.
        """
        state = node.state

        # Helper to find facts for quick queries:
        def facts_with_pred(pred):
            return {fact for fact in state if fact.startswith("(" + pred)}

        # Children that must be served from the goal but are not served yet:
        children_to_serve = set()
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                child = get_parts(goal_fact)[1]
                served_fact = f"(served {child})"
                if served_fact not in state:
                    children_to_serve.add(child)

        # Children currently waiting and not served:
        waiting_children = set()
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                if f"(served {child})" not in state:
                    waiting_children.add(child)

        # Intersection of children to serve and actually waiting (should be same, but safe)
        children_needed = waiting_children.intersection(children_to_serve)

        # Categorize children by allergy in those needing service
        allergic_needed = children_needed.intersection(self.allergic_children)
        non_allergic_needed = children_needed.intersection(self.non_allergic_children)

        # Sets of breads and contents currently available in kitchen that have no gluten or regular
        breads_no_gluten = {get_parts(f)[1] for f in state if match(f, "no_gluten_bread", "*")}
        contents_no_gluten = {get_parts(f)[1] for f in state if match(f, "no_gluten_content", "*")}
        breads_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        contents_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}

        # Sandwiches no gluten or regular currently at kitchen
        sandwiches_no_gluten_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_sandwich", "*")}
        sandwiches_no_gluten = {get_parts(f)[1] for f in state if match(f, "no_gluten_sandwich", "*")}
        sandwiches_regular_kitchen = {
            get_parts(f)[1]
            for f in state
            if match(f, "at_kitchen_sandwich", "*") and get_parts(f)[1] not in sandwiches_no_gluten
        }

        # Sandwiches currently on trays
        sandwiches_ontray = {get_parts(f)[1] for f in state if match(f, "ontray", "*", "*")}
        sandwiches_no_gluten_ontray = {s for s in sandwiches_ontray if f"(no_gluten_sandwich {s})" in state}
        sandwiches_regular_ontray = sandwiches_ontray - sandwiches_no_gluten_ontray

        # Sandwiches already made (in kitchen or on trays)
        sandwiches_no_gluten_made = sandwiches_no_gluten_kitchen.union(sandwiches_no_gluten_ontray)
        sandwiches_regular_made = sandwiches_regular_kitchen.union(sandwiches_regular_ontray)

        # Count sandwiches made but not yet on trays (at kitchen only sandwiches)
        sandwiches_no_gluten_kitchen_only = sandwiches_no_gluten_kitchen - sandwiches_ontray
        sandwiches_regular_kitchen_only = sandwiches_regular_kitchen - sandwiches_ontray

        # Count trays, their locations
        trays = set()
        tray_locations = dict()  # tray -> place
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj.startswith("tray"):
                    trays.add(obj)
                    tray_locations[obj] = loc
        # There could be trays without explicit location in facts (rare but possible)
        # To be safe, we only consider trays that appear in at or ontray facts:
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1], get_parts(fact)[2]
                trays.add(tray)
                if tray not in tray_locations:
                    tray_locations[tray] = "unknown"  # unknown location if no (at ...)

        # Count sandwiches that are served or serving status
        served_children = {get_parts(f)[1] for f in state if match(f, "served", "*")}

        # Sandwiches that are not existent (not made yet)
        sandwiches_not_exist = {get_parts(f)[1] for f in state if match(f, "notexist", "*")}

        # Heuristic calculation variables:
        # Number of sandwiches needed for each category = children needing service
        num_needed_no_gluten = len(allergic_needed)
        num_needed_regular = len(non_allergic_needed)

        # Number of sandwiches already made and available usable for each category.
        # We assume sandwiches made and tagged no-gluten can serve allergic children,
        # sandwiches regular serve non-allergic.
        available_no_gluten = len(sandwiches_no_gluten_made.intersection(
            sandwiches_no_gluten_kitchen.union(sandwiches_no_gluten_ontray)))
        available_regular = len(sandwiches_regular_made)

        # But we need sandwiches on trays to serve: sandwiches made but still in kitchen only need an extra put_on_tray action before serving.
        # Sandwiches on trays can be at kitchen or at child's place.
        # Trays at kitchen need to be moved to child's place before serving.
        # This complexity is simplified by counting:
        #   - # sandwiches needed but not made: need to make sandwich (make_sandwich_no_gluten or make_sandwich)
        #   - # sandwiches made but only in kitchen (not on trays): must put_on_tray
        #   - # trays that must move to child's place: count minimum number of moves to serve all children given trays location.
        #   - # serve actions for each child needing service

        # Calculate sandwiches still needed to be made
        sandwiches_to_make_no_gluten = max(0, num_needed_no_gluten - len(sandwiches_no_gluten_made))
        sandwiches_to_make_regular = max(0, num_needed_regular - len(sandwiches_regular_made))

        # Calculate sandwiches made but not on trays
        sandwiches_no_gluten_kitchen_only_num = len(sandwiches_no_gluten_kitchen_only)
        sandwiches_regular_kitchen_only_num = len(sandwiches_regular_kitchen_only)

        # Total sandwiches on trays
        sandwiches_on_tray_no_gluten = len(sandwiches_no_gluten_ontray)
        sandwiches_on_tray_regular = len(sandwiches_regular_ontray)
        sandwiches_on_tray_total = sandwiches_on_tray_no_gluten + sandwiches_on_tray_regular

        # Estimate trays at kitchen and away (not kitchen)
        trays_at_kitchen = sum(1 for t, loc in tray_locations.items() if loc == "kitchen")
        trays_not_kitchen = len(trays) - trays_at_kitchen

        # Count number of children still to be served
        num_children_to_serve = len(children_needed)

        # Estimation of needed trays to deliver sandwiches:
        # We assume multiple sandwiches can be placed on one tray,
        # so minimal trays needed is max number of sandwiches / capacity.
        # Since no explicit capacity, assume unlimited or 1 sandwich per child.
        # Since children are distinct places, assume each child requires a tray at their place.

        # Count distinct places of waiting children (known from waiting facts)
        child_places = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in children_needed
        }

        # Number of trays optimally needed at children's places is at least the number of distinct places (worst case one per place).
        trays_needed_to_place = len(child_places)

        # Trays currently at kitchen or at places
        trays_at_places = trays_not_kitchen

        # Number of trays that must be moved to places
        trays_to_move = max(0, trays_needed_to_place - trays_at_places)

        # Number of put_on_tray actions needed:
        # Each sandwich not on a tray needs a put_on_tray.
        put_on_tray_actions = sandwiches_no_gluten_kitchen_only_num + sandwiches_regular_kitchen_only_num

        # Number of make sandwich actions needed:
        make_no_gluten_actions = sandwiches_to_make_no_gluten
        make_regular_actions = sandwiches_to_make_regular
        make_actions = make_no_gluten_actions + make_regular_actions

        # Number of serve actions equals number of children to serve
        serve_actions = num_children_to_serve

        # Number of move_tray actions estimated as trays_to_move
        move_tray_actions = trays_to_move

        # Sum all actions to estimate heuristic cost:
        total_estimated_actions = (
            make_actions
            + put_on_tray_actions
            + move_tray_actions
            + serve_actions
        )

        # For goal states (all children served), heuristic must be zero
        if num_children_to_serve == 0:
            return 0

        # Return the heuristic estimate
        return total_estimated_actions
