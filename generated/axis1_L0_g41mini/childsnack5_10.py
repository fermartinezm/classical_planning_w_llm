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
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    The heuristic estimates the number of actions required to serve all waiting children
    with appropriate sandwiches, considering gluten allergies and sandwich preparation.

    # Assumptions
    - Each child must be served exactly one sandwich.
    - Children are either allergic to gluten or not, determining the type of sandwich needed.
    - Sandwiches must be made from bread and content portions available in the kitchen.
    - There are gluten-free breads and contents for allergic children and regular ones for others.
    - Preparing a sandwich consumes one bread and one content portion.
    - Sandwiches must be placed on trays present in the kitchen before serving.
    - Serving requires the tray with the sandwich to be at the child's location (waiting place).
    - Trays can be moved between kitchen and places.
    - The heuristic does not need to be admissible but should be as accurate and efficient as possible.

    # Heuristic Initialization
    - Extract the set of children that must be served from the goal.
    - Extract which children are allergic and which are not from static facts.
    - Extract the sets of breads and content portions that are gluten-free.
    - Count the available trays in the kitchen.
    - No gluten-free sandwiches are initially assumed in the state; detected from facts.
    - Similarly, track regular sandwiches from facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify children that are still not served (waiting children).
    2. Classify waiting children into gluten-allergic and non-allergic groups.
    3. Count sandwiches on trays and sandwiches available in kitchen, split by gluten-free or not.
    4. Count bread and content portions available in kitchen, split by gluten-free or not.
    5. Count trays currently in kitchen (available for loading sandwiches).
    6. For each group (allergic and non-allergic):
       a) Determine how many children need sandwiches of that type (unserved).
       b) Determine how many sandwiches of that type are already on trays ready to be served.
       c) Calculate how many sandwiches need to be placed on trays (if there are sandwiches in kitchen not on trays).
       d) Calculate how many sandwiches need to be made if insufficient sandwiches exist.
           - To make sandwiches, sufficient bread and content portions of appropriate type are needed.
           - We assume making sandwiches to exactly fulfill the deficit up to the number of children to serve.
    7. Calculate the minimal number of required tray moves:
       - Sandwiches to put on trays must use trays in kitchen; if insufficient trays,
         assume some trays must be moved back to kitchen (which costs actions).
    8. Serving each child requires at least one serve action.
    9. The heuristic sums:
       - Estimated sandwich making actions (one per sandwich made).
       - Estimated tray loading actions (one put_on_tray per sandwich put on tray).
       - Estimated tray moving actions to position trays at children's places.
       - Serving actions (one per child).
    10. This heuristic assumes optimal batch preparation and tray usage ignoring intricate ordering,
        but reflects main costs and is efficiently computable.

    The returned heuristic is 0 if all children are served (goal state).
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract children from goals (served children)
        self.children = set()
        for fact in self.goals:
            if match(fact, "served", "*"):
                parts = get_parts(fact)
                self.children.add(parts[1])

        # Extract allergy info from static facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                parts = get_parts(fact)
                self.allergic_children.add(parts[1])
            elif match(fact, "not_allergic_gluten", "*"):
                parts = get_parts(fact)
                self.not_allergic_children.add(parts[1])

        # Extract gluten-free breads and contents
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                parts = get_parts(fact)
                self.no_gluten_breads.add(parts[1])
            elif match(fact, "no_gluten_content", "*"):
                parts = get_parts(fact)
                self.no_gluten_contents.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Helper to find objects by a predicate in state
        def objects_with_predicate(pred, gluten_free_required=None):
            # gluten_free_required: True for gluten-free only, False for gluten-containing only,
            # None for all.
            objs = set()
            for fact in state:
                if match(fact, pred, "*"):
                    parts = get_parts(fact)
                    obj = parts[1]
                    if gluten_free_required is None:
                        objs.add(obj)
                    elif gluten_free_required is True:
                        # Check if object is gluten-free type
                        if pred == "at_kitchen_bread":
                            if obj in self.no_gluten_breads:
                                objs.add(obj)
                        elif pred == "at_kitchen_content":
                            if obj in self.no_gluten_contents:
                                objs.add(obj)
                        elif pred == "no_gluten_sandwich":
                            objs.add(obj)  # sandwiches are already known gluten-free by predicate
                    elif gluten_free_required is False:
                        if pred == "at_kitchen_bread":
                            if obj not in self.no_gluten_breads:
                                objs.add(obj)
                        elif pred == "at_kitchen_content":
                            if obj not in self.no_gluten_contents:
                                objs.add(obj)
                        elif pred == "no_gluten_sandwich":
                            continue  # gluten sandwiches do not have this predicate
            return objs

        # Count waiting children by allergy type
        waiting_allergic = set()
        waiting_not_allergic = set()
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child = parts[1]
                if child in self.allergic_children:
                    waiting_allergic.add(child)
                elif child in self.not_allergic_children:
                    waiting_not_allergic.add(child)

        # How many children to serve of each type?
        num_allergic_waiting = len(waiting_allergic)
        num_not_allergic_waiting = len(waiting_not_allergic)

        # Count sandwiches in kitchen and on trays by gluten type
        # Gluten-free sandwiches have predicate no_gluten_sandwich
        kitchen_gluten_free_sandwiches = set()
        kitchen_regular_sandwiches = set()
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                parts = get_parts(fact)
                s = parts[1]
                if ("(no_gluten_sandwich " + s + ")") in state:
                    kitchen_gluten_free_sandwiches.add(s)
                else:
                    kitchen_regular_sandwiches.add(s)

        # Sandwiches on trays by gluten type
        tray_gluten_free_sandwiches = set()
        tray_regular_sandwiches = set()
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                parts = get_parts(fact)
                s = parts[1]
                if ("(no_gluten_sandwich " + s + ")") in state:
                    tray_gluten_free_sandwiches.add(s)
                else:
                    tray_regular_sandwiches.add(s)

        # Total sandwiches available (kitchen + on trays)
        total_gluten_free_sandwiches = len(kitchen_gluten_free_sandwiches) + len(tray_gluten_free_sandwiches)
        total_regular_sandwiches = len(kitchen_regular_sandwiches) + len(tray_regular_sandwiches)

        # Count breads and contents currently in kitchen by gluten type
        kitchen_gluten_free_breads = objects_with_predicate("at_kitchen_bread", gluten_free_required=True)
        kitchen_regular_breads = objects_with_predicate("at_kitchen_bread", gluten_free_required=False)

        kitchen_gluten_free_contents = objects_with_predicate("at_kitchen_content", gluten_free_required=True)
        kitchen_regular_contents = objects_with_predicate("at_kitchen_content", gluten_free_required=False)

        # Count trays in kitchen available for placing sandwiches
        trays_in_kitchen = set()
        for fact in state:
            if match(fact, "at", "*", "kitchen"):
                parts = get_parts(fact)
                if parts[1].startswith("tray"):
                    trays_in_kitchen.add(parts[1])

        num_trays_in_kitchen = len(trays_in_kitchen)

        # Serve actions: One per waiting child to serve
        serve_actions = num_allergic_waiting + num_not_allergic_waiting

        # Determine sandwiches needed per group
        # Allergic children require gluten-free sandwiches
        # Non-allergic require regular sandwiches
        # Calculate how many sandwiches of proper type must still be served

        # Number of sandwiches on trays ready to serve per type (can be served immediately)
        tray_sandwiches_allergic = len(tray_gluten_free_sandwiches)
        tray_sandwiches_not_allergic = len(tray_regular_sandwiches)

        # Sandwiches in kitchen per type (prepared but not on tray)
        kitchen_sandwiches_allergic = len(kitchen_gluten_free_sandwiches)
        kitchen_sandwiches_not_allergic = len(kitchen_regular_sandwiches)

        # Sandwich deficit per allergy type (children waiting minus sandwiches already on trays)
        sandwich_deficit_allergic = max(0, num_allergic_waiting - tray_sandwiches_allergic)
        sandwich_deficit_not_allergic = max(0, num_not_allergic_waiting - tray_sandwiches_not_allergic)

        # Determine how many sandwiches need to be put on trays from kitchen sandwiches
        put_on_tray_allergic = min(kitchen_sandwiches_allergic, sandwich_deficit_allergic)
        put_on_tray_not_allergic = min(kitchen_sandwiches_not_allergic, sandwich_deficit_not_allergic)

        # Remaining deficit after using sandwich-in-kitchen to put on trays:
        residual_deficit_allergic = sandwich_deficit_allergic - put_on_tray_allergic
        residual_deficit_not_allergic = sandwich_deficit_not_allergic - put_on_tray_not_allergic

        # Now calculate how many sandwiches must be made to cover residual deficits
        # Limited by available breads and contents in kitchen of respective types

        # Maximum sandwiches possible to make from available bread and contents
        max_make_allergic = min(len(kitchen_gluten_free_breads), len(kitchen_gluten_free_contents))
        max_make_not_allergic = min(len(kitchen_regular_breads), len(kitchen_regular_contents))

        make_allergic = min(residual_deficit_allergic, max_make_allergic)
        make_not_allergic = min(residual_deficit_not_allergic, max_make_not_allergic)

        # After making the new sandwiches, they must also be put on trays
        # We assume all freshly made sandwiches immediately get put on trays
        put_on_tray_allergic += make_allergic
        put_on_tray_not_allergic += make_not_allergic

        # Calculate tray usage total needed now (sandwiches that must be on trays)
        total_put_on_tray = put_on_tray_allergic + put_on_tray_not_allergic

        # If not enough trays are in kitchen, some tray moves are needed
        tray_moves = max(0, total_put_on_tray - num_trays_in_kitchen)

        # Actions count:
        # - Making sandwiches: Each sandwich made corresponds to 1 make_sandwich[_no_gluten] action.
        # - Putting sandwiches on trays: 1 put_on_tray per sandwich.
        # - Moving trays: 1 move_tray per tray to bring it to kitchen if tray not available.
        # - Serving: 1 serve_sandwich[_no_gluten] per child.

        total_actions = (
            make_allergic +  # make gluten-free sandwiches
            make_not_allergic +  # make regular sandwiches
            total_put_on_tray +  # loading sandwiches onto trays
            tray_moves +          # moving trays back to kitchen as needed
            serve_actions         # serving actions
        )

        # Return zero if goal reached (all children served)
        if serve_actions == 0:
            return 0
        return total_actions
