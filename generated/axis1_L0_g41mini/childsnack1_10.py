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
    A domain-dependent heuristic for the childsnack domain.

    Summary:
    This heuristic estimates the minimum number of actions required to serve 
    all children in the goal by preparing and serving suitable sandwiches efficiently.
    It considers the current availability of breads, contents, sandwiches on trays,
    trays locations, and children's allergies.

    Assumptions:
    - Serving each child requires exactly one sandwich.
    - Sandwiches can be gluten-containing or gluten-free depending on the child's allergy.
    - Making a gluten-free sandwich requires gluten-free bread and content portions.
    - All sandwiches must come from bread and content currently available in the kitchen.
    - Trays are initially at the kitchen or some place; trays must carry sandwiches to the children.
    - Actions:
      * make_sandwich_no_gluten and make_sandwich create sandwiches,
      * put_on_tray places sandwiches on trays (tray must be at kitchen)
      * move_tray moves the tray between places,
      * serve_sandwich_no_gluten or serve_sandwich serves children (tray must be at the child's place)
    - Each action counts as 1 toward the heuristic; estimated actions include making sandwiches,
      putting them on trays, moving trays to waiting places, and serving them.
    - Sandwiches existing and already on trays reduce needed steps.

    Heuristic Initialization:
    - Extract sets of children who are allergic / not allergic to gluten from static facts.
    - Collect all breads and contents currently no_gluten or not.
    - Initialize goal children (children that are not yet served).
    - Extract trays from initial static info (if needed).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify which children still need serving (goal children minus those already served).
    2. Count how many children who are allergic to gluten remain to be served,
       and how many are non-allergic.
    3. Count sandwiches already made and on trays, distinguishing gluten-free and regular.
       Deduct those "ready" sandwiches from children requiring them.
    4. Estimate number of gluten-free sandwiches that can still be made with available gluten-free breads and contents.
       Likewise estimate regular sandwiches possible from available regular breads and contents.
       These indicate sandwich production capacity left.
    5. If sandwich capacity is insufficient for remaining children, estimate how many sandwiches must still be made.
    6. Estimate minimal number of actions as sum of:
       - Actions to make remaining sandwiches (1 per sandwich),
       - Putting sandwiches on trays (1 per sandwich),
       - Moving trays to children's waiting places (count unique places with waiting children),
       - Serving sandwiches to children (1 per child).
    7. Additional refinements:
       - If trays are already at the children's places with sandwiches on them, reduce move_tray costs accordingly.
       - Group serving by place: serving multiple children in the same place does not require multiple moves.
    8. Return 0 if all children have been served (goal state).
    """

    def __init__(self, task):
        """
        Extract static info and goals:
        - Identify allergic and non-allergic children.
        - Store gluten-free breads and contents.
        - Store all trays.
        - Extract goal children.
        """
        self.goals = task.goals
        self.static = task.static

        # Children sets
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.not_allergic_children.add(child)

        # Gluten-free breads and contents available (static info, so fixed)
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                bread = get_parts(fact)[1]
                self.no_gluten_breads.add(bread)
            elif match(fact, "no_gluten_content", "*"):
                content = get_parts(fact)[1]
                self.no_gluten_contents.add(content)

        # Extract trays (we only need trays present in the domain)
        self.trays = set()
        for fact in self.static:
            if match(fact, "at", "*"):
                obj = get_parts(fact)[1]
                if obj.startswith("tray"):
                    self.trays.add(obj)

        # Extract goal children (all children that must be served)
        self.goal_children = set()
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                child = get_parts(goal_fact)[1]
                self.goal_children.add(child)

    def __call__(self, node):
        """Compute heuristic value (estimated action cost) in the given node state."""

        state = node.state

        # Helper functions to test presence of facts for specific objects:

        # Children already served
        served_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }

        # Children still to serve
        children_to_serve = self.goal_children - served_children
        if not children_to_serve:
            # All served => heuristic 0
            return 0

        # Children waiting and their places
        waiting_places = dict()  # child -> place
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child, place = parts[1], parts[2]
                waiting_places[child] = place

        # Count how many allergic vs not_allergic children remain to serve
        allergic_to_serve = [c for c in children_to_serve if c in self.allergic_children]
        not_allergic_to_serve = [c for c in children_to_serve if c in self.not_allergic_children]

        # Current availability in kitchen
        breads_in_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        contents_in_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        # Current gluten-free breads and contents available (intersection with static gluten-free sets)
        gluten_free_breads_available = breads_in_kitchen & self.no_gluten_breads
        gluten_free_contents_available = contents_in_kitchen & self.no_gluten_contents

        # Count sandwiches already made and on trays or kitchen (distinguish gluten-free and normal)
        gluten_free_sandwiches_ready = 0  # made sandwiches gluten-free available either on tray or kitchen
        normal_sandwiches_ready = 0

        # Sandwich presence tracking
        sandwiches_in_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*")
        }
        sandwiches_on_tray = dict()  # sandwich -> tray
        sandwiches_gluten_free = set()
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                parts = get_parts(fact)
                sandwich, tray = parts[1], parts[2]
                sandwiches_on_tray[sandwich] = tray
            elif match(fact, "no_gluten_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                sandwiches_gluten_free.add(sandwich)

        # Classify made sandwiches by gluten-free and normal, on tray or in kitchen
        for s in sandwiches_in_kitchen | sandwiches_on_tray.keys():
            if s in sandwiches_gluten_free:
                gluten_free_sandwiches_ready += 1
            else:
                normal_sandwiches_ready += 1

        # How many gluten-free sandwiches still need to be made to serve allergic children
        gf_needed = max(0, len(allergic_to_serve) - gluten_free_sandwiches_ready)

        # How many normal sandwiches still need to be made to serve non-allergic children
        normal_needed = max(0, len(not_allergic_to_serve) - normal_sandwiches_ready)

        # Calculate max number of sandwiches that can be made from available breads and contents:
        # Gluten-free sandwiches possible:
        gluten_free_possible = min(len(gluten_free_breads_available), len(gluten_free_contents_available))

        # Normal sandwiches possible:
        # Normal breads and contents are breads_in_kitchen / gluten_free_breads, contents_in_kitchen / gluten_free_contents
        normal_breads_available = breads_in_kitchen - self.no_gluten_breads
        normal_contents_available = contents_in_kitchen - self.no_gluten_contents
        normal_possible = min(len(normal_breads_available), len(normal_contents_available))

        # If needed is greater than possible, make sure estimate includes that gap (heuristic isn't admissible so safe to overestimate)
        gf_to_produce = max(0, gf_needed - gluten_free_possible)
        normal_to_produce = max(0, normal_needed - normal_possible)

        # Actions needed to produce sandwiches (1 make action per sandwich)
        produce_actions = gf_to_produce + normal_to_produce

        # Actions to put sandwiches on trays
        # Estimate number: sandwiches needed to be served (all children minus sandwiches already on trays serving them)
        # Count sandwiches already on trays
        sandwiches_on_trays_count = len(sandwiches_on_tray)

        # We assume each child needs one sandwich served, so plates needed is children_to_serve count
        # Some sandwiches may already be on trays, subtract those from need to put on trays
        put_on_tray_actions = max(0, len(children_to_serve) - sandwiches_on_trays_count)

        # Moving trays: estimate number of distinct places trays must be moved to serve children waiting there
        # For trays already at correct places with sandwiches, no move needed for those children

        # Map trays currently locations
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                tray, place = get_parts(fact)[1], get_parts(fact)[2]
                if tray.startswith("tray"):
                    tray_locations[tray] = place

        # Determine which children are already served by sandwiches on trays at their waiting places:
        # For each sandwich on a tray at place p, a child waiting at p can be served by it.
        # However, children are distinct, so assume one sandwich serves one child at that place.
        # Count how many children have sandwiches on trays at their places

        # Build place -> number of sandwiches on trays at that place
        sandwiches_per_place = {}
        for sandwich, tray in sandwiches_on_tray.items():
            place = tray_locations.get(tray, None)
            if place is not None:
                sandwiches_per_place[place] = sandwiches_per_place.get(place, 0) + 1

        # Group children by place
        children_by_place = {}
        for child in children_to_serve:
            place = waiting_places.get(child, None)
            if place is not None:
                children_by_place.setdefault(place, set()).add(child)

        # Count for how many children there are sandwiches already on trays at their place
        children_already_served_by_tray = 0
        for place, children_in_place in children_by_place.items():
            sandwiches_available = sandwiches_per_place.get(place, 0)
            children_at_place = len(children_in_place)
            # Sandwiches available can serve that many children
            children_already_served_by_tray += min(sandwiches_available, children_at_place)

        # Children that still need serving by moving trays (at places without suitable sandwiches on trays)
        children_need_tray_moves = len(children_to_serve) - children_already_served_by_tray

        # Actually, we may move trays once per distinct place, not per child
        # So number of moves is number of distinct places with children without sandwiches on trays
        places_need_tray_moves = 0
        for place, children_in_place in children_by_place.items():
            sandwiches_available = sandwiches_per_place.get(place, 0)
            if sandwiches_available < len(children_in_place):
                places_need_tray_moves += 1

        # Each tray move counts as 1 action
        move_tray_actions = places_need_tray_moves

        # Serving actions: 1 per child still to serve
        serve_actions = len(children_to_serve)

        # Total estimated actions
        total_actions = produce_actions + put_on_tray_actions + move_tray_actions + serve_actions

        # Make sure heuristic is finite and non-negative
        return max(total_actions, 0)
