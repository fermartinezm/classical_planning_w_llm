from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract parts of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnack domain.

    Summary:
    This heuristic estimates the number of actions required to serve all children by making or obtaining sandwiches 
    suitable for their gluten allergy status, placing sandwiches on trays, moving trays if necessary, and serving children.

    Assumptions:
    - Each child must be served exactly one sandwich.
    - Sandwiches must have the correct gluten property matching the child's allergy (no gluten sandwiches for allergic children).
    - Bread portions and content portions are consumed when making a sandwich.
    - Sandwiches must be put on trays before serving.
    - Trays start at the kitchen and may need to be moved to the children's waiting location.
    - Each serving action serves exactly one child.
    - The heuristic assumes that sandwiches are made optimally with available bread and content.
    - The heuristic is not guaranteed admissible but attempts to be informative and efficient.

    Heuristic Initialization:
    - Extract from static facts:
        * The gluten allergy status of each child.
        * Which breads and contents are gluten-free.
        * The waiting place for each child.
    - Extract from goal conditions the set of children to be served (to know which children remain).
    - Extract the set of trays in the domain (to track their locations).
    - Note: Sandwiches and ingredients are dynamic and assessed in the current state.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify children not yet served.
    2. For each child, determine the type of sandwich needed (gluten-free if allergic, otherwise any).
    3. Count number of sandwiches of the required type already prepared and available at kitchen or on trays.
    4. Compute how many sandwiches still need to be made for each category:
        - For gluten-free allergic children, sandwiches must be gluten-free.
        - For non-allergic children, sandwiches may be any type (prefer using existing non-gluten-free sandwiches if available).
    5. Check available bread portions and content portions at kitchen to estimate how many sandwiches can still be made.
    6. Estimate actions needed for making missing sandwiches:
        - One make_sandwich or make_sandwich_no_gluten action per sandwich.
    7. Estimate actions for placing sandwiches on trays:
        - Sandwiches not yet on any tray require at least one put_on_tray action.
    8. Estimate actions to move trays from kitchen to children waiting places if trays are not yet at the required places.
        - For each child, check if the tray holding their sandwich is at their waiting place; if not, at least one move_tray action.
        - Multiple children at same place sharing trays can reduce number of moves; so count minimal moves per tray.
    9. Estimate serving actions:
        - Each child requires one serve_sandwich or serve_sandwich_no_gluten action.
    10. Sum all estimated actions:
        - make sandwich actions
        - put on tray actions
        - move tray actions
        - serve actions
    This heuristic provides an informed estimate that accounts for current sandwich and tray locations,
    and the need to produce appropriate sandwiches given allergies.

    The heuristic returns 0 if all children are served (goal state), otherwise the computed estimate.
    """

    def __init__(self, task):
        # Extract static info
        self.goals = task.goals

        # Set of children: allergic or not
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.child_waiting_place = {}
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.trays = set()

        # Parse static info
        for fact in task.static:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]

            if pred == "allergic_gluten" and len(parts) == 2:
                self.allergic_children.add(parts[1])
            elif pred == "not_allergic_gluten" and len(parts) == 2:
                self.non_allergic_children.add(parts[1])
            elif pred == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                self.child_waiting_place[child] = place
            elif pred == "no_gluten_bread" and len(parts) == 2:
                self.no_gluten_breads.add(parts[1])
            elif pred == "no_gluten_content" and len(parts) == 2:
                self.no_gluten_contents.add(parts[1])
            elif pred == "at" and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Objects of type tray appear in at facts, so collect trays from initial static info as well.
                # This will capture trays given in problem objects or initial state.
                # Since static facts may not include this, we will accumulate trays also dynamically later.
                # But to start with, collect trays mentioned in static facts.
                # Otherwise, we build tray set from initial_state also.
                # Here we store any tray name seen in static.
                # Another way is to identify trays by domain, but we have only fact strings here.
                # We only add trays if object name starts with tray prefix.
                if obj.startswith("tray"):
                    self.trays.add(obj)

        # Because some trays appear only in initial_state, not static, we add them later in __call__.

        # Extract children in goal
        # The goal is conjunction of served facts per child
        self.goal_children = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "served" and len(parts) == 2:
                self.goal_children.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Helper funcs
        def has_fact(*args):
            """Check if fact matching args is in current state."""
            pattern = "(" + " ".join(args) + ")"
            return pattern in state

        # Update trays from current state (some trays appear only here)
        trays_in_state = {get_parts(f)[1] for f in state if match(f, "at", "tray*", "*")}
        trays_in_state.update({get_parts(f)[1] for f in state if match(f, "ontray", "*", "tray*")})
        self.trays.update(trays_in_state)

        # Children not yet served
        served_children = {get_parts(f)[1] for f in state if match(f, "served", "*")}
        unserved_children = self.goal_children - served_children
        if not unserved_children:
            return 0  # Goal reached

        # Identify breads and contents at kitchen (available for making sandwiches)
        breads_at_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        contents_at_kitchen = {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}

        # Sandwiches at kitchen (available) and their properties
        # For each sandwich, note if gluten-free or not, and its location (kitchen or ontray)
        sandwiches_at_kitchen = set()
        sandwiches_on_tray = set()
        gluten_free_sandwiches = set()
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                sandwiches_at_kitchen.add(s)
            elif match(fact, "ontray", "*", "*"):
                s = get_parts(fact)[1]
                sandwiches_on_tray.add(s)
            elif match(fact, "no_gluten_sandwich", "*"):
                s = get_parts(fact)[1]
                gluten_free_sandwiches.add(s)

        # Group sandwiches by gluten-free and non-gluten-free that are available (kitchen or on trays)
        gf_available_sandwiches = (sandwiches_at_kitchen | sandwiches_on_tray) & gluten_free_sandwiches
        nongf_available_sandwiches = (sandwiches_at_kitchen | sandwiches_on_tray) - gluten_free_sandwiches

        # Count children needing gluten-free or non-gluten-free sandwiches
        children_need_gf = [c for c in unserved_children if c in self.allergic_children]
        children_need_nongf = [c for c in unserved_children if c in self.non_allergic_children]

        # Sandwiches needed to serve children (count)
        num_gf_needed = len(children_need_gf)
        num_nongf_needed = len(children_need_nongf)

        # Use already available sandwiches first to reduce sandwiches to make
        gf_to_make = max(0, num_gf_needed - len(gf_available_sandwiches))
        nongf_to_make = max(0, num_nongf_needed - len(nongf_available_sandwiches))

        # Total sandwiches to make
        total_sandwiches_to_make = gf_to_make + nongf_to_make

        # Check available breads and contents for making sandwiches
        # Each sandwich consumes 1 bread and 1 content portion
        max_sandwiches_possible = min(len(breads_at_kitchen), len(contents_at_kitchen))

        # Limit sandwiches to make by available ingredients
        total_sandwiches_to_make = min(total_sandwiches_to_make, max_sandwiches_possible)

        # Estimate actions:

        # 1) Make sandwich actions (one per sandwich to make)
        actions_make = total_sandwiches_to_make

        # 2) Put on tray actions:
        # Count sandwiches not already on a tray (those at kitchen)
        # All sandwiches on trays do not require this action
        # So, count sandwiches not on trays: these are sandwiches at kitchen (including those made or available)
        # But consider only sandwiches that will be used to serve the children

        # To approximate which sandwiches will be used, combine sandwiches currently on trays and at kitchen:
        # Let's track sandwiches available (at kitchen or on tray) as candidates for serving.

        # All sandwiches currently on trays are ready to serve from trays.

        # Sandwiches at kitchen (available or new made ones) need to be put on trays.
        # Estimate how many sandwiches must be put on trays:

        # How many children need sandwiches served?
        num_total_unserved = len(unserved_children)

        # Total sandwiches available (on trays or at kitchen):
        total_sandwiches_available = len(gf_available_sandwiches) + len(nongf_available_sandwiches) + total_sandwiches_to_make

        # Sandwiches already on trays that can be used:
        sandwiches_on_tray_usable = len(gf_available_sandwiches & sandwiches_on_tray) + len(nongf_available_sandwiches & sandwiches_on_tray)

        # Sandwiches needed to be put on tray:
        # At least the difference: min(all needed sandwiches, total unserved) - sandwiches already on trays
        sandwiches_to_put_on_tray = max(0, num_total_unserved - sandwiches_on_tray_usable)

        actions_put_on_tray = sandwiches_to_put_on_tray

        # 3) Move tray actions:
        # We need to move trays from kitchen to each child's waiting place if not already there.
        # For each tray, check where it is.
        # For each child, find the tray holding their sandwich.
        # If tray is not at child's waiting place, count one move_tray action for that tray.
        # Multiple children served by the same tray in same place count once.

        # Map sandwich to tray (from ontray facts)
        sandwich_to_tray = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1], get_parts(fact)[2]
                sandwich_to_tray[s] = t

        # Map tray to its location
        tray_location = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1], get_parts(fact)[2]
                if obj in self.trays:
                    tray_location[obj] = loc

        # For children already served, no moves needed.

        # For unserved children, assign sandwiches from available ones.
        # For simplification, assign sandwiches greedily:
        # first gluten-free sandwiches to allergic children,
        # then non-gluten-free sandwiches to non-allergic children.

        # Collect all sandwiches usable for serving
        available_sandwiches = list((gf_available_sandwiches | nongf_available_sandwiches) & (sandwich_to_tray.keys() | sandwiches_at_kitchen))

        # Create list of sandwiches on trays and at kitchen separately
        sandwiches_on_tray_list = [s for s in available_sandwiches if s in sandwiches_on_tray]
        sandwiches_at_kitchen_list = [s for s in available_sandwiches if s in sandwiches_at_kitchen]

        # Assign sandwiches to children; track which tray will serve which children and at which place
        tray_to_target_places = {}  # tray -> set of places to move to

        # NOTE: To keep heuristic efficient, we assign sandwiches based on gluten property only,
        # ignoring exact matching of sandwiches to children.

        def assign_children(children, gf_required):
            """Assign sandwiches to children needing gf_required property."""
            nonlocal sandwiches_on_tray_list, sandwiches_at_kitchen_list, tray_to_target_places

            unassigned = children[:]
            # Use sandwiches on trays first
            for s in sandwiches_on_tray_list[:]:
                if gf_required and s not in gluten_free_sandwiches:
                    continue
                if not gf_required and s in gluten_free_sandwiches:
                    # non gluten-free children can eat gluten-free sandwiches as well (assumed safe)
                    pass
                if not unassigned:
                    break
                child = unassigned.pop()
                tray = sandwich_to_tray.get(s)
                if tray is not None:
                    place = self.child_waiting_place.get(child)
                    if place is not None:
                        if tray not in tray_to_target_places:
                            tray_to_target_places[tray] = set()
                        tray_to_target_places[tray].add(place)
                sandwiches_on_tray_list.remove(s)

            # Use sandwiches at kitchen (need to be put on trays and trays moved)
            for s in sandwiches_at_kitchen_list[:]:
                if gf_required and s not in gluten_free_sandwiches:
                    continue
                if not gf_required and s in gluten_free_sandwiches:
                    # non gluten-free children can eat gluten-free sandwiches as well
                    pass
                if not unassigned:
                    break
                child = unassigned.pop()
                # For these sandwiches, tray is not known yet, so assume a tray move for that child
                # We'll count one move per distinct target place per tray later.
                # Here, we temporarily assign a dummy tray name per sandwich to count moves accordingly.
                dummy_tray = f"dummy_tray_for_{s}"
                place = self.child_waiting_place.get(child)
                if place is not None:
                    if dummy_tray not in tray_to_target_places:
                        tray_to_target_places[dummy_tray] = set()
                    tray_to_target_places[dummy_tray].add(place)
                sandwiches_at_kitchen_list.remove(s)

            # Remaining unassigned children probably will need sandwiches made,
            # We must also assume newly made sandwiches will be put on trays and moved.
            for child in unassigned:
                dummy_tray = f"dummy_tray_for_new_sandwich_{child}"
                place = self.child_waiting_place.get(child)
                if place is not None:
                    if dummy_tray not in tray_to_target_places:
                        tray_to_target_places[dummy_tray] = set()
                    tray_to_target_places[dummy_tray].add(place)

        # Assign children needing gluten-free sandwiches
        assign_children(children_need_gf, gf_required=True)
        # Assign children needing non-gluten-free sandwiches
        assign_children(children_need_nongf, gf_required=False)

        # For each tray, count if at least one move_tray action needed
        # If tray location includes kitchen, and places needed are some other place(s),
        # count number of moves equal to number of distinct target places different from kitchen (worst case).
        # To keep heuristic simple and admissible, count one move per distinct place different from kitchen per tray.
        # We do NOT model multiple moves between places; optimistic estimate.

        actions_move = 0
        for tray, places in tray_to_target_places.items():
            if tray in tray_location:
                loc = tray_location[tray]
                # count the number of distinct places different from current tray location
                moves_needed = len({p for p in places if p != loc})
                actions_move += moves_needed
            else:
                # For dummy trays or trays without known location, assume one move per distinct place
                actions_move += len(places)

        # 4) Serving actions:
        # One serve action per unserved child
        actions_serve = len(unserved_children)

        # Estimate sum of actions:
        # make + put on tray + move trays + serve
        heuristic_value = actions_make + actions_put_on_tray + actions_move + actions_serve

        # Return finite heuristic > 0 for non-goal states
        return max(heuristic_value, 1)
