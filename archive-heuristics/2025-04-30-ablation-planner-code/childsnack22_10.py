from fnmatch import fnmatch
# Assuming heuristics.heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy base class for testing purposes."""
        def __init__(self, task):
            self.task = task
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError


# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False # Pattern length must match fact length
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are waiting and not yet served. It considers the steps needed to make
    sandwiches, put them on trays, move trays to the children's locations, and
    finally serve the children, taking into account resource availability (bread,
    content, sandwich objects, trays) and allergy constraints.

    # Assumptions
    - The goal is to serve all children specified in the task goals.
    - Children specified in the goal are initially in a 'waiting' state at a fixed location.
    - Sandwiches on trays at a child's location can be used to serve that child
      if the sandwich type matches the child's allergy requirements.
    - Sandwiches not currently on a tray at a child's location must come from
      the kitchen (either already made or made from ingredients).
    - Making a sandwich requires one bread, one content, and one 'notexist'
      sandwich object. Gluten-free sandwiches require gluten-free ingredients.
    - Putting a sandwich on a tray requires the sandwich to be in the kitchen
      and a tray to be in the kitchen.
    - Moving a tray requires the tray to be at a place.
    - Serving a child requires a suitable sandwich on a tray at the child's
      waiting location, and the child waiting at that location.
    - Tray capacity is sufficient to hold needed sandwiches for a location.
    - Tray movement cost from kitchen to a place is 1 action per destination place,
      limited by the number of trays available in the kitchen.
    - Sandwiches on trays at locations where no goal children are waiting are ignored
      or considered unavailable for the current set of unsatisfied children.

    # Heuristic Initialization
    - Extract the set of children that need to be served from the task goals.
    - Extract static facts about children's allergies (`allergic_gluten`, `not_allergic_gluten`)
      and their waiting locations (`waiting`).
    - Extract static facts about gluten-free ingredients (`no_gluten_bread`, `no_gluten_content`).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of estimated costs for:
    1.  Serving all unsatisfied children.
    2.  Making any necessary sandwiches that are not already available or creatable.
    3.  Putting necessary sandwiches onto trays in the kitchen.
    4.  Moving trays with sandwiches from the kitchen to the places where children need them.

    Detailed steps:
    1.  Identify the set of children who are in the goal but not yet served (`unsatisfied_children`). If this set is empty, the heuristic is 0.
    2.  The base heuristic cost is the number of unsatisfied children (`|unsatisfied_children|`), representing the 'serve' action cost for each.
    3.  Identify which unsatisfied children can be served by sandwiches already on trays at their waiting locations. Iterate through unsatisfied children and check if a suitable sandwich exists on a tray at their location that hasn't been 'used' by another child in this calculation. Mark these children as 'sandwich found locally' and the sandwich/tray as 'used'.
    4.  Identify the set of children who still need a sandwich delivered (`children_needing_delivery`). These are the unsatisfied children not marked as 'sandwich found locally'.
    5.  Count the number of gluten-free and regular sandwiches required for these children (`Needed_gf_delivery`, `Needed_reg_delivery`).
    6.  Count the available resources in the kitchen from the current state: bread, content (GF and regular), already made sandwiches (GF and regular), 'notexist' sandwich objects, and trays.
    7.  Calculate the maximum number of gluten-free and regular sandwiches that can be made in the kitchen given the available ingredients and 'notexist' objects (`Can_make_gf`, `Can_make_reg`).
    8.  Calculate the total number of gluten-free and regular sandwiches available in the kitchen pool (already made + creatable) (`Avail_gf_kitchen`, `Avail_reg_kitchen`).
    9.  Determine how many GF and Regular sandwiches must be sourced from this kitchen pool to satisfy the needs of `children_needing_delivery` (`From_kitchen_gf`, `From_kitchen_reg`). Prioritize GF needs with GF sandwiches.
    10. The total number of sandwiches coming from the kitchen pool is `From_kitchen_gf + From_kitchen_reg`.
    11. Calculate how many of these `From_kitchen_total` sandwiches must actually be *made* (`Actual_make_gf`, `Actual_make_reg`) based on how many were already made in the kitchen vs. how many need to be sourced from the kitchen pool. Add `Actual_make_gf + Actual_make_reg` to the heuristic cost.
    12. Each of the `From_kitchen_total` sandwiches needs to be put on a tray. Add `From_kitchen_total` to the heuristic cost (for 'put_on_tray' actions).
    13. These sandwiches need to be moved to the places where the children in `children_needing_delivery` are waiting. Identify the set of distinct places needing delivery (`Places_needing_delivery`). The number of tray movement actions required from the kitchen is estimated as the minimum of the number of distinct places needing delivery and the number of trays available in the kitchen. Add this minimum to the heuristic cost.
    14. The total heuristic value is the sum of costs from steps 2, 11, 12, and 13.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Extract goal children
        self.goal_children = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "served":
                child = args[0]
                self.goal_children.add(child)

        # Extract static child info (allergy, waiting location)
        self.child_allergy = {}
        self.child_location = {}
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "allergic_gluten":
                child = args[0]
                self.child_allergy[child] = True
            elif predicate == "not_allergic_gluten":
                child = args[0]
                self.child_allergy[child] = False
            elif predicate == "waiting":
                child, place = args
                self.child_location[child] = place

        # Extract static gluten-free ingredient info
        self.gluten_free_bread_names = set()
        self.gluten_free_content_names = set()
        for fact in static_facts:
            predicate, *args = get_parts(fact)
            if predicate == "no_gluten_bread":
                self.gluten_free_bread_names.add(args[0])
            elif predicate == "no_gluten_content":
                self.gluten_free_content_names.add(args[0])

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unsatisfied children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unsatisfied_children = self.goal_children - served_children

        if not unsatisfied_children:
            return 0  # Goal state reached

        # 2. Base cost: 1 action per child for serving
        h = len(unsatisfied_children)

        # State information extraction
        kitchen_bread = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        kitchen_content = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        ontray_facts = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "ontray", "*", "*")}
        sandwich_is_gf = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}

        # Count kitchen resources
        N_gf_bread_kitchen = len(kitchen_bread.intersection(self.gluten_free_bread_names))
        N_reg_bread_kitchen = len(kitchen_bread) - N_gf_bread_kitchen
        N_gf_content_kitchen = len(kitchen_content.intersection(self.gluten_free_content_names))
        N_reg_content_kitchen = len(kitchen_content) - N_gf_content_kitchen
        N_sandwich_notexist = len(notexist_sandwiches)
        N_gf_kitchen = len(kitchen_sandwiches.intersection(sandwich_is_gf))
        N_reg_kitchen = len(kitchen_sandwiches) - N_gf_kitchen
        N_trays_kitchen = sum(1 for tray, loc in tray_locations.items() if loc == "kitchen")

        # 3. & 4. Identify children needing delivery vs. those served locally
        children_needing_delivery = set(unsatisfied_children)
        used_sandwiches_ontray = set()
        used_trays_at_place = set()

        # Create a list of available sandwiches on trays at places (excluding kitchen)
        available_ontray_at_places = []
        for s, t in ontray_facts.items():
            if t in tray_locations and tray_locations[t] != "kitchen":
                 available_ontray_at_places.append((s, t, tray_locations[t], s in sandwich_is_gf))

        # Try to satisfy children with local sandwiches first
        # Sort children and available sandwiches for deterministic selection
        sorted_unsatisfied_children = sorted(list(unsatisfied_children))
        sorted_available_ontray = sorted(available_ontray_at_places)

        for child in sorted_unsatisfied_children:
            child_place = self.child_location.get(child) # Get place from static init
            if not child_place: continue # Should not happen if init is correct

            child_allergic = self.child_allergy.get(child, False) # Default to not allergic

            found_local_sandwich = False
            for s, t, p, is_gf in sorted_available_ontray:
                if p == child_place and s not in used_sandwiches_ontray and t not in used_trays_at_place:
                    is_suitable = (child_allergic and is_gf) or (not child_allergic)
                    if is_suitable:
                        # Mark as served locally and remove from needing delivery
                        if child in children_needing_delivery: # Check if not already covered by another sandwich
                             children_needing_delivery.remove(child)
                             used_sandwiches_ontray.add(s)
                             used_trays_at_place.add(t)
                             found_local_sandwich = True
                             break # Move to the next child

        # 5. Count needed sandwiches for delivery
        Needed_gf_delivery = sum(1 for child in children_needing_delivery if self.child_allergy.get(child, False))
        Needed_reg_delivery = len(children_needing_delivery) - Needed_gf_delivery

        # 7. Calculate creatable sandwiches
        Can_make_gf = min(N_gf_bread_kitchen, N_gf_content_kitchen, N_sandwich_notexist)

        remaining_notexist = N_sandwich_notexist - Can_make_gf
        remaining_gf_bread = N_gf_bread_kitchen - Can_make_gf
        remaining_gf_content = N_gf_content_kitchen - Can_make_gf

        total_bread_for_reg = N_reg_bread_kitchen + remaining_gf_bread
        total_content_for_reg = N_reg_content_kitchen + remaining_gf_content

        Can_make_reg = min(total_bread_for_reg, total_content_for_reg, remaining_notexist)

        # 8. Calculate available kitchen pool sandwiches
        Avail_gf_kitchen = N_gf_kitchen + Can_make_gf
        Avail_reg_kitchen = N_reg_kitchen + Can_make_reg

        # 9. Determine how many must come from kitchen pool
        From_kitchen_gf = min(Needed_gf_delivery, Avail_gf_kitchen)
        # Use remaining GF sandwiches from kitchen pool for regular needs
        remaining_avail_gf_kitchen = Avail_gf_kitchen - From_kitchen_gf
        From_kitchen_reg = min(Needed_reg_delivery, Avail_reg_kitchen + remaining_avail_gf_kitchen)
        From_kitchen_total = From_kitchen_gf + From_kitchen_reg

        # 11. Calculate how many must be made
        # Need From_kitchen_gf, have N_gf_kitchen made
        Actual_make_gf = max(0, From_kitchen_gf - N_gf_kitchen)
        # Limited by what we can actually make
        Actual_make_gf = min(Actual_make_gf, Can_make_gf)

        # Need From_kitchen_reg, have N_reg_kitchen made + surplus GF from kitchen pool
        # The surplus GF from kitchen pool is Avail_gf_kitchen - From_kitchen_gf
        Actual_make_reg = max(0, From_kitchen_reg - (N_reg_kitchen + max(0, Avail_gf_kitchen - From_kitchen_gf)))
        # Limited by what we can actually make
        Actual_make_reg = min(Actual_make_reg, Can_make_reg)

        h += Actual_make_gf + Actual_make_reg # Cost for making

        # 12. Cost for putting on tray
        h += From_kitchen_total

        # 13. Cost for moving trays
        Places_needing_delivery = {self.child_location.get(child) for child in children_needing_delivery if self.child_location.get(child)}
        h += min(len(Places_needing_delivery), N_trays_kitchen)

        return h
