from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args contains wildcards
    if len(parts) != len(args) and '*' not in args:
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It breaks down the problem into stages for each required sandwich type (gluten-free
    and regular) and location: making the sandwich, putting it on a tray, moving
    the tray to the child's location, and finally serving the child. It sums the
    estimated number of actions needed for each stage across all requirements.

    # Assumptions
    - All children listed in the goal must be served.
    - Children's allergy status and waiting locations are static and available in static facts.
    - Sufficient bread, content, and tray objects exist to solve the problem
      (the heuristic doesn't explicitly check counts of these, only whether
       ingredients are available *somewhere* in the kitchen for making, which is not strictly checked either;
       it assumes making is possible if a sandwich is needed and doesn't exist).
    - Each 'make', 'put-on-tray', and 'serve' action contributes 1 to the cost.
    - Each distinct (location, allergy_status) pair where children are waiting
      and lack a suitable sandwich on a tray at that location requires at least
      one 'move-tray' action.

    # Heuristic Initialization
    - Extract static information: Identify all children that need to be served (from the goal),
      and their allergy status and waiting places from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of children who are in the goal but not yet served. Let this be `UnservedChildren`.
    2. If `UnservedChildren` is empty, the heuristic is 0.
    3. Initialize heuristic `h = 0`.
    4. Add the number of unserved children to `h`. This accounts for the final 'serve' action for each child. `h += |UnservedChildren|`.
    5. Categorize unserved children by allergy status: `NeedsGF` (allergic) and `NeedsReg` (not allergic).
    6. Identify all existing sandwich objects in the state (in kitchen or on trays).
    7. Identify which existing sandwiches are gluten-free (`gf_sandwiches`). Regular sandwiches are those existing sandwiches that are not gluten-free.
    8. Calculate the number of GF and Regular sandwiches that *must* be made to satisfy the needs of unserved children:
       `MakeGF = max(0, |NeedsGF| - |ExistingGF|)`
       `MakeReg = max(0, |NeedsReg| - |ExistingReg|)`
       Add these counts to `h`. `h += MakeGF + MakeReg`.
    9. Identify existing sandwiches that are currently on trays (`ExistingGFOnTray`, `ExistingRegOnTray`).
    10. Calculate the number of GF and Regular sandwiches that *must* be put on trays (either newly made or from the kitchen) to satisfy the needs of unserved children:
        `PutOnTrayGF = max(0, |NeedsGF| - |ExistingGFOnTray|)`
        `PutOnTrayReg = max(0, |NeedsReg| - |ExistingRegOnTray|)`
        Add these counts to `h`. `h += PutOnTrayGF + PutOnTrayReg`.
    11. Identify which distinct (location, allergy_status) pairs, corresponding to unserved children, still need a suitable sandwich delivered on a tray to that location.
        Iterate through `UnservedChildren`. For each child `c` at place `p` with allergy `A`:
        Check if there is *any* tray `t` at location `p` (`(at t p)` is true) that contains *any* sandwich `s` (`(ontray s t)` is true) that is suitable for child `c` (based on allergy and sandwich type).
        If no such sandwich/tray/location combination exists for child `c`, add the pair `(p, A)` to a set `LocationsNeedingDelivery`.
    12. Add the number of distinct pairs in `LocationsNeedingDelivery` to `h`. This accounts for the 'move-tray' actions (one per location/type needing delivery). `h += |LocationsNeedingDelivery|`.

    The final value of `h` is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Goal children (those who need to be served).
        - Child allergy status and waiting places.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all children that need to be served from the goal state.
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Extract static child information: allergy status and waiting place.
        self.child_info = {}
        for fact in self.static_facts:
            if match(fact, "waiting", "?c", "?p"):
                child, place = get_parts(fact)[1:]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['place'] = place
            elif match(fact, "allergic_gluten", "?c"):
                child = get_parts(fact)[1]
                if child not in self.child_info:
                     self.child_info[child] = {}
                self.child_info[child]['allergic'] = True
            elif match(fact, "not_allergic_gluten", "?c"):
                child = get_parts(fact)[1]
                if child not in self.child_info:
                     self.child_info[child] = {}
                self.child_info[child]['allergic'] = False

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = {c for c in self.goal_children if f"(served {c})" not in state}

        # 2. If all goal children are served, heuristic is 0.
        if not unserved_children:
            return 0

        # 3. Initialize heuristic
        h = 0

        # 4. Add cost for serving each child
        h += len(unserved_children) # Each needs a 'serve' action

        # 5. Categorize unserved children by allergy status
        needs_gf = {c for c in unserved_children if self.child_info[c]['allergic']}
        needs_reg = {c for c in unserved_children if not self.child_info[c]['allergic']}

        # 6. Identify all existing sandwich objects
        existing_sandwiches = set()
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "?s"):
                 s = get_parts(fact)[1]
                 existing_sandwiches.add(s)
             elif match(fact, "ontray", "?s", "?t"):
                 s = get_parts(fact)[1]
                 existing_sandwiches.add(s)

        # 7. Identify GF sandwiches among existing ones
        gf_sandwiches = {s for s in existing_sandwiches if f"(no_gluten_sandwich {s})" in state}
        existing_gf = gf_sandwiches
        existing_reg = {s for s in existing_sandwiches if s not in gf_sandwiches} # Regular sandwiches are those that exist and are not GF

        # Helper to check if a sandwich 's' is suitable for allergy status 'A'
        def is_suitable(s, is_allergic):
            if is_allergic:
                return s in gf_sandwiches
            else: # Not allergic, any existing sandwich is suitable
                return s in existing_sandwiches # Check if it's a known sandwich object

        # 8. Calculate sandwiches that must be made
        make_gf = max(0, len(needs_gf) - len(existing_gf))
        make_reg = max(0, len(needs_reg) - len(existing_reg))
        h += make_gf + make_reg # Cost for 'make_sandwich' actions

        # 9. Identify existing sandwiches on trays
        existing_gf_ontray = set()
        existing_reg_ontray = set()
        for fact in state:
            if match(fact, "ontray", "?s", "?t"):
                s = get_parts(fact)[1]
                if s in gf_sandwiches:
                    existing_gf_ontray.add(s)
                else:
                    existing_reg_ontray.add(s)

        # 10. Calculate sandwiches that must be put on trays
        # This counts how many *needed* sandwiches are not yet on trays.
        # It's max(0, needed_total - existing_ontray).
        put_on_tray_gf = max(0, len(needs_gf) - len(existing_gf_ontray))
        put_on_tray_reg = max(0, len(needs_reg) - len(existing_reg_ontray))
        h += put_on_tray_gf + put_on_tray_reg # Cost for 'put_on_tray' actions

        # 11. Identify locations needing delivery
        locations_needing_delivery = set() # Stores (place, allergy_status) tuples

        # Get current tray locations
        tray_locations = {}
        for fact in state:
            if match(fact, "at", "?t", "?p"):
                tray, place = get_parts(fact)[1:]
                tray_locations[tray] = place

        # Get sandwiches on trays mapping tray -> {sandwiches}
        sandwiches_on_trays_map = {}
        for fact in state:
            if match(fact, "ontray", "?s", "?t"):
                s, t = get_parts(fact)[1:]
                if t not in sandwiches_on_trays_map:
                    sandwiches_on_trays_map[t] = set()
                sandwiches_on_trays_map[t].add(s)


        for child in unserved_children:
            place = self.child_info[child]['place']
            is_allergic = self.child_info[child]['allergic']

            location_satisfied = False
            # Check if any tray at this location has a suitable sandwich
            for tray, current_place in tray_locations.items():
                if current_place == place:
                    if tray in sandwiches_on_trays_map:
                        for s in sandwiches_on_trays_map[tray]:
                            if is_suitable(s, is_allergic):
                                location_satisfied = True
                                break # Found a suitable sandwich at this location
                if location_satisfied:
                    break # No need to check other trays for this child

            if not location_satisfied:
                # This child needs a delivery to their location
                locations_needing_delivery.add((place, is_allergic))

        # 12. Add cost for moving trays
        h += len(locations_needing_delivery) # Cost for 'move_tray' actions (one per location/type needing delivery)

        return h
