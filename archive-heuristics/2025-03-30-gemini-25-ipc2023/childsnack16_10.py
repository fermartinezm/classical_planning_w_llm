import itertools
from fnmatch import fnmatch
# Assuming the Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts represented as strings
def get_parts(fact):
    """
    Extract the components of a PDDL fact string by removing parentheses
    and splitting by space.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    """
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact string matches a given pattern.
    Uses fnmatch for wildcard support in the pattern arguments.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: A sequence of strings representing the pattern, e.g., ("at", "*", "rooma").
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the pattern length
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern element
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It calculates the cost by summing estimates for the necessary steps: making sandwiches,
    putting sandwiches on trays, moving trays to children's locations, and serving the children.
    The heuristic tries to utilize existing sandwiches and tray positions to reduce the estimate,
    aiming for informativeness rather than admissibility.

    # Assumptions
    - The goal is always to have all children served (`(served ?c)` for all children defined in the problem).
    - Children's locations (`waiting ?c ?p`) and allergies (`allergic_gluten`, `not_allergic_gluten`) are static facts.
    - Gluten-free status of bread/content (`no_gluten_bread`, `no_gluten_content`) is static.
    - The heuristic ignores limits on bread/content ingredients when calculating the cost of making sandwiches.
      It assumes enough ingredients are always available.
    - The heuristic uses a simplified model for tray movement costs. It counts 1 action if a tray needs to be brought
      to the kitchen for pickups (if none is there), and 1 action for each destination location that needs a delivery
      but does not currently have a tray. It does not model optimal routing or capacity.
    - It assumes any tray can be used for any sandwich/child.
    - It assumes objects found in `(at ?obj ?loc)` facts are trays, which might require adjustment if other object
      types use the `at` predicate.

    # Heuristic Initialization
    - Extracts static information about children (location, allergy status), gluten-free ingredients,
      and the set of all children expected to be served from the task's static facts and goals.
    - Stores the goal predicates (assumed to be `(served ?c)` for all children).

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Unserved Children:** Determine which children still need the `(served ?c)` goal fact to be true,
        based on the current state. If all children listed in the goals are served, the heuristic value is 0.
    2.  **Identify Available Resources:** Scan the current state to find:
        - Existing sandwiches and their locations (kitchen or on a tray).
        - The gluten-free status of existing sandwiches (`no_gluten_sandwich`).
        - The current location of all trays (`at ?t ?p`).
    3.  **Match Needs to Resources:** For each unserved child, determine their sandwich need (gluten-free or regular).
        Greedily assign available, suitable sandwiches (preferring regular for non-allergic children, but allowing
        gluten-free if necessary) to meet these needs. Keep track of which assigned sandwiches come from the kitchen
        supply versus those already on a tray.
    4.  **Calculate Make Cost:** Count how many children could not be assigned an existing sandwich. Each of these
        children requires a sandwich to be made, costing 1 `make_sandwich` (or `make_sandwich_no_gluten`) action each.
    5.  **Calculate Put-on-Tray Cost:** Count how many sandwiches need a `put_on_tray` action. This includes all
        sandwiches that need to be made (as they appear in the kitchen) plus any existing sandwiches that were assigned
        from the kitchen stock. Each costs 1 action.
    6.  **Calculate Serve Cost:** Each unserved child requires one `serve_sandwich` (or `serve_sandwich_no_gluten`)
        action to achieve their goal. The cost is 1 per unserved child.
    7.  **Calculate Move Cost:** Estimate the minimum number of `move_tray` actions:
        a. Check if any sandwiches need to be picked up from the kitchen (either newly made or assigned from kitchen stock).
        b. Check if any tray is currently located at the kitchen. If pickups are needed but no tray is at the kitchen,
           add 1 move cost (assuming at least one tray exists in the problem).
        c. Identify all unique non-kitchen locations where unserved children are waiting.
        d. Check if a tray is currently present at each of these destination locations. For each required destination
           location that currently lacks a tray, add 1 move cost (assuming a tray can be moved there).
    8.  **Total Heuristic Value:** The final heuristic estimate is the sum of the costs calculated in steps 4, 5, 6, and 7.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information from the task.
        """
        self.goals = task.goals
        static_facts = task.static

        self.child_locations = {} # Map: child -> place
        self.child_needs_gf = {}  # Map: child -> bool (True if allergic)
        self.children = set()     # Set of all child names
        self.gf_bread = set()     # Set of gluten-free bread portions
        self.gf_content = set()   # Set of gluten-free content portions

        # Parse static facts to populate child info and ingredient info
        for fact in static_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "waiting":
                child, place = parts[1], parts[2]
                self.child_locations[child] = place
                self.children.add(child)
            elif predicate == "allergic_gluten":
                child = parts[1]
                self.child_needs_gf[child] = True
                self.children.add(child)
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                self.child_needs_gf[child] = False
                self.children.add(child)
            elif predicate == "no_gluten_bread":
                self.gf_bread.add(parts[1])
            elif predicate == "no_gluten_content":
                self.gf_content.add(parts[1])

        # Ensure all children mentioned in goals are included in our set
        for goal in self.goals:
             if match(goal, "served", "*"):
                 child_name = get_parts(goal)[1]
                 self.children.add(child_name)
                 # If a child is in goal but not static facts, we might lack info.
                 # Assume well-formed problems where static facts cover all children.
                 if child_name not in self.child_needs_gf:
                     # Default assumption if allergy info is missing - might need refinement
                     # based on typical problem structure. Let's assume false for now.
                     self.child_needs_gf[child_name] = False
                 if child_name not in self.child_locations:
                     # This would be problematic, heuristic might fail. Assume well-formed.
                     pass


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.
        """
        state = node.state

        # 1. Identify Unserved Children
        served_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.children - served_in_state

        # If all children are served, the goal is reached.
        if not unserved_children:
            return 0

        # 2. Identify Available Resources from the current state
        sandwiches_ontray = {} # Map: sandwich -> tray
        sandwiches_at_kitchen = set() # Set: sandwich
        sandwich_is_gf = {} # Map: sandwich -> bool (True if GF)
        tray_locations = {} # Map: tray -> place
        trays = set()       # Set: tray names

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "ontray":
                s, t = parts[1], parts[2]
                sandwiches_ontray[s] = t
            elif predicate == "at_kitchen_sandwich":
                sandwiches_at_kitchen.add(parts[1])
            elif predicate == "no_gluten_sandwich":
                # This fact indicates a sandwich exists and is GF
                sandwich_is_gf[parts[1]] = True
            elif predicate == "at":
                obj, loc = parts[1], parts[2]
                # Assumption: Objects involved in 'at' predicates are trays.
                # This could be fragile if other objects (e.g., robot) use 'at'.
                # A robust solution would use type information if available.
                # For now, proceed with this assumption based on domain structure.
                if loc != 'kitchen' or not obj.startswith('kitchen'): # Avoid matching kitchen constant itself
                    tray_locations[obj] = loc
                    trays.add(obj)

        # Determine GF status for all existing sandwiches (those not 'notexist')
        existing_sandwiches = sandwiches_at_kitchen.union(sandwiches_ontray.keys())
        for s in existing_sandwiches:
            if s not in sandwich_is_gf:
                 # If not explicitly marked as GF, assume it's regular
                 sandwich_is_gf[s] = False

        # Separate existing sandwiches by type for easier matching
        sandwiches_in_play_gf = {s for s in existing_sandwiches if sandwich_is_gf[s]}
        sandwiches_in_play_reg = {s for s in existing_sandwiches if not sandwich_is_gf[s]}

        # 3. Match Needs to Resources
        sandwiches_to_make_gf = 0
        sandwiches_to_make_reg = 0
        # Track assignments: child -> (sandwich_id, is_gf, source_is_kitchen)
        assigned_sandwiches = {}

        # Process children deterministically
        unserved_children_list = sorted(list(unserved_children))

        for child in unserved_children_list:
            # Ensure we have info for the child (should be guaranteed by __init__)
            needs_gf = self.child_needs_gf.get(child, False)
            target_place = self.child_locations.get(child, None)
            if target_place is None: continue # Should not happen in well-formed problems

            found_sandwich = False
            source_is_kitchen = False
            assigned_s = None
            assigned_is_gf = False

            if needs_gf:
                # Allergic child needs a GF sandwich
                if sandwiches_in_play_gf:
                    assigned_s = sandwiches_in_play_gf.pop() # Assign an available GF sandwich
                    assigned_is_gf = True
                    found_sandwich = True
            else:
                # Non-allergic child prefers regular, but GF is acceptable
                if sandwiches_in_play_reg:
                    assigned_s = sandwiches_in_play_reg.pop() # Assign available regular sandwich
                    assigned_is_gf = False
                    found_sandwich = True
                elif sandwiches_in_play_gf:
                    assigned_s = sandwiches_in_play_gf.pop() # Assign available GF sandwich if no regular
                    assigned_is_gf = True
                    found_sandwich = True

            if found_sandwich:
                # Check if the assigned sandwich came from the kitchen or a tray
                source_is_kitchen = (assigned_s in sandwiches_at_kitchen)
                assigned_sandwiches[child] = (assigned_s, assigned_is_gf, source_is_kitchen)
            else:
                # No suitable existing sandwich found, mark one for creation
                if needs_gf:
                    sandwiches_to_make_gf += 1
                else:
                    sandwiches_to_make_reg += 1

        # 4. Calculate Make Cost
        # Cost is 1 per sandwich that needs to be made
        cost_make = sandwiches_to_make_gf + sandwiches_to_make_reg

        # 5. Calculate Put-on-Tray Cost
        # Cost is 1 if sandwich is made (appears at kitchen) or assigned from kitchen stock
        cost_put = cost_make + sum(1 for s, is_gf, source_is_kitchen in assigned_sandwiches.values() if source_is_kitchen)

        # 6. Calculate Serve Cost
        # Cost is 1 per child that needs serving
        cost_serve = len(unserved_children)

        # 7. Calculate Move Cost (Simplified Estimation)
        cost_move = 0
        # Check if any pickups are needed from the kitchen
        needs_pickup_from_kitchen = (cost_make > 0) or any(source_is_kitchen for _, _, source_is_kitchen in assigned_sandwiches.values())

        # Check if a tray is already at the kitchen
        trays_at_kitchen = any(loc == 'kitchen' for loc in tray_locations.values())

        # If pickup needed and no tray at kitchen, add cost (if trays exist at all)
        if needs_pickup_from_kitchen and not trays_at_kitchen and trays:
             cost_move += 1

        # Identify unique non-kitchen locations needing deliveries
        locations_needing_delivery = {
            self.child_locations[c]
            for c in unserved_children
            if c in self.child_locations and self.child_locations[c] != 'kitchen'
        }

        # Check if trays are present at these locations
        trays_at_target_location = {loc: False for loc in locations_needing_delivery}
        for t, loc in tray_locations.items():
            if loc in trays_at_target_location:
                trays_at_target_location[loc] = True

        # Add cost for each location needing delivery but lacking a tray (if trays exist)
        for loc, has_tray in trays_at_target_location.items():
            if not has_tray and trays:
                 cost_move += 1

        # 8. Total Heuristic Value
        # Sum the costs of the necessary action types
        h = cost_make + cost_put + cost_serve + cost_move
        return h
