from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed inputs gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def is_sandwich_suitable_for_any_unserved(sandwich, unserved_children, allergy_status, gluten_free_sandwiches):
    """
    Checks if a given sandwich is suitable for at least one unserved child.
    A sandwich is suitable if it's gluten-free and there's an allergic unserved child,
    OR if there's any non-allergic unserved child (who can eat any sandwich).
    """
    is_gf = sandwich in gluten_free_sandwiches
    any_allergic_unserved = any(allergy_status.get(c) == 'allergic' for c in unserved_children)
    any_not_allergic_unserved = any(allergy_status.get(c) == 'not_allergic' for c in unserved_children)

    # A GF sandwich is suitable if any allergic child needs one OR any non-allergic child needs one.
    # A non-GF sandwich is suitable only if any non-allergic child needs one.
    return (is_gf and (any_allergic_unserved or any_not_allergic_unserved)) or (not is_gf and any_not_allergic_unserved)


def suitable_sandwich_on_tray_at_location(child, child_waiting_location, child_allergy, gluten_free_sandwiches, sandwiches_on_trays_at_location):
    """
    Checks if a suitable sandwich for the given child is currently on a tray at the child's waiting location.
    """
    child_loc = child_waiting_location.get(child)
    if not child_loc:
        return False # Child is not waiting anywhere? Should not happen based on domain.

    sandwiches_at_loc = sandwiches_on_trays_at_location.get(child_loc, set())
    if not sandwiches_at_loc:
        return False # No sandwiches on trays at this location

    child_allergy_status = child_allergy.get(child)

    for s in sandwiches_at_loc:
        is_gf = s in gluten_free_sandwiches
        if child_allergy_status == 'allergic':
            if is_gf:
                return True # Found a GF sandwich for an allergic child
        elif child_allergy_status == 'not_allergic':
            return True # Any sandwich is suitable for a non-allergic child

    return False # No suitable sandwich found at the location

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of `serve` actions needed, the number of new sandwiches
    that must be made, the number of sandwiches that need to be put on trays
    (those starting in the kitchen or newly made), and the number of distinct
    locations that require a tray delivery.

    # Assumptions
    - All children specified in the goal must be served.
    - Children's allergy status and waiting locations are static.
    - Trays can hold multiple sandwiches.
    - Enough bread and content portions exist in the kitchen to make any required sandwiches.
    - Enough `notexist` sandwich objects exist to represent newly made sandwiches.
    - Trays are available in the kitchen to put sandwiches on.

    # Heuristic Initialization
    - Extracts the set of children that need to be served (from the goal).
    - Extracts static information: child allergy status and waiting locations.
    - Extracts the names of all sandwich and place objects from the task definition.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of unserved children (`N_unserved`). If none, heuristic is 0.
    2. For each unserved child, determine their waiting location and allergy status using pre-calculated static information.
    3. Parse the current state to find:
       - Which sandwiches are on which trays (`ontray`).
       - Where each tray is located (`at`).
       - Which sandwiches are in the kitchen (`at_kitchen_sandwich`).
       - Which sandwiches are gluten-free (`no_gluten_sandwich`).
       - Which sandwich objects have not yet been created (`notexist`).
    4. Determine for each unserved child whether a suitable sandwich is already on a tray at their waiting location. This is done by checking the `ontray` and `at` facts in the state and comparing sandwich suitability with the child's allergy status.
    5. Count the number of unserved children who *do not* have a suitable sandwich ready at their location (`N_needs_delivery_to_location`). These children require a sandwich delivery.
    6. Identify the set of distinct locations where at least one unserved child needs a sandwich delivery (`LocationsNeedingDelivery`). The number of `move_tray` actions needed is estimated as the size of this set, assuming one trip per location is sufficient to deliver all needed sandwiches there.
    7. Count the total number of suitable sandwiches currently available *anywhere* (in kitchen or on trays) that are *not* yet at the location of an unserved child who needs them. (Simplified: count all suitable sandwiches that exist). Let this be `AvailableSuitableAnywhere`. A sandwich is considered suitable if it can satisfy the need of at least one unserved child (GF for allergic/non-allergic, Any for non-allergic).
    8. Calculate the number of sandwiches that *must* be newly made to satisfy the delivery needs: `N_must_make = max(0, N_needs_delivery_to_location - AvailableSuitableAnywhere)`. This assumes we prioritize using existing sandwiches before making new ones.
    9. Count the number of suitable sandwiches currently `at_kitchen_sandwich` (`AvailableSuitableKitchen`). Suitability is checked as in step 7.
    10. Calculate the number of `put_on_tray` actions needed: Each sandwich sourced from the kitchen or newly made that is needed for delivery requires a `put_on_tray` action. This is estimated as the number of needed deliveries that can be fulfilled by kitchen sandwiches (`min(N_needs_delivery_to_location, AvailableSuitableKitchen)`) plus the number of needed deliveries that must be fulfilled by newly made sandwiches (`N_must_make`).
    11. The total heuristic is the sum of the estimated actions:
        - `N_unserved` (for the final `serve` action for each child)
        - `N_must_make` (for the `make_sandwich` action for each new sandwich)
        - The estimated `put_on_tray` actions
        - The estimated `move_tray` actions (`len(LocationsNeedingDelivery)`)
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goal_children = {get_parts(g)[1] for g in task.goals if match(g, "served", "*")}

        self.child_allergy = {}
        self.child_waiting_location = {}
        for fact in task.static:
            if match(fact, "allergic_gluten", "*"):
                self.child_allergy[get_parts(fact)[1]] = 'allergic'
            elif match(fact, "not_allergic_gluten", "*"):
                self.child_allergy[get_parts(fact)[1]] = 'not_allergic'
            elif match(fact, "waiting", "*", "*"):
                self.child_waiting_location[get_parts(fact)[1]] = get_parts(fact)[2]

        self.all_sandwich_objects = {obj.split(' - ')[0] for obj in task.objects if obj.split(' - ')[1] == 'sandwich'}
        self.all_places = {obj.split(' - ')[0] for obj in task.objects if obj.split(' - ')[1] == 'place'}
        # Add the constant kitchen place if it's not explicitly in objects (though it should be)
        if 'kitchen' not in self.all_places:
             self.all_places.add('kitchen')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = [c for c in self.goal_children if '(served ' + c + ')' not in state]
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal state

        # Parse state facts
        sandwiches_on_trays = {} # {sandwich: tray}
        tray_locations = {} # {tray: place}
        kitchen_sandwiches = set() # {sandwich}
        gluten_free_sandwiches = set() # {sandwich}
        notexist_sandwiches = set() # {sandwich}

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or malformed facts

            predicate = parts[0]
            if predicate == 'ontray' and len(parts) == 3:
                sandwiches_on_trays[parts[1]] = parts[2]
            elif predicate == 'at' and len(parts) == 3:
                 # In childsnack, only trays have 'at' predicate with a place
                 tray_locations[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_sandwich' and len(parts) == 2:
                kitchen_sandwiches.add(parts[1])
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                gluten_free_sandwiches.add(parts[1])
            elif predicate == 'notexist' and len(parts) == 2:
                notexist_sandwiches.add(parts[1])

        # Map sandwiches on trays to their locations
        sandwiches_on_trays_at_location = {} # {place: {sandwich}}
        for s, t in sandwiches_on_trays.items():
            if t in tray_locations:
                p = tray_locations[t]
                sandwiches_on_trays_at_location.setdefault(p, set()).add(s)

        # 4. Determine which children need delivery
        N_needs_delivery_to_location = 0
        LocationsNeedingDelivery = set()

        for child in unserved_children:
            if not suitable_sandwich_on_tray_at_location(child, self.child_waiting_location, self.child_allergy, gluten_free_sandwiches, sandwiches_on_trays_at_location):
                 N_needs_delivery_to_location += 1
                 child_loc = self.child_waiting_location.get(child)
                 if child_loc:
                     LocationsNeedingDelivery.add(child_loc)

        # 7. Count available suitable sandwiches anywhere (that exist)
        # A sandwich is suitable if it can serve *any* unserved child.
        existing_sandwiches = self.all_sandwich_objects - notexist_sandwiches
        AvailableSuitableAnywhere = len([
            s for s in existing_sandwiches
            if is_sandwich_suitable_for_any_unserved(s, unserved_children, self.child_allergy, gluten_free_sandwiches)
        ])

        # 8. Calculate sandwiches that must be newly made
        N_must_make = max(0, N_needs_delivery_to_location - AvailableSuitableAnywhere)
        # Note: This assumes enough notexist sandwiches and kitchen resources exist.

        # 9. Count suitable sandwiches in the kitchen
        AvailableSuitableKitchen = len([
             s for s in kitchen_sandwiches
             if is_sandwich_suitable_for_any_unserved(s, unserved_children, self.child_allergy, gluten_free_sandwiches)
        ])

        # 10. Calculate put_on_tray actions needed
        # Sandwiches sourced from kitchen or newly made need put_on_tray
        # We need N_needs_delivery_to_location sandwiches delivered.
        # min(N_needs_delivery_to_location, AvailableSuitableKitchen) can come from kitchen (need put).
        # N_must_make must be made (need put).
        # The rest come from trays elsewhere (don't need put).
        N_put_actions = min(N_needs_delivery_to_location, AvailableSuitableKitchen) + N_must_make

        # 11. Calculate move_tray actions needed
        N_move_actions = len(LocationsNeedingDelivery)

        # Total heuristic estimate
        # N_unserved (serve) + N_must_make (make) + N_put_actions (put) + N_move_actions (move)
        heuristic_value = N_unserved + N_must_make + N_put_actions + N_move_actions

        return heuristic_value
