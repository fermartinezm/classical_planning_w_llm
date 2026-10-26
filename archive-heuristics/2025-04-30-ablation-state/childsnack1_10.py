from heuristics.heuristic_base import Heuristic

# Utility function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace or multiple spaces
    return fact.strip()[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It decomposes the problem into stages: making sandwiches, putting them on trays,
    moving trays to children's locations, and finally serving the children. The
    heuristic is the sum of estimated actions for each stage, based on the number
    of items (sandwiches, children, places) that need to transition to the next stage.

    # Assumptions
    - Each unserved child requires one suitable sandwich (gluten-free for allergic, any for non-allergic).
    - Sandwiches must be made, put on a tray, and the tray moved to the child's location before serving.
    - Trays can carry multiple sandwiches.
    - The kitchen is a fixed place.
    - The heuristic counts the number of items/locations needing attention at each stage, providing an estimate of required actions.

    # Heuristic Initialization
    - Extracts static information about child allergies from the task's static facts.
    - Identifies all children and places involved in the problem by parsing initial state facts and adding the 'kitchen' constant.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated as the sum of four components:
    h = h_make + h_put + h_move + h_serve

    1.  **h_serve:** The number of children who are not yet served. Each unserved child requires one 'serve' action as the final step.
        - Count facts `(served ?c)` in the state.
        - Subtract this count from the total number of children in the problem.

    2.  **h_make:** The number of sandwiches that still need to be made.
        - Count sandwiches that exist (`at_kitchen_sandwich` or `ontray`).
        - The number of sandwiches needed is equal to the number of unserved children.
        - The number of sandwiches to make is the maximum of 0 and (needed sandwiches - existing sandwiches).

    3.  **h_put:** The number of sandwiches that need to be put on a tray.
        - Sandwiches must be on a tray before they can be moved or served.
        - Count sandwiches that are currently on trays (`ontray`).
        - The number of sandwiches that need to be put on trays is the maximum of 0 and (needed sandwiches - sandwiches already on trays). These sandwiches must pass through the 'put_on_tray' stage.

    4.  **h_move:** The number of tray movements required to bring sandwiches to the children.
        - For each place where unserved children are waiting, determine the deficit of suitable sandwiches on trays currently at that location.
        - A suitable sandwich is gluten-free for allergic children and any sandwich for non-allergic children.
        - Count how many GF and Regular sandwiches are needed at each place based on unserved children's allergies.
        - Count how many GF and Regular sandwiches are available on trays currently located at each place.
        - Calculate the deficit for GF and Regular sandwiches at each place. Surplus GF sandwiches can cover Regular needs.
        - Sum the deficits for GF and Regular sandwiches at each place to get the total sandwich deficit at that place.
        - The `h_move` component is the count of distinct places that have a total sandwich deficit greater than 0. This estimates the number of destinations that require at least one tray delivery trip.

    The total heuristic is the sum of these four components. It is 0 if and only if all children are served (goal state).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals
        self.child_allergy = {}  # {child: 'allergic' or 'not_allergic'}
        self.all_children = set()
        self.all_places = set()

        # Extract static child allergy information
        for fact in task.static:
            parts = get_parts(fact)
            if parts and parts[0] == 'allergic_gluten' and len(parts) > 1:
                self.child_allergy[parts[1]] = 'allergic'
            elif parts and parts[0] == 'not_allergic_gluten' and len(parts) > 1:
                self.child_allergy[parts[1]] = 'not_allergic'

        # Extract all children and places involved from initial state facts
        # This is more reliable than parsing object list which might not be standard
        # and ensures we only consider objects relevant to the instance.
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue # Skip empty facts
             predicate = parts[0]
             if predicate in ['waiting', 'served', 'allergic_gluten', 'not_allergic_gluten']:
                 if len(parts) > 1: self.all_children.add(parts[1])
                 if predicate == 'waiting' and len(parts) > 2:
                     self.all_places.add(parts[2])
             elif predicate == 'at' and len(parts) > 2:
                 # (at ?t ?p)
                 self.all_places.add(parts[2])

        # Add the constant kitchen place
        self.all_places.add('kitchen')


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Component 1: h_serve ---
        # Count unserved children
        unserved_children_count = 0
        unserved_children_info = {} # {child: {'place': p, 'allergic': bool}}
        for child in self.all_children:
            if f'(served {child})' not in state:
                unserved_children_count += 1
                # Find child's waiting place and allergy info
                place = None
                for fact in state:
                    parts = get_parts(fact)
                    if parts and parts[0] == 'waiting' and len(parts) > 2 and parts[1] == child:
                        place = parts[2]
                        break
                allergy = self.child_allergy.get(child, 'not_allergic') # Default to not_allergic
                unserved_children_info[child] = {'place': place, 'allergic': (allergy == 'allergic')}

        h_serve = unserved_children_count

        # If no children unserved, goal reached, heuristic is 0
        if h_serve == 0:
            return 0

        # --- Component 2: h_make & Component 3: h_put ---
        # Count existing sandwiches (made) and sandwiches on trays
        made_sandwiches_count = 0
        sandwiches_on_trays_count = 0

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'ontray':
                made_sandwiches_count += 1
                sandwiches_on_trays_count += 1
            elif parts[0] == 'at_kitchen_sandwich':
                made_sandwiches_count += 1

        # Number of sandwiches needed is equal to the number of unserved children
        h_make = max(0, unserved_children_count - made_sandwiches_count)

        # Number of sandwiches that need to be put on a tray
        # These are the needed sandwiches that are not yet on trays.
        h_put = max(0, unserved_children_count - sandwiches_on_trays_count)


        # --- Component 4: h_move ---
        # Number of places needing delivery
        gf_req_at_place = {p: 0 for p in self.all_places}
        reg_req_at_place = {p: 0 for p in self.all_places}
        gf_on_tray_at_place = {p: 0 for p in self.all_places}
        reg_on_tray_at_place = {p: 0 for p in self.all_places}
        tray_location = {} # {tray: place}
        sandwich_is_gf = {} # {sandwich: bool}


        # Populate requirements per place based on unserved children
        for child_info in unserved_children_info.values():
            p = child_info['place']
            if p: # Ensure child has a waiting place
                if child_info['allergic']:
                    gf_req_at_place[p] += 1
                else:
                    reg_req_at_place[p] += 1

        # Populate tray locations and sandwich types
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'at' and len(parts) > 2:
                t, p = parts[1], parts[2]
                tray_location[t] = p
            elif parts[0] == 'no_gluten_sandwich' and len(parts) > 1:
                s = parts[1]
                sandwich_is_gf[s] = True

        # Populate available sandwiches on trays at places
        for fact in state:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'ontray' and len(parts) > 2:
                 s, t = parts[1], parts[2]
                 if t in tray_location: # Tray must have a location
                     p = tray_location[t]
                     if sandwich_is_gf.get(s, False):
                         gf_on_tray_at_place[p] += 1
                     else:
                         reg_on_tray_at_place[p] += 1


        places_with_deficit_count = 0
        for p in self.all_places:
            gf_needed = gf_req_at_place.get(p, 0)
            reg_needed = reg_req_at_place.get(p, 0)
            gf_available = gf_on_tray_at_place.get(p, 0)
            reg_available = reg_on_tray_at_place.get(p, 0)

            # Calculate deficit considering GF can satisfy Reg needs
            gf_deficit = max(0, gf_needed - gf_available)
            # How many Reg needs are left after using available Reg and surplus GF?
            reg_deficit = max(0, reg_needed - (reg_available + max(0, gf_available - gf_needed)))

            if gf_deficit > 0 or reg_deficit > 0:
                # This place needs sandwiches delivered on a tray
                places_with_deficit_count += 1 # Count this destination


        h_move = places_with_deficit_count

        # Total heuristic is the sum of components
        total_heuristic = h_make + h_put + h_move + h_serve

        return total_heuristic
