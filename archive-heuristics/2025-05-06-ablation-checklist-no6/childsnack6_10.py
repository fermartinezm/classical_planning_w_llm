from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import copy

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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all unserved children.
    It counts the number of sandwiches needed from different stages of the
    preparation/delivery pipeline and assigns a cost based on the stage:
    1. Sandwich on tray at child's location (Cost: 1 - serve)
    2. Sandwich on tray elsewhere (Cost: 2 - move tray + serve)
    3. Sandwich in kitchen (Cost: 3 - put on tray + move tray + serve)
    4. Sandwich needs to be made (Cost: 4 - make + put on tray + move tray + serve)

    Prioritizes gluten-free sandwiches for allergic children.
    Assumes sufficient resources (trays, notexist sandwiches, ingredients)
    exist in solvable problems beyond what's explicitly counted in the state
    for making sandwiches.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goal_children = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'served':
                self.goal_children.add(parts[1])

        self.child_info = {} # child -> {'place': p, 'allergic': bool}
        self.places = {'kitchen'} # kitchen is a constant place

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                child, place = parts[1], parts[2]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['place'] = place
                self.places.add(place)
            elif parts[0] == 'allergic_gluten':
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = True
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = False

        # Ensure all goal children have necessary info (fallback defaults)
        for child in self.goal_children:
             if child not in self.child_info:
                 # Child is in goal but not in static waiting/allergy facts? Should not happen in valid PDDL.
                 # Defaulting to not allergic and kitchen place as a fallback.
                 self.child_info[child] = {'allergic': False, 'place': 'kitchen'}
             if 'allergic' not in self.child_info[child]:
                  self.child_info[child]['allergic'] = False # Default if only waiting fact exists
             if 'place' not in self.child_info[child]:
                  self.child_info[child]['place'] = 'kitchen' # Default if only allergy fact exists


        # Add places from initial state tray locations
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1].startswith('tray'):
                 self.places.add(parts[2])


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs
        unserved_children = {c for c in self.goal_children if '(served ' + c + ')' not in state}

        if not unserved_children:
            return 0 # Goal reached

        # Count unserved children by location and allergy status
        num_unserved_allergic_at = {p: 0 for p in self.places}
        num_unserved_not_allergic_at = {p: 0 for p in self.places}

        for child in unserved_children:
            # Assuming (waiting child place) is in static facts for all goal children
            # Fallback to 'kitchen' if place info is missing (should not happen in valid PDDL)
            place = self.child_info.get(child, {}).get('place', 'kitchen')
            is_allergic = self.child_info.get(child, {}).get('allergic', False) # Fallback to not allergic

            if is_allergic:
                num_unserved_allergic_at[place] += 1
            else:
                num_unserved_not_allergic_at[place] += 1

        needed_gf = sum(num_unserved_allergic_at.values())
        needed_any = sum(num_unserved_not_allergic_at.values()) # This is the *additional* non-GF need

        # 2. Count available sandwiches by type and state
        tray_location = {}
        sandwich_is_gf = {}
        num_gf_bread_kitchen = 0
        num_gf_content_kitchen = 0
        num_any_bread_kitchen = 0
        num_any_content_kitchen = 0

        # First pass to get tray locations, GF status of sandwiches, and ingredient counts
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('tray'):
                tray_location[parts[1]] = parts[2]
            elif parts[0] == 'no_gluten_sandwich':
                sandwich_is_gf[parts[1]] = True
            elif parts[0] == 'at_kitchen_bread':
                num_any_bread_kitchen += 1
                # Check if the specific bread is GF
                if '(no_gluten_bread ' + parts[1] + ')' in state:
                    num_gf_bread_kitchen += 1
            elif parts[0] == 'at_kitchen_content':
                num_any_content_kitchen += 1
                 # Check if the specific content is GF
                if '(no_gluten_content ' + parts[1] + ')' in state:
                    num_gf_content_kitchen += 1


        num_gf_ontray_at_p = {p: 0 for p in self.places}
        num_nongf_ontray_at_p = {p: 0 for p in self.places}
        num_gf_kitchen = 0
        num_nongf_kitchen = 0
        num_notexist = 0

        # Second pass to count sandwiches in different states
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                s, t = parts[1], parts[2]
                p = tray_location.get(t)
                if p: # Tray must have a location
                    if sandwich_is_gf.get(s, False):
                        num_gf_ontray_at_p[p] += 1
                    else:
                        num_nongf_ontray_at_p[p] += 1
            elif parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                if sandwich_is_gf.get(s, False):
                    num_gf_kitchen += 1
                else:
                    num_nongf_kitchen += 1
            elif parts[0] == 'notexist':
                num_notexist += 1

        # Available make capacity (ingredient pairs)
        avail_gf_make_count = min(num_gf_bread_kitchen, num_gf_content_kitchen)
        avail_any_make_count = min(num_any_bread_kitchen, num_any_content_kitchen)


        # 3. Calculate heuristic cost by greedily assigning sandwiches from cheapest stages
        h = 0

        # Track remaining needs and resources
        rem_needed_gf = needed_gf
        rem_needed_any = needed_any # Additional non-GF need

        # Use copies for counts that are consumed within a stage calculation loop
        rem_num_gf_ontray_at_p = copy.copy(num_gf_ontray_at_p)
        rem_num_nongf_ontray_at_p = copy.copy(num_nongf_ontray_at_p)

        # Stage 4: Serve (cost 1) - Sandwiches already on tray at child's location
        served_gf_at_p = 0
        served_any_at_p = 0
        for place in self.places:
            # Serve allergic children at place
            can_serve_gf = min(num_unserved_allergic_at[place], rem_num_gf_ontray_at_p[place])
            served_gf_at_p += can_serve_gf
            rem_num_gf_ontray_at_p[place] -= can_serve_gf # Sandwiches used

            # Serve non-allergic children at place using remaining GF and non-GF
            can_serve_any = min(num_unserved_not_allergic_at[place], rem_num_gf_ontray_at_p[place] + rem_num_nongf_ontray_at_p[place])
            served_any_at_p += can_serve_any
            # Update remaining counts based on which sandwiches were used for 'any' serves
            used_gf_for_any = min(rem_num_gf_ontray_at_p[place], can_serve_any)
            rem_num_gf_ontray_at_p[place] -= used_gf_for_any
            rem_num_nongf_ontray_at_p[place] -= (can_serve_any - used_gf_for_any)


        h += (served_gf_at_p + served_any_at_p) * 1
        rem_needed_gf -= served_gf_at_p
        rem_needed_any -= served_any_at_p

        # Count total remaining ontray sandwiches (now considered "elsewhere")
        rem_gf_ontray_elsewhere = sum(rem_num_gf_ontray_at_p.values())
        rem_nongf_ontray_elsewhere = sum(rem_num_nongf_ontray_at_p.values())

        # Stage 3: Move + Serve (cost 2) - Sandwiches on trays elsewhere
        use_gf_elsewhere = min(rem_needed_gf, rem_gf_ontray_elsewhere)
        h += use_gf_elsewhere * 2
        rem_needed_gf -= use_gf_elsewhere
        rem_gf_ontray_elsewhere -= use_gf_elsewhere # Sandwiches used

        use_any_elsewhere = min(rem_needed_any, rem_gf_ontray_elsewhere + rem_nongf_ontray_elsewhere)
        h += use_any_elsewhere * 2
        rem_needed_any -= use_any_elsewhere
        # No need to track remaining ontray elsewhere, they are used up conceptually

        # Stage 2: Put + Move + Serve (cost 3) - Sandwiches in kitchen
        rem_num_gf_kitchen = num_gf_kitchen
        rem_num_nongf_kitchen = num_nongf_kitchen

        use_gf_kitchen = min(rem_needed_gf, rem_num_gf_kitchen)
        h += use_gf_kitchen * 3
        rem_needed_gf -= use_gf_kitchen
        rem_num_gf_kitchen -= use_gf_kitchen # Sandwiches used

        use_any_kitchen = min(rem_needed_any, rem_num_gf_kitchen + rem_num_nongf_kitchen)
        h += use_any_kitchen * 3
        rem_needed_any -= use_any_kitchen
        # No need to track remaining kitchen, they are used up conceptually

        # Stage 1: Make + Put + Move + Serve (cost 4) - Sandwiches to be made
        rem_num_notexist = num_notexist
        rem_avail_gf_make = avail_gf_make_count
        rem_avail_any_make = avail_any_make_count # Total ingredient pairs

        # Use GF make capacity for needed GF
        can_make_gf = min(rem_num_notexist, rem_avail_gf_make)
        use_gf_make = min(rem_needed_gf, can_make_gf)
        h += use_gf_make * 4
        rem_needed_gf -= use_gf_make
        rem_num_notexist -= use_gf_make
        rem_avail_gf_make -= use_gf_make # These GF ingredients are used

        # Use remaining make capacity (ingredients) for needed Any
        # Total ingredient pairs used so far = use_gf_make
        # Remaining total ingredient pairs = avail_any_make_count - use_gf_make
        rem_avail_any_ingredients = avail_any_make_count - use_gf_make

        can_make_any = min(rem_num_notexist, rem_avail_any_ingredients)
        use_any_make = min(rem_needed_any, can_make_any)
        h += use_any_make * 4
        rem_needed_any -= use_any_make
        # rem_num_notexist -= use_any_make # Not needed after this step

        # If rem_needed_gf > 0 or rem_needed_any > 0, it means we couldn't find/make enough sandwiches.
        # For solvable problems, this shouldn't happen if resources are sufficient.
        # We can assert or return infinity if needed, but for a non-admissible heuristic,
        # just returning the calculated value is usually fine, assuming the problem is solvable.

        return h
