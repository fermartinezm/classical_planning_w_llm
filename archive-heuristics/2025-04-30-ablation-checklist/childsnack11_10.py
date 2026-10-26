from fnmatch import fnmatch
# Assuming heuristic_base is available in the execution environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not available
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.task = task
        def __call__(self, node):
            raise NotImplementedError
        def __str__(self):
            return self.__class__.__name__
        def __repr__(self):
            return self.__str__()


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic sums up the estimated cost for each unserved child,
    based on the current state of the required sandwich:
    - If the required sandwich is already on a tray at the child's location: cost 0 (delivery steps) + 1 (serve) = 1
    - If the required sandwich is on a tray at another location: cost 1 (move_tray) + 1 (serve) = 2
    - If the required sandwich is in the kitchen: cost 1 (put_on_tray) + 1 (move_tray) + 1 (serve) = 3
    - If the required sandwich needs to be made: cost 1 (make) + 1 (put_on_tray) + 1 (move_tray) + 1 (serve) = 4

    The heuristic counts the total number of GF and Regular sandwiches needed,
    and then "satisfies" these needs by consuming available sandwiches from
    the pools closest to the goal state first (ontray@loc, then ontray@other,
    then kitchen, then needing_make).

    Assumes sufficient trays, ingredients, and notexist sandwich objects are
    eventually available for the 'make' step cost calculation.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child allergies and initial locations
        from static facts and identifying goal children.
        """
        super().__init__(task) # Call base class constructor

        # Pre-process static facts to map children to allergies and initial locations
        self.child_allergy = {}
        self.child_location = {}
        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            if parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = 'gf'
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = 'reg'
            elif parts[0] == 'waiting' and len(parts) == 3:
                 # Waiting location is static in this domain based on examples
                self.child_location[parts[1]] = parts[2]

        # Get all children mentioned in the goal (those that need to be served)
        self.goal_children = {get_parts(g)[1] for g in self.goals if match(g, 'served', '*')}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        h = 0

        # 1. Identify unserved children and their needs (location, allergy)
        unserved_children = []
        needed_at_loc = {} # {location: {'gf': count, 'reg': count}}
        for child in self.goal_children:
            if f'(served {child})' not in state:
                unserved_children.append(child)
                loc = self.child_location.get(child)
                allergy = self.child_allergy.get(child) # 'gf' or 'reg'
                if loc: # Ensure location is known from static facts
                    if loc not in needed_at_loc:
                        needed_at_loc[loc] = {'gf': 0, 'reg': 0}
                    if allergy == 'gf':
                        needed_at_loc[loc]['gf'] += 1
                    else: # Assume not_allergic_gluten if not allergic_gluten
                        needed_at_loc[loc]['reg'] += 1
                # else: Child is in goal but no waiting location in static? Problematic instance?
                # For heuristic, we can ignore children without a known waiting location.

        num_unserved = len(unserved_children)

        # Base cost: 1 action per unserved child for the final 'serve' action
        h += num_unserved

        if num_unserved == 0:
            return 0 # Goal reached

        # Calculate total needed sandwiches by type
        Needed_gf = sum(loc_needs['gf'] for loc_needs in needed_at_loc.values())
        Needed_reg = sum(loc_needs['reg'] for loc_needs in needed_at_loc.values())

        # 2. Count available sandwiches by state and type
        tray_location = {get_parts(f)[1]: get_parts(f)[2] for f in state if match(f, 'at', '*', '*')}
        is_gf_sandwich = {get_parts(f)[1] for f in state if match(f, 'no_gluten_sandwich', '*')}

        Avail_gf_ontray_at_loc = 0
        Avail_reg_ontray_at_loc = 0
        Avail_gf_ontray_total = 0
        Avail_reg_ontray_total = 0
        Avail_gf_kitchen = 0
        Avail_reg_kitchen = 0

        ontray_sandwiches_list = [] # Store (sandwich, tray, is_gf)

        for f in state:
            if match(f, 'at_kitchen_sandwich', '*'):
                s = get_parts(f)[1]
                if s in is_gf_sandwich:
                    Avail_gf_kitchen += 1
                else:
                    Avail_reg_kitchen += 1
            elif match(f, 'ontray', '*', '*'):
                s, t = get_parts(f)[1:3]
                is_gf = s in is_gf_sandwich
                if is_gf:
                    Avail_gf_ontray_total += 1
                else:
                    Avail_reg_ontray_total += 1
                ontray_sandwiches_list.append((s, t, is_gf))

        # Count Avail_ontray_at_loc by matching against needs at locations
        # Use a copy of needed_at_loc counts to simulate consumption by available sandwiches
        needed_at_loc_copy = {loc: needs.copy() for loc, needs in needed_at_loc.items()}
        for s, t, is_gf in ontray_sandwiches_list:
            p = tray_location.get(t)
            if p and p in needed_at_loc_copy:
                if is_gf and needed_at_loc_copy[p]['gf'] > 0:
                    Avail_gf_ontray_at_loc += 1
                    needed_at_loc_copy[p]['gf'] -= 1 # This sandwich can cover one GF need at p
                elif not is_gf and needed_at_loc_copy[p]['reg'] > 0:
                    Avail_reg_ontray_at_loc += 1
                    needed_at_loc_copy[p]['reg'] -= 1 # This sandwich can cover one Reg need at p


        # 3. Calculate costs based on remaining needs and available sandwiches
        # We prioritize using sandwiches that are closer to the goal state (ontray@loc > ontray@other > kitchen > need_make)
        rem_gf = Needed_gf
        rem_reg = Needed_reg

        # Sandwiches already on tray at correct location (cost 0 for delivery steps)
        # These needs are covered by Avail_ontray_at_loc
        served_by_ontray_at_loc_gf = min(rem_gf, Avail_gf_ontray_at_loc)
        rem_gf -= served_by_ontray_at_loc_gf

        served_by_ontray_at_loc_reg = min(rem_reg, Avail_reg_ontray_at_loc)
        rem_reg -= served_by_ontray_at_loc_reg

        # Sandwiches on tray at other locations (cost 1: move_tray)
        # Total ontray minus those at correct location gives those at other locations
        Avail_gf_ontray_other = Avail_gf_ontray_total - Avail_gf_ontray_at_loc
        Avail_reg_ontray_other = Avail_reg_ontray_total - Avail_reg_ontray_at_loc

        served_by_ontray_other_gf = min(rem_gf, Avail_gf_ontray_other)
        h += served_by_ontray_other_gf * 1 # Cost for move_tray
        rem_gf -= served_by_ontray_other_gf

        served_by_ontray_other_reg = min(rem_reg, Avail_reg_ontray_other)
        h += served_by_ontray_other_reg * 1 # Cost for move_tray
        rem_reg -= served_by_ontray_other_reg

        # Sandwiches in kitchen (cost 2: put_on_tray + move_tray)
        served_by_kitchen_gf = min(rem_gf, Avail_gf_kitchen)
        h += served_by_kitchen_gf * 2 # Cost for put_on_tray + move_tray
        rem_gf -= served_by_kitchen_gf

        served_by_kitchen_reg = min(rem_reg, Avail_reg_kitchen)
        h += served_by_kitchen_reg * 2 # Cost for put_on_tray + move_tray
        rem_reg -= served_by_kitchen_reg

        # Sandwiches needing to be made (cost 3: make + put_on_tray + move_tray)
        # This assumes ingredients and notexist sandwiches are available for making.
        h += rem_gf * 3 # Cost for make + put_on_tray + move_tray
        h += rem_reg * 3 # Cost for make + put_on_tray + move_tray

        return h

