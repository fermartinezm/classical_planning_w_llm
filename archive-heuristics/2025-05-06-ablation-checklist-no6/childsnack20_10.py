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
    # Handle trailing wildcard
    if args and args[-1] == '*':
         if len(parts) < len(args) - 1: return False
         args = args[:-1] # Match up to the wildcard
         parts = parts[:len(args)]

    if len(parts) != len(args):
         return False

    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic counts the number of unserved children (cost 1 each for serving)
    and adds the estimated cost to get a suitable sandwich to each child's location
    on a tray, based on the sandwich's current state (on tray elsewhere, in kitchen,
    or needs making). It prioritizes using sandwiches that are closer to being ready
    and prioritizes gluten-free sandwiches for allergic children.

    Stages for a sandwich to reach "on tray at child's location" (cost to reach this stage):
    - Stage 1: On tray elsewhere (cost 1: move_tray)
    - Stage 2: In kitchen (cost 2: put_on_tray + move_tray)
    - Stage 3: Needs making (cost 3: make + put_on_tray + move_tray)

    Heuristic = (Number of unserved children * 1 [serve])
              + (Number of sandwiches needed from Stage 1 * 1)
              + (Number of sandwiches needed from Stage 2 * 2)
              + (Number of sandwiches needed from Stage 3 * 3)

    Assumes:
    - A tray is available in the kitchen when needed for put_on_tray.
    - Ingredients are sufficient if (at_kitchen_bread) and (at_kitchen_content) exist (checked for GF/Any).
    - notexist sandwich objects are consumed when making sandwiches.
    - Resource contention for trays and ingredients is simplified.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child allergy information and GF ingredient info.
        """
        self.goals = task.goals
        self.static = task.static

        self.child_allergy = {}
        self.all_children = set()

        # Extract child allergy information from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = True
                self.all_children.add(child)
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = False
                self.all_children.add(child)

        # Extract static GF ingredient information
        self.static_gf_breads = {p[1] for p in (get_parts(f) for f in self.static) if p[0] == 'no_gluten_bread'}
        self.static_gf_contents = {p[1] for p in (get_parts(f) for f in self.static) if p[0] == 'no_gluten_content'}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify unserved children and their needs
        unserved_children = {c for c in self.all_children if f'(served {c})' not in state}
        N_unserved = len(unserved_children)

        # If all children are served, the goal is reached, heuristic is 0.
        if N_unserved == 0:
            return 0

        N_ua = sum(1 for c in unserved_children if self.child_allergy.get(c, False)) # Default to False if allergy status unknown
        N_un = N_unserved - N_ua

        # 2. Count available sandwiches by state and type
        # Need initial counts for calculation
        N_gf_ontray_init = 0
        N_any_ontray_init = 0 # Total ontray
        N_gf_kitchen_init = 0
        N_any_kitchen_init = 0 # Total kitchen
        N_notexist_init = 0

        # Pre-calculate GF status for sandwiches mentioned in the state for efficiency
        sandwich_is_gf = {}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'no_gluten_sandwich':
                 sandwich_is_gf[parts[1]] = True

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'ontray':
                s = parts[1]
                if sandwich_is_gf.get(s, False):
                    N_gf_ontray_init += 1
                N_any_ontray_init += 1
            elif parts[0] == 'at_kitchen_sandwich':
                s = parts[1]
                if sandwich_is_gf.get(s, False):
                    N_gf_kitchen_init += 1
                N_any_kitchen_init += 1
            elif parts[0] == 'notexist':
                N_notexist_init += 1

        # Count ingredient availability in the kitchen
        breads_in_kitchen = set()
        contents_in_kitchen = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at_kitchen_bread':
                breads_in_kitchen.add(parts[1])
            elif parts[0] == 'at_kitchen_content':
                contents_in_kitchen.add(parts[1])

        num_any_bread_kitchen = len(breads_in_kitchen)
        num_gf_bread_kitchen = len(breads_in_kitchen.intersection(self.static_gf_breads))

        num_any_content_kitchen = len(contents_in_kitchen)
        num_gf_content_kitchen = len(contents_in_kitchen.intersection(self.static_gf_contents))

        can_make_gf = num_gf_bread_kitchen > 0 and num_gf_content_kitchen > 0
        can_make_any = num_any_bread_kitchen > 0 and num_any_content_kitchen > 0


        # 3. Calculate heuristic cost based on needed sandwiches and available stages
        h = N_unserved # Cost for the final 'serve' action for each unserved child

        needed_gf = N_ua
        needed_any = N_un

        # Use available sandwiches from stages, prioritizing GF for GF needs, then Any for Any needs.
        # Stage 1: On tray elsewhere (cost 1: move_tray)
        use_s1_gf = min(needed_gf, N_gf_ontray_init)
        needed_gf -= use_s1_gf
        N_gf_ontray_rem = N_gf_ontray_init - use_s1_gf
        h += use_s1_gf * 1

        # Available Any on tray = (Initial Any on tray - Initial GF on tray) + Remaining GF on tray
        avail_any_s1 = (N_any_ontray_init - N_gf_ontray_init) + N_gf_ontray_rem
        use_s1_any = min(needed_any, avail_any_s1)
        needed_any -= use_s1_any
        h += use_s1_any * 1

        # Stage 2: In kitchen (cost 2: put_on_tray + move_tray)
        use_s2_gf = min(needed_gf, N_gf_kitchen_init)
        needed_gf -= use_s2_gf
        N_gf_kitchen_rem = N_gf_kitchen_init - use_s2_gf
        h += use_s2_gf * 2

        # Available Any in kitchen = (Initial Any in kitchen - Initial GF in kitchen) + Remaining GF in kitchen
        avail_any_s2 = (N_any_kitchen_init - N_gf_kitchen_init) + N_gf_kitchen_rem
        use_s2_any = min(needed_any, avail_any_s2)
        needed_any -= use_s2_any
        h += use_s2_any * 2

        # Stage 3: Needs making (cost 3: make + put_on_tray + move_tray)
        use_s3_gf = 0
        notexist_rem = N_notexist_init
        if can_make_gf:
            use_s3_gf = min(needed_gf, N_notexist_init)
            needed_gf -= use_s3_gf
            notexist_rem -= use_s3_gf
            h += use_s3_gf * 3

        use_s3_any = 0
        if can_make_any:
            use_s3_any = min(needed_any, notexist_rem)
            needed_any -= use_s3_any
            h += use_s3_any * 3

        # If after exhausting all available sandwiches (in all stages), there are still needed sandwiches,
        # it implies the problem is unsolvable from this state with current resources.
        if needed_gf > 0 or needed_any > 0:
            return float('inf') # Indicate unsolvable or very high cost

        return h
