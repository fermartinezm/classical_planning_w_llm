from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    The heuristic is non-admissible and designed for greedy best-first search.

    Components of the heuristic:
    1.  Number of unserved children (each needs a 'serve' action).
    2.  Number of locations with unserved children that currently lack a tray
        (each needs a 'move_tray' action to bring a tray).
    3.  Number of suitable sandwiches currently in the kitchen that need to be
        put on a tray (each needs a 'put_on_tray' action). Includes a cost
        to move a tray to the kitchen if needed.
    4.  Number of suitable sandwiches that need to be made because there aren't
        enough available anywhere (kitchen or tray), respecting gluten constraints
        and ingredient availability.

    Assumptions:
    - Ingredients are only ever in the kitchen.
    - Trays can be moved between any places.
    - Sandwiches are made in the kitchen.
    - Gluten-free sandwiches are required for allergic children.
    - Any sandwich can be served to non-allergic children.
    - The heuristic prioritizes using available GF sandwiches for allergic children first.
    - Assumes any object appearing as the first argument of an `(at ?obj ?place)` fact is a tray if ?place is not kitchen, or if ?place is kitchen. This is a domain-specific shortcut.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by pre-processing static facts to get child info.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Pre-process static facts to get child info (waiting location, allergy status)
        self.child_info = {} # child -> {'place': p, 'allergic': bool}
        self.all_children = set() # Set of all child objects mentioned in static facts or goals

        # Get child info from static facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue
            pred = parts[0]
            if pred == "waiting" and len(parts) == 3:
                child, place = parts[1:]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['place'] = place
                self.all_children.add(child)
            elif pred == "allergic_gluten" and len(parts) == 2:
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = True
                self.all_children.add(child)
            elif pred == "not_allergic_gluten" and len(parts) == 2:
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = False
                self.all_children.add(child)

        # Ensure all children from goals are included (they should be covered by waiting facts)
        for goal in self.goals:
             if match(goal, "served", "*") and len(get_parts(goal)) == 2:
                 child = get_parts(goal)[1]
                 if child not in self.child_info:
                      # This case indicates an unusual problem definition
                      print(f"Warning: Child {child} in goal but not in static waiting/allergy facts.")
                      # Add default info, though problem might be ill-defined
                      self.child_info[child] = {'place': 'unknown', 'allergic': False}
                 self.all_children.add(child)


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state

        # 1. Identify Unserved Children and their Locations/Allergies
        unserved_children = set()
        unserved_locations = set() # Places where at least one unserved child is waiting

        for child in self.all_children:
            if f'(served {child})' not in state:
                unserved_children.add(child)
                # Get place from pre-processed info
                place = self.child_info.get(child, {}).get('place')
                if place:
                    unserved_locations.add(place)
                # Note: If place is None, the child was in goals/allergy but not waiting.
                # This shouldn't happen in valid problems, but we proceed assuming it's okay.

        Num_Unserved = len(unserved_children)
        if Num_Unserved == 0:
            return 0 # Goal reached

        Num_Unserved_Allergic = sum(1 for c in unserved_children if self.child_info.get(c, {}).get('allergic', False))
        Num_Unserved_NonAllergic = Num_Unserved - Num_Unserved_Allergic

        # 2. Count Available Sandwiches (Kitchen or Tray) and Ingredients
        S_gf_avail = 0 # Sandwiches marked no_gluten_sandwich, anywhere
        S_nongf_avail = 0 # Sandwiches NOT marked no_gluten_sandwich, anywhere
        S_suitable_kitchen = 0 # Sandwiches in kitchen, suitable for *any* unserved child

        trays_at_location = {} # place -> count (excluding kitchen)
        trays_at_kitchen = 0

        B_gf_kitchen_count = 0
        C_gf_kitchen_count = 0
        B_nongf_kitchen_count = 0
        C_nongf_kitchen_count = 0

        # Parse state facts
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts if any

            pred = parts[0]

            if pred == "at_kitchen_sandwich" and len(parts) == 2:
                s = parts[1]
                is_gf = f'(no_gluten_sandwich {s})' in state
                if is_gf:
                    S_gf_avail += 1
                    if Num_Unserved > 0: # GF is suitable for any unserved child
                        S_suitable_kitchen += 1
                else:
                    S_nongf_avail += 1
                    if Num_Unserved_NonAllergic > 0: # Non-GF only suitable for non-allergic
                        S_suitable_kitchen += 1

            elif pred == "ontray" and len(parts) == 3:
                s, t = parts[1:]
                is_gf = f'(no_gluten_sandwich {s})' in state
                if is_gf:
                    S_gf_avail += 1
                else:
                    S_nongf_avail += 1

            elif pred == "at" and len(parts) == 3: # Tray location - Assuming any object in (at ?t ?p) is a tray
                t, p = parts[1:]
                if p == "kitchen":
                    trays_at_kitchen += 1
                else:
                    trays_at_location[p] = trays_at_location.get(p, 0) + 1

            elif pred == "at_kitchen_bread" and len(parts) == 2:
                b = parts[1]
                if f'(no_gluten_bread {b})' in state:
                    B_gf_kitchen_count += 1
                else:
                    B_nongf_kitchen_count += 1

            elif pred == "at_kitchen_content" and len(parts) == 2:
                c = parts[1]
                if f'(no_gluten_content {c})' in state:
                    C_gf_kitchen_count += 1
                else:
                    C_nongf_kitchen_count += 1

        # 3. Calculate Heuristic Components
        Heuristic = 0

        # Component 1: Serve actions
        # Each unserved child needs one final serve action.
        Heuristic += Num_Unserved

        # Component 2: Tray moves to locations with unserved children
        # Count unique locations with unserved children that have no tray.
        locations_needing_tray = sum(1 for p in unserved_locations if p != "kitchen" and trays_at_location.get(p, 0) == 0)
        Heuristic += locations_needing_tray

        # Component 3: Put on tray actions
        # Number of suitable sandwiches currently in the kitchen that need to be put on a tray.
        # Each such sandwich needs a 'put_on_tray' action.
        Heuristic += S_suitable_kitchen
        # If there are suitable sandwiches to put on a tray but no tray in the kitchen,
        # we need one 'move_tray' action to bring a tray to the kitchen first.
        if S_suitable_kitchen > 0 and trays_at_kitchen == 0:
             Heuristic += 1 # Cost to move a tray to the kitchen

        # Component 4: Make sandwich actions
        # Calculate the minimum number of suitable sandwiches that must be made.
        # This is based on the number of unserved children and available sandwiches,
        # respecting the gluten constraint.

        gf_needed = Num_Unserved_Allergic
        nongf_needed = Num_Unserved_NonAllergic

        # Available sandwiches (kitchen or tray)
        gf_avail = S_gf_avail
        nongf_avail = S_nongf_avail

        # How many GF sandwiches must be made?
        # This is the number of allergic children not covered by available GF sandwiches.
        gf_to_make = max(0, gf_needed - gf_avail)

        # How many non-GF sandwiches must be made?
        # Non-allergic children can use remaining available GF sandwiches first.
        remaining_gf_avail_for_nongf = max(0, gf_avail - gf_needed) # GF sandwiches left after potentially serving allergic
        any_avail_for_nongf = nongf_avail + remaining_gf_avail_for_nongf
        nongf_to_make = max(0, nongf_needed - any_avail_for_for_nongf)

        # Check ingredient limits for making the required sandwiches.
        max_can_make_gf = min(B_gf_kitchen_count, C_gf_kitchen_count)
        max_can_make_nongf = min(B_nongf_kitchen_count, C_nongf_kitchen_count)

        # If we need more GF sandwiches than we can make with current GF ingredients,
        # the state is likely unsolvable from here.
        if gf_to_make > max_can_make_gf:
            return float('inf')

        # If we need more non-GF sandwiches (for non-allergic children) than we can make
        # with remaining GF ingredients capacity + non-GF ingredients, it's unsolvable.
        remaining_gf_can_make = max(0, max_can_make_gf - gf_to_make) # GF capacity left after making needed GF
        total_can_make_for_nongf = max_can_make_nongf + remaining_gf_can_make
        if nongf_to_make > total_can_make_for_nongf:
            return float('inf')

        # If solvable based on ingredients, add the cost of making the necessary sandwiches.
        # Each sandwich making action costs 1.
        Heuristic += gf_to_make + nongf_to_make

        return Heuristic
