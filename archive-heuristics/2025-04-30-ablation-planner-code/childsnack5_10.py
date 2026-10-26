from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all unserved
    children. It calculates the cost for each unserved child independently based
    on the current state of suitable sandwiches and trays, and sums these costs.
    It ignores resource conflicts (like multiple children needing the same tray
    simultaneously) but captures the sequential steps needed for each child.

    # Assumptions
    - Each unserved child requires one suitable sandwich (gluten-free for allergic,
      any for non-allergic).
    - The steps to serve a child involve getting a suitable sandwich onto a tray,
      moving the tray to the child's location, and performing the serve action.
    - The heuristic assigns a fixed cost based on the 'closest' available suitable
      sandwich for each child:
        - 1 action: Sandwich is already on a tray at the child's location (`serve`).
        - 2 actions: Sandwich is on a tray elsewhere (`move_tray` + `serve`).
        - 3 actions: Sandwich is in the kitchen (`put_on_tray` + `move_tray` + `serve`).
        - 4 actions: Sandwich must be made (`make` + `put_on_tray` + `move_tray` + `serve`).
    - It tracks the available ingredients and `notexist` sandwich objects in the
      current state to determine if a sandwich can be made when needed for the
      heuristic calculation. It assumes ingredients/objects used for making
      sandwiches for one child in the heuristic calculation are unavailable for
      subsequent children in the same calculation pass.
    - It assumes a tray is always available when needed for 'put_on_tray' or 'move_tray'
      actions for the heuristic calculation, ignoring dynamic tray locations and
      availability conflicts.
    - If a sandwich cannot be made when needed (due to lack of ingredients/objects
      in the current state), the heuristic returns infinity for that child, resulting
      in infinity for the state.

    # Heuristic Initialization
    - Extracts static information: which children are allergic/not allergic,
      where each child is waiting, and which bread/content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children and their properties (allergy status, waiting place)
       from static facts.
    2. Identify which children are currently unserved by checking the state facts
       against the set of all children.
    3. Identify available sandwiches in the current state: those `(at_kitchen_sandwich ?s)`
       or `(ontray ?s ?t)`. Determine if each available sandwich is gluten-free.
    4. Identify the location of all trays `(at ?t ?p)`.
    5. Count available ingredients (`(at_kitchen_bread ?b)`, `(at_kitchen_content ?c)`)
       and `(notexist ?s)` sandwich objects in the current state. Determine which
       ingredients are gluten-free based on static facts.
    6. Initialize total heuristic cost `h = 0`. Track how many GF and total makes have been 'assigned'
       to children during the heuristic calculation (`makes_assigned_gf`, `total_makes_assigned`).
    7. For each unserved child (processing allergic children first):
       a. Determine the type of sandwich needed (GF or Any).
       b. Find the minimum cost to get a suitable sandwich to this child, considering
          the following possibilities in order of increasing cost:
          - Cost 1: A suitable sandwich is `(ontray S T)` and `(at T P)` for the child's place `P`.
          - Cost 2: A suitable sandwich is `(ontray S T)` at a place `P'` different from `P`.
          - Cost 3: A suitable sandwich is `(at_kitchen_sandwich S)`.
          - Cost 4: A suitable sandwich can be made *given the remaining ingredients and notexist objects*
            after accounting for makes assigned to previous children in this heuristic calculation.
          - Cost infinity: A suitable sandwich cannot be made.
       c. Add this minimum cost to the total heuristic `h`. If cost is 4, increment the
          appropriate `makes_assigned_gf` counter (if needed GF) and `total_makes_assigned`
          counter for tracking resource consumption within the heuristic calculation.
    8. Return the total heuristic cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        self.goals = task.goals # Goal conditions
        static_facts = task.static # Facts that are not affected by actions

        self.allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")}
        self.child_waiting_place = {get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")}
        self.gf_bread_static = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")}
        self.gf_content_static = {get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state # Current world state facts as a frozenset of strings

        # Identify unserved children
        all_children = self.allergic_children.union(self.not_allergic_children)
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = [(child, self.child_waiting_place.get(child), child in self.allergic_children)
                             for child in all_children if child not in served_children]

        # If all children are served, the heuristic is 0
        if not unserved_children:
            return 0

        # Identify available sandwiches and their locations/types in the current state
        sandwiches_on_tray = {} # {sandwich: (tray, place)}
        sandwiches_kitchen = set() # {sandwich}
        sandwich_is_gf = {} # {sandwich: bool}

        # First pass to get sandwich locations and GF status
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]

            if predicate == "ontray":
                s, t = parts[1], parts[2]
                sandwiches_on_tray[s] = (t, None) # Placeholder for place

            elif predicate == "at_kitchen_sandwich":
                s = parts[1]
                sandwiches_kitchen.add(s)

            elif predicate == "no_gluten_sandwich":
                s = parts[1]
                sandwich_is_gf[s] = True

        # Second pass to get tray locations and link them
        tray_locations = {} # {tray: place}
        for fact in state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 if len(parts) == 3: # (at tray place)
                     tray, place = parts[1], parts[2]
                     tray_locations[tray] = place

        # Link tray locations to sandwiches on trays
        for s, (t, _) in list(sandwiches_on_tray.items()): # Iterate over copy
             if t in tray_locations:
                 sandwiches_on_tray[s] = (t, tray_locations[t])
             else:
                 # Tray location unknown, assume it's not at a child's place for heuristic
                 sandwiches_on_tray[s] = (t, 'unknown_place')

        # Count available ingredients and notexist objects in the current state
        current_notexist_count = len({get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")})
        current_bread_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        current_content_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        current_gf_bread_kitchen_count = len([b for b in current_bread_kitchen if b in self.gf_bread_static])
        current_gf_content_kitchen_count = len([c for c in current_content_kitchen if c in self.gf_content_static])
        current_any_bread_kitchen_count = len(current_bread_kitchen)
        current_any_content_kitchen_count = len(current_content_kitchen)

        # Calculate current makeable capacities
        current_makeable_total_slots = min(current_any_bread_kitchen_count + current_gf_bread_kitchen_count,
                                           current_any_content_kitchen_count + current_gf_content_kitchen_count,
                                           current_notexist_count)
        current_makeable_gf_only_slots = min(current_gf_bread_kitchen_count,
                                             current_gf_content_kitchen_count,
                                             current_notexist_count)


        # Calculate heuristic as sum of minimum costs per unserved child
        h = 0
        makes_assigned_gf = 0 # Number of GF makes assigned to children in this heuristic pass
        total_makes_assigned = 0 # Total number of makes assigned in this heuristic pass

        # Process allergic children first to prioritize GF sandwiches
        unserved_children_sorted = sorted(unserved_children, key=lambda x: x[2], reverse=True) # Allergic first

        for child, place, needs_gf in unserved_children_sorted:
            child_cost = float('inf')

            # Check Case 1: Suitable sandwich on tray at child's location (Cost 1)
            found_case1 = False
            for s, (t, p) in sandwiches_on_tray.items():
                if p == place:
                    is_gf = sandwich_is_gf.get(s, False)
                    if (needs_gf and is_gf) or (not needs_gf): # Suitable sandwich
                        child_cost = 1
                        found_case1 = True
                        break # Found one option for this child

            if found_case1:
                h += child_cost
                continue # Move to next child

            # Check Case 2: Suitable sandwich on tray elsewhere (Cost 2)
            found_case2 = False
            for s, (t, p) in sandwiches_on_tray.items():
                 if p != place:
                    is_gf = sandwich_is_gf.get(s, False)
                    if (needs_gf and is_gf) or (not needs_gf): # Suitable sandwich
                        child_cost = 2
                        found_case2 = True
                        break # Found one option for this child

            if found_case2:
                h += child_cost
                continue # Move to next child

            # Check Case 3: Suitable sandwich in kitchen (Cost 3)
            found_case3 = False
            for s in sandwiches_kitchen:
                is_gf = sandwich_is_gf.get(s, False)
                if (needs_gf and is_gf) or (not needs_gf): # Suitable sandwich
                    child_cost = 3
                    found_case3 = True
                    break # Found one option for this child

            if found_case3:
                h += child_cost
                continue # Move to next child

            # Check Case 4: Must make a suitable sandwich (Cost 4 or inf)
            can_make_needed = False
            if needs_gf:
                # Can make GF if enough GF ingredients and notexist objects remain
                # and total makes capacity is not exceeded
                if makes_assigned_gf < current_makeable_gf_only_slots and total_makes_assigned < current_makeable_total_slots:
                    can_make_needed = True
                    makes_assigned_gf += 1 # Assign this GF make
                    total_makes_assigned += 1 # Assign this total make
            else: # Needs Any sandwich
                # Can make Any if enough total ingredients and notexist objects remain
                if total_makes_assigned < current_makeable_total_slots:
                     can_make_needed = True
                     total_makes_assigned += 1 # Assign this total make

            # Add the determined cost for this child
            if can_make_needed:
                 child_cost = 4
            else:
                 child_cost = float('inf') # Cannot make needed sandwich


            h += child_cost

        # If any child cost was infinity, the total heuristic is infinity
        if h == float('inf'):
             return float('inf')

        return h
