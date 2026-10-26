from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    This heuristic estimates the number of actions required to serve all children.
    It models the process as a pipeline where sandwiches are made, put on trays,
    moved to the children's locations, and finally served. The heuristic counts
    the number of unserved children and the number of sandwiches in different
    stages of this pipeline, assigning a cost based on how many steps are
    remaining from that stage to being served. It prioritizes using sandwiches
    that are closer to the final serving stage.

    # Assumptions
    - The problem is solvable (enough ingredients and sandwich objects exist in total).
    - Enough trays are available to move sandwiches.
    - Children remain waiting at their initial locations until served.
    - The cost of each action (make, put, move, serve) is 1.
    - Resource contention (e.g., multiple children needing the same tray) is simplified
      by aggregating costs based on the number of items/needs in each stage.

    # Heuristic Initialization
    - Extracts all objects by type (children, sandwiches, trays, bread, content, places)
      from the task's fact list.
    - Identifies static properties like which children are allergic and which
      ingredients are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by considering the "state" of the sandwiches
    needed for the unserved children. For each unserved child, a suitable sandwich
    must eventually be made (if necessary), put on a tray, moved to the child's
    location, and served. This forms a pipeline with distinct stages:

    Stage 4: Needs Making (Cost 4: make + put + move + serve)
    Stage 3: At Kitchen (Cost 3: put + move + serve)
    Stage 2: On Tray in Kitchen (Cost 2: move + serve)
    Stage 1: On Tray at Location (Cost 1: serve)

    The heuristic calculates the number of unserved children who need a gluten-free
    sandwich and those who need any sandwich. It then counts the number of
    available sandwiches in each stage, categorized by whether they are gluten-free
    or not, and also counts available ingredients and sandwich objects for making.

    It then allocates the "needs" (unserved children) to the available sandwiches
    and makable sandwiches, prioritizing those in stages closer to the goal (Stage 1
    first, then Stage 2, etc.). Gluten-free sandwiches are prioritized for allergic
    children, but can be used for non-allergic children if there's a surplus.

    The total heuristic value is the sum of (number of needs satisfied by this stage) * (cost of this stage)
    for each stage, considering both gluten-free and non-gluten-free requirements and availability.

    Specifically:
    1. Count unserved allergic and non-allergic children. These represent the needs for GF and Any sandwiches.
    2. Count available sandwiches in each stage (Ontray-Location, OnTray-Kitchen, Kitchen), separated by GF/Non-GF.
    3. Count available ingredients (GF bread, Non-GF bread, GF content, Non-GF content) and sandwich objects (`notexist`).
    4. Calculate how many GF and Regular sandwiches *can* be made based on ingredient/object availability.
    5. Allocate the `needed_gf` and `needed_nongf` counts to the available sandwiches in stages 1 through 3, prioritizing GF for `needed_gf` and using surplus GF for `needed_nongf`.
    6. Allocate the remaining `needed_gf` and `needed_nongf` to the makable sandwiches (Stage 4).
    7. Sum the costs: (allocated_stage1_gf + allocated_stage1_nongf) * 1 + (allocated_stage2_gf + allocated_stage2_nongf) * 2 + (allocated_stage3_gf + allocated_stage3_nongf) * 3 + (allocated_stage4_gf + allocated_stage4_nongf) * 4.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and object lists."""
        self.goals = task.goals
        static_facts = task.static

        # Extract objects by type
        self.children = set()
        self.sandwiches = set()
        self.trays = set()
        self.bread_portions = set()
        self.content_portions = set()
        self.places = {'kitchen'} # kitchen is a constant

        # Extract objects and static properties from all possible facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()

        for fact_str in task.facts:
            parts = get_parts(fact_str)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == 'allergic_gluten' and len(args) == 1:
                self.children.add(args[0])
                self.allergic_children.add(args[0])
            elif predicate == 'not_allergic_gluten' and len(args) == 1:
                self.children.add(args[0])
                self.not_allergic_children.add(args[0])
            elif predicate == 'waiting' and len(args) == 2:
                 # Child type inferred from predicate
                self.children.add(args[0])
                # Place type inferred from predicate
                self.places.add(args[1])
            elif predicate == 'served' and len(args) == 1:
                 # Child type inferred from predicate
                self.children.add(args[0])
            elif predicate == 'at_kitchen_bread' and len(args) == 1:
                self.bread_portions.add(args[0])
            elif predicate == 'at_kitchen_content' and len(args) == 1:
                self.content_portions.add(args[0])
            elif predicate == 'at_kitchen_sandwich' and len(args) == 1:
                self.sandwiches.add(args[0])
            elif predicate == 'no_gluten_bread' and len(args) == 1:
                self.bread_portions.add(args[0])
                self.no_gluten_breads.add(args[0])
            elif predicate == 'no_gluten_content' and len(args) == 1:
                self.content_portions.add(args[0])
                self.no_gluten_contents.add(args[0])
            elif predicate == 'ontray' and len(args) == 2:
                self.sandwiches.add(args[0])
                self.trays.add(args[1])
            elif predicate == 'no_gluten_sandwich' and len(args) == 1:
                self.sandwiches.add(args[0])
            elif predicate == 'at' and len(args) == 2:
                # Tray type inferred from predicate
                self.trays.add(args[0])
                # Place type inferred from predicate
                self.places.add(args[1])
            elif predicate == 'notexist' and len(args) == 1:
                self.sandwiches.add(args[0])
            # Add other predicates if they introduce new object types

        # Ensure all children found are categorized
        for child in self.children:
             if child not in self.allergic_children and child not in self.not_allergic_children:
                 # This case shouldn't happen based on domain, but good practice
                 pass # Child type unknown, heuristic might be less accurate

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children and their needs
        unserved_allergic = {c for c in self.allergic_children if f'(served {c})' not in state}
        unserved_non_allergic = {c for c in self.not_allergic_children if f'(served {c})' not in state}

        needed_gf = len(unserved_allergic)
        needed_nongf = len(unserved_non_allergic)

        # If no children need serving, goal is reached
        if needed_gf == 0 and needed_nongf == 0:
            return 0

        # 2. Count available sandwiches by type and stage
        N_gf_otl = 0 # On Tray at Location (not kitchen)
        N_nongf_otl = 0
        N_gf_otk = 0 # On Tray at Kitchen
        N_nongf_otk = 0
        N_gf_k = 0   # At Kitchen
        N_nongf_k = 0

        # Map tray locations
        tray_locs = {}
        for tray in self.trays:
            for place in self.places:
                if f'(at {tray} {place})' in state:
                    tray_locs[tray] = place
                    break # Assume tray is only at one place

        # Map sandwich to tray and check if GF
        sandwich_ontray = {}
        sandwich_is_gf = {s: True for s in self.sandwiches if f'(no_gluten_sandwich {s})' in state}

        for sandwich in self.sandwiches:
            if f'(at_kitchen_sandwich {sandwich})' in state:
                if sandwich_is_gf.get(sandwich, False):
                    N_gf_k += 1
                else:
                    N_nongf_k += 1
            elif any(match(fact, "ontray", sandwich, "*") for fact in state):
                 # Find the tray and its location
                 tray = next((t for t in self.trays if f'(ontray {sandwich} {t})' in state), None)
                 if tray and tray in tray_locs:
                     location = tray_locs[tray]
                     if location == 'kitchen':
                         if sandwich_is_gf.get(sandwich, False):
                             N_gf_otk += 1
                         else:
                             N_nongf_otk += 1
                     else: # On tray at a non-kitchen location
                         if sandwich_is_gf.get(sandwich, False):
                             N_gf_otl += 1
                         else:
                             N_nongf_otl += 1

        # 3. Count available ingredients and sandwich objects
        available_gf_bread = len({b for b in self.bread_portions if f'(at_kitchen_bread {b})' in state and f'(no_gluten_bread {b})' in state})
        available_nongf_bread = len({b for b in self.bread_portions if f'(at_kitchen_bread {b})' in state and f'(no_gluten_bread {b})' not in state})
        available_gf_content = len({c for c in self.content_portions if f'(at_kitchen_content {c})' in state and f'(no_gluten_content {c})' in state})
        available_nongf_content = len({c for c in self.content_portions if f'(at_kitchen_content {c})' in state and f'(no_gluten_content {c})' not in state})
        available_sandwich_objects = len({s for s in self.sandwiches if f'(notexist {s})' in state})

        # 4. Calculate how many sandwiches can be made
        can_make_gf = min(available_gf_bread, available_gf_content, available_sandwich_objects)
        
        # Ingredients/objects remaining after potential GF makes
        rem_gf_bread = available_gf_bread - can_make_gf
        rem_gf_content = available_gf_content - can_make_gf
        rem_objects = available_sandwich_objects - can_make_gf

        available_bread_for_regular = available_nongf_bread + rem_gf_bread
        available_content_for_regular = available_nongf_content + rem_gf_content

        can_make_regular = min(available_bread_for_regular, available_content_for_regular, rem_objects)


        # 5. Allocate needs to stages, prioritizing closer stages and GF for allergic
        h = 0
        current_needed_gf = needed_gf
        current_needed_nongf = needed_nongf

        # Stage 1: On Tray at Location (Cost 1: serve)
        from_gf_otl = min(current_needed_gf, N_gf_otl)
        h += from_gf_otl * 1
        current_needed_gf -= from_gf_otl

        from_nongf_otl = min(current_needed_nongf, N_nongf_otl)
        h += from_nongf_otl * 1
        current_needed_nongf -= from_nongf_otl

        from_gf_otl_for_nongf = min(current_needed_nongf, N_gf_otl - from_gf_otl)
        h += from_gf_otl_for_nongf * 1
        current_needed_nongf -= from_gf_otl_for_nongf

        # Stage 2: On Tray in Kitchen (Cost 2: move + serve)
        from_gf_otk = min(current_needed_gf, N_gf_otk)
        h += from_gf_otk * 2
        current_needed_gf -= from_gf_otk

        from_nongf_otk = min(current_needed_nongf, N_nongf_otk)
        h += from_nongf_otk * 2
        current_needed_nongf -= from_nongf_otk

        from_gf_otk_for_nongf = min(current_needed_nongf, N_gf_otk - from_gf_otk)
        h += from_gf_otk_for_nongf * 2
        current_needed_nongf -= from_gf_otk_for_nongf

        # Stage 3: At Kitchen (Cost 3: put + move + serve)
        from_gf_k = min(current_needed_gf, N_gf_k)
        h += from_gf_k * 3
        current_needed_gf -= from_gf_k

        from_nongf_k = min(current_needed_nongf, N_nongf_k)
        h += from_nongf_k * 3
        current_needed_nongf -= from_nongf_k

        from_gf_k_for_nongf = min(current_needed_nongf, N_gf_k - from_gf_k)
        h += from_gf_k_for_nongf * 3
        current_needed_nongf -= from_gf_k_for_nongf

        # Stage 4: Needs Make (Cost 4: make + put + move + serve)
        # We can only make up to the calculated 'can_make_gf' and 'can_make_regular' limits
        makes_gf_allocated = min(current_needed_gf, can_make_gf)
        h += makes_gf_allocated * 4
        current_needed_gf -= makes_gf_allocated

        # Need to recalculate available_for_regular based on *allocated* GF makes
        rem_gf_bread_after_alloc = available_gf_bread - makes_gf_allocated
        rem_gf_content_after_alloc = available_gf_content - makes_gf_allocated
        rem_objects_after_alloc = available_sandwich_objects - makes_gf_allocated

        available_bread_for_regular_after_alloc = available_nongf_bread + rem_gf_bread_after_alloc
        available_content_for_regular_after_alloc = available_nongf_content + rem_gf_content_after_alloc

        can_make_regular_after_alloc = min(available_bread_for_regular_after_alloc, available_content_for_regular_after_alloc, rem_objects_after_alloc)

        makes_regular_allocated = min(current_needed_nongf, can_make_regular_after_alloc)
        h += makes_regular_allocated * 4
        current_needed_nongf -= makes_regular_allocated

        # If current_needed_gf or current_needed_nongf are still > 0, it implies unsolvability
        # or a state unreachable in a solvable problem. For a non-admissible heuristic,
        # we can return a large value or let the remaining cost contribute.
        # Given the problem description implies solvable instances, this shouldn't happen
        # in states reachable from the initial state of a solvable problem.
        # We can add the remaining needs with a high cost or just assume they are covered
        # by the counts above if the problem is well-formed and solvable.
        # Let's add them with the highest cost (4) as a fallback, though ideally they are 0.
        h += current_needed_gf * 4
        h += current_needed_nongf * 4


        return h

