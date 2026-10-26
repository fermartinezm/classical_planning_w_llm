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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and estimates the minimum steps
    needed to get a suitable sandwich to each, based on the sandwich's current
    location and readiness (on tray at location, on tray elsewhere, in kitchen,
    or needs making). It prioritizes serving allergic children with gluten-free
    sandwiches and accounts for the need to move a tray to the kitchen if
    sandwiches need to be put on trays there.

    # Assumptions
    - Each unserved child needs exactly one suitable sandwich.
    - Sufficient trays exist globally to serve all children eventually.
    - Sufficient ingredients and 'notexist' sandwich objects exist globally
      to make all needed sandwiches eventually (if the instance is solvable).
    - The cost of moving a tray between any two places is 1.
    - The cost of making a sandwich, putting it on a tray, and serving is 1 each.
    - The heuristic assumes a greedy assignment of the "most ready" suitable
      sandwiches to unserved children, prioritizing allergic children.

    # Heuristic Initialization
    - Extracts static information about which children are allergic/not allergic
      and where children are waiting.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are not yet served based on the goal state and current state.
    2. Categorize unserved children into allergic and non-allergic groups and record their waiting locations.
    3. Count available ingredients (gluten-free and regular bread/content) and 'notexist' sandwich objects in the kitchen from the current state.
    4. Count available sandwiches based on their current state and type (gluten-free or regular):
       - On a tray at the location of *any* unserved child.
       - On a tray elsewhere (not at an unserved child's location, including on tray at kitchen).
       - In the kitchen (not on a tray).
    5. Count the number of trays currently at the kitchen and elsewhere.
    6. Initialize heuristic cost `h = 0`.
    7. Greedily assign available sandwiches to unserved children, adding the
       estimated cost for each child based on the sandwich's readiness stage:
       - Prioritize allergic children (who require GF sandwiches).
       - For each child, check for the most ready suitable sandwich first:
         - Cost 1: Suitable sandwich found on a tray already at the child's location.
         - Cost 2: Suitable sandwich found on a tray elsewhere.
         - Cost 3: Suitable sandwich found in the kitchen (needs put on tray + move tray + serve).
         - Cost 4: Suitable sandwich needs to be made (needs make + put on tray + move tray + serve).
       - Consume available sandwiches/makeable counts as they are assigned. Prioritize using GF sandwiches for allergic children first, then for non-allergic children, before using regular sandwiches for non-allergic children.
    8. After assigning sandwiches from kitchen stock or makeable ones, count how many
       assignments required the 'put_on_tray' step at the kitchen. If this count is
       greater than zero and no tray was initially at the kitchen but trays exist
       elsewhere, add 1 to the heuristic for moving a tray to the kitchen.
    9. The total accumulated cost `h` is the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals # Goal conditions (served children)
        static_facts = task.static # Facts that are not affected by actions.

        self.waiting_children = {} # child -> place
        self.allergic_children = set()
        self.not_allergic_children = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                self.waiting_children[parts[1]] = parts[2]
            elif parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])

        # Get all children from the waiting list (static)
        self.all_children = set(self.waiting_children.keys())


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse State ---
        served_children = set()
        notexist_sandwiches = set()
        gf_bread_kitchen = 0
        reg_bread_kitchen = 0
        gf_content_kitchen = 0
        reg_content_kitchen = 0
        kitchen_sandwiches = {} # sandwich -> is_gf
        ontray_sandwiches = {} # sandwich -> tray
        tray_locations = {} # tray -> place
        is_gf_sandwich = {} # sandwich -> bool

        for fact in state:
            parts = get_parts(fact)

            if parts[0] == 'served':
                served_children.add(parts[1])
            elif parts[0] == 'notexist':
                notexist_sandwiches.add(parts[1])
            elif parts[0] == 'at_kitchen_bread':
                bread = parts[1]
                is_gf = f'(no_gluten_bread {bread})' in state
                if is_gf:
                    gf_bread_kitchen += 1
                else:
                    reg_bread_kitchen += 1
            elif parts[0] == 'at_kitchen_content':
                content = parts[1]
                is_gf = f'(no_gluten_content {content})' in state
                if is_gf:
                    gf_content_kitchen += 1
                else:
                    reg_content_kitchen += 1
            elif parts[0] == 'at_kitchen_sandwich':
                sandwich = parts[1]
                is_gf = f'(no_gluten_sandwich {sandwich})' in state
                kitchen_sandwiches[sandwich] = is_gf
                is_gf_sandwich[sandwich] = is_gf # Store GF status for later lookup
            elif parts[0] == 'ontray':
                sandwich, tray = parts[1], parts[2]
                ontray_sandwiches[sandwich] = tray
                # Ensure GF status is recorded if sandwich is on tray
                if f'(no_gluten_sandwich {sandwich})' in state:
                     is_gf_sandwich[sandwich] = True
                elif sandwich not in is_gf_sandwich: # Assume regular if not marked GF
                     is_gf_sandwich[sandwich] = False

            elif parts[0] == 'at':
                tray, place = parts[1], parts[2]
                tray_locations[tray] = place

        trays_at_kitchen_count = sum(1 for t, p in tray_locations.items() if p == 'kitchen')
        trays_elsewhere_count = sum(1 for t, p in tray_locations.items() if p != 'kitchen')


        # --- Identify Unserved Children ---
        unserved_allergic_needs = [] # list of (child, place)
        unserved_not_allergic_needs = [] # list of (child, place)

        for child in self.all_children:
            if child not in served_children:
                place = self.waiting_children[child]
                if child in self.allergic_children:
                    unserved_allergic_needs.append((child, place))
                else: # not_allergic_gluten
                    unserved_not_allergic_needs.append((child, place))

        # If all children are served, heuristic is 0
        if not unserved_allergic_needs and not unserved_not_allergic_needs:
            return 0

        # --- Categorize Available Sandwiches ---
        available_gf_ontray_at_location = [] # list of sandwich
        available_reg_ontray_at_location = [] # list of sandwich
        available_gf_ontray_elsewhere = [] # list of sandwich
        available_reg_ontray_elsewhere = [] # list of sandwich
        available_gf_kitchen = [] # list of sandwich (not ontray)
        available_reg_kitchen = [] # list of sandwich (not ontray)

        # Sandwiches on trays
        for s, t in ontray_sandwiches.items():
            p = tray_locations.get(t)
            is_gf = is_gf_sandwich.get(s, False)

            if p is not None: # Tray has a location
                # Check if this tray is at the location of *any* unserved child
                is_at_any_unserved_location = False
                for (uc, up) in unserved_allergic_needs + unserved_not_allergic_needs:
                    if p == up:
                        is_at_any_unserved_location = True
                        break

                if is_at_any_unserved_location:
                    if is_gf:
                        available_gf_ontray_at_location.append(s)
                    else:
                        available_reg_ontray_at_location.append(s)
                else: # On tray, but not at any unserved child's location
                    if is_gf:
                        available_gf_ontray_elsewhere.append(s)
                    else:
                        available_reg_ontray_elsewhere.append(s)

        # Sandwiches in kitchen (not ontray)
        for s, is_gf in kitchen_sandwiches.items():
             if s not in ontray_sandwiches: # Only count if not already on a tray
                 if is_gf:
                     available_gf_kitchen.append(s)
                 else:
                     available_reg_kitchen.append(s)

        # --- Count Makeable Sandwiches Potential ---
        num_notexist = len(notexist_sandwiches)
        total_bread_k = gf_bread_kitchen + reg_bread_kitchen
        total_content_k = gf_content_kitchen + reg_content_kitchen

        # Potential makeable sandwiches based on ingredients and notexist objects
        makeable_gf_potential = min(gf_bread_kitchen, gf_content_kitchen, num_notexist)
        makeable_total_potential = min(total_bread_k, total_content_k, num_notexist)


        # --- Greedy Assignment and Cost Calculation ---
        h = 0
        sandwiches_needing_put_on_tray = 0 # Counter for the +1 tray move cost

        # Available counts (will be consumed during assignment)
        avail_gf_ontray_at_loc = len(available_gf_ontray_at_location)
        avail_reg_ontray_at_loc = len(available_reg_ontray_at_location)
        avail_gf_ontray_elsewhere = len(available_gf_ontray_elsewhere)
        avail_reg_ontray_elsewhere = len(available_reg_ontray_elsewhere)
        avail_gf_kitchen = len(available_gf_kitchen)
        avail_reg_kitchen = len(available_reg_kitchen)
        avail_makeable_gf = makeable_gf_potential
        avail_makeable_total = makeable_total_potential

        # Process allergic children first (need GF)
        needed_allergic = len(unserved_allergic_needs)

        # Stage 1: GF ontray at location (cost 1)
        num = min(needed_allergic, avail_gf_ontray_at_loc)
        h += num * 1
        needed_allergic -= num
        avail_gf_ontray_at_loc -= num

        # Stage 2: GF ontray elsewhere (cost 2)
        num = min(needed_allergic, avail_gf_ontray_elsewhere)
        h += num * 2
        needed_allergic -= num
        avail_gf_ontray_elsewhere -= num

        # Stage 3: GF in kitchen (cost 3)
        num = min(needed_allergic, avail_gf_kitchen)
        h += num * 3
        needed_allergic -= num
        avail_gf_kitchen -= num
        sandwiches_needing_put_on_tray += num

        # Stage 4: Makeable GF (cost 4)
        num = min(needed_allergic, avail_makeable_gf)
        h += num * 4
        needed_allergic -= num
        avail_makeable_gf -= num # Consume makeable potential


        # Process non-allergic children (need any suitable)
        needed_not_allergic = len(unserved_not_allergic_needs)

        # Stage 1: Any suitable ontray at location (cost 1)
        # Use remaining GF first, then regular
        num_gf = min(needed_not_allergic, avail_gf_ontray_at_loc)
        h += num_gf * 1
        needed_not_allergic -= num_gf
        avail_gf_ontray_at_loc -= num_gf

        num_reg = min(needed_not_allergic, avail_reg_ontray_at_loc)
        h += num_reg * 1
        needed_not_allergic -= num_reg
        avail_reg_ontray_at_loc -= num_reg


        # Stage 2: Any suitable ontray elsewhere (cost 2)
        # Use remaining GF first, then regular
        num_gf = min(needed_not_allergic, avail_gf_ontray_elsewhere)
        h += num_gf * 2
        needed_not_allergic -= num_gf
        avail_gf_ontray_elsewhere -= num_gf

        num_reg = min(needed_not_allergic, avail_reg_ontray_elsewhere)
        h += num_reg * 2
        needed_not_allergic -= num_reg
        avail_reg_ontray_elsewhere -= num_reg


        # Stage 3: Any suitable in kitchen (cost 3)
        # Use remaining GF first, then regular
        num_gf = min(needed_not_allergic, avail_gf_kitchen)
        h += num_gf * 3
        needed_not_allergic -= num_gf
        avail_gf_kitchen -= num_gf
        sandwiches_needing_put_on_tray += num_gf

        num_reg = min(needed_not_allergic, avail_reg_kitchen)
        h += num_reg * 3
        needed_not_allergic -= num_reg
        avail_reg_kitchen -= num_reg
        sandwiches_needing_put_on_tray += num_reg


        # Stage 4: Makeable (GF or Reg) (cost 4)
        # Use remaining makeable GF first, then makeable Regular
        num_gf = min(needed_not_allergic, avail_makeable_gf)
        h += num_gf * 4
        needed_not_allergic -= num_gf
        avail_makeable_gf -= num_gf # Consume makeable potential
        sandwiches_needing_put_on_tray += num_gf

        # Remaining makeable potential is for regular sandwiches
        # Total makeable sandwiches available for non-allergic = Total makeable - GF assigned to allergic
        # The number of GF sandwiches assigned to allergic from the makeable pool is (makeable_gf_potential - avail_makeable_gf)
        makeable_gf_used_for_allergic = makeable_gf_potential - avail_makeable_gf
        avail_makeable_for_reg = avail_makeable_total - makeable_gf_used_for_allergic

        num_reg = min(needed_not_allergic, avail_makeable_for_reg)
        h += num_reg * 4
        needed_not_allergic -= num_reg
        # avail_makeable_for_reg -= num_reg # No need to track remaining makeable reg potential
        sandwiches_needing_put_on_tray += num_reg


        # Add cost for moving tray to kitchen if needed
        # This cost is incurred if any sandwich needs the 'put_on_tray' action at the kitchen
        # (i.e., it was in the kitchen or needed making) AND no tray is currently at the kitchen,
        # but there is at least one tray available elsewhere to move.
        if sandwiches_needing_put_on_tray > 0 and trays_at_kitchen_count == 0 and trays_elsewhere_count > 0:
            h += 1

        return h
