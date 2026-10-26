from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Helper function to split a PDDL fact string into its predicate and arguments."""
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the cost to reach the goal state (all children served)
    by summing up the estimated costs for each unserved child. It calculates the
    minimum number of actions (make, put, move, serve) required to get a suitable
    sandwich to each unserved child, prioritizing sandwiches that are closer to
    being served. It considers gluten allergies and assumes sufficient ingredients,
    sandwich slots, and trays are available when needed for the minimum cost path.

    Assumptions:
    - The heuristic assumes that in solvable instances, there are enough ingredients,
      sandwich slots, and trays to make and deliver all necessary sandwiches.
    - It assumes that a tray can be moved between any two places in one action.
    - It assumes that a sandwich on a tray at a location where *any* unserved child
      is waiting is "closer" (cost 0 for put/move steps) than one elsewhere (cost 1
      for move) or at the kitchen (cost 2 for put+move). This is a simplification;
      a sandwich might be at the wrong waiting location for a specific child.
    - It assumes each sandwich serves exactly one child.

    Heuristic Initialization:
    The constructor extracts static information from the task:
    - Which children are allergic to gluten.
    - The waiting location for each child.
    - The set of all children involved in the task (from goals and waiting facts).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all unserved children in the current state by comparing the goal
       facts (all children served) with the current state facts.
    2. If all children are served, the heuristic is 0.
    3. Count the number of unserved allergic children (`unserved_gf_count`) and
       unserved non-allergic children (`unserved_any_count`).
    4. The base heuristic value is the total number of unserved children, representing
       the minimum 1 'serve' action needed for each.
    5. Identify the set of locations where unserved children are waiting in the current state.
    6. Count the available sandwiches in the current state, categorized by type
       (gluten-free or not) and their current state/location, assigning a cost
       multiplier for the steps needed *before* the 'serve' action:
       - Cost 0: Sandwich is on a tray at a location where *some* unserved child is waiting.
         (Needs 0 additional put/move actions).
       - Cost 1: Sandwich is on a tray at a location where *no* unserved child is waiting.
         (Needs 1 'move_tray' action).
       - Cost 2: Sandwich is at the kitchen (`at_kitchen_sandwich`).
         (Needs 1 'put_on_tray' + 1 'move_tray' actions).
       - Cost 3: Sandwich needs to be made (not currently existing or available).
         (Needs 1 'make_sandwich' + 1 'put_on_tray' + 1 'move_tray' actions).
    7. Greedily satisfy the needs of allergic children first, using the available
       gluten-free sandwiches from the cheapest cost bucket (Cost 0) upwards.
       Add the corresponding cost multiplier (1, 2, or 3) for each sandwich used
       from cost buckets 1, 2, or 3 respectively.
    8. Any unused gluten-free sandwiches are added to the pool of available
       sandwiches for non-allergic children, maintaining their cost bucket.
    9. Greedily satisfy the needs of non-allergic children using the available
       any-type sandwiches (including remaining GF ones) from the cheapest cost
       bucket (Cost 0) upwards. Add the corresponding cost multiplier (1, 2, or 3)
       for each sandwich used from cost buckets 1, 2, or 3 respectively.
    10. The total heuristic value is the sum of the base serve cost and the
        additional costs calculated in steps 7 and 9.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.allergic_children = set()
        self.waiting_loc = {}
        self.all_children = set()

        # Extract static information
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.waiting_loc[child] = place
                self.all_children.add(child)

        # Identify all children from goals as well
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == "served":
                 self.all_children.add(parts[1])


    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = {c for c in self.all_children if c not in served_children}

        # 2. If all children are served, heuristic is 0
        if not unserved_children:
            return 0

        # 3. Count unserved allergic and non-allergic children
        unserved_gf_count = len([c for c in unserved_children if c in self.allergic_children])
        unserved_any_count = len([c for c in unserved_children if c not in self.allergic_children])

        # 4. Base heuristic: 1 action (serve) per unserved child
        h = unserved_gf_count + unserved_any_count

        # 5. Identify waiting places for unserved children in the current state
        waiting_places_in_state = {
            place for child, place in self.waiting_loc.items()
            if child in unserved_children
        }

        # 6. Count available sandwiches by type and state/location
        gf_sandwiches_ontray = set()
        any_sandwiches_ontray = set()
        gf_sandwiches_kitchen = set()
        any_sandwiches_kitchen = set()
        sandwich_ontray_tray = {} # Map sandwich to tray if ontray
        tray_locations = {} # Map tray to location

        # First pass to find tray locations and sandwich types
        sandwich_is_gf = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                tray, place = parts[1], parts[2]
                tray_locations[tray] = place
            elif parts[0] == "no_gluten_sandwich":
                 sandwich = parts[1]
                 sandwich_is_gf[sandwich] = True
            # We don't need to explicitly track non-gluten status, absence implies it.

        # Second pass to categorize sandwiches based on state and type
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "ontray":
                sandwich, tray = parts[1], parts[2]
                sandwich_ontray_tray[sandwich] = tray
                if sandwich_is_gf.get(sandwich, False):
                    gf_sandwiches_ontray.add(sandwich)
                else:
                    any_sandwiches_ontray.add(sandwich)
            elif parts[0] == "at_kitchen_sandwich":
                 sandwich = parts[1]
                 if sandwich_is_gf.get(sandwich, False):
                     gf_sandwiches_kitchen.add(sandwich)
                 else:
                     any_sandwiches_kitchen.add(sandwich)

        # Count sandwiches in cost buckets based on the revised definition
        S_gf_cost0 = 0 # ontray at a waiting_places_in_state
        S_any_cost0 = 0 # ontray at a waiting_places_in_state
        S_gf_cost1 = 0 # ontray elsewhere
        S_any_cost1 = 0 # ontray elsewhere
        S_gf_cost2 = len(gf_sandwiches_kitchen) # at kitchen
        S_any_cost2 = len(any_sandwiches_kitchen) # at kitchen

        for s in gf_sandwiches_ontray:
            t = sandwich_ontray_tray.get(s)
            if t and t in tray_locations:
                p = tray_locations[t]
                if p in waiting_places_in_state:
                    S_gf_cost0 += 1
                else:
                    S_gf_cost1 += 1

        for s in any_sandwiches_ontray:
            t = sandwich_ontray_tray.get(s)
            if t and t in tray_locations:
                p = tray_locations[t]
                if p in waiting_places_in_state:
                    S_any_cost0 += 1
                else:
                    S_any_cost1 += 1

        # 7. Satisfy GF needs first using cheapest available GF sandwiches
        needed_gf = unserved_gf_count
        used_gf_cost0 = min(needed_gf, S_gf_cost0)
        needed_gf -= used_gf_cost0
        used_gf_cost1 = min(needed_gf, S_gf_cost1)
        needed_gf -= used_cost1
        used_gf_cost2 = min(needed_gf, S_gf_cost2)
        needed_gf -= used_cost2
        used_gf_cost3 = needed_gf # Remaining must be made

        h += used_gf_cost1 * 1 + used_gf_cost2 * 2 + used_gf_cost3 * 3

        # 8. Calculate remaining GF sandwiches available for Any needs
        S_gf_cost0_rem = S_gf_cost0 - used_gf_cost0
        S_gf_cost1_rem = S_gf_cost1 - used_gf_cost1
        S_gf_cost2_rem = S_gf_cost2 - used_cost2

        # 9. Satisfy Any needs using cheapest available Any or remaining GF sandwiches
        needed_any = unserved_any_count
        total_avail_any_cost0 = S_any_cost0 + S_gf_cost0_rem
        total_avail_any_cost1 = S_any_cost1 + S_gf_cost1_rem
        total_avail_any_cost2 = S_any_cost2 + S_gf_cost2_rem

        used_any_cost0 = min(needed_any, total_avail_any_cost0)
        needed_any -= used_cost0
        used_any_cost1 = min(needed_any, total_avail_any_cost1)
        needed_any -= used_cost1
        used_any_cost2 = min(needed_any, total_avail_any_cost2)
        needed_any -= used_cost2
        used_any_cost3 = needed_any # Remaining must be made

        h += used_any_cost1 * 1 + used_any_cost2 * 2 + used_any_cost3 * 3

        # 10. Total heuristic is the sum calculated
        return h
