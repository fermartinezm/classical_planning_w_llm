from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic counts the number of unserved children and adds costs
    based on the "stage" of the sandwich needed for them:
    - Sandwich on tray at child's location: 1 (serve)
    - Sandwich on tray elsewhere: 1 (move tray) + 1 (serve) = 2
    - Sandwich in kitchen: 1 (put on tray) + 1 (move tray) + 1 (serve) = 3
    - Sandwich needs making: 1 (make) + 1 (put on tray) + 1 (move tray) + 1 (serve) = 4

    It prioritizes using sandwiches that are closer to being served.
    It accounts for gluten allergies.
    It simplifies resource constraints (trays, ingredients, slots) assuming they are available when needed,
    which is acceptable for a non-admissible heuristic.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal: all children to be served.
        - Static facts: child allergies and waiting places.
        """
        self.goals = task.goals
        self.static = task.static

        # Map child to allergy status
        self.child_allergy = {}
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.child_allergy[child] = False

        # Map child to waiting place
        self.child_place = {}
        self.waiting_places = set()
        # Children are listed in goals. Waiting facts are static.
        # We only care about children that are in the goals.
        all_children_in_goals = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:3]
                # Only consider children that are actually in the goals
                if child in all_children_in_goals:
                    self.child_place[child] = place
                    self.waiting_places.add(place)

        # The set of all children we care about (those in the goals)
        self.all_children = all_children_in_goals


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
             return 0

        # Identify served children in the current state
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify unserved children and their needs/locations
        unserved_children_gf = {} # place -> count
        unserved_children_any = {} # place -> count
        for child in self.all_children:
            if child not in served_children:
                place = self.child_place.get(child)
                if place: # Ensure child has a waiting place defined (should be true for goal children)
                    if self.child_allergy.get(child, False): # Default to not allergic if info missing
                        unserved_children_gf[place] = unserved_children_gf.get(place, 0) + 1
                    else:
                        unserved_children_any[place] = unserved_children_any.get(place, 0) + 1

        # Count available sandwiches and trays
        sandwiches_ontray = {} # sandwich -> tray
        trays_at_place = {} # tray -> place
        sandwiches_kitchen = set() # sandwich
        sandwich_is_gf = {} # sandwich -> bool

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:3]
                sandwiches_ontray[s] = t
            elif match(fact, "at", "*", "*"):
                 # Only track tray locations
                 parts = get_parts(fact)
                 if len(parts) == 3 and parts[1].startswith('tray'):
                    t, p = parts[1:3]
                    trays_at_place[t] = p
            elif match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                sandwiches_kitchen.add(s)
            elif match(fact, "no_gluten_sandwich", "*"):
                 s = get_parts(fact)[1]
                 sandwich_is_gf[s] = True

        # Default non-GF sandwiches to False for sandwiches we know exist
        all_known_sandwiches = set(sandwiches_ontray.keys()) | sandwiches_kitchen
        for s in all_known_sandwiches:
             sandwich_is_gf.setdefault(s, False)


        # Count available sandwiches by location/stage and type
        avail_gf_at_p = {p: 0 for p in self.waiting_places}
        avail_reg_at_p = {p: 0 for p in self.waiting_places}
        avail_gf_ontray_other = 0
        avail_reg_ontray_other = 0
        avail_gf_kitchen = 0
        avail_reg_kitchen = 0

        for s, t in sandwiches_ontray.items():
            p = trays_at_place.get(t)
            if p: # Tray location is known
                if sandwich_is_gf[s]:
                    if p in self.waiting_places:
                        avail_gf_at_p[p] += 1
                    else:
                        avail_gf_ontray_other += 1
                else: # Regular sandwich
                    if p in self.waiting_places:
                        avail_reg_at_p[p] += 1
                    else:
                        avail_reg_ontray_other += 1

        for s in sandwiches_kitchen:
            if sandwich_is_gf[s]:
                avail_gf_kitchen += 1
            else:
                avail_reg_kitchen += 1

        # Calculate heuristic cost
        total_cost = 0

        # Children needing GF sandwiches per place
        rem_children_gf_at_p = {p: count for p, count in unserved_children_gf.items()}
        # Children needing Any sandwiches per place
        rem_children_any_at_p = {p: count for p, count in unserved_children_any.items()}

        # Stage 4: Serve (Sandwich on tray at location) - Cost 1
        # Prioritize GF needs with GF sandwiches at location
        for p in self.waiting_places:
            use_gf_s4 = min(rem_children_gf_at_p.get(p, 0), avail_gf_at_p.get(p, 0))
            total_cost += use_gf_s4 * 1
            rem_children_gf_at_p[p] = rem_children_gf_at_p.get(p, 0) - use_gf_s4 # Update remaining needs at this location
            avail_gf_at_p[p] -= use_gf_s4 # Consume the sandwich

        # Prioritize Any needs with Regular sandwiches at location
        for p in self.waiting_places:
            use_any_s4_reg = min(rem_children_any_at_p.get(p, 0), avail_reg_at_p.get(p, 0))
            total_cost += use_any_s4_reg * 1
            rem_children_any_at_p[p] = rem_children_any_at_p.get(p, 0) - use_any_s4_reg # Update remaining needs at this location
            avail_reg_at_p[p] -= use_any_s4_reg # Consume the sandwich

        # Prioritize Any needs with remaining GF sandwiches at location
        for p in self.waiting_places:
             use_any_s4_gf = min(rem_children_any_at_p.get(p, 0), avail_gf_at_p.get(p, 0)) # Use remaining GF at p
             total_cost += use_any_s4_gf * 1
             rem_children_any_at_p[p] = rem_children_any_at_p.get(p, 0) - use_any_s4_gf # Update remaining needs at this location
             avail_gf_at_p[p] -= use_any_s4_gf # Consume the sandwich

        # Recalculate total remaining needs after Stage 4 assignments
        rem_gf = sum(rem_children_gf_at_p.values())
        rem_any = sum(rem_children_any_at_p.values())


        # Stage 3: Move Tray + Serve (Sandwich on tray elsewhere) - Cost 2
        # Prioritize GF needs with GF sandwiches on tray elsewhere
        use_gf_s3 = min(rem_gf, avail_gf_ontray_other)
        total_cost += use_gf_s3 * 2
        rem_gf -= use_gf_s3
        avail_gf_ontray_other -= use_gf_s3 # Consume the sandwich

        # Prioritize Any needs with remaining GF sandwiches on tray elsewhere
        use_any_s3_gf = min(rem_any, avail_gf_ontray_other)
        total_cost += use_any_s3_gf * 2
        rem_any -= use_any_s3_gf
        avail_gf_ontray_other -= use_any_s3_gf # Consume the sandwich

        # Prioritize Any needs with Regular sandwiches on tray elsewhere
        use_any_s3_reg = min(rem_any, avail_reg_ontray_other)
        total_cost += use_any_s3_reg * 2
        rem_any -= use_any_s3_reg
        avail_reg_ontray_other -= use_any_s3_reg # Consume the sandwich


        # Stage 2: Put on Tray + Move Tray + Serve (Sandwich in kitchen) - Cost 3
        # Prioritize GF needs with GF sandwiches in kitchen
        use_gf_s2 = min(rem_gf, avail_gf_kitchen)
        total_cost += use_gf_s2 * 3
        rem_gf -= use_gf_s2
        avail_gf_kitchen -= use_gf_s2 # Consume the sandwich

        # Prioritize Any needs with remaining GF sandwiches in kitchen
        use_any_s2_gf = min(rem_any, avail_gf_kitchen)
        total_cost += use_any_s2_gf * 3
        rem_any -= use_any_s2_gf
        avail_gf_kitchen -= use_any_s2_gf # Consume the sandwich

        # Prioritize Any needs with Regular sandwiches in kitchen
        use_any_s2_reg = min(rem_any, avail_reg_kitchen)
        total_cost += use_any_s2_reg * 3
        rem_any -= use_any_s2_reg
        avail_reg_kitchen -= use_any_s2_reg # Consume the sandwich


        # Stage 1: Make + Put on Tray + Move Tray + Serve (Sandwich needs making) - Cost 4
        # Total remaining needs must be made
        rem_total = rem_gf + rem_any

        total_cost += rem_total * 4

        return total_cost
