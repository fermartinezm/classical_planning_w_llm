from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
         # This case should ideally not happen with valid PDDL facts in the state/goal/static
         return []
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic is the sum of estimated costs for sequential steps required
    to get each unserved child their correct sandwich:
    1. The number of unserved children (cost of 'serve' action).
    2. The number of sandwiches of each type (GF/REG) that still need to be made.
    3. The number of sandwiches currently at the kitchen that are needed on trays.
    4. The number of sandwiches on trays that are not yet at the correct child's location.

    This heuristic is non-admissible but aims to guide a greedy best-first search
    by prioritizing states where more prerequisites for serving children are met.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals # Goal conditions
        self.static_facts = task.static # Static facts

        # Extract child allergy status from static facts
        self.child_allergy = {} # child -> 'gf' or 'reg'
        for fact in self.static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip invalid facts
            if parts[0] == 'allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = 'gf'
            elif parts[0] == 'not_allergic_gluten' and len(parts) == 2:
                self.child_allergy[parts[1]] = 'reg'

        # Extract the set of all children that need to be served in the goal
        self.goal_children = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == 'served'}


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state
        h = 0

        # 1. Identify unserved children and their needs (place, type).
        served_children = {get_parts(f)[1] for f in state if get_parts(f) and get_parts(f)[0] == 'served'}
        children_to_serve = self.goal_children - served_children

        # If all goal children are served, heuristic is 0.
        if not children_to_serve:
            return 0

        child_waiting_place = {} # child -> place
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'waiting' and len(parts) == 3:
                child_waiting_place[parts[1]] = parts[2]

        unserved_needs = {} # (place, type) -> count
        for child in children_to_serve:
            place = child_waiting_place.get(child)
            allergy_type = self.child_allergy.get(child)
            # Ensure child is waiting and allergy status is known (should be true in valid problems)
            if place and allergy_type:
                unserved_needs[(place, allergy_type)] = unserved_needs.get((place, allergy_type), 0) + 1
                h += 1 # Cost for the final 'serve' action
            # Note: If a child in goal is not waiting or allergy unknown, they are unservable
            # in this state according to the domain rules. The heuristic counts the 'serve'
            # action for them, which is a reasonable estimate of remaining work.


        # 2. Count available sandwiches by type and location (kitchen, on_tray_at_place).
        available_sandwiches_kitchen = {'gf': 0, 'reg': 0}
        available_sandwiches_ontray_at = {} # place -> {'gf': count, 'reg': count}
        sandwich_is_gf = {} # sandwich -> True/False

        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'no_gluten_sandwich' and len(parts) == 2:
                sandwich_is_gf[parts[1]] = True

        tray_locations = {} # tray -> place
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) == 3:
                 # Assuming 'at' predicate is only for trays and places based on domain
                 tray_locations[parts[1]] = parts[2]


        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'at_kitchen_sandwich' and len(parts) == 2:
                s = parts[1]
                s_type = 'gf' if sandwich_is_gf.get(s, False) else 'reg'
                available_sandwiches_kitchen[s_type] += 1
            elif parts[0] == 'ontray' and len(parts) == 3:
                s, t = parts[1], parts[2]
                s_type = 'gf' if sandwich_is_gf.get(s, False) else 'reg'
                tray_place = tray_locations.get(t) # Find tray location
                if tray_place:
                    if tray_place not in available_sandwiches_ontray_at:
                        available_sandwiches_ontray_at[tray_place] = {'gf': 0, 'reg': 0}
                    available_sandwiches_ontray_at[tray_place][s_type] += 1

        # 3. Calculate total needed sandwiches.
        total_needed_gf = sum(count for (p, type), count in unserved_needs.items() if type == 'gf')
        total_needed_reg = sum(count for (p, type), count in unserved_needs.items() if type == 'reg')

        # 4. Calculate deficit for `make_sandwich`.
        # This is the number of sandwiches of each type that need to be created.
        total_available_gf = available_sandwiches_kitchen['gf'] + sum(d.get('gf', 0) for d in available_sandwiches_ontray_at.values())
        total_available_reg = available_sandwiches_kitchen['reg'] + sum(d.get('reg', 0) for d in available_sandwiches_ontray_at.values())
        deficit_make_gf = max(0, total_needed_gf - total_available_gf)
        deficit_make_reg = max(0, total_needed_reg - total_available_reg)
        h += deficit_make_gf + deficit_make_reg

        # 5. Calculate deficit for `put_on_tray`.
        # This is the number of sandwiches currently at the kitchen that are needed
        # on trays to satisfy the total demand not already met by sandwiches on trays.
        needed_gf_from_kitchen = min(available_sandwiches_kitchen['gf'], max(0, total_needed_gf - sum(d.get('gf', 0) for d in available_sandwiches_ontray_at.values())))
        needed_reg_from_kitchen = min(available_sandwiches_kitchen['reg'], max(0, total_needed_reg - sum(d.get('reg', 0) for d in available_sandwiches_ontray_at.values())))
        deficit_put_on_tray = needed_gf_from_kitchen + needed_reg_from_kitchen
        h += deficit_put_on_tray

        # 6. Calculate deficit for `move_tray`.
        # This is the number of sandwiches of type T needed at place P that are
        # currently on trays but not at place P. This simplifies to the total
        # deficit of sandwiches needed at each specific place.
        deficit_move = 0
        for (place, s_type), needed_count in unserved_needs.items():
            have_count = available_sandwiches_ontray_at.get(place, {}).get(s_type, 0)
            deficit_at_place = max(0, needed_count - have_count)
            deficit_move += deficit_at_place # Each deficit needs a delivery action (move_tray)

        h += deficit_move

        return h
