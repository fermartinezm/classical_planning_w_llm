import math
from heuristics.heuristic_base import Heuristic
# Assuming Task and Operator classes are available in the environment
# from task import Operator, Task

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to reach a goal state by summing
    up the estimated costs for each unserved child. The cost for a child
    depends on whether a suitable sandwich is already available on a tray
    at their location, or if it needs to be moved from elsewhere, is at the
    kitchen, or needs to be made. It also accounts for the final 'serve'
    action for each child and the potential need to move trays to the kitchen
    to put sandwiches on them. The heuristic is non-admissible and aims to
    guide a greedy best-first search efficiently. Returns infinity if the
    goal is detected to be unreachable based on available ingredients and
    sandwiches.

    Assumptions:
    - The domain is childsnacks as described in the PDDL file.
    - Facts are represented as strings like '(predicate arg1 arg2)'.
    - Static facts (allergies, waiting locations, gluten status of ingredients)
      are available in self.static.
    - The goal is to serve all children specified in task.goals.
    - A tray can hold multiple sandwiches (implicitly, as actions are per sandwich).
    - The cost of each action is 1.
    - Unserved children in the goal are assumed to be waiting at a location
      specified in the static facts.

    Heuristic Initialization:
    The constructor extracts static information from task.static:
    - Which children are allergic to gluten (stored in self.child_allergy).
    - The waiting location for each child (stored in self.child_location).
    - Which bread and content portions are gluten-free (stored in self.is_ng_bread,
      self.is_ng_content).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all children that need to be served (those in task.goals
       whose '(served child)' fact is not in the current state). Store their
       waiting locations and allergy statuses.
    2. If no children need serving, the heuristic is 0 (goal state).
    3. Count available ingredients (bread, content, NG bread, NG content) at the kitchen.
    4. Count available sandwiches (at kitchen, on trays) and determine their
       gluten status and location (if on a tray).
    5. Calculate the total number of suitable sandwiches required (`num_unserved`)
       and the number of NG sandwiches required (`len(unserved_ng_children)`).
    6. Calculate the total number of sandwiches available (existing + makeable)
       and the number of NG sandwiches available (existing NG + makeable NG).
    7. Check if the total number of available sandwiches is less than the total
       number of unserved children, or if the total number of available NG
       sandwiches is less than the number of unserved allergic children. If either
       is true, the goal is likely unreachable, return infinity.
    8. Calculate the number of sandwiches that still need to be made (`sandwiches_to_make`)
       to meet the total demand, assuming existing sandwiches are used first.
    9. Count how many unserved children already have a suitable sandwich on a tray
       at their waiting location (`children_immediately_servable`). Add 1 to the
       heuristic for each of these children (the 'serve' action).
    10. The remaining unserved children (`children_needing_delivery`) need a
        sandwich delivered to their location. Add 1 to the heuristic for each
        of these children (the final 'serve' action).
    11. Estimate the 'delivery_cost' to get the `children_needing_delivery`
        sandwiches onto trays and moved to the correct locations.
        - Identify the pool of sandwiches available for delivery: those that need
          making (`sandwiches_to_make`), those at the kitchen (`sandwiches_at_kitchen`),
          and those on trays elsewhere (not at a location where an unserved child
          needs them immediately).
        - Assign sandwiches from this pool to satisfy the delivery needs,
          prioritizing sources with lower estimated action costs:
            - Sandwiches on trays elsewhere: cost 1 (move_tray).
            - Sandwiches at kitchen: cost 2 (put_on_tray + move_tray).
            - Sandwiches needing making: cost 3 (make + put_on_tray + move_tray).
        - Sum the costs for the assigned sandwiches.
    12. Account for the need to move trays to the kitchen if there aren't enough
        trays already there to handle all the 'put_on_tray' actions required for
        sandwiches starting at the kitchen or needing to be made. Add 1 for each
        tray movement needed to bring a tray to the kitchen. Add this cost to the
        `delivery_cost`.
    13. The total heuristic value is the sum of the costs from steps 9, 10, and 12.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Extract static information
        self.child_allergy = {}
        self.child_location = {}
        self.is_ng_bread = {}
        self.is_ng_content = {}

        for fact in self.static:
            parts = fact[1:-1].split()
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = True
            elif predicate == 'not_allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = False
            elif predicate == 'waiting':
                child = parts[1]
                place = parts[2]
                self.child_location[child] = place
            elif predicate == 'no_gluten_bread':
                bread = parts[1]
                self.is_ng_bread[bread] = True
            elif predicate == 'no_gluten_content':
                content = parts[1]
                self.is_ng_content[content] = True


    def __call__(self, node):
        state = node.state
        goals = self.goals

        # 1. Identify unserved children and their locations/needs
        unserved_children = []
        unserved_children_by_place = {}
        unserved_ng_children = []
        unserved_reg_children = []

        served_facts = {fact for fact in state if fact.startswith('(served ')}

        for goal in goals:
            if goal.startswith('(served '):
                child = goal[len('(served '):-1]
                if goal not in served_facts:
                    unserved_children.append(child)
                    # Child must be in static waiting facts if they need serving
                    place = self.child_location.get(child)
                    if place is None:
                         # Unserved child is not waiting anywhere, cannot be served
                         return float('inf')

                    unserved_children_by_place.setdefault(place, []).append(child)
                    if self.child_allergy.get(child, False):
                        unserved_ng_children.append(child)
                    else:
                        unserved_reg_children.append(child)

        num_unserved = len(unserved_children)

        # 2. If no children need serving, goal reached
        if num_unserved == 0:
            return 0

        # 3. Count available ingredients at kitchen
        num_bread_kitchen = 0
        num_content_kitchen = 0
        num_ng_bread_kitchen = 0
        num_ng_content_kitchen = 0
        for fact in state:
            if fact.startswith('(at_kitchen_bread '):
                bread = fact[len('(at_kitchen_bread '):-1]
                num_bread_kitchen += 1
                if self.is_ng_bread.get(bread, False):
                    num_ng_bread_kitchen += 1
            elif fact.startswith('(at_kitchen_content '):
                content = fact[len('(at_kitchen_content '):-1]
                num_content_kitchen += 1
                if self.is_ng_content.get(content, False):
                    num_ng_content_kitchen += 1

        # 4. Count available sandwiches and their status/location.
        sandwiches_at_kitchen = set()
        sandwiches_ontray = {} # {sandwich: tray}
        trays_location = {} # {tray: place}
        sandwich_is_ng = {} # {sandwich: is_ng}

        for fact in state:
            if fact.startswith('(at_kitchen_sandwich '):
                s = fact[len('(at_kitchen_sandwich '):-1]
                sandwiches_at_kitchen.add(s)
            elif fact.startswith('(ontray '):
                parts = fact[len('(ontray '):-1].split()
                if len(parts) == 2:
                    s = parts[0] # Corrected index
                    t = parts[1] # Corrected index
                    sandwiches_ontray[s] = t
            elif fact.startswith('(at '):
                 parts = fact[len('(at '):-1].split()
                 if len(parts) == 2:
                    t = parts[0] # Corrected index
                    p = parts[1] # Corrected index
                    trays_location[t] = p
            elif fact.startswith('(no_gluten_sandwich '):
                s = fact[len('(no_gluten_sandwich '):-1]
                sandwich_is_ng[s] = True

        sandwiches_on_trays_location = {} # {place: set of sandwiches}
        for s, t in sandwiches_ontray.items():
            if t in trays_location:
                p = trays_location[t]
                sandwiches_on_trays_location.setdefault(p, set()).add(s)

        # 5. Calculate sandwich needs and availability.
        all_avail_sandwiches = sandwiches_at_kitchen | set(sandwiches_ontray.keys())
        num_avail_ng_sandwich = len([s for s in all_avail_sandwiches if sandwich_is_ng.get(s, False)])
        num_avail_reg_sandwich = len([s for s in all_avail_sandwiches if not sandwich_is_ng.get(s, False)])
        num_avail_total_sandwich = num_avail_ng_sandwich + num_avail_reg_sandwich

        need_ng_sandwich = max(0, len(unserved_ng_children) - num_avail_ng_sandwich)
        surplus_ng_avail_for_reg = max(0, num_avail_ng_sandwich - len(unserved_ng_children))
        need_reg_sandwich = max(0, len(unserved_reg_children) - (num_avail_reg_sandwich + surplus_ng_avail_for_reg))
        total_sandwich_need_to_obtain = need_ng_sandwich + need_reg_sandwich

        # 6. Check if enough sandwiches can exist in total and for NG needs
        num_can_make_ng = min(num_ng_bread_kitchen, num_ng_content_kitchen)
        num_can_make_total = min(num_bread_kitchen, num_content_kitchen)

        if num_avail_total_sandwich + num_can_make_total < num_unserved:
            return float('inf') # Not enough ingredients + existing sandwiches total

        if num_avail_ng_sandwich + num_can_make_ng < len(unserved_ng_children):
             return float('inf') # Not enough ingredients + existing sandwiches for NG needs

        # 7. Calculate the number of sandwiches that still need to be made
        sandwiches_to_make = total_sandwich_need_to_obtain # Assume we make exactly what's needed

        # 8. Count children immediately servable.
        children_immediately_servable = 0
        # Keep track of sandwiches used for immediate service to avoid double counting
        served_sandwiches_at_location = {} # {place: set of sandwiches used}

        for child in unserved_children:
            place = self.child_location[child]
            is_allergic = self.child_allergy.get(child, False)

            suitable_sandwich_found = False
            if place in sandwiches_on_trays_location:
                for s in sandwiches_on_trays_location[place]:
                    is_s_ng = sandwich_is_ng.get(s, False)
                    if (is_allergic and is_s_ng) or (not is_allergic):
                       if s not in served_sandwiches_at_location.get(place, set()):
                           suitable_sandwich_found = True
                           children_immediately_servable += 1
                           served_sandwiches_at_location.setdefault(place, set()).add(s)
                           break
            # Note: This greedy assignment might not be optimal, but it's a heuristic.

        # 9. Calculate children needing delivery.
        children_needing_delivery = num_unserved - children_immediately_servable

        # 10. Estimate delivery costs (make/put/move)
        delivery_cost = 0
        needs_to_satisfy = children_needing_delivery

        # Identify sandwiches on trays that are NOT at the location of a child needing immediate service
        sandwiches_on_trays_used_for_immediate_service = set().union(*served_sandwiches_at_location.values()) if served_sandwiches_at_location else set()
        avail_ontray_elsewhere_for_delivery = set()
        for s, t in sandwiches_ontray.items():
            if s not in sandwiches_on_trays_used_for_immediate_service:
                 avail_ontray_elsewhere_for_delivery.add(s)

        N_make = sandwiches_to_make
        N_kitchen = len(sandwiches_at_kitchen)
        N_ontray_elsewhere = len(avail_ontray_elsewhere_for_delivery)

        # Use sandwiches on trays elsewhere first (cost 1: move_tray)
        use_ontray_elsewhere = min(needs_to_satisfy, N_ontray_elsewhere)
        delivery_cost += use_ontray_elsewhere * 1
        needs_to_satisfy -= use_ontray_elsewhere

        # Use sandwiches at kitchen next (cost 2: put_on_tray + move_tray)
        use_kitchen = min(needs_to_satisfy, N_kitchen)
        delivery_cost += use_kitchen * 2
        needs_to_satisfy -= use_kitchen

        # Make new sandwiches last (cost 3: make + put_on_tray + move_tray)
        use_make = min(needs_to_satisfy, N_make)
        delivery_cost += use_make * 3
        needs_to_satisfy -= use_make

        # needs_to_satisfy should be 0 here if unsolvability checks passed

        # 11. Add cost for moving trays to kitchen if needed for put_on_tray actions
        num_trays_kitchen = len([t for t, p in trays_location.items() if p == 'kitchen'])
        put_on_tray_actions_needed_at_kitchen = use_kitchen + use_make
        trays_to_move_to_kitchen = max(0, put_on_tray_actions_needed_at_kitchen - num_trays_kitchen)
        delivery_cost += trays_to_move_to_kitchen * 1


        # 12. Total heuristic value.
        # Cost = (serve actions for ready children) + (delivery actions) + (serve actions for children needing delivery)
        h = children_immediately_servable * 1
        h += delivery_cost
        h += children_needing_delivery * 1

        return h
