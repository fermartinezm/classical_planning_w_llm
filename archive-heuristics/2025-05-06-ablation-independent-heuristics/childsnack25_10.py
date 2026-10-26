import re
from collections import defaultdict

# Assume Heuristic base class is available from the planner environment
# from heuristics.heuristic_base import Heuristic
# Assume Task class is available from the planner environment
# from task import Task, Operator # Operator might not be strictly needed in heuristic

# Helper function to parse a fact string like "(predicate obj1 obj2)"
def parse_fact(fact_string):
    """Parses a PDDL fact string into a predicate and a tuple of objects."""
    # Remove parentheses and split by space
    parts = fact_string[1:-1].split()
    predicate = parts[0]
    objects = tuple(parts[1:])
    return predicate, objects

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the cost to reach a goal state by summing the
    estimated minimum number of actions required for each unserved child.
    For each unserved child, it calculates the minimum cost to get a suitable
    sandwich onto a tray at the child's location, plus the final 'serve' action.
    The cost estimation for getting the sandwich ready is based on the current
    state of the "closest" suitable sandwich and tray:
    - If a suitable sandwich is already on a tray at the child's location: cost 0.
    - If a suitable sandwich is on a tray elsewhere: cost 1 (move tray).
    - If a suitable sandwich is at the kitchen, and a tray is at the kitchen: cost 2 (put on tray, move tray).
    - If a suitable sandwich is at the kitchen, and no tray is at the kitchen but one exists elsewhere: cost 3 (move tray to kitchen, put on tray, move tray).
    - If a suitable sandwich needs to be made (resources available), and a tray is at the kitchen: cost 3 (make, put on tray, move tray).
    - If a suitable sandwich needs to be made (resources available), and no tray is at the kitchen but one exists elsewhere: cost 4 (make, move tray to kitchen, put on tray, move tray).
    - If a suitable sandwich cannot be made (resources unavailable): cost infinity.
    The total heuristic is the sum of (estimated_prep_cost + 1) for all unserved children.

    Assumptions:
    - The heuristic assumes that any available suitable sandwich can be used for any child requiring that type.
    - It assumes that any available tray can be used for any sandwich or child location.
    - It assumes that if a tray is needed at the kitchen or elsewhere, any existing tray can be moved there in one step (if not already there).
    - It assumes that if resources for making a sandwich are available at the kitchen, a 'make' action is possible.
    - The heuristic does not attempt to solve the resource assignment problem optimally; it takes the minimum cost option independently for each child based on the *existence* of resources/items in certain states. This makes it non-admissible but potentially effective for greedy search.
    - The heuristic assumes action costs are 1.
    - All children in the goal are assumed to be initially waiting at a place, and the 'waiting' fact persists.
    - All relevant trays are assumed to have their location specified by an 'at' fact in the initial state or state.

    Heuristic Initialization:
    The constructor extracts static information from the task, specifically
    which children are allergic to gluten and where each child is waiting.
    It also identifies all children that are part of the goal.
    Information about which bread/content items are gluten-free is also extracted.
    This information is stored in sets and dictionaries for quick lookup
    during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Initialize total heuristic value `h` to 0.
    2.  Parse the current state facts to build temporary data structures:
        - Set of served children.
        - Set of sandwiches at the kitchen.
        - Map from sandwich on tray to the tray object.
        - Set of sandwiches currently on any tray.
        - Map from tray object to its current location.
        - Set of sandwich objects that do not exist (`notexist`).
        - Set of sandwiches that are gluten-free.
        - Count of gluten-free bread portions at the kitchen.
        - Count of gluten-free content portions at the kitchen.
        - Count of any bread portions at the kitchen.
        - Count of any content portions at the kitchen.
        - Count of sandwich objects that do not exist.
        - Boolean flags indicating if any tray is at the kitchen or exists elsewhere (based on tray_location map).
    3.  Identify unserved children by checking the set of all goal children against the set of served children in the state.
    4.  If there are no unserved children, the goal is reached, return 0.
    5.  For each unserved child `c`:
        a.  Determine the child's waiting place `p` using the precomputed static information.
        b.  Determine if the child is allergic using the precomputed static information.
        c.  Initialize `min_prep_cost_c` to infinity. This will store the minimum cost to get a suitable sandwich on a tray at place `p`.
        d.  Check for suitable sandwiches in order of "closeness" and update `min_prep_cost_c`:
            - Iterate through sandwiches on trays (`ontray_sandwiches`). For each sandwich `s`:
              Get its tray `t` from `ontray_map`.
              If `tray_location.get(t) == p` and `s` is suitable for `c` (based on allergy and `no_gluten_sandwiches` set):
                  `min_prep_cost_c = min(min_prep_cost_c, 0)`. Break loops, found best case.
            - If `min_prep_cost_c` is still infinity, iterate through sandwiches on trays (`ontray_sandwiches`) again:
              For each sandwich `s`: Get its tray `t`.
              If `tray_location.get(t) is not None and tray_location.get(t) != p` and `s` is suitable for `c`:
                  `min_prep_cost_c = min(min_prep_cost_c, 1)`. Break loops, found a cost 1 case.
            - If `min_prep_cost_c` is still infinity, iterate through sandwiches at kitchen (`kitchen_sandwiches`):
              For each sandwich `s`:
              If `s` is suitable for `c`:
                  if bool(tray_location): # Check if any tray exists anywhere
                      if any_tray_at_kitchen:
                          min_prep_cost_c = min(min_prep_cost_c, 2) # Needs put, move
                      else:
                          min_prep_cost_c = min(min_prep_cost_c, 3) # Needs move tray to kitchen, put, move
                  # If no tray exists, cannot put on tray, cost remains infinity from this path.
                  if min_prep_cost_c <= 3: break # Found a case for cost 2 or 3
            - If `min_prep_cost_c` is still infinity, check if a suitable sandwich can be made:
              Determine if resources exist at the kitchen to make *at least one* suitable sandwich (check counts of bread, content, notexist sandwich objects based on allergy status).
              If resources are available (`can_make_suitable` is True):
                  if bool(tray_location): # Check if any tray exists anywhere
                      if any_tray_at_kitchen:
                          min_prep_cost_c = min(min_prep_cost_c, 3) # Needs make, put, move
                      else:
                          min_prep_cost_c = min(min_prep_cost_c, 4) # Needs make, move tray to kitchen, put, move
                  # If no tray exists, cannot put on tray after making, cost remains infinity from this path.

        e.  If `min_prep_cost_c` is still infinity after all checks, it means the child cannot be served in this state (e.g., missing resources, no trays, or sandwich/tray in an unreachable state). Return float('inf').
        f.  Add `min_prep_cost_c + 1` (for the serve action) to the total heuristic `h`.
    6.  Return the total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        # Extract static information from task.static
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.child_waiting_place = {}
        self.goal_children = set()

        # Collect all children from the goals first
        for goal_fact in self.task.goals:
             pred, objs = parse_fact(goal_fact)
             if pred == 'served':
                 self.goal_children.add(objs[0])

        # Extract static properties for goal children and relevant objects
        self.no_gluten_bread_static = set()
        self.no_gluten_content_static = set()

        for fact in self.static_facts:
            pred, objs = parse_fact(fact)
            if pred == 'allergic_gluten':
                child = objs[0]
                if child in self.goal_children:
                    self.allergic_children.add(child)
            elif pred == 'not_allergic_gluten':
                child = objs[0]
                if child in self.goal_children:
                    self.not_allergic_children.add(child)
            elif pred == 'waiting':
                child, place = objs
                if child in self.goal_children:
                    self.child_waiting_place[child] = place
            elif pred == 'no_gluten_bread':
                 self.no_gluten_bread_static.add(objs[0])
            elif pred == 'no_gluten_content':
                 self.no_gluten_content_static.add(objs[0])

        # Check if all goal children have a waiting place defined in static facts
        # If not, they can never be served. This state is unsolvable.
        # The heuristic will return infinity for such a child in __call__.


    def __call__(self, node):
        state = node.state

        # --- Build temporary state data structures ---
        served_children = set()
        kitchen_sandwiches = set()
        ontray_map = {} # {sandwich: tray}
        ontray_sandwiches = set()
        tray_location = {} # {tray: place}
        notexist_sandwiches = set()
        no_gluten_sandwiches = set()

        kitchen_bread_gf_count = 0
        kitchen_content_gf_count = 0
        kitchen_bread_any_count = 0
        kitchen_content_any_count = 0
        notexist_sandwich_count = 0

        any_tray_at_kitchen = False

        for fact in state:
            pred, objs = parse_fact(fact)
            if pred == 'served':
                served_children.add(objs[0])
            elif pred == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(objs[0])
            elif pred == 'ontray':
                s, t = objs
                ontray_map[s] = t
                ontray_sandwiches.add(s)
            elif pred == 'at':
                t, p = objs
                tray_location[t] = p
                if p == 'kitchen':
                    any_tray_at_kitchen = True
            elif pred == 'notexist':
                notexist_sandwiches.add(objs[0])
                notexist_sandwich_count += 1
            elif pred == 'no_gluten_sandwich':
                 no_gluten_sandwiches.add(objs[0])
            elif pred == 'at_kitchen_bread':
                 bread = objs[0]
                 kitchen_bread_any_count += 1
                 if bread in self.no_gluten_bread_static:
                     kitchen_bread_gf_count += 1
            elif pred == 'at_kitchen_content':
                 content = objs[0]
                 kitchen_content_any_count += 1
                 if content in self.no_gluten_content_static:
                     kitchen_content_gf_count += 1

        # Check if any tray exists anywhere based on the state
        any_tray_exists = bool(tray_location)

        # --- Calculate Heuristic ---
        h = 0
        unserved_children = self.goal_children - served_children

        if not unserved_children:
            return 0 # Goal state

        for child in unserved_children:
            place = self.child_waiting_place.get(child)
            # If a goal child is not waiting anywhere (based on static info), they cannot be served.
            # This state is unsolvable.
            if place is None:
                 # print(f"Debug: Unserved goal child {child} has no waiting place in static info.")
                 return float('inf')

            is_allergic = child in self.allergic_children

            min_prep_cost_c = float('inf')

            # Check for suitable sandwiches in order of "closeness"

            # 1. Suitable sandwich on tray at child's location (Cost 0)
            found_cost_0 = False
            for s in ontray_sandwiches:
                t = ontray_map[s]
                if tray_location.get(t) == place:
                    is_gf_sandwich = s in no_gluten_sandwiches
                    if (is_allergic and is_gf_sandwich) or (not is_allergic):
                        min_prep_cost_c = 0
                        found_cost_0 = True
                        break # Found the best case for this child

            if found_cost_0:
                 h += min_prep_cost_c + 1
                 continue # Move to the next unserved child

            # 2. Suitable sandwich on tray elsewhere (Cost 1)
            found_cost_1 = False
            for s in ontray_sandwiches:
                 t = ontray_map[s]
                 if tray_location.get(t) is not None and tray_location.get(t) != place: # Check if location is known and different
                     is_gf_sandwich = s in no_gluten_sandwiches
                     if (is_allergic and is_gf_sandwich) or (not is_allergic):
                         min_prep_cost_c = 1
                         found_cost_1 = True
                         break # Found a case for cost 1

            if found_cost_1:
                 h += min_prep_cost_c + 1
                 continue # Move to the next unserved child

            # 3. Suitable sandwich at kitchen (Cost 2 or 3)
            found_kitchen_sandwich = False
            for s in kitchen_sandwiches:
                 is_gf_sandwich = s in no_gluten_sandwiches
                 if (is_allergic and is_gf_sandwich) or (not is_allergic):
                     found_kitchen_sandwich = True
                     break # Found a suitable sandwich at kitchen

            if found_kitchen_sandwich:
                 if any_tray_exists: # Tray exists anywhere to potentially move to kitchen
                      if any_tray_at_kitchen:
                          min_prep_cost_c = 2 # Needs put, move
                      else:
                          min_prep_cost_c = 3 # Needs move tray to kitchen, put, move
                 # If no tray exists, cannot put on tray, cost remains infinity from this path.

            if min_prep_cost_c <= 3: # If we found a case for cost 2 or 3
                 h += min_prep_cost_c + 1
                 continue # Move to the next unserved child

            # 4. Suitable sandwich needs to be made (Cost 3 or 4)
            can_make_suitable = False
            if notexist_sandwich_count > 0: # Need an unused sandwich object
                if is_allergic:
                    # Need GF bread, GF content
                    if kitchen_bread_gf_count > 0 and kitchen_content_gf_count > 0:
                        can_make_suitable = True
                else: # Not allergic, can use any
                    # Need any bread, any content
                    if kitchen_bread_any_count > 0 and kitchen_content_any_count > 0:
                        can_make_suitable = True

            if can_make_suitable:
                 if any_tray_exists: # Tray exists anywhere to potentially move to kitchen
                      if any_tray_at_kitchen:
                          min_prep_cost_c = 3 # Needs make, put, move
                      else:
                          min_prep_cost_c = 4 # Needs make, move tray to kitchen, put, move
                 # If no tray exists, cannot put on tray after making, cost remains infinity from this path.

            if min_prep_cost_c <= 4: # If we found a case for cost 3 or 4
                 h += min_prep_cost_c + 1
                 continue # Move to the next unserved child

            # If we reached here, min_prep_cost_c is still infinity
            # This child cannot be served in this state
            # print(f"Debug: Child {child} cannot be served. min_prep_cost_c remains inf.")
            return float('inf') # State is likely a dead end

        return h
