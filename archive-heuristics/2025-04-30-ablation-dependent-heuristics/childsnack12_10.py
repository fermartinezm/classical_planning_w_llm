# Import necessary modules
import collections
import math # Import math for infinity

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to serve all unserved children.
    The heuristic breaks down the process for each unserved child into steps:
    1. Get a suitable sandwich (gluten-free for allergic, any for others).
    2. Get the sandwich onto a tray.
    3. Get the tray with the sandwich to the child's location.
    4. Serve the child.
    The heuristic sums up the estimated minimum actions needed to fulfill the
    sandwich-at-location requirement for all unserved children, plus one action
    for the final 'serve' step for each child. It greedily allocates available
    sandwiches (on trays elsewhere, at kitchen, or needing to be made) to
    satisfy the total demand, prioritizing cheaper sources and gluten-free
    sandwiches for allergic children.

    Assumptions:
    - Trays can be moved between any two places with a cost of 1 action.
    - Tray availability at specific locations for intermediate steps (like
      put_on_tray at kitchen) is implicitly handled by assuming a tray can be
      moved there if needed, adding to the cost. The heuristic does not
      explicitly track individual tray objects or their availability beyond
      their current location.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task definition.
    It identifies:
    - Which children are allergic or not allergic to gluten.
    - Which bread and content portions are gluten-free.
    - The waiting place for each child.
    - All possible places in the domain (including 'kitchen' and object places).
    - The set of children that need to be served according to the goal.
    This information is stored in instance variables for quick access during
    heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Check if the goal is already reached. If yes, return 0.
    2.  Identify all children who are in the goal state but not yet served in
        the current state. Count the total number of such unserved children
        (N_unserved). This contributes at least N_unserved to the heuristic
        (for the final 'serve' action for each).
    3.  For each unserved child, determine their location and sandwich type
        requirement (gluten-free or any). Aggregate the demand for each
        sandwich type at each location (demand_gf_at_p, demand_nongf_at_p).
    4.  Count the number of suitable sandwiches already on trays at the correct
        location for immediate serving (avail_gf_ontray_at_p,
        avail_nongf_ontray_at_p). This requires checking 'ontray', 'at', and
        'no_gluten_sandwich' facts in the current state.
    5.  Calculate the remaining deficit of suitable sandwiches on trays at the
        correct locations after accounting for sandwiches already at the right
        place. This is done per place, considering that GF sandwiches can satisfy
        non-GF demand at that location. Sum these deficits globally
        (total_deficit_gf, total_deficit_nongf).
    6.  Count available suitable sandwiches that are *not* yet on trays at the
        correct location. These are sandwiches:
        - On trays at other locations (avail_gf_ontray_elsewhere,
          avail_nongf_ontray_elsewhere). Getting these to the child's location
          costs 1 'move_tray' action per sandwich needed.
        - At the kitchen (at_kitchen_sandwich) (avail_gf_kitchen,
          avail_nongf_kitchen). Getting these to the child's location costs
          1 'put_on_tray' + 1 'move_tray' = 2 actions per sandwich needed
          (assuming a tray is available at the kitchen or can be moved there).
        - That can be made (from at_kitchen_bread, at_kitchen_content, notexist)
          (can_make_gf, can_make_nongf). Getting these to the child's location
          costs 1 'make' + 1 'put_on_tray' + 1 'move_tray' = 3 actions per
          sandwich needed (assuming resources and a tray are available at the
          kitchen or can be moved there).
    7.  Greedily satisfy the total remaining deficits (total_deficit_gf,
        total_deficit_nongf) using the available sandwiches from the sources
        identified in step 6, starting with the cheapest sources (+1 cost),
        then the next cheapest (+2 cost), and finally the most expensive (+3
        cost). Prioritize using gluten-free sandwiches for the gluten-free
        deficit first. Add the corresponding costs to the heuristic value.
    8.  The final heuristic value is the sum of the base cost (N_unserved) and
        the costs calculated in step 7 for moving/preparing sandwiches.
    9.  If, after exhausting all available resources, there is still a deficit,
        it implies the state is unsolvable, and the heuristic returns infinity.
    """

    def __init__(self, task):
        self.task = task
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_bread_types = set()
        self.no_gluten_content_types = set()
        self.waiting_places = {}  # child -> place
        self.all_places = {'kitchen'} # Start with the constant kitchen

        # Extract static information from task.static
        for fact_string in task.static:
            pred, objs = self._parse_fact(fact_string)
            if pred == 'allergic_gluten':
                if len(objs) > 0: self.allergic_children.add(objs[0])
            elif pred == 'not_allergic_gluten':
                if len(objs) > 0: self.not_allergic_children.add(objs[0])
            elif pred == 'no_gluten_bread':
                if len(objs) > 0: self.no_gluten_bread_types.add(objs[0])
            elif pred == 'no_gluten_content':
                if len(objs) > 0: self.no_gluten_content_types.add(objs[0])
            elif pred == 'waiting':
                if len(objs) > 1:
                    self.waiting_places[objs[0]] = objs[1]
                    self.all_places.add(objs[1]) # Add waiting place to all places

        # Extract all places mentioned in initial state 'at' facts
        for fact_string in task.initial_state:
             pred, objs = self._parse_fact(fact_string)
             if pred == 'at':
                 if len(objs) > 1: # Ensure it's (at tray place)
                    self.all_places.add(objs[1]) # tray is at place

        # Extract children to serve from task.goals
        self.children_to_serve = set()
        for fact_string in task.goals:
            pred, objs = self._parse_fact(fact_string)
            if pred == 'served':
                if len(objs) > 0: self.children_to_serve.add(objs[0])

    def _parse_fact(self, fact_string):
        """Helper to parse a fact string into predicate and objects."""
        # Removes surrounding brackets and splits by space
        parts = fact_string[1:-1].split()
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def __call__(self, state):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        # 1. Check if the goal is already reached.
        if self.task.goal_reached(state):
            return 0

        state_facts = set(state)

        # 2. Identify unserved children
        unserved_children = {c for c in self.children_to_serve if f'(served {c})' not in state_facts}
        N_unserved = len(unserved_children)

        # If no children need serving, but goal not reached, something is wrong or it's an empty goal.
        # Assuming goal is always to serve children.
        if N_unserved == 0:
             return 0 # Should be covered by task.goal_reached, but safe fallback.

        # 3. Aggregate demand for sandwiches at each location
        demand_gf_at_p = collections.Counter()
        demand_nongf_at_p = collections.Counter()
        for child in unserved_children:
            place = self.waiting_places.get(child) # Get place from static info
            if place: # Child must have a waiting place
                if child in self.allergic_children:
                    demand_gf_at_p[place] += 1
                else:
                    demand_nongf_at_p[place] += 1

        # 4. Count available sandwiches on trays at each location and their type
        avail_gf_ontray_at_p = collections.Counter()
        avail_nongf_ontray_at_p = collections.Counter()
        sandwich_is_gf = {} # Map sandwich object to boolean (is_gluten_free)
        sandwich_on_tray = {} # Map sandwich object to tray object
        tray_at_place = {} # Map tray object to place object

        # First pass to determine sandwich types and tray locations
        for fact_string in state_facts:
            pred, objs = self._parse_fact(fact_string)
            if pred == 'no_gluten_sandwich':
                if len(objs) > 0: sandwich_is_gf[objs[0]] = True
            elif pred == 'ontray':
                 if len(objs) > 1: sandwich_on_tray[objs[0]] = objs[1]
            elif pred == 'at':
                 if len(objs) > 1: tray_at_place[objs[0]] = objs[1]

        # Assume sandwiches not marked as no_gluten are non-gluten
        # Collect all sandwich objects mentioned in relevant state facts
        all_sandwiches_in_state = set(sandwich_on_tray.keys()).union(
            {self._parse_fact(f)[1][0] for f in state_facts if self._parse_fact(f)[0] == 'at_kitchen_sandwich' and len(self._parse_fact(f)[1]) > 0}
        ).union(
             {self._parse_fact(f)[1][0] for f in state_facts if self._parse_fact(f)[0] == 'notexist' and len(self._parse_fact(f)[1]) > 0}
        )

        for s in all_sandwiches_in_state:
             if s not in sandwich_is_gf:
                 sandwich_is_gf[s] = False # Assume non-GF if not marked GF

        # Count sandwiches on trays at specific places
        for s, t in sandwich_on_tray.items():
            place = tray_at_place.get(t)
            if place:
                if sandwich_is_gf.get(s, False):
                    avail_gf_ontray_at_p[place] += 1
                else:
                    avail_nongf_ontray_at_p[place] += 1

        # 5. Calculate remaining deficit per place after using sandwiches already there
        deficit_gf_at_p = collections.Counter()
        deficit_nongf_at_p = collections.Counter()

        for place in self.all_places:
            needed_gf = demand_gf_at_p[place]
            needed_nongf = demand_nongf_at_p[place]
            avail_gf = avail_gf_ontray_at_p[place]
            avail_nongf = avail_nongf_ontray_at_p[place]

            # Satisfy GF demand first with GF sandwiches at this place
            served_gf_now = min(needed_gf, avail_gf)
            needed_gf -= served_gf_now
            avail_gf -= served_gf_now

            # Satisfy non-GF demand with non-GF sandwiches at this place
            served_nongf_now = min(needed_nongf, avail_nongf)
            needed_nongf -= served_nongf_now
            avail_nongf -= served_nongf_now

            # Satisfy remaining non-GF demand with remaining GF sandwiches at this place
            served_nongf_with_gf_now = min(needed_nongf, avail_gf)
            needed_nongf -= served_nongf_with_gf_now
            avail_gf -= served_nongf_with_gf_now

            # The remaining needed are the deficits at this place
            deficit_gf_at_p[place] = needed_gf
            deficit_nongf_at_p[place] = needed_nongf

        total_deficit_gf = sum(deficit_gf_at_p.values())
        total_deficit_nongf = sum(deficit_nongf_at_p.values())

        # 6. Count available sandwiches not yet used for immediate service (globally)
        # Total sandwiches on trays anywhere
        total_ontray_gf = sum(1 for s in sandwich_on_tray if sandwich_is_gf.get(s, False))
        total_ontray_nongf = sum(1 for s in sandwich_on_tray if not sandwich_is_gf.get(s, False))

        # Count sandwiches at kitchen
        avail_gf_kitchen = sum(1 for fact_string in state_facts if self._parse_fact(fact_string)[0] == 'at_kitchen_sandwich' and len(self._parse_fact(fact_string)[1]) > 0 and sandwich_is_gf.get(self._parse_fact(fact_string)[1][0], False))
        avail_nongf_kitchen = sum(1 for fact_string in state_facts if self._parse_fact(fact_string)[0] == 'at_kitchen_sandwich' and len(self._parse_fact(fact_string)[1]) > 0 and not sandwich_is_gf.get(self._parse_fact(fact_string)[1][0], False))

        # 6. Count potential new sandwiches.
        avail_bread_counts = collections.Counter()
        avail_content_counts = collections.Counter()
        avail_notexist_sandwich_count = 0

        for fact_string in state_facts:
            pred, objs = self._parse_fact(fact_string)
            if pred == 'at_kitchen_bread':
                if len(objs) > 0:
                    bread_type = objs[0]
                    if bread_type in self.no_gluten_bread_types:
                        avail_bread_counts['gf'] += 1
                    else:
                        avail_bread_counts['nongf'] += 1
            elif pred == 'at_kitchen_content':
                if len(objs) > 0:
                    content_type = objs[0]
                    if content_type in self.no_gluten_content_types:
                        avail_content_counts['gf'] += 1
                    else:
                        avail_content_counts['nongf'] += 1
            elif pred == 'notexist':
                 if len(objs) > 0: avail_notexist_sandwich_count += 1 # Count available notexist objects

        can_make_gf = min(avail_bread_counts['gf'], avail_content_counts['gf'], avail_notexist_sandwich_count)
        can_make_nongf = min(avail_bread_counts['nongf'], avail_content_counts['nongf'], avail_notexist_sandwich_count - can_make_gf)


        # Available pools (globally, not yet satisfying a deficit at the right place)
        # Cost +1: On tray elsewhere (needs move)
        # Sandwiches on trays NOT at a location with demand (these are candidates for moving)
        total_ontray_at_demand_places_gf = sum(avail_gf_ontray_at_p[p] for p in demand_gf_at_p.keys() | demand_nongf_at_p.keys())
        total_ontray_at_demand_places_nongf = sum(avail_nongf_ontray_at_p[p] for p in demand_gf_at_p.keys() | demand_nongf_at_p.keys())

        avail_gf_ontray_elsewhere = total_ontray_gf - total_ontray_at_demand_places_gf
        avail_nongf_ontray_elsewhere = total_ontray_nongf - total_ontray_at_demand_places_nongf


        # 7. Greedily satisfy deficits from cheapest sources
        h = 0
        deficit_gf = total_deficit_gf
        deficit_nongf = total_deficit_nongf

        # Source: On tray elsewhere (GF) - Cost +1
        num = min(deficit_gf, avail_gf_ontray_elsewhere)
        h += num * 1
        deficit_gf -= num
        avail_gf_ontray_elsewhere -= num

        # Source: At kitchen (GF) - Cost +2
        num = min(deficit_gf, avail_gf_kitchen)
        h += num * 2
        deficit_gf -= num
        avail_gf_kitchen -= num

        # Source: Can make (GF) - Cost +3
        num = min(deficit_gf, can_make_gf)
        h += num * 3
        deficit_gf -= num
        can_make_gf -= num

        # Satisfy non-GF deficit
        # Available Any sources (remaining counts after GF deficit is considered)
        avail_any_ontray_elsewhere = avail_gf_ontray_elsewhere + avail_nongf_ontray_elsewhere
        avail_any_kitchen = avail_gf_kitchen + avail_nongf_kitchen
        can_make_any = can_make_gf + can_make_nongf

        # Source: On tray elsewhere (Any remaining) - Cost +1
        num = min(deficit_nongf, avail_any_ontray_elsewhere)
        h += num * 1
        deficit_nongf -= num
        # No need to decrement underlying pools here, just the combined pool

        # Source: At kitchen (Any remaining) - Cost +2
        num = min(deficit_nongf, avail_any_kitchen)
        h += num * 2
        deficit_nongf -= num
        # No need to decrement underlying pools here

        # Source: Can make (Any remaining) - Cost +3
        num = min(deficit_nongf, can_make_any)
        h += num * 3
        deficit_nongf -= num
        # No need to decrement underlying pools here

        # 8. Add the base cost for serving
        h += N_unserved

        # 9. Check for unsolvable state
        if deficit_gf > 0 or deficit_nongf > 0:
             return math.inf # Problem is unsolvable from this state

        return h
