import math # For infinity

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class childsnackHeuristic(Heuristic):
    """
    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    in the childsnacks domain. The goal is to serve all children. The heuristic
    breaks down the process into four main stages: making sandwiches, putting
    sandwiches on trays at the kitchen, moving trays to the children's locations,
    and serving the children. It sums the estimated minimum number of actions
    needed for each stage, based on the current state and available resources.
    It also performs resource availability checks at each stage; if the required
    resources (ingredients, sandwich objects, trays, sandwiches at kitchen)
    are insufficient based on the initial state and the output of previous
    stages, the heuristic returns infinity, indicating a likely dead end within
    the heuristic's simplified resource model.

    Assumptions:
    - The heuristic assumes a simplified linear flow of resources:
      notexist_sandwich + ingredients -> at_kitchen_sandwich -> ontray_at_kitchen -> ontray_at_location -> served_child.
    - It assumes trays can hold multiple sandwiches (implicitly, by counting
      sandwiches and trays separately).
    - It assumes trays used for putting sandwiches at the kitchen are the same
      trays that are subsequently moved to locations.
    - It assumes resource counts from earlier stages (e.g., sandwiches made)
      become available for subsequent stages (e.g., putting on trays).
    - It assumes static facts remain true and dynamic facts are correctly
      represented in the state.
    - It assumes 'kitchen' is a constant place and is the origin for trays and sandwiches.
    - The resource checks, particularly for trays at the kitchen, are based on
      a simplified model where the total demand for trays at the kitchen for
      putting sandwiches and moving trays must be met by the initial supply
      of trays at the kitchen. This does not account for tray reuse (returning
      to the kitchen) which is possible in the full domain but not in a simple
      relaxation.

    Heuristic Initialization:
    The constructor parses the static facts from the task description and the
    objects defined in the task to build dictionaries and sets containing
    information that does not change during planning. This includes which
    children are allergic, where children are waiting, which bread/content
    portions are gluten-free, and lists of all objects by type. This static
    information is stored for quick access during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to determine the status of dynamic facts:
        - Identify served children.
        - Count available bread and content at the kitchen.
        - Count sandwiches at the kitchen.
        - Identify sandwiches on trays.
        - Determine the location of each tray.
        - Count available 'notexist' sandwich objects.
        - Identify which sandwiches are currently marked as no-gluten.
    2.  Calculate derived state information:
        - Identify unserved children, count them (`N_unserved`), and count
          how many are allergic (`N_unserved_allergic`).
        - Identify locations where unserved children are waiting
          (`waiting_places_with_unserved`) and count them (`Num_waiting_locations`).
        - Identify trays currently at locations other than the kitchen
          (`trays_at_locations`) and count them (`Num_trays_at_locations`).
    3.  Calculate initial counts of available resources from the state,
        using static info for gluten status where applicable (bread/content).
    4.  Estimate the minimum number of actions needed for each stage, assuming
        resources flow through the pipeline:
        - **Stage 1 (Make Sandwiches):** Estimate the number of sandwiches that
          need to be made (`cost_make`). This is the number of unserved children
          minus the number of sandwiches already available (at kitchen or on tray),
          prioritizing no-gluten sandwiches for allergic children.
        - **Stage 2 (Put on Tray):** Estimate the number of sandwiches that need
          to be moved from the kitchen onto trays (`cost_put`). This is the total
          unserved children minus those already on trays (anywhere).
        - **Stage 3 (Move Tray):** Estimate the number of trays that need to be
          moved from the kitchen to locations where children are waiting (`cost_move`).
          This is the number of waiting locations minus the number of trays already
          at locations.
        - **Stage 4 (Serve Children):** Estimate the number of children that need
          to be served (`cost_serve`). This is simply the number of unserved children.
    5.  Perform resource availability checks based on the estimated costs and
        initial resources, considering the flow between stages:
        - Check if enough bread (gluten-free and any), content (gluten-free and any),
          and 'notexist' sandwich objects are available at the kitchen to make
          `cost_make` sandwiches, ensuring the required number are gluten-free.
          If not, return infinity.
        - Calculate the total number of sandwiches available at the kitchen after
          making the necessary ones. Check if this count is sufficient for the
          `cost_put` actions. If not, return infinity.
        - Check if enough trays are available at the kitchen (initially) to support
          both the `cost_put` actions (as trays are needed to put sandwiches on)
          and the `cost_move` actions (as trays are moved from the kitchen). The
          heuristic assumes the trays used for putting are the ones moved, and
          the total demand on trays at the kitchen is the sum of `cost_put` and
          `cost_move`. If not enough trays are available at the kitchen, return infinity.
        - Check if the total number of trays is sufficient to have one at each
          location where children are waiting. If not, return infinity.
    6.  If all resource checks pass, the heuristic value is the sum of the
        estimated costs for the four stages: `cost_make + cost_put + cost_move + cost_serve`.
    7.  If any resource check failed, return `math.inf`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self._parse_static_info(task.static, task.objects)

    def _parse_static_info(self, static_facts, task_objects):
        self.child_is_allergic = {}
        self.child_waiting_place = {}
        self.bread_is_no_gluten = {}
        self.content_is_no_gluten = {}

        # Get all objects from task definition (most reliable source)
        self.all_children = set(task_objects.get('child', []))
        self.all_trays = set(task_objects.get('tray', []))
        self.all_sandwiches = set(task_objects.get('sandwich', []))
        self.all_bread = set(task_objects.get('bread-portion', []))
        self.all_content = set(task_objects.get('content-portion', []))
        self.all_places = set(task_objects.get('place', []))
        self.all_places.discard('kitchen') # kitchen is a constant, not usually in :objects

        for fact_str in static_facts:
            predicate, objects = self._parse_fact(fact_str)
            if predicate == 'allergic_gluten':
                child = objects[0]
                self.child_is_allergic[child] = True
            elif predicate == 'not_allergic_gluten':
                child = objects[0]
                self.child_is_allergic[child] = False
            elif predicate == 'waiting':
                child, place = objects
                self.child_waiting_place[child] = place
            elif predicate == 'no_gluten_bread':
                bread = objects[0]
                self.bread_is_no_gluten[bread] = True
            elif predicate == 'no_gluten_content':
                content = objects[0]
                self.content_is_no_gluten[content] = True

        # Ensure all children mentioned in static are in all_children (if not listed in :objects)
        self.all_children.update(self.child_is_allergic.keys())
        self.all_children.update(self.child_waiting_place.keys())

        # Default NG status to False if not specified in static
        for bread in self.all_bread:
            self.bread_is_no_gluten.setdefault(bread, False)
        for content in self.all_content:
            self.content_is_no_gluten.setdefault(content, False)


    def _parse_fact(self, fact_str):
        # Removes leading/trailing parens and splits by space
        parts = fact_str[1:-1].split()
        predicate = parts[0]
        objects = parts[1:]
        return predicate, objects

    def __call__(self, node):
        state = node.state

        # --- Parse State ---
        served_children = set()
        at_kitchen_bread = set()
        at_kitchen_content = set()
        at_kitchen_sandwich = set()
        ontray_sandwiches_by_tray = {} # tray -> set of sandwich objects
        sandwich_is_no_gluten = set() # dynamic status
        tray_location = {} # tray -> place
        notexist_sandwiches = set()

        for fact_str in state:
            predicate, objects = self._parse_fact(fact_str)
            if predicate == 'served':
                served_children.add(objects[0])
            elif predicate == 'at_kitchen_bread':
                at_kitchen_bread.add(objects[0])
            elif predicate == 'at_kitchen_content':
                at_kitchen_content.add(objects[0])
            elif predicate == 'at_kitchen_sandwich':
                at_kitchen_sandwich.add(objects[0])
            elif predicate == 'ontray':
                sandwich, tray = objects
                ontray_sandwiches_by_tray.setdefault(tray, set()).add(sandwich)
            elif predicate == 'no_gluten_sandwich':
                sandwich_is_no_gluten.add(objects[0])
            elif predicate == 'at':
                tray, place = objects
                tray_location[tray] = place
            elif predicate == 'notexist':
                notexist_sandwiches.add(objects[0])

        # --- Calculate Derived State Info ---
        unserved_children = self.all_children - served_children
        N_unserved = len(unserved_children)

        if N_unserved == 0:
            return 0 # Goal reached

        N_unserved_allergic = sum(1 for c in unserved_children if self.child_is_allergic.get(c, False))

        waiting_places_with_unserved = set()
        for child in unserved_children:
             if child in self.child_waiting_place:
                 waiting_places_with_unserved.add(self.child_waiting_place[child])
        Num_waiting_locations = len(waiting_places_with_unserved)

        trays_at_locations = {t for t, p in tray_location.items() if p != 'kitchen'}
        Num_trays_at_locations = len(trays_at_locations)

        # --- Calculate Available Resources (Initial) ---
        # Sandwiches available anywhere (at kitchen or on tray)
        avail_ng_init = len([s for s in at_kitchen_sandwich if s in sandwich_is_no_gluten]) + \
                        sum(len([s for s in sandwiches if s in sandwich_is_no_gluten]) for sandwiches in ontray_sandwiches_by_tray.values())
        avail_any_init = len(at_kitchen_sandwich) + sum(len(sandwiches) for sandwiches in ontray_sandwiches_by_tray.values())

        # Sandwiches available specifically on trays (any location)
        avail_ontray_init = sum(len(sandwiches) for sandwiches in ontray_sandwiches_by_tray.values())

        # Sandwiches available specifically at kitchen
        avail_kitchen_init = len(at_kitchen_sandwich)

        # Ingredients at kitchen
        avail_bread_ng_kitchen_init = len([b for b in at_kitchen_bread if self.bread_is_no_gluten.get(b, False)])
        avail_bread_any_kitchen_init = len(at_kitchen_bread)
        avail_content_ng_kitchen_init = len([c for c in at_kitchen_content if self.content_is_no_gluten.get(c, False)])
        avail_content_any_kitchen_init = len(at_kitchen_content)

        # Notexist sandwich objects
        avail_notexist_init = len(notexist_sandwiches)

        # Trays
        avail_trays_kitchen_init = len([t for t, p in tray_location.items() if p == 'kitchen'])
        avail_trays_total_init = len(self.all_trays)


        # --- Estimate Costs for Stages ---

        # Stage 1: Make Sandwiches
        # Number of NG sandwiches that MUST be made
        make_ng = max(0, N_unserved_allergic - avail_ng_init)
        # Number of ANY sandwiches needed in total (including NG), minus those already available
        needed_any_total = max(0, N_unserved - avail_any_init)
        # Number of additional ANY sandwiches to make after making the necessary NG ones
        make_any_additional = max(0, needed_any_total - make_ng)
        # Total sandwiches to make
        cost_make = make_ng + make_any_additional

        # Stage 2: Put on Tray (at kitchen)
        # Number of sandwiches that need to transition from 'at_kitchen_sandwich' to 'ontray'
        cost_put = max(0, N_unserved - avail_ontray_init)

        # Stage 3: Move Trays to Locations
        # Number of trays that need to transition from 'at kitchen' to 'at location'
        cost_move = max(0, Num_waiting_locations - Num_trays_at_locations)

        # Stage 4: Serve Children
        # Number of children that need to transition from 'waiting' to 'served'
        cost_serve = N_unserved

        # --- Resource Availability Checks ---
        is_possible = True

        # Check resources for Stage 1 (Make)
        # Enough NG ingredients for NG sandwiches
        if make_ng > avail_bread_ng_kitchen_init: is_possible = False
        if make_ng > avail_content_ng_kitchen_init: is_possible = False
        # Enough Any ingredients for all sandwiches to be made
        if cost_make > avail_bread_any_kitchen_init: is_possible = False
        if cost_make > avail_content_any_kitchen_init: is_possible = False
        # Enough notexist objects
        if cost_make > avail_notexist_init: is_possible = False

        # Calculate sandwiches available at kitchen after making
        avail_kitchen_after_make = avail_kitchen_init + cost_make

        # Check resources for Stage 2 (Put)
        # Enough sandwiches at kitchen to be put on trays
        if cost_put > avail_kitchen_after_make: is_possible = False
        # Enough trays at kitchen for putting sandwiches on.
        # These trays are the same ones that will be moved.
        # The total demand on trays at kitchen is the sum of trays needed for putting
        # and trays needed for moving, as these are distinct actions consuming
        # the 'at tray kitchen' predicate in a relaxed world.
        trays_needed_at_kitchen_for_put_and_move = cost_put + cost_move
        if trays_needed_at_kitchen_for_put_and_move > avail_trays_kitchen_init: is_possible = False


        # Check resources for Stage 3 (Move)
        # Enough total trays to cover all waiting locations
        if Num_waiting_locations > avail_trays_total_init: is_possible = False
        # Enough trays at kitchen to be moved (This check is implicitly covered
        # by the combined trays_needed_at_kitchen_for_put_and_move check above,
        # as cost_move is part of the sum).


        if not is_possible:
            return math.inf

        # --- Calculate Total Heuristic ---
        h_value = cost_make + cost_put + cost_move + cost_serve

        return h_value
