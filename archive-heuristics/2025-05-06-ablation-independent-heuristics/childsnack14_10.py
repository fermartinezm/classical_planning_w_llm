from heuristics.heuristic_base import Heuristic
from task import Task
from collections import defaultdict

class childsnackHeuristic(Heuristic):
    """
    Summary:
    Domain-dependent heuristic for the childsnacks domain.
    Estimates the minimum number of actions required to serve all waiting children.
    It calculates the cost for each unserved child based on the cheapest available
    source of an appropriate sandwich (considering allergy status) and the steps
    needed to get that sandwich on a tray to the child's location. Resources
    (sandwiches, ingredients, trays) are allocated greedily, prioritizing
    gluten-free needs and cheaper resource sources.

    Assumptions:
    - The heuristic assumes action costs are 1.
    - It assumes the structure of PDDL facts allows parsing object names by splitting
      the string and removing parentheses, and that object types can be inferred
      from the predicate name and argument position based on the domain definition.
    - It assumes 'kitchen' is the constant name for the kitchen place.
    - It assumes trays are needed for putting sandwiches on them (from kitchen)
      and for moving sandwiches to children's locations. Trays already at a
      child's location or elsewhere with a sandwich on them are assumed to
      stay with that sandwich until served. Trays in the kitchen are available
      for new sandwiches or sandwiches in the kitchen.

    Heuristic Initialization:
    The constructor parses the static facts from the task to identify:
    - Allergic and non-allergic children.
    - Gluten-free bread and content portions.
    - The waiting place for each child.
    It also collects the names of all objects (children, bread, content, sandwiches,
    trays, places) present in the initial state, static facts, and goal facts
    to know the full set of objects in the problem instance.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify unserved allergic and non-allergic children. These represent the goals.
    2.  Identify available resources in the current state:
        - Sandwiches on trays at places where unserved children are waiting (separated by GF/Reg).
        - Sandwiches on trays at other places (separated by GF/Reg).
        - Sandwiches in the kitchen (separated by GF/Reg).
        - Available ingredient pairs in the kitchen (GF bread+content, Any bread+content).
        - Trays in the kitchen.
    3.  Categorize available sandwich/ingredient resources by the minimum actions needed
        to get them onto a tray and to a child's location, plus the final serve action:
        - Cost 1: Sandwich is on a tray already at a waiting child's location. (Serve)
        - Cost 2: Sandwich is on a tray elsewhere. (Move tray + Serve)
        - Cost 3: Sandwich is in the kitchen. (Put on tray + Move tray + Serve). Requires a tray from the kitchen.
        - Cost 4: Sandwich needs to be made from ingredients in the kitchen. (Make + Put on tray + Move tray + Serve). Requires ingredients and a tray from the kitchen.
    4.  Count the number of resources available at each cost level, distinguishing
        between Gluten-Free (GF) and Any (Reg) sandwiches/ingredients. Also count
        available trays in the kitchen needed for Cost 3 and 4 actions.
    5.  Initialize heuristic value `h = 0`.
    6.  Greedily allocate resources to satisfy the needs of unserved allergic (GF) children first:
        - For each GF child need, try to use the cheapest available GF resource (Cost 1, then 2, then 3, then 4).
        - If a Cost 3 or 4 resource is used, consume one available tray from the kitchen pool.
        - Add the cost (1, 2, 3, or 4) to `h` for each child served this way.
        - If GF needs remain after exhausting all GF resources, the state is likely a dead end or very far, return infinity.
    7.  Greedily allocate remaining resources to satisfy the needs of unserved non-allergic (Reg) children:
        - For each Reg child need, try to use the cheapest available *any* remaining resource (Cost 1, then 2, then 3, then 4). Note that remaining GF resources can satisfy Reg needs.
        - If a Cost 3 or 4 resource is used, consume one available tray from the kitchen pool (using the pool remaining after GF allocation).
        - Add the cost (1, 2, 3, or 4) to `h` for each child served this way.
        - If Reg needs remain after exhausting all remaining resources, return infinity.
    8.  Return the total calculated heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self.task = task
        self.kitchen_place = 'kitchen' # Assuming 'kitchen' is the constant name

        # --- Parse Static Information and Collect All Objects ---
        self.all_children = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = {self.kitchen_place} # Start with kitchen constant

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_bread_names = set()
        self.no_gluten_content_names = set()
        self.child_waiting_place = {} # child -> place

        # Collect objects and static predicates from static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'allergic_gluten':
                if args: self.allergic_children.add(args[0])
            elif pred == 'not_allergic_gluten':
                if args: self.not_allergic_children.add(args[0])
            elif pred == 'no_gluten_bread':
                if args: self.no_gluten_bread_names.add(args[0])
            elif pred == 'no_gluten_content':
                if args: self.no_gluten_content_names.add(args[0])
            elif pred == 'waiting':
                if len(args) > 1: self.child_waiting_place[args[0]] = args[1]

        # Collect all object names from all facts in the task (initial, goals, all possible facts)
        all_facts_set = set(task.facts) | set(task.initial_state) | set(task.goals)

        for fact in all_facts_set:
             parts = fact.strip('()').split()
             if not parts: continue
             pred = parts[0]
             args = parts[1:]

             # Infer object types based on predicate argument positions from domain
             if pred in ['at_kitchen_bread', 'no_gluten_bread']:
                 if args: self.all_bread.add(args[0])
             elif pred in ['at_kitchen_content', 'no_gluten_content']:
                 if args: self.all_content.add(args[0])
             elif pred in ['at_kitchen_sandwich', 'ontray', 'no_gluten_sandwich', 'notexist']:
                 if args: self.all_sandwiches.add(args[0])
             elif pred in ['ontray', 'at']:
                 if pred == 'ontray' and len(args) > 1: self.all_trays.add(args[1])
                 if pred == 'at' and len(args) > 0: self.all_trays.add(args[0])
             elif pred in ['allergic_gluten', 'not_allergic_gluten', 'served', 'waiting']:
                 if args: self.all_children.add(args[0])
                 if pred == 'waiting' and len(args) > 1: self.all_places.add(args[1])
             elif pred == 'at':
                 if len(args) > 1: self.all_places.add(args[1])


    def __call__(self, node):
        state = node.state

        # --- Parse Dynamic Information from State ---
        served_children_state = set()
        trays_at_place_state = defaultdict(set) # place -> {tray}
        sandwiches_on_tray_state = defaultdict(set) # tray -> {sandwich}
        sandwiches_kitchen_state = set() # {sandwich}
        bread_kitchen_state = set() # {bread}
        content_kitchen_state = set() # {content}
        no_gluten_sandwich_names_state = set() # {sandwich}

        for fact in state:
            parts = fact.strip('()').split()
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'served':
                if args: served_children_state.add(args[0])
            elif pred == 'at':
                if len(args) > 1: trays_at_place_state[args[1]].add(args[0])
            elif pred == 'ontray':
                if len(args) > 1: sandwiches_on_tray_state[args[1]].add(args[0])
            elif pred == 'at_kitchen_sandwich':
                if args: sandwiches_kitchen_state.add(args[0])
            elif pred == 'at_kitchen_bread':
                if args: bread_kitchen_state.add(args[0])
            elif pred == 'at_kitchen_content':
                if args: content_kitchen_state.add(args[0])
            elif pred == 'no_gluten_sandwich':
                if args: no_gluten_sandwich_names_state.add(args[0])

        # --- Calculate Needs ---
        unserved_gf_children = [c for c in self.allergic_children if c not in served_children_state]
        unserved_reg_children = [c for c in self.not_allergic_children if c not in served_children_state]

        if not unserved_gf_children and not unserved_reg_children:
            return 0 # Goal state

        # Places where unserved children are waiting
        waiting_places = {self.child_waiting_place[c] for c in unserved_gf_children + unserved_reg_children if c in self.child_waiting_place}


        # --- Calculate Available Resources by Cost and Type ---

        # Sandwiches currently on any tray
        sandwiches_on_trays_state = {s for tray, ss in sandwiches_on_tray_state.items() for s in ss}
        gf_sandwiches_on_trays_state = sandwiches_on_trays_state & no_gluten_sandwich_names_state
        reg_sandwiches_on_trays_state = sandwiches_on_trays_state - no_gluten_sandwich_names_state

        # Cost 1: Sandwich on tray at a waiting place
        # A sandwich is C1 if it's on a tray located at *any* place where an unserved child is waiting.
        gf_on_tray_at_waiting_place_set = {s for s in gf_sandwiches_on_trays_state
                                           if any(t in trays_at_place_state[p]
                                                  for t in sandwiches_on_tray_state if s in sandwiches_on_tray_state[t]
                                                  for p in waiting_places)}
        reg_on_tray_at_waiting_place_set = {s for s in reg_sandwiches_on_trays_state
                                           if any(t in trays_at_place_state[p]
                                                  for t in sandwiches_on_tray_state if s in sandwiches_on_tray_state[t]
                                                  for p in waiting_places)}

        num_c1_gf = len(gf_on_tray_at_waiting_place_set)
        num_c1_reg = len(reg_on_tray_at_waiting_place_set)

        # Cost 2: Sandwich on tray elsewhere (not at any waiting place)
        gf_on_tray_elsewhere_set = gf_sandwiches_on_trays_state - gf_on_tray_at_waiting_place_set
        reg_on_tray_elsewhere_set = reg_sandwiches_on_trays_state - reg_on_tray_at_waiting_place_set

        num_c2_gf = len(gf_on_tray_elsewhere_set)
        num_c2_reg = len(reg_on_tray_elsewhere_set)

        # Cost 3: Sandwich in kitchen
        gf_kitchen_sandwiches_state = sandwiches_kitchen_state & no_gluten_sandwich_names_state
        reg_kitchen_sandwiches_state = sandwiches_kitchen_state - no_gluten_sandwich_names_state

        num_c3_gf = len(gf_kitchen_sandwiches_state)
        num_c3_reg = len(reg_kitchen_sandwiches_state)

        # Cost 4: Ingredients in kitchen
        gf_bread_kitchen_state = bread_kitchen_state & self.no_gluten_bread_names
        gf_content_kitchen_state = content_kitchen_state & self.no_gluten_content_names
        # Any bread + any content for regular sandwiches
        any_bread_kitchen_state = bread_kitchen_state
        any_content_kitchen_state = content_kitchen_state

        num_c4_gf_ing = min(len(gf_bread_kitchen_state), len(gf_content_kitchen_state))
        num_c4_any_ing = min(len(any_bread_kitchen_state), len(any_content_kitchen_state))

        # Trays in kitchen (needed for Cost 3 and 4)
        trays_kitchen_state = trays_at_place_state[self.kitchen_place]
        num_trays_kitchen = len(trays_kitchen_state)

        # Store initial counts for Reg allocation step
        init_num_c1_gf = num_c1_gf
        init_num_c1_reg = num_c1_reg
        init_num_c2_gf = num_c2_gf
        init_num_c2_reg = num_c2_reg
        init_num_c3_gf = num_c3_gf
        init_num_c3_reg = num_c3_reg
        init_num_c4_gf_ing = num_c4_gf_ing
        init_num_c4_reg_ing = num_c4_any_ing # Use any ingredients for initial reg count
        init_num_trays_kitchen = num_trays_kitchen

        # --- Greedy Allocation ---
        h = 0
        needs_gf = len(unserved_gf_children)
        needs_reg = len(unserved_reg_children)

        # Allocate for GF needs
        use = min(needs_gf, num_c1_gf)
        h += use * 1
        needs_gf -= use
        num_c1_gf -= use # Consume GF resource from C1

        use = min(needs_gf, num_c2_gf)
        h += use * 2
        needs_gf -= use
        num_c2_gf -= use # Consume GF resource from C2

        use = min(needs_gf, num_c3_gf, num_trays_kitchen)
        h += use * 3
        needs_gf -= use
        num_c3_gf -= use # Consume GF sandwich from C3
        num_trays_kitchen -= use # Consume tray

        use = min(needs_gf, num_c4_gf_ing, num_trays_kitchen)
        h += use * 4
        needs_gf -= use
        num_c4_gf_ing -= use # Consume GF ingredients
        num_trays_kitchen -= use # Consume tray

        if needs_gf > 0:
            return float('inf') # Cannot satisfy all GF needs

        # Allocate for Reg needs (using remaining resources)
        # Remaining 'Any' resources are the sum of remaining GF and initial Reg resources
        rem_num_c1_any = num_c1_gf + init_num_c1_reg
        rem_num_c2_any = num_c2_gf + init_num_c2_reg
        rem_num_c3_any = num_c3_gf + init_num_c3_reg
        rem_num_c4_any_ing = num_c4_gf_ing + init_num_c4_reg_ing
        rem_num_trays_kitchen = num_trays_kitchen # Trays kitchen remaining after GF allocation

        use = min(needs_reg, rem_num_c1_any)
        h += use * 1
        needs_reg -= use
        rem_num_c1_any -= use

        use = min(needs_reg, rem_num_c2_any)
        h += use * 2
        needs_reg -= use
        rem_num_c2_any -= use

        use = min(needs_reg, rem_num_c3_any, rem_num_trays_kitchen)
        h += use * 3
        needs_reg -= use
        rem_num_c3_any -= use
        rem_num_trays_kitchen -= use

        use = min(needs_reg, rem_num_c4_any_ing, rem_num_trays_kitchen)
        h += use * 4
        needs_reg -= use
        rem_num_c4_any_ing -= use
        rem_trays_kitchen -= use

        if needs_reg > 0:
            return float('inf') # Cannot satisfy all Reg needs

        return h
