import heapq
import logging

from heuristics.heuristic_base import Heuristic
from task import Operator, Task


class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to serve all
    unserved children. It does this by counting the number of unserved
    allergic (needing gluten-free) and non-allergic (needing regular or
    gluten-free) children. It then counts the available sandwiches and
    ingredients in the state, categorized by type (gluten-free/regular)
    and location/status (on tray at child's place, on tray elsewhere,
    in kitchen, makeable from ingredients). The heuristic value is calculated
    by greedily assigning the most "ready" available sandwiches/resources
    to satisfy the needs of the unserved children, summing the estimated
    minimum actions required for each assignment.

    Assumptions:
    - The heuristic assumes a greedy assignment of available sandwiches/resources
      to unserved children, prioritizing resources that are closer to being
      served (e.g., sandwiches already on trays at the child's location are
      used first).
    - The heuristic simplifies the handling of trays. It counts sandwiches
      on trays at child locations or elsewhere, and sandwiches in the kitchen
      or makeable. The cost estimates (1, 2, 3, 4) implicitly assume that
      trays are available at the kitchen when needed for putting sandwiches
      on trays and moving them. A more complex heuristic could explicitly
      track tray availability and location, but this simplified approach
      aims for efficiency while still being domain-aware.
    - The heuristic assumes that any sandwich on a tray not at a child's
      waiting place requires one 'move_tray' action to reach a child's
      place, regardless of its current location (e.g., kitchen or another
      non-child place).

    Heuristic Initialization:
    The constructor processes the static facts from the task description.
    It identifies:
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - The waiting place for each child.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    It also extracts the names of all possible children, trays, sandwiches,
    bread portions, content portions, and places from the task's ground facts
    to facilitate state parsing.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Parse the current state to identify:
        - Which children are already served.
        - Which bread and content portions are in the kitchen.
        - Which sandwiches are in the kitchen.
        - Which sandwiches are on which trays.
        - The location of each tray.
        - Which sandwiches do not yet exist (`notexist`).
        - Which sandwiches are gluten-free.
    2.  Identify the set of unserved children, separating them into allergic
        (needing GF sandwiches) and non-allergic (needing Reg or GF sandwiches).
    3.  If there are no unserved children, the state is a goal state, and the
        heuristic value is 0.
    4.  Count the total number of GF and Reg sandwiches needed (`needed_gf`, `needed_reg`).
    5.  Count the available ingredients (GF/Reg bread and content) in the kitchen.
    6.  Count the number of sandwiches of each type (GF/Reg) that can be made
        given the available ingredients and `notexist` slots (`can_make_gf`, `can_make_reg`).
    7.  Count the available sandwiches of each type (GF/Reg) based on their
        current status/location:
        - On a tray located at a child's waiting place (`avail_ontray_at_place_gf`, `avail_ontray_at_place_reg`).
        - On a tray located elsewhere (not at a child's waiting place) (`avail_ontray_elsewhere_gf`, `avail_ontray_elsewhere_reg`).
        - In the kitchen (`avail_kitchen_sandwich_gf`, `avail_kitchen_sandwich_reg`).
    8.  Initialize the heuristic value `h` to 0.
    9.  Greedily satisfy the needed sandwiches using the available resources,
        starting with the "most ready" and adding the corresponding estimated
        cost to `h`:
        - Satisfy `needed_gf` using `avail_ontray_at_place_gf` (cost 1 per sandwich). Update counts.
        - Satisfy remaining `needed_reg` using remaining `avail_ontray_at_place_reg` and `avail_ontray_at_place_gf` (cost 1 per sandwich). Update counts.
        - Satisfy remaining `needed_gf` using remaining `avail_ontray_elsewhere_gf` (cost 2 per sandwich). Update counts.
        - Satisfy remaining `needed_reg` using remaining `avail_ontray_elsewhere_reg` and `avail_ontray_elsewhere_gf` (cost 2 per sandwich). Update counts.
        - Satisfy remaining `needed_gf` using remaining `avail_kitchen_sandwich_gf` (cost 3 per sandwich). Update counts.
        - Satisfy remaining `needed_reg` using remaining `avail_kitchen_sandwich_reg` and `avail_kitchen_sandwich_gf` (cost 3 per sandwich). Update counts.
        - Satisfy remaining `needed_gf` using remaining `can_make_gf` (cost 4 per sandwich). Update counts.
        - Satisfy remaining `needed_reg` using remaining `can_make_reg` and `can_make_gf` (cost 4 per sandwich). Update counts.
    10. If, after exhausting all available resources, there are still children
        who need sandwiches (`needed_gf > 0` or `needed_reg > 0`), the state
        is considered a dead end or unsolvable from this point with available
        resources, and the heuristic returns infinity (`float('inf')`).
    11. Otherwise, return the calculated total cost `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals # Keep track of goal facts for h=0 check

        # Data structures to store static information
        self.allergic_children = set()
        self.non_allergic_children = set()
        self.child_place_map = {}  # child -> place
        self.gf_bread = set()
        self.gf_content = set()

        # Data structures to store all possible objects by type
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_places = set()
        self.all_bread = set()
        self.all_content = set()

        # Map predicate names to argument indices and their types for object extraction
        pred_arg_types = {
            'at_kitchen_bread': [(0, 'bread-portion')],
            'at_kitchen_content': [(0, 'content-portion')],
            'at_kitchen_sandwich': [(0, 'sandwich')],
            'no_gluten_bread': [(0, 'bread-portion')],
            'no_gluten_content': [(0, 'content-portion')],
            'ontray': [(0, 'sandwich'), (1, 'tray')],
            'no_gluten_sandwich': [(0, 'sandwich')],
            'allergic_gluten': [(0, 'child')],
            'not_allergic_gluten': [(0, 'child')],
            'served': [(0, 'child')],
            'waiting': [(0, 'child'), (1, 'place')],
            'at': [(0, 'tray'), (1, 'place')],
            'notexist': [(0, 'sandwich')],
        }

        # Extract objects by type from all possible ground facts in the task
        for fact_str in task.facts:
            pred, args = self._parse_fact(fact_str)
            if pred in pred_arg_types:
                for idx, obj_type in pred_arg_types[pred]:
                    if idx < len(args):
                        # Map object type string to the corresponding set
                        obj_set = getattr(self, f'all_{obj_type.replace("-", "_")}s', None)
                        if obj_set is not None:
                             obj_set.add(args[idx])
                        # Handle 'place' type separately as it's not pluralized consistently
                        elif obj_type == 'place':
                             self.all_places.add(args[idx])


        # Parse static facts using the extracted objects
        for fact_str in task.static:
            pred, args = self._parse_fact(fact_str)
            if pred == 'allergic_gluten':
                self.allergic_children.add(args[0])
            elif pred == 'not_allergic_gluten':
                self.non_allergic_children.add(args[0])
            elif pred == 'waiting':
                self.child_place_map[args[0]] = args[1]
            elif pred == 'no_gluten_bread':
                self.gf_bread.add(args[0])
            elif pred == 'no_gluten_content':
                self.gf_content.add(args[0])

        # Determine all child waiting places
        self.child_waiting_places = set(self.child_place_map.values())


    def _parse_fact(self, fact_str):
        """Helper to parse a fact string into predicate and arguments."""
        # Remove leading/trailing brackets and split by space
        parts = fact_str.strip()[1:-1].split()
        if not parts:
            return None, [] # Handle empty fact string if necessary
        return parts[0], parts[1:] # predicate, arguments


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        """
        state = node.state

        # Check if goal is reached (heuristic is 0)
        if self.goals <= state:
             return 0

        # --- Parse State Facts ---
        served_children = set()
        kitchen_bread = set()
        kitchen_content = set()
        kitchen_sandwich = set()
        ontray_map = {}  # sandwich -> tray
        tray_location_map = {}  # tray -> place
        notexist_sandwich = set()
        gf_sandwich = set() # Sandwiches that are gluten-free

        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'served':
                served_children.add(args[0])
            elif pred == 'at_kitchen_bread':
                kitchen_bread.add(args[0])
            elif pred == 'at_kitchen_content':
                kitchen_content.add(args[0])
            elif pred == 'at_kitchen_sandwich':
                kitchen_sandwich.add(args[0])
            elif pred == 'ontray':
                if len(args) == 2:
                    ontray_map[args[0]] = args[1]
            elif pred == 'at':
                 if len(args) == 2:
                    tray_location_map[args[0]] = args[1]
            elif pred == 'notexist':
                notexist_sandwich.add(args[0])
            elif pred == 'no_gluten_sandwich':
                gf_sandwich.add(args[0])

        # --- Identify Unserved Children and Needs ---
        unserved_allergic = self.allergic_children - served_children
        unserved_non_allergic = self.non_allergic_children - served_children

        needed_gf = len(unserved_allergic)
        needed_reg = len(unserved_non_allergic)

        # If no children need serving, but goal check above failed, something is wrong.
        # However, the goal is just (and (served child1) ...), so if needed_gf/reg are 0, goal is met.
        # This check is redundant if goal check is done first.

        # --- Count Available Ingredients ---
        num_bread_kitchen_gf = len(kitchen_bread.intersection(self.gf_bread))
        num_bread_kitchen_reg = len(kitchen_bread - self.gf_bread)
        num_content_kitchen_gf = len(kitchen_content.intersection(self.gf_content))
        num_content_kitchen_reg = len(kitchen_content - self.gf_content)
        num_notexist_sandwich_slots = len(notexist_sandwich)

        # --- Count Makeable Sandwiches ---
        # Making a GF sandwich requires GF bread, GF content, and a notexist slot.
        can_make_gf = min(num_bread_kitchen_gf, num_content_kitchen_gf, num_notexist_sandwich_slots)
        remaining_notexist_slots = num_notexist_sandwich_slots - can_make_gf
        # Making a Reg sandwich requires Reg bread, Reg content, and a notexist slot.
        # Note: Domain allows making Reg sandwich from any bread/content if not specified GF.
        # The PDDL shows make_sandwich uses any bread/content, make_sandwich_no_gluten requires GF.
        # So, Reg sandwich can be made from Reg+Reg, Reg+GF, GF+Reg, GF+GF ingredients if not making GF sandwich.
        # Let's assume Reg sandwich uses Reg ingredients if possible, otherwise any available.
        # A simpler approach: count total available ingredients for *any* sandwich.
        # Total bread = num_bread_kitchen_gf + num_bread_kitchen_reg
        # Total content = num_content_kitchen_gf + num_content_kitchen_reg
        # Total makeable sandwiches (any type) = min(Total bread, Total content, num_notexist_sandwich_slots)
        # This doesn't distinguish between makeable GF and Reg accurately based on ingredients.

        # Let's stick to the PDDL: make_sandwich_no_gluten needs GF+GF. make_sandwich needs any+any.
        # If we need a Reg sandwich, we can use make_sandwich. This consumes any bread and any content.
        # The PDDL doesn't restrict make_sandwich based on GF properties of ingredients.
        # So, `make_sandwich` can use any `at_kitchen_bread` and `at_kitchen_content`.
        # Let's refine `can_make_reg`: it uses available bread/content *not* designated as GF, or if those run out, it can use GF ones *if* they are not needed for `can_make_gf`.

        # Simplified approach for makeable:
        # can_make_gf = min(num_bread_kitchen_gf, num_content_kitchen_gf, num_notexist_sandwich_slots)
        # remaining_notexist = num_notexist_sandwich_slots - can_make_gf
        # remaining_bread_gf = num_bread_kitchen_gf - can_make_gf # Assumes GF bread used for GF sandwich first
        # remaining_content_gf = num_content_kitchen_gf - can_make_gf # Assumes GF content used for GF sandwich first
        # remaining_bread_reg = num_bread_kitchen_reg
        # remaining_content_reg = num_content_kitchen_reg
        # total_remaining_bread = remaining_bread_gf + remaining_bread_reg
        # total_remaining_content = remaining_content_gf + remaining_content_reg
        # can_make_reg = min(total_remaining_bread, total_remaining_content, remaining_notexist)

        # Let's use the simpler interpretation first: `make_sandwich` uses *any* bread/content.
        # So, `can_make_reg` is limited by total available bread, total available content, and remaining notexist slots.
        # This seems more aligned with the `make_sandwich` precondition `(and (at_kitchen_bread ?b) (at_kitchen_content ?c) (notexist ?s))`.
        # It doesn't specify *which* bread/content, just *that* they exist in the kitchen.

        can_make_gf = min(num_bread_kitchen_gf, num_content_kitchen_gf, num_notexist_sandwich_slots)
        remaining_notexist = num_notexist_sandwich_slots - can_make_gf
        # For regular sandwiches, we can use any remaining bread and content.
        remaining_bread = num_bread_kitchen_gf + num_bread_kitchen_reg - can_make_gf # Subtract bread used for GF
        remaining_content = num_content_kitchen_gf + num_content_kitchen_reg - can_make_gf # Subtract content used for GF
        can_make_reg = min(remaining_bread, remaining_content, remaining_notexist)


        # --- Count Available Sandwiches by Status ---
        avail_ontray_at_place_gf = 0
        avail_ontray_at_place_reg = 0
        avail_ontray_elsewhere_gf = 0
        avail_ontray_elsewhere_reg = 0
        avail_kitchen_sandwich_gf = 0
        avail_kitchen_sandwich_reg = 0

        # Sandwiches that exist are either in the kitchen or on a tray.
        # We need to consider all possible sandwiches defined in the problem.
        existing_sandwiches = self.all_sandwiches - notexist_sandwich

        for s in existing_sandwiches:
            is_gf = s in gf_sandwich

            if s in kitchen_sandwich:
                if is_gf:
                    avail_kitchen_sandwich_gf += 1
                else:
                    avail_kitchen_sandwich_reg += 1
            elif s in ontray_map:
                t = ontray_map[s]
                place_t = tray_location_map.get(t)
                if place_t is not None:
                    if place_t in self.child_waiting_places:
                        if is_gf:
                            avail_ontray_at_place_gf += 1
                        else:
                            avail_ontray_at_place_reg += 1
                    else: # Tray is at kitchen or some other non-child place
                        if is_gf:
                            avail_ontray_elsewhere_gf += 1
                        else:
                            avail_ontray_elsewhere_reg += 1
                # else: tray location unknown, ignore this sandwich for availability counts?
                # Assuming valid states, trays always have a location.

        # --- Calculate Heuristic by Satisfying Needs ---
        h = 0

        # Cost 1: Serve (sandwich on tray at child place)
        served_gf = min(needed_gf, avail_ontray_at_place_gf)
        h += served_gf * 1
        needed_gf -= served_gf
        avail_ontray_at_place_gf -= served_gf # Used up for GF needs

        served_reg = min(needed_reg, avail_ontray_at_place_reg + avail_ontray_at_place_gf) # Use remaining GF first, then Reg
        h += served_reg * 1
        needed_reg -= served_reg
        # avail_ontray_at_place_reg and remaining avail_ontray_at_place_gf are conceptually used up.

        # Cost 2: Move + Serve (sandwich on tray elsewhere)
        moved_gf = min(needed_gf, avail_ontray_elsewhere_gf)
        h += moved_gf * 2
        needed_gf -= moved_gf
        avail_ontray_elsewhere_gf -= moved_gf # Used up for GF needs

        moved_reg = min(needed_reg, avail_ontray_elsewhere_reg + avail_ontray_elsewhere_gf) # Use remaining GF first, then Reg
        h += moved_reg * 2
        needed_reg -= moved_reg
        # avail_ontray_elsewhere_reg and remaining avail_ontray_elsewhere_gf are conceptually used up.

        # Cost 3: Put + Move + Serve (kitchen sandwich)
        kitchen_gf = min(needed_gf, avail_kitchen_sandwich_gf)
        h += kitchen_gf * 3
        needed_gf -= kitchen_gf
        avail_kitchen_sandwich_gf -= kitchen_gf # Used up for GF needs

        kitchen_reg = min(needed_reg, avail_kitchen_sandwich_reg + avail_kitchen_sandwich_gf) # Use remaining GF first, then Reg
        h += kitchen_reg * 3
        needed_reg -= kitchen_reg
        # avail_kitchen_sandwich_reg and remaining avail_kitchen_sandwich_gf are conceptually used up.

        # Cost 4: Make + Put + Move + Serve (makeable)
        make_gf = min(needed_gf, can_make_gf)
        h += make_gf * 4
        needed_gf -= make_gf
        can_make_gf -= make_gf # Used up for GF needs

        make_reg = min(needed_reg, can_make_reg + can_make_gf) # Use remaining makeable GF first, then Reg
        h += make_reg * 4
        needed_reg -= make_reg
        # can_make_reg and remaining can_make_gf are conceptually used up.

        # --- Check for Impossibility ---
        if needed_gf > 0 or needed_reg > 0:
            return float('inf')

        return h

