# The base Heuristic class and Task class are expected to be available
# in the environment where this code is run.
# from heuristics.heuristic_base import Heuristic
# from task import Task

import math

class childsnackHeuristic(Heuristic):
    """
    Summary:
    Domain-dependent heuristic for the childsnacks domain.
    It estimates the number of actions required to serve all unserved children.
    The heuristic is based on a relaxed view of the problem where delete
    effects are ignored and resources (sandwiches, trays, components) are
    matched greedily to unserved children based on the "stage" of readiness
    of the sandwich, prioritizing sandwiches that are closer to being served.
    The stages represent the effort needed to get a suitable sandwich onto
    a tray and delivered to the child's location, plus the final serve action.

    Assumptions:
    - The heuristic assumes that once a sandwich is matched to a child's need
      at a certain stage, it contributes that stage's cost to the heuristic.
      It does not track individual sandwich/tray objects beyond counting
      available resources at each stage.
    - It assumes trays at the kitchen are reusable for different sandwiches
      in stages 3 and 4 within the same state evaluation.
    - It assumes components (bread/content) at the kitchen are sufficient
      to make sandwiches up to the counted available amount in stage 4.
    - It assumes 'notexist' sandwich objects are available for making new
      sandwiches in stage 4.
    - It assumes the gluten status of sandwiches is either explicitly stated
      via '(no_gluten_sandwich S)' in the initial state/static facts or
      is unknown (treated as regular).

    Heuristic Initialization:
    The constructor extracts static information from the task description:
    - Which children are allergic to gluten.
    - The waiting place for each child.
    - Which bread types are gluten-free.
    - Which content types are gluten-free.
    - Which initial/static sandwiches are gluten-free.
    It also stores the set of goal facts (all children served).

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify all unserved children from the goal facts and the current state.
    2.  Count the total number of allergic and non-allergic unserved children.
    3.  Initialize the heuristic value `h` to 0.
    4.  If there are no unserved children, return 0.
    5.  Parse available resources from the current state:
        - Sandwiches on trays, their location, and gluten status (derived).
        - Sandwiches at the kitchen and their gluten status (derived).
        - Bread at the kitchen and its gluten status (static).
        - Content at the kitchen and its gluten status (static).
        - Trays at the kitchen.
        - 'notexist' sandwich objects.
    6.  **Stage 1 (Cost 1: Serve):**
        - Count suitable sandwiches already on trays located at the waiting
          place of any remaining unserved child. A sandwich is suitable if
          it's GF for an allergic child or any type for a non-allergic child.
        - Match these available sandwiches to the remaining unserved children
          (allergic first with GF, then non-allergic with remaining GF and Reg).
        - Add (number of children served in this stage) * 1 to `h`.
        - Update the counts of remaining allergic and non-allergic children needed.
        - Track how many GF/Reg sandwiches on trays were conceptually "used"
          in this stage matching to exclude them from Stage 2.
    7.  **Stage 2 (Cost 2: Move + Serve):**
        - Count suitable sandwiches on trays at *any* location that were *not*
          matched in Stage 1.
        - Match these available sandwiches to the remaining unserved children
          (allergic first with GF, then non-allergic with remaining GF and Reg).
        - Add (number of children served in this stage) * 2 to `h`.
        - Update the counts of remaining allergic and non-allergic children needed.
    8.  **Stage 3 (Cost 3: Put + Move + Serve):**
        - Count suitable sandwiches available at the kitchen.
        - Count trays available at the kitchen.
        - Determine how many sandwiches can be put on trays at the kitchen,
          limited by the number of sandwiches and the number of trays.
          Prioritize putting GF sandwiches on trays.
        - Match these available sandwich-on-tray opportunities to the remaining
          unserved children (allergic first with GF, then non-allergic with
          remaining GF and Reg).
        - Add (number of children served in this stage) * 3 to `h`.
        - Update the counts of remaining allergic and non_allergic children needed.
    9.  **Stage 4 (Cost 4: Make + Put + Move + Serve):**
        - Count available suitable bread and content at the kitchen.
        - Count available 'notexist' sandwich objects.
        - Determine how many new suitable sandwiches can be made, limited by
          components and 'notexist' objects. Prioritize making GF sandwiches.
        - Count trays available at the kitchen (trays are reusable).
        - Determine how many new sandwiches can be put on trays at the kitchen,
          limited by the number of makable sandwiches and the number of trays.
          Prioritize putting new GF sandwiches on trays.
        - Match these available new-sandwich-on-tray opportunities to the remaining
          unserved children (allergic first with GF, then non_allergic with
          remaining GF and Reg).
        - Add (number of children served in this stage) * 4 to `h`.
        - Update the counts of remaining allergic and non_allergic children needed.
    10. **Final Check:**
        - If any allergic or non-allergic children still remain unserved after
          considering all stages, it implies insufficient resources (components,
          'notexist' objects, or trays) to serve them from the current state
          through the considered action sequences. Return `math.inf`.
        - Otherwise, return the calculated total heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.goals = task.goals

        # Extract static information
        self.allergic_children = set()
        self.waiting_places = {}  # child -> place
        self.no_gluten_bread_types = set()
        self.no_gluten_content_types = set()
        self.no_gluten_sandwich_types_static = set() # Explicitly marked GF in static

        for fact_str in task.static:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty facts
            predicate = parts[0]
            if predicate == 'allergic_gluten':
                if len(parts) > 1: self.allergic_children.add(parts[1])
            elif predicate == 'waiting':
                if len(parts) > 2: self.waiting_places[parts[1]] = parts[2]
            elif predicate == 'no_gluten_bread':
                if len(parts) > 1: self.no_gluten_bread_types.add(parts[1])
            elif predicate == 'no_gluten_content':
                if len(parts) > 1: self.no_gluten_content_types.add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                 if len(parts) > 1: self.no_gluten_sandwich_types_static.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Parse state facts
        served_children = set()
        sandwiches_on_trays = {}  # sandwich -> tray
        tray_locations = {}  # tray -> place
        sandwiches_at_kitchen = set()
        bread_at_kitchen = set()
        content_at_kitchen = set()
        notexist_sandwiches = set()
        is_sandwich_gf_state = set() # Explicitly marked GF in state

        for fact_str in state:
            parts = fact_str.strip('()').split()
            if not parts: continue # Skip empty facts
            predicate = parts[0]
            if predicate == 'served':
                if len(parts) > 1: served_children.add(parts[1])
            elif predicate == 'ontray':
                if len(parts) > 2: sandwiches_on_trays[parts[1]] = parts[2]
            elif predicate == 'at':
                 if len(parts) > 2: tray_locations[parts[1]] = parts[2]
            elif predicate == 'at_kitchen_sandwich':
                if len(parts) > 1: sandwiches_at_kitchen.add(parts[1])
            elif predicate == 'at_kitchen_bread':
                if len(parts) > 1: bread_at_kitchen.add(parts[1])
            elif predicate == 'at_kitchen_content':
                if len(parts) > 1: content_at_kitchen.add(parts[1])
            elif predicate == 'notexist':
                if len(parts) > 1: notexist_sandwiches.add(parts[1])
            elif predicate == 'no_gluten_sandwich':
                if len(parts) > 1: is_sandwich_gf_state.add(parts[1])

        # Combine GF sandwich info from static and state
        is_sandwich_gf = self.no_gluten_sandwich_types_static | is_sandwich_gf_state

        # Identify unserved children and their needs
        unserved_children_list = [] # List of (child, place, is_allergic)
        allergic_needed = 0
        non_allergic_needed = 0

        # Goals are like '(served child1)'
        for goal_str in self.goals:
            parts = goal_str.strip('()').split()
            if not parts or len(parts) < 2: continue
            child_name = parts[1]
            if child_name not in served_children:
                place = self.waiting_places.get(child_name)
                is_allergic = child_name in self.allergic_children
                unserved_children_list.append((child_name, place, is_allergic))
                if is_allergic:
                    allergic_needed += 1
                else:
                    non_allergic_needed += 1

        if not unserved_children_list:
            return 0

        h = 0

        # --- Count total sandwiches on trays ---
        total_gf_on_tray = 0
        total_reg_on_tray = 0
        for s in sandwiches_on_trays:
             if s in is_sandwich_gf:
                 total_gf_on_tray += 1
             else:
                 total_reg_on_tray += 1

        # --- Stage 1 (Cost 1: Serve) ---
        # Count suitable sandwiches on trays at the correct location for any remaining unserved child
        stage1_gf_at_correct_place = 0
        stage1_reg_at_correct_place = 0
        # Use a set to track sandwich-tray pairs already counted for Stage 1 availability
        correct_place_sandwich_tray_pairs_counted = set()

        for s, t in sandwiches_on_trays.items():
            p = tray_locations.get(t)
            if p is not None:
                 # Check if this location `p` has any unserved children
                 children_at_p = [c for c, cp, ca in unserved_children_list if cp == p]
                 if children_at_p:
                     # Check if sandwich `s` is suitable for any child at `p`
                     s_is_gf = s in is_sandwich_gf
                     is_suitable_for_any_at_p = False
                     for child_name in children_at_p:
                         child_is_allergic = child_name in self.allergic_children
                         if (child_is_allergic and s_is_gf) or (not child_is_allergic):
                             is_suitable_for_any_at_p = True
                             break
                     if is_suitable_for_any_at_p:
                         # Count this specific sandwich-tray pair only once for Stage 1 availability
                         if (s, t) not in correct_place_sandwich_tray_pairs_counted:
                             if s_is_gf:
                                 stage1_gf_at_correct_place += 1
                             else:
                                 stage1_reg_at_correct_place += 1
                             correct_place_sandwich_tray_pairs_counted.add((s, t))


        # Match available stage 1 sandwiches to needs
        served_allergic_stage1 = min(allergic_needed, stage1_gf_at_correct_place)
        allergic_needed -= served_allergic_stage1
        stage1_gf_used_for_allergic = served_allergic_stage1 # Track GF used

        stage1_gf_available_for_non_allergic = stage1_gf_at_correct_place - stage1_gf_used_for_allergic
        served_non_allergic_stage1 = min(non_allergic_needed, stage1_gf_available_for_non_allergic + stage1_reg_at_correct_place)
        non_allergic_needed -= served_non_allergic_stage1

        stage1_gf_used_for_non_allergic = min(served_non_allergic_stage1, stage1_gf_available_for_non_allergic) # Track GF used
        stage1_reg_used_for_non_allergic = served_non_allergic_stage1 - stage1_gf_used_for_non_allergic # Track Reg used

        h += (served_allergic_stage1 + served_non_allergic_stage1) * 1

        # --- Stage 2 (Cost 2: Move + Serve) ---
        # Count suitable sandwiches on trays at *any* location not used in Stage 1 matching
        stage2_gf_available = total_gf_on_tray - stage1_gf_used_for_allergic - stage1_gf_used_for_non_allergic
        stage2_reg_available = total_reg_on_tray - stage1_reg_used_for_non_allergic

        # Match available stage 2 sandwiches to remaining needs
        served_allergic_stage2 = min(allergic_needed, stage2_gf_available)
        allergic_needed -= served_allergic_stage2
        stage2_gf_available -= served_allergic_stage2 # Remaining GF available for non-allergic

        served_non_allergic_stage2 = min(non_allergic_needed, stage2_gf_available + stage2_reg_available)
        non_allergic_needed -= served_non_allergic_stage2

        h += (served_allergic_stage2 + served_non_allergic_stage2) * 2

        # --- Stage 3 (Cost 3: Put + Move + Serve) ---
        # Count suitable sandwiches at kitchen and available trays at kitchen
        stage3_gf_kitchen = sum(1 for s in sandwiches_at_kitchen if s in is_sandwich_gf)
        stage3_reg_kitchen = len(sandwiches_at_kitchen) - stage3_gf_kitchen
        trays_at_kitchen = sum(1 for t, p in tray_locations.items() if p == 'kitchen')

        # Determine how many sandwiches can be put on trays at kitchen, limited by sandwiches and trays.
        # Prioritize putting GF sandwiches on trays for allergic needs.
        available_gf_for_stage3 = min(stage3_gf_kitchen, trays_at_kitchen)
        trays_available_after_gf = trays_at_kitchen - available_gf_for_stage3
        available_reg_for_stage3 = min(stage3_reg_kitchen, trays_available_after_gf)

        # Match available stage 3 sandwich-on-tray opportunities to remaining needs
        served_allergic_stage3 = min(allergic_needed, available_gf_for_stage3)
        allergic_needed -= served_allergic_stage3
        available_gf_for_stage3 -= served_allergic_stage3 # Remaining GF available for non-allergic

        served_non_allergic_stage3 = min(non_allergic_needed, available_gf_for_stage3 + available_reg_for_stage3)
        non_allergic_needed -= served_non_allergic_stage3

        h += (served_allergic_stage3 + served_non_allergic_stage3) * 3

        # --- Stage 4 (Cost 4: Make + Put + Move + Serve) ---
        # Count makable sandwiches and available trays at kitchen
        stage4_gf_bread = sum(1 for b in bread_at_kitchen if b in self.no_gluten_bread_types)
        stage4_reg_bread = len(bread_at_kitchen) - stage4_gf_bread
        stage4_gf_content = sum(1 for c in content_at_kitchen if c in self.no_gluten_content_types)
        stage4_reg_content = len(content_at_kitchen) - stage4_gf_content
        stage4_notexist = len(notexist_sandwiches)
        # Trays at kitchen are reusable, count total again.
        trays_at_kitchen = sum(1 for t, p in tray_locations.items() if p == 'kitchen')

        # Determine how many new suitable sandwiches can be made
        num_new_gf_possible = min(stage4_gf_bread, stage4_gf_content, stage4_notexist)
        num_new_reg_possible = min(stage4_reg_bread, stage4_reg_content, max(0, stage4_notexist - num_new_gf_possible))

        # Determine how many new sandwiches can be put on trays at kitchen, limited by makable and trays.
        # Prioritize putting new GF sandwiches on trays.
        available_new_gf_for_stage4 = min(num_new_gf_possible, trays_at_kitchen)
        trays_available_after_new_gf = trays_at_kitchen - available_new_gf_for_stage4
        available_new_reg_for_stage4 = min(num_new_reg_possible, trays_available_after_new_gf)

        # Match available stage 4 new-sandwich-on-tray opportunities to remaining needs
        served_allergic_stage4 = min(allergic_needed, available_new_gf_for_stage4)
        allergic_needed -= served_allergic_stage4
        available_new_gf_for_stage4 -= served_allergic_stage4 # Remaining GF available for non-allergic

        served_non_allergic_stage4 = min(non_allergic_needed, available_new_gf_for_stage4 + available_new_reg_for_stage4)
        non_allergic_needed -= served_non_allergic_stage4

        h += (served_allergic_stage4 + served_non_allergic_stage4) * 4

        # --- Final Check ---
        # If any children still need serving, resources are insufficient via these stages
        if allergic_needed > 0 or non_allergic_needed > 0:
            return math.inf
        else:
            return h
