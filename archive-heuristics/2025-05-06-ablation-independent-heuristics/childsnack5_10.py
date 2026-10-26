from heuristics.heuristic_base import Heuristic
from task import Task
import logging

# Configure logging for debugging if needed
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to serve all children.
    It calculates the deficit of suitable sandwiches at different stages
    of the delivery pipeline (on tray at child's location, on remote tray,
    in kitchen not on tray, needing to be made) and assigns a cost based on the minimum
    actions required to move a sandwich from that stage to being served:
    - Stage 1 (Cost 1): Sandwich is on a tray at the child's waiting location. Needs 1 'serve' action.
    - Stage 2 (Cost 2): Sandwich is on a tray but not at the child's waiting location (includes kitchen). Needs 1 'move_tray' + 1 'serve' action.
    - Stage 3 (Cost 3): Sandwich is in the kitchen, not on a tray. Needs 1 'put_on_tray' + 1 'move_tray' + 1 'serve' action.
    - Stage 4 (Cost 4): Sandwich needs to be made. Needs 1 'make_sandwich' + 1 'put_on_tray' + 1 'move_tray' + 1 'serve' action.

    It greedily satisfies the needs of unserved children using sandwiches from the
    "closest" stages first (Stage 1, then Stage 2, Stage 3, Stage 4).

    Assumptions:
    - The problem instance is solvable, implying sufficient initial resources
      (bread, content, notexist sandwich objects) to make all necessary sandwiches
      over the course of the plan. The heuristic assumes that if a sandwich is needed
      and resources are available in the current state, it can be made.
    - Tray capacity is effectively infinite (or sufficient).
    - A tray can be moved between any two places.
    - Children stay at their waiting locations.
    - Gluten status of bread, content, and children is static.
    - The heuristic is not admissible, but aims to be informative for greedy search.

    Heuristic Initialization:
    - Parses static facts from the task description to determine:
        - Allergy status for each child (`child_allergy`).
        - Waiting location for each child (`child_location`).
        - Gluten status for each bread type (`is_no_gluten_bread`).
        - Gluten status for each content type (`is_no_no_gluten_content`).
    - Identifies all children and all relevant places (kitchen and waiting locations).
    - Stores the set of goal facts (all children served).

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize heuristic value `h = 0`.
    2. Parse the current state to identify all true facts and build mappings/counts
       of available resources and sandwiches by type (GF/Regular) and status/location:
       - Available bread and content in the kitchen by type.
       - Available `notexist` sandwich objects.
       - Sandwiches in the kitchen not yet on a tray.
       - Sandwiches on trays, noting their location.
       - Gluten status of existing sandwiches.
    3. Identify all unserved children based on the goal facts and current state.
    4. If no children are unserved, the state is a goal state, return 0.
    5. Count the total number of GF and Regular sandwiches needed by unserved children.
    6. Calculate the heuristic cost by iterating through the stages (1 to 4),
       greedily satisfying the remaining needs at each stage:
       - **Stage 1 (Cost 1):** For each location where children are waiting, count how many GF/Reg children can be served by suitable sandwiches already on trays at that location. Add `count * 1` to `h` and reduce the remaining needed counts.
       - **Stage 2 (Cost 2):** Count the total number of remaining needed GF/Reg sandwiches that are on trays but not at the child's waiting location (i.e., on any tray minus those used in Stage 1). Add `count * 2` to `h` and reduce remaining needed counts.
       - **Stage 3 (Cost 3):** Count the total number of remaining needed GF/Reg sandwiches that are in the kitchen and not on a tray. Add `count * 3` to `h` and reduce remaining needed counts.
       - **Stage 4 (Cost 4):** Count the total number of remaining needed GF/Reg sandwiches that can be made from available resources (`at_kitchen_bread`, `at_kitchen_content`, `notexist`). Prioritize making GF sandwiches if needed, considering the shared `notexist` resource. Add `count * 4` to `h` and reduce remaining needed counts.
    7. If, after considering all available and potential sandwiches across all stages,
       there are still children needing sandwiches, it implies the state is likely
       unsolvable from this point with the current resources. Return `float('inf')`.
    8. Otherwise, return the calculated heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.child_allergy = {}
        self.child_location = {}
        self.is_no_gluten_bread = {}
        self.is_no_gluten_content = {}
        self.all_children = set()
        self.all_places = {'kitchen'} # Start with kitchen, add others from waiting facts

        # Parse static facts
        for fact_str in task.static:
            parts = self._parse_fact(fact_str)
            if not parts:
                continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = True
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten':
                child = parts[1]
                self.child_allergy[child] = False
                self.all_children.add(child)
            elif predicate == 'waiting':
                child = parts[1]
                place = parts[2]
                self.child_location[child] = place
                self.all_places.add(place)
            elif predicate == 'no_gluten_bread':
                bread = parts[1]
                self.is_no_gluten_bread[bread] = True
            elif predicate == 'no_gluten_content':
                content = parts[1]
                self.is_no_gluten_content[content] = True

        # logger.info(f"Initialized heuristic with {len(self.all_children)} children and {len(self.all_places)} places.")
        # logger.info(f"Child allergy: {self.child_allergy}")
        # logger.info(f"Child location: {self.child_location}")


    def _parse_fact(self, fact_str):
        """Helper to parse a fact string like '(predicate arg1 arg2)'."""
        # Remove surrounding parentheses and split by space
        parts = fact_str.strip('()').split()
        if not parts:
            return None
        return parts

    def __call__(self, node):
        state = node.state
        state_facts = set(state)

        # 1. Check if goal is reached
        unserved_children = {c for c in self.all_children if f'(served {c})' not in state_facts}
        if not unserved_children:
            # logger.debug("Goal reached, heuristic = 0")
            return 0

        # 2. Count needed sandwiches by type
        total_gf_needed = sum(1 for c in unserved_children if self.child_allergy.get(c, False))
        total_reg_needed = sum(1 for c in unserved_children if not self.child_allergy.get(c, False))

        # logger.debug(f"Unserved children: {len(unserved_children)}. GF needed: {total_gf_needed}, Reg needed: {total_reg_needed}")

        # 3. Count available sandwiches and resources in current state
        available_gf_bread_kitchen = 0
        available_reg_bread_kitchen = 0
        available_gf_content_kitchen = 0
        available_reg_content_kitchen = 0
        available_notexist_sandwiches = 0

        at_kitchen_sandwich_gf_set = set()
        at_kitchen_sandwich_reg_set = set()
        sandwiches_on_trays_gf_set = set()
        sandwiches_on_trays_reg_set = set()

        tray_locations = {} # Map tray object to place object
        sandwich_tray_map = {} # Map sandwich object to tray object
        sandwich_is_gf = {} # Map sandwich object name to True/False

        # Single pass through state facts to populate initial structures
        for fact_str in state_facts:
            parts = self._parse_fact(fact_str)
            if not parts: continue
            predicate = parts[0]

            if predicate == 'no_gluten_sandwich':
                sandwich_is_gf[parts[1]] = True
            elif predicate == 'at_kitchen_bread':
                bread = parts[1]
                if self.is_no_gluten_bread.get(bread, False):
                    available_gf_bread_kitchen += 1
                else:
                    available_reg_bread_kitchen += 1
            elif predicate == 'at_kitchen_content':
                content = parts[1]
                if self.is_no_gluten_content.get(content, False):
                    available_gf_content_kitchen += 1
                else:
                    available_reg_content_kitchen += 1
            elif predicate == 'notexist':
                available_notexist_sandwiches += 1
            elif predicate == 'at_kitchen_sandwich':
                sandwich = parts[1]
                is_gf = sandwich_is_gf.get(sandwich, False) # Default to False if not explicitly marked GF
                if is_gf:
                    at_kitchen_sandwich_gf_set.add(sandwich)
                else:
                    at_kitchen_sandwich_reg_set.add(sandwich)
            elif predicate == 'ontray':
                sandwich = parts[1]
                tray = parts[2]
                sandwich_tray_map[sandwich] = tray
                is_gf = sandwich_is_gf.get(sandwich, False) # Default to False
                if is_gf:
                    sandwiches_on_trays_gf_set.add(sandwich)
                else:
                    sandwiches_on_trays_reg_set.add(sandwich)
            elif predicate == 'at':
                tray = parts[1]
                place = parts[2]
                tray_locations[tray] = place
                self.all_places.add(place) # Add any place a tray is located at

        # Calculate counts based on parsed data
        sandwiches_gf_in_kitchen_not_on_tray = len(at_kitchen_sandwich_gf_set - sandwiches_on_trays_gf_set)
        sandwiches_reg_in_kitchen_not_on_tray = len(at_kitchen_sandwich_reg_set - sandwiches_on_trays_reg_set)

        sandwiches_gf_on_tray_at_loc = {p: 0 for p in self.all_places}
        sandwiches_reg_on_tray_at_loc = {p: 0 for p in self.all_places}

        for sandwich in sandwiches_on_trays_gf_set:
            tray = sandwich_tray_map.get(sandwich)
            if tray:
                location = tray_locations.get(tray)
                if location in self.all_places: # Only count if tray location is known and relevant
                    sandwiches_gf_on_tray_at_loc[location] += 1

        for sandwich in sandwiches_on_trays_reg_set:
            tray = sandwich_tray_map.get(sandwich)
            if tray:
                location = tray_locations.get(tray)
                if location in self.all_places: # Only count if tray location is known and relevant
                    sandwiches_reg_on_tray_at_loc[location] += 1

        sandwiches_gf_on_any_tray = len(sandwiches_on_trays_gf_set)
        sandwiches_reg_on_any_tray = len(sandwiches_on_trays_reg_set)

        # logger.debug(f"Resources: GF Bread={available_gf_bread_kitchen}, Reg Bread={available_reg_bread_kitchen}, GF Content={available_gf_content_kitchen}, Reg Content={available_reg_content_kitchen}, Notexist={available_notexist_sandwiches}")
        # logger.debug(f"Sandwiches: GF Kitchen={sandwiches_gf_in_kitchen_not_on_tray}, Reg Kitchen={sandwiches_reg_in_kitchen_not_on_tray}")
        # logger.debug(f"Sandwiches on tray at loc (GF): {sandwiches_gf_on_tray_at_loc}")
        # logger.debug(f"Sandwiches on tray at loc (Reg): {sandwiches_reg_on_tray_at_loc}")
        # logger.debug(f"Sandwiches on any tray: GF={sandwiches_gf_on_any_tray}, Reg={sandwiches_reg_on_any_tray}")


        # 4. Calculate heuristic cost by satisfying needs from closest stages first

        h = 0
        remaining_gf_needed = total_gf_needed
        remaining_reg_needed = total_reg_needed

        # Stage 1: Serve (Cost 1) - Sandwich on tray at child's location.
        # Match children at location P with sandwiches on trays at location P.
        for loc in self.all_places:
             if loc == 'kitchen': continue # Children don't wait at kitchen

             needed_gf_at_loc = sum(1 for c in unserved_children if self.child_location.get(c) == loc and self.child_allergy.get(c, False))
             needed_reg_at_loc = sum(1 for c in unserved_children if self.child_location.get(c) == loc and not self.child_allergy.get(c, False))

             can_serve_gf_at_loc = min(needed_gf_at_loc, sandwiches_gf_on_tray_at_loc.get(loc, 0))
             can_serve_reg_at_loc = min(needed_reg_at_loc, sandwiches_reg_on_tray_at_loc.get(loc, 0))

             h += (can_serve_gf_at_loc + can_serve_reg_at_loc) * 1
             remaining_gf_needed -= can_serve_gf_at_loc
             remaining_reg_needed -= can_serve_reg_at_loc

        # logger.debug(f"After Stage 1 (Serve): h={h}, Rem GF={remaining_gf_needed}, Rem Reg={remaining_reg_needed}")

        # Stage 2: Move Tray + Serve (Cost 2) - Sandwich on tray NOT at child's location.
        # These are sandwiches on trays at locations other than where the remaining children are waiting.
        # This includes trays at kitchen and trays at other waiting locations.
        # Total sandwiches on trays = sandwiches_gf_on_any_tray + sandwiches_reg_on_any_tray.
        # Sandwiches already used in Stage 1 are implicitly removed by reducing remaining_needed.
        # The remaining sandwiches on trays can satisfy remaining needs.
        # Sandwiches on trays at kitchen are also available for moving.
        available_gf_on_remote_trays = sandwiches_gf_on_any_tray - sum(sandwiches_gf_on_tray_at_loc.values())
        available_reg_on_remote_trays = sandwiches_reg_on_any_tray - sum(sandwiches_reg_on_tray_at_loc.values())

        can_satisfy_remote_gf = min(remaining_gf_needed, available_gf_on_remote_trays)
        can_satisfy_remote_reg = min(remaining_reg_needed, available_reg_on_remote_trays)
        h += (can_satisfy_remote_gf + can_satisfy_remote_reg) * 2
        remaining_gf_needed -= can_satisfy_remote_gf
        remaining_reg_needed -= can_satisfy_remote_reg

        # logger.debug(f"After Stage 2 (Move+Serve): h={h}, Rem GF={remaining_gf_needed}, Rem Reg={remaining_reg_needed}")


        # Stage 3: Put on Tray + Move Tray + Serve (Cost 3) - Sandwich in kitchen (not on tray).
        can_satisfy_kitchen_gf = min(remaining_gf_needed, sandwiches_gf_in_kitchen_not_on_tray)
        can_satisfy_kitchen_reg = min(remaining_reg_needed, sandwiches_reg_in_kitchen_not_on_tray)
        h += (can_satisfy_kitchen_gf + can_satisfy_kitchen_reg) * 3
        remaining_gf_needed -= can_satisfy_kitchen_gf
        remaining_reg_needed -= can_satisfy_kitchen_reg

        # logger.debug(f"After Stage 3 (Put+Move+Serve): h={h}, Rem GF={remaining_gf_needed}, Rem Reg={remaining_reg_needed}")


        # Stage 4: Make + Put on Tray + Move Tray + Serve (Cost 4) - Needs making.
        potential_make_gf = min(available_gf_bread_kitchen, available_gf_content_kitchen, available_notexist_sandwiches)
        # Need to be careful with notexist sharing. Prioritize GF if needed.
        remaining_notexist_after_gf_potential = max(0, available_notexist_sandwiches - potential_make_gf)
        potential_make_reg = min(available_reg_bread_kitchen, available_reg_content_kitchen, remaining_notexist_after_gf_potential)

        can_make_gf = min(remaining_gf_needed, potential_make_gf)
        can_make_reg = min(remaining_reg_needed, potential_make_reg)

        h += (can_make_gf + can_make_reg) * 4
        remaining_gf_needed -= can_make_gf
        remaining_reg_needed -= can_make_reg

        # logger.debug(f"After Stage 4 (Make+Put+Move+Serve): h={h}, Rem GF={remaining_gf_needed}, Rem Reg={remaining_reg_needed}")


        # 5. Check if all needs are satisfied
        if remaining_gf_needed > 0 or remaining_reg_needed > 0:
             # This state is likely unsolvable from here with current resources
             # or implies a resource bottleneck not fully captured by this heuristic.
             # As per requirement "finite for solvable states", this case
             # should ideally not be reached for states in a solvable problem.
             # Returning infinity indicates a dead end in the search space.
             # logger.debug("Remaining needs > 0, returning infinity")
             return float('inf')

        # logger.debug(f"Final heuristic value: {h}")
        return h

