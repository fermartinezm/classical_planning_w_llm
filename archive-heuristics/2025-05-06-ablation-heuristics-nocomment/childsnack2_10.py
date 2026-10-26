from fnmatch import fnmatch
from collections import defaultdict
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
def get_parts(fact):
    """Splits a PDDL fact string into predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Checks if a fact matches a pattern of predicate and arguments."""
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions required to serve all unserved children
    by considering the current state of sandwiches and trays and the steps
    needed to get a suitable sandwich to each child.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static information from the task.

        Args:
            task: The planning task object.
        """
        super().__init__(task) # Call base class constructor

        static_facts = task.static

        # Extract static information
        self.allergic_children_static = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children_static = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "not_allergic_gluten", "*")
        }
        self.no_gluten_bread_static = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_content_static = {
            get_parts(fact)[1] for fact in static_facts if match(fact, "no_gluten_content", "*")
        }
        self.waiting_children_loc_static = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in static_facts if match(fact, "waiting", "*", "*")
        }
        # Note: no_gluten_sandwich is not static, it's an effect of an action.

    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            An estimate of the remaining cost to reach a goal state, or float('inf')
            if the state is estimated to be unsolvable.

        Summary:
            The heuristic estimates the cost to serve all unserved children. It does this
            by counting the number of unserved allergic and non-allergic children and
            then estimating the cost to get a suitable sandwich to each, prioritizing
            sandwiches that are closer to being served (e.g., already on a tray at the
            child's location, then on a tray elsewhere, then in the kitchen, then needing
            to be made). The cost estimation for each stage is based on the minimum
            number of actions required (serve, move+serve, put+move+serve, make+put+move+serve),
            considering bottlenecks like tray availability in the kitchen.

        Assumptions:
            - Actions have the following estimated base costs:
                - serve_sandwich: 1
                - move_tray: 1
                - put_on_tray: 1
                - make_sandwich: 1
            - The heuristic assumes a greedy assignment of available sandwiches and trays
              to unserved children based on the stages described below.
            - The heuristic assumes that solvable instances will have sufficient ingredients
              (GF bread/content, any bread/content) and 'notexist' sandwich objects to
              make any required sandwiches. If not, it returns infinity.
            - Tray movement between any two places costs 1 action.
            - Putting a sandwich on a tray requires the tray to be in the kitchen.

        Heuristic Initialization:
            - Identifies static facts: which children are allergic/not allergic,
              which bread/content is gluten-free, and where each child is waiting.

        Step-By-Step Thinking for Computing Heuristic:
            1. Identify all unserved children based on the goal and current state,
               and separate them into allergic (need GF) and non-allergic (need any)
               using static information.
            2. Extract relevant dynamic facts from the current state: served children,
               sandwiches on trays and their locations, sandwiches in the kitchen,
               bread/content in the kitchen, 'notexist' sandwiches, and tray locations.
               Also identify which existing sandwiches are gluten-free.
            3. Initialize heuristic value `h = 0`.
            4. Count available suitable sandwiches at different stages of readiness:
               - On trays at the correct location for an unserved child.
               - On trays at a wrong location.
               - In the kitchen.
               - Needing to be made (based on remaining unserved children).
            5. Greedily assign available sandwiches to unserved children based on the
               following stages, adding the estimated cost for each served child:
               - **Stage 1 (Cost 1: Serve):** Children whose suitable sandwich is
                 already on a tray at their waiting location. Prioritize allergic
                 children for GF sandwiches.
               - **Stage 2 (Cost 2: Move + Serve):** Remaining children whose suitable
                 sandwich is on a tray at a different location. Prioritize allergic
                 children for GF sandwiches.
               - **Stage 3 (Cost 3 or 4: Put + Move + Serve):** Remaining children
                 whose suitable sandwich is currently `at_kitchen_sandwich`. These
                 need to be put on a tray and moved. The cost depends on the number
                 of trays available `at kitchen`. If insufficient kitchen trays,
                 some trays need to be moved to the kitchen first (cost 4 vs 3).
                 Prioritize allergic children for GF sandwiches.
               - **Stage 4 (Cost 1 + 3 or 4: Make + Put + Move + Serve):** Remaining
                 children need sandwiches to be made. Check if sufficient ingredients
                 (GF bread/content, any bread/content) and 'notexist' sandwich objects
                 are available. If not, return infinity. Add cost for making (1 per sandwich).
                 These newly made sandwiches are `at_kitchen_sandwich`, so add the cost
                 for putting them on trays and serving, similar to Stage 3, considering
                 remaining kitchen tray availability.
            6. If, after accounting for all available and makeable sandwiches, there are
               still unserved children, return infinity (indicates an unsolvable state
               not caught by the ingredient check, or a logic error).
            7. Return the total calculated heuristic value `h`.
        """
        state = node.state

        # --- Extract Dynamic Information from State ---
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        ontray_facts = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "ontray", "*", "*")}
        at_tray_facts = {(get_parts(fact)[1], get_parts(fact)[2]) for fact in state if match(fact, "at", "*", "*")}
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        no_gluten_sandwiches_state = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        at_kitchen_bread_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        at_kitchen_content_state = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}
        notexist_sandwiches_state = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}

        # Map sandwich on tray to tray location
        sandwich_on_tray_loc = {}
        tray_locations = dict(at_tray_facts)
        for s, t in ontray_facts:
            if t in tray_locations:
                sandwich_on_tray_loc[s] = tray_locations[t]

        # --- Identify Unserved Children ---
        unserved_children = {c for c in self.waiting_children_loc_static if c not in served_children}
        unserved_gf_children = {c for c in unserved_children if c in self.allergic_children_static}
        unserved_any_children = {c for c in unserved_children if c in self.not_allergic_children_static}

        unserved_gf_count = len(unserved_gf_children)
        unserved_any_count = len(unserved_any_children)

        # If no children are unserved, the goal is reached
        if unserved_gf_count == 0 and unserved_any_count == 0:
            return 0

        h = 0

        # --- Count Available Sandwiches by Stage ---

        # Stage 1: On trays at the correct location (cost 1)
        # Count GF sandwiches on trays at allergic children's locations
        allergic_locations = {self.waiting_children_loc_static[c] for c in unserved_gf_children}
        gf_sandwiches_at_allergic_loc = {s for s, loc in sandwich_on_tray_loc.items() if s in no_gluten_sandwiches_state and loc in allergic_locations}

        # Count any sandwiches on trays at non-allergic children's locations
        non_allergic_locations = {self.waiting_children_loc_static[c] for c in unserved_any_children}
        any_sandwiches_at_non_allergic_loc = {s for s, loc in sandwich_on_tray_loc.items() if loc in non_allergic_locations}

        num_gf_at_correct_loc = len(gf_sandwiches_at_allergic_loc)
        num_any_at_correct_loc = len(any_sandwiches_at_non_allergic_loc)

        # Assign sandwiches greedily (prioritize GF for allergic, then GF for any, then any for any)
        served_gf_stage1 = min(unserved_gf_count, num_gf_at_correct_loc)
        h += served_gf_stage1 * 1
        unserved_gf_count -= served_gf_stage1
        num_gf_at_correct_loc -= served_gf_stage1 # Remaining GF at correct loc

        served_any_stage1_gf = min(unserved_any_count, num_gf_at_correct_loc) # Use remaining GF
        h += served_any_stage1_gf * 1
        unserved_any_count -= served_any_stage1_gf
        num_gf_at_correct_loc -= served_any_stage1_gf # Consume GF sandwiches

        served_any_stage1_any = min(unserved_any_count, num_any_at_correct_loc - served_any_stage1_gf) # Use non-GF (or GF already counted in potential)
        h += served_any_stage1_any * 1
        unserved_any_count -= served_any_stage1_any


        # Stage 2: On trays at wrong location (cost 2)
        # Count sandwiches on trays anywhere that are *not* at a correct location for any unserved child
        all_correct_locations = allergic_locations | non_allergic_locations

        ontray_wrong_loc_gf = {s for s, loc in sandwich_on_tray_loc.items() if loc not in all_correct_locations and s in no_gluten_sandwiches_state}
        ontray_wrong_loc_any = {s for s, loc in sandwich_on_tray_loc.items() if loc not in all_correct_locations} # Includes GF

        num_gf_ontray_wrong = len(ontray_wrong_loc_gf)
        num_any_ontray_wrong = len(ontray_wrong_loc_any)

        # Assign sandwiches greedily
        served_gf_stage2 = min(unserved_gf_count, num_gf_ontray_wrong)
        h += served_gf_stage2 * 2
        unserved_gf_count -= served_gf_stage2
        num_gf_ontray_wrong -= served_gf_stage2

        served_any_stage2_gf = min(unserved_any_count, num_gf_ontray_wrong) # Use remaining GF
        h += served_any_stage2_gf * 2
        unserved_any_count -= served_any_stage2_gf
        num_gf_ontray_wrong -= served_any_stage2_gf

        served_any_stage2_any = min(unserved_any_count, num_any_ontray_wrong - served_any_stage2_gf) # Use non-GF (or GF already counted in potential)
        h += served_any_stage2_any * 2
        unserved_any_count -= served_any_stage2_any


        # Stage 3: Kitchen sandwiches (cost varies)
        kitchen_sandwich_gf_count = len({s for s in kitchen_sandwiches if s in no_gluten_sandwiches_state})
        kitchen_sandwich_any_count = len(kitchen_sandwiches) # Includes GF

        # Assign sandwiches greedily
        served_gf_stage3 = min(unserved_gf_count, kitchen_sandwich_gf_count)
        unserved_gf_count -= served_gf_stage3
        kitchen_sandwich_gf_count -= served_gf_stage3

        served_any_stage3_gf = min(unserved_any_count, kitchen_sandwich_gf_count) # Use remaining GF
        unserved_any_count -= served_any_stage3_gf
        kitchen_sandwich_gf_count -= served_any_stage3_gf

        served_any_stage3_any = min(unserved_any_count, kitchen_sandwich_any_count - served_any_stage3_gf) # Use non-GF (or GF already counted in potential)
        unserved_any_count -= served_any_stage3_any

        num_stage3_sandwiches_used = served_gf_stage3 + served_any_stage3_gf + served_any_stage3_any


        # Stage 4: Make and serve (cost varies + make cost)
        num_to_make_gf = unserved_gf_count
        num_to_make_any = unserved_any_count
        num_to_make = num_to_make_gf + num_to_make_any

        if num_to_make > 0:
            # Ingredient and notexist check
            gf_bread_count = len({b for b in at_kitchen_bread_state if b in self.no_gluten_bread_static})
            gf_content_count = len({c for c in at_kitchen_content_state if c in self.no_gluten_content_static})
            any_bread_count = len(at_kitchen_bread_state)
            any_content_count = len(at_kitchen_content_state)
            notexist_sandwich_count = len(notexist_sandwiches_state)

            # Check if enough GF ingredients for GF makes
            if num_to_make_gf > gf_bread_count or num_to_make_gf > gf_content_count:
                 return float('inf')

            # Check if enough total ingredients/notexist for total makes
            # Total bread needed = num_to_make_gf + num_to_make_any
            # Total content needed = num_to_make_gf + num_to_make_any
            # Total notexist needed = num_to_make_gf + num_to_make_any

            if (num_to_make_gf + num_to_make_any) > any_bread_count or \
               (num_to_make_gf + num_to_make_any) > any_content_count or \
               (num_to_make_gf + num_to_make_any) > notexist_sandwich_count:
                return float('inf')

            # Add make cost
            h += num_to_make * 1

        # --- Calculate Tray-dependent Costs for Stage 3 and 4 Sandwiches ---
        num_sandwiches_needing_put = num_stage3_sandwiches_used + num_to_make

        if num_sandwiches_needing_put > 0:
            trays_kitchen_count = len({t for t, p in at_tray_facts if p == 'kitchen'})
            can_use_existing_tray = min(num_sandwiches_needing_put, trays_kitchen_count)
            need_move_tray_to_kitchen = max(0, num_sandwiches_needing_put - trays_kitchen_count)

            # Cost for sandwiches using existing kitchen trays: 1 put + 1 move + 1 serve = 3
            h += can_use_existing_tray * 3
            # Cost for sandwiches needing tray moved to kitchen: 1 move_tray_to_kitchen + 1 put + 1 move_tray_to_child + 1 serve = 4
            h += need_move_tray_to_kitchen * 4

        # --- Final Check ---
        # If any children remain unserved, it implies an unsolvable state
        # not caught by the ingredient check, or a logic error in counting.
        # For solvable states, unserved_gf_count and unserved_any_count should be 0 here.
        if unserved_gf_count > 0 or unserved_any_count > 0:
             return float('inf')

        return h
