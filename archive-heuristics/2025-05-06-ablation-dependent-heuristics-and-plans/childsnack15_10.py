from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and handle empty facts if necessary, though PDDL facts aren't empty
    if not isinstance(fact, str) or len(fact) < 2:
         return [] # Or raise an error, depending on expected input
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    by counting the number of unserved children and assigning a cost based on
0    the "readiness" stage of suitable sandwiches needed for them. It greedily
    uses the most readily available suitable sandwiches first.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Gluten-allergic children require gluten-free sandwiches.
    - Non-allergic children can be served with any sandwich (regular or gluten-free).
    - Sandwiches progress through stages: needs making -> in kitchen -> on tray (not at child) -> on tray (at child) -> served.
    - Costs are assigned based on the stage of the *most available* suitable sandwich for a child's need type (GF/Regular).
    - Sufficient trays are assumed to be available in the kitchen or movable to the kitchen when needed for 'put_on_tray' or 'make' actions.
    - Sufficient bread, content, and sandwich objects are assumed to exist initially to make all required sandwiches (solvable problem).

    # Heuristic Initialization
    - Identify all children that need to be served from the goal conditions.
    - Extract static information about which children are allergic or not.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are in the goal state but are not yet marked as `served`. These are the unserved children.
    2. If there are no unserved children, the heuristic is 0 (goal state).
    3. Separate the unserved children into those who are gluten-allergic (needing GF sandwiches) and those who are not (needing Regular sandwiches). Count the total number needed for each type (`N_gf_unserved`, `N_reg_unserved`).
    4. Identify the waiting location for each unserved child.
    5. Count the number of available suitable sandwiches by type (GF/Regular) and their current "readiness" stage:
       - Stage 1 (Cost 1: Serve): Sandwiches on a tray located at a waiting location of a child who needs that type of sandwich.
       - Stage 2 (Cost 2: Move Tray + Serve): Sandwiches on a tray not located at a waiting location of a child who needs that type of sandwich.
       - Stage 3 (Cost 3: Put on Tray + Move Tray + Serve): Sandwiches in the kitchen (`at_kitchen_sandwich`).
       - Stage 4 (Cost 4: Make + Put on Tray + Move Tray + Serve): Sandwiches that can be made (`notexist` object available, and suitable bread/content available in the kitchen).
    6. Greedily assign the available sandwiches to the needed sandwiches, starting with the lowest cost stage (Stage 1), then Stage 2, Stage 3, and finally Stage 4.
    7. Sum the costs for the assigned sandwiches: `cost = (sandwiches_from_S1 * 1) + (sandwiches_from_S2 * 2) + (sandwiches_from_S3 * 3) + (sandwiches_from_S4 * 4)`.
    8. The total sum is the heuristic value. If after exhausting all available and makeable sandwiches there are still children unserved, add a cost of 4 for each remaining child (assuming they would need to be made).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and static allergy info.
        """
        self.goals = task.goals
        self.static = task.static

        # Identify all children that need to be served from the goal conditions.
        self.children_to_serve = {
            get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "served"
        }

        # Identify which children are allergic from static facts.
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "allergic_gluten", "*")
        }
        # Identify which children are not allergic from static facts.
        self.non_allergic_children = {
            get_parts(fact)[1] for fact in self.static if match(fact, "not_allergic_gluten", "*")
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unserved children
        unserved_children = {
            c for c in self.children_to_serve if f"(served {c})" not in state
        }

        # 2. If no unserved children, goal is reached
        if not unserved_children:
            return 0

        # 3. Separate unserved children by allergy status and count needed sandwiches
        unserved_gf_needed = {c for c in unserved_children if c in self.allergic_children}
        unserved_reg_needed = {c for c in unserved_children if c in self.non_allergic_children}

        N_gf_unserved = len(unserved_gf_needed)
        N_reg_unserved = len(unserved_reg_needed)

        # Map children to their waiting locations
        child_waiting_locs = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in state
            if match(fact, "waiting", "*", "*") and get_parts(fact)[1] in unserved_children
        }

        # Identify relevant waiting locations for each type
        waiting_locations_gf = {child_waiting_locs[c] for c in unserved_gf_needed if c in child_waiting_locs}
        waiting_locations_reg = {child_waiting_locs[c] for c in unserved_reg_needed if c in child_waiting_locs}


        # Map trays to locations
        tray_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")
        }

        # Identify GF sandwiches that exist
        is_gf_sandwich = {
            get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")
        }

        # 5. Count available sandwiches by stage and type
        avail_gf_s1 = 0  # on tray at a relevant waiting child's location
        avail_reg_s1 = 0 # on tray at a relevant waiting child's location
        avail_gf_s2 = 0  # on tray not at a relevant waiting child's location
        avail_reg_s2 = 0 # on tray not at a relevant waiting child's location
        avail_gf_s3 = 0  # in kitchen
        avail_reg_s3 = 0 # in kitchen

        counted_sandwiches = set() # Prevent double counting sandwiches across stages

        # Count S1 (on tray at relevant waiting location)
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s = get_parts(fact)[1]
                t = get_parts(fact)[2]
                if s in counted_sandwiches: continue
                tray_loc = tray_locations.get(t)
                if tray_loc is None: continue # Tray location unknown

                is_gf = s in is_gf_sandwich

                is_at_relevant_waiting_loc = False
                if is_gf and tray_loc in waiting_locations_gf:
                    is_at_relevant_waiting_loc = True
                if not is_gf and tray_loc in waiting_locations_reg:
                     is_at_relevant_waiting_loc = True

                if is_at_relevant_waiting_loc:
                    if is_gf: avail_gf_s1 += 1
                    else: avail_reg_s1 += 1
                    counted_sandwiches.add(s)

        # Count S2 (on tray not at relevant waiting location)
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s = get_parts(fact)[1]
                if s in counted_sandwiches: continue # Already counted in S1
                t = get_parts(fact)[2]
                tray_loc = tray_locations.get(t)
                if tray_loc is None: continue # Tray location unknown

                is_gf = s in is_gf_sandwich

                # Check if this tray location is *not* a waiting location for *any* relevant child
                is_at_any_relevant_waiting_loc = False
                if is_gf and tray_loc in waiting_locations_gf: is_at_any_relevant_waiting_loc = True
                if not is_gf and tray_loc in waiting_locations_reg: is_at_any_relevant_waiting_loc = True

                if not is_at_any_relevant_waiting_loc:
                     if is_gf: avail_gf_s2 += 1
                     else: avail_reg_s2 += 1
                     counted_sandwiches.add(s)

        # Count S3 (in kitchen)
        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                if s in counted_sandwiches: continue # Already counted in S1 or S2
                counted_sandwiches.add(s)

                is_gf = s in is_gf_sandwich
                if is_gf: avail_gf_s3 += 1
                else: avail_reg_s3 += 1

        # Count S4 (makeable)
        num_available_sandwich_objects = sum(1 for fact in state if match(fact, "notexist", "*"))
        num_available_gf_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and match(fact, "no_gluten_bread", "*"))
        num_available_reg_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and not match(fact, "no_gluten_bread", "*"))
        num_available_gf_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and match(fact, "no_gluten_content", "*"))
        num_available_reg_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and not match(fact, "no_gluten_content", "*"))

        # Makeable GF sandwiches require GF bread, GF content, and a sandwich object
        makeable_gf = min(num_available_sandwich_objects, num_available_gf_bread, num_available_gf_content)

        # Ingredients used for GF sandwiches
        bread_used_for_gf = makeable_gf
        content_used_for_gf = makeable_gf

        # Remaining resources for regular sandwiches
        remaining_sandwich_objects = num_available_sandwich_objects - makeable_gf
        remaining_bread = (num_available_gf_bread + num_available_reg_bread) - bread_used_for_gf
        remaining_content = (num_available_gf_content + num_available_reg_content) - content_used_for_gf

        # Makeable Regular sandwiches require any remaining bread, any remaining content, and a remaining sandwich object
        makeable_reg = min(remaining_sandwich_objects, remaining_bread, remaining_content)


        # 6. & 7. Greedily assign sandwiches and sum costs
        h = 0
        needed_gf = N_gf_unserved
        needed_reg = N_reg_unserved

        # Stage 1: Serve (cost 1)
        used = min(needed_gf, avail_gf_s1)
        h += used * 1
        needed_gf -= used
        used = min(needed_reg, avail_reg_s1)
        h += used * 1
        needed_reg -= used

        # Stage 2: Move + Serve (cost 2)
        used = min(needed_gf, avail_gf_s2)
        h += used * 2
        needed_gf -= used
        used = min(needed_reg, avail_reg_s2)
        h += used * 2
        needed_reg -= used

        # Stage 3: Put + Move + Serve (cost 3)
        used = min(needed_gf, avail_gf_s3)
        h += used * 3
        needed_gf -= used
        used = min(needed_reg, avail_reg_s3)
        h += used * 3
        needed_reg -= used

        # Stage 4: Make + Put + Move + Serve (cost 4)
        used = min(needed_gf, makeable_gf)
        h += used * 4
        needed_gf -= used
        used = min(needed_reg, makeable_reg)
        h += used * 4
        needed_reg -= used

        # 8. Add cost for any remaining unserved children (should not happen in solvable problems with sufficient initial resources)
        # If it happens, it means we ran out of makeable sandwiches. Assume they still need 4 steps if resources were somehow replenished.
        h += needed_gf * 4
        h += needed_reg * 4

        return h
