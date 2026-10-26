from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or not fact:
        return []
    # Remove leading/trailing parentheses and split by space
    return fact.strip('()').split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments, unless args contains wildcards
    # A simpler check is just to zip and compare, zip stops at the shortest sequence
    return all(fnmatch(part, arg) for part, arg in zip(parts, args)) and len(parts) == len(args)


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are waiting and need a snack. It calculates the cost based on the current
    state of required sandwiches (already at table, on tray in kitchen, made in
    kitchen, or needing to be made) and prioritizes using sandwiches that are
    closer to being served.

    # Assumptions
    - Each child needing service requires exactly one safe sandwich.
    - A sandwich is safe for a child if the child is not allergic to gluten,
      or if the sandwich is gluten-free.
    - A sandwich is gluten-free if it was made with gluten-free bread and
      gluten-free content.
    - Actions have a cost of 1.
    - The agent can perform one action at a time.
    - Trays are needed to move sandwiches from the kitchen to tables and to serve.
    - Resources (bread, content, sandwich objects, trays) are sufficient in
      solvable problems to make all necessary sandwiches.
    - The only locations are 'kitchen' and various 'table' objects.
    - Children are initially waiting at specific tables and stay there until served.

    # Heuristic Initialization
    The heuristic extracts the following static information from the task:
    - The set of children that need to be served (from the goal).
    - The allergy status (allergic_gluten or not_allergic_gluten) for each child.
    - The gluten status (no_gluten_bread or has_gluten_bread inferred) for each bread portion.
    - The gluten status (no_gluten_content or has_gluten_content inferred) for each content portion.
    - The table where each child is waiting.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  Identify Children Needing Service: Determine which children are in the goal
        state's served list but are not currently marked as served in the current state.
        Count the total number of such children (`N_total_need`).
        Categorize these children based on whether they are allergic to gluten
        (`N_need_gf`) or not (`N_need_any`).

    2.  Identify Sandwiches and Ingredients: Scan the current state to find
        information about existing sandwiches (made, on tray, notexist) and
        available ingredients (bread, content) in the kitchen.

    3.  Determine Gluten Status of Made Sandwiches: For sandwiches that have been
        made (`(made S B C')`), determine if they are gluten-free by checking the
        gluten status of the bread (B) and content (C') used, based on static facts.

    4.  Count Children Already Served at Tables: For each child needing service,
        check if there is already a safe sandwich on a tray at their specific
        waiting table. Count how many children can be served immediately this way
        (`N_served_now`). Each such child contributes 1 action (the 'serve_child' action)
        to the heuristic cost.

    5.  Calculate Children Needing Delivery: The number of children who still need
        a sandwich delivered is `N_deliver = N_total_need - N_served_now`.
        Split this into `N_deliver_gf` (children needing GF) and `N_deliver_any`
        (children needing Any, not covered by `N_served_now`).

    6.  Count Available Sandwiches (Not at Tables): Count sandwiches that are
        either on a tray in the kitchen, made and in the kitchen, or not yet made
        (`notexist`). Categorize these by their state and gluten status (GF or Any).
        Also, count available ingredients in the kitchen to determine how many
        GF and Any sandwiches can still be made from scratch.

    7.  Estimate Delivery Costs: For the `N_deliver` children, estimate the cost
        to get a safe sandwich to their table and serve them. This cost depends
        on the sandwich's current state:
        - On tray in kitchen: 1 (move_tray) + 1 (serve_child) = 2 actions.
        - Made in kitchen: 1 (put_on_tray) + 1 (move_tray) + 1 (serve_child) = 3 actions.
        - Needs making: 1 (make_sandwich) + 1 (put_on_tray) + 1 (move_tray) + 1 (serve_child) = 4 actions.

    8.  Fulfill Delivery Needs Greedily: Satisfy the `N_deliver_gf` needs first
        using available GF sandwiches from the cheapest sources (on tray kitchen,
        then made kitchen, then makeable GF). Add the corresponding costs.
        Then, satisfy the `N_deliver_any` needs using the remaining available
        sandwiches (GF or Any) from the cheapest remaining sources. Add the
        corresponding costs.

    9.  Sum Costs: The total heuristic value is the sum of costs from step 4
        (immediate serving) and step 8 (delivery).

    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        self.static = task.static  # Facts that are not affected by actions.

        # Extract children that need to be served from the goal
        self.goal_children = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "*")
        }

        # Extract child allergy information
        self.child_allergies = {}
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                self.child_allergies[get_parts(fact)[1]] = 'gluten'
            elif match(fact, "not_allergic_gluten", "*"):
                 self.child_allergies[get_parts(fact)[1]] = 'none'

        # Extract ingredient gluten information
        self.bread_gluten = {}
        self.content_gluten = {}
        for fact in self.static:
            if match(fact, "no_gluten_bread", "*"):
                self.bread_gluten[get_parts(fact)[1]] = 'none'
            # We don't know all bread/content objects from static, only those with special properties.
            # We will infer gluten status for others when processing state facts.

        # Extract child waiting tables
        self.child_tables = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                child, table = get_parts(fact)[1:3]
                self.child_tables[child] = table

    def is_gf_sandwich(self, sandwich, state_made_facts, bread_gluten_map, content_gluten_map):
        """Check if a sandwich is gluten-free based on how it was made."""
        # Find the (made S B C') fact for this sandwich in the current state
        made_fact_parts = state_made_facts.get(sandwich)

        if made_fact_parts:
            b, c = made_fact_parts
            # A sandwich is GF if both bread and content are GF
            bread_is_gf = bread_gluten_map.get(b, 'gluten') == 'none' # Default to gluten if unknown
            content_is_gf = content_gluten_map.get(c, 'gluten') == 'none' # Default to gluten if unknown
            return bread_is_gf and content_is_gf
        else:
            # If the sandwich hasn't been made yet, it's not GF *in the current state*
            # Its potential GF status is handled by the makeable counts
            return False

    def is_safe_for_child(self, sandwich, child, state_made_facts, bread_gluten_map, content_gluten_map):
        """Check if a sandwich is safe for a specific child."""
        if self.child_allergies.get(child) == 'gluten':
            # Child is allergic, sandwich must be GF
            return self.is_gf_sandwich(sandwich, state_made_facts, bread_gluten_map, content_gluten_map)
        else:
            # Child is not allergic, any sandwich is safe
            return True

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        static_info = self.static # Access static info stored in __init__

        # 1. Identify Children Needing Service
        children_to_serve = {c for c in self.goal_children if f"(served {c})" not in state}
        n_total_need = len(children_to_serve)

        if n_total_need == 0:
            return 0 # Goal reached

        # Categorize children needing service by allergy
        children_need_gf = {c for c in children_to_serve if self.child_allergies.get(c) == 'gluten'}
        children_need_any = {c for c in children_to_serve if self.child_allergies.get(c) == 'none'}
        n_need_gf = len(children_need_gf)
        n_need_any = len(children_need_any)

        # Precompute ingredient gluten properties from static facts and state facts
        bread_gluten_map = self.bread_gluten.copy()
        content_gluten_map = self.content_gluten.copy()

        # Identify all bread and content objects mentioned in the state (e.g., in made facts or at_kitchen)
        all_breads_in_state = {get_parts(f)[2] for f in state if match(f, "made", "*", "*", "*")} | {get_parts(f)[1] for f in state if match(f, "at_kitchen_bread", "*")}
        all_contents_in_state = {get_parts(f)[3] for f in state if match(f, "made", "*", "*", "*")} | {get_parts(f)[1] for f in state if match(f, "at_kitchen_content", "*")}

        # Assume gluten if not explicitly marked no_gluten in static
        for b in all_breads_in_state:
             if b not in bread_gluten_map: bread_gluten_map[b] = 'gluten'
        for c in all_contents_in_state:
             if c not in content_gluten_map: content_gluten_map[c] = 'gluten'

        # Build map of made sandwiches to ingredients from state
        state_made_facts_map = {get_parts(fact)[1]: (get_parts(fact)[2], get_parts(fact)[3]) for fact in state if match(fact, "made", "*", "*", "*")}

        # Helper to check GF status using current state's made facts and ingredient maps
        def is_gf(s):
             return self.is_gf_sandwich(s, state_made_facts_map, bread_gluten_map, content_gluten_map)

        # Helper to check safety using current state's made facts and ingredient maps
        def is_safe(s, c):
             return self.is_safe_for_child(s, c, state_made_facts_map, static_info)


        # 4. Count Children Already Served at Tables
        children_served_at_table = set()
        for child in children_to_serve:
            table = self.child_tables.get(child)
            if not table: continue # Child not waiting at a known table? (Shouldn't happen in valid problems)

            # Find trays at this table
            trays_at_table = {get_parts(fact)[1] for fact in state if match(fact, "at", "*", table)}

            # Find sandwiches on these trays
            sandwiches_on_trays_at_table = {
                get_parts(fact)[1]
                for fact in state
                if match(fact, "ontray", "*", "*") and get_parts(fact)[2] in trays_at_table
            }

            # Check if any of these sandwiches are safe for the child
            if any(is_safe(s, child) for s in sandwiches_on_trays_at_table):
                children_served_at_table.add(child)

        n_served_now = len(children_served_at_table)
        cost = n_served_now * 1 # Cost for serving children already ready

        # 5. Calculate Children Needing Delivery
        children_need_delivery = children_to_serve - children_served_at_table
        n_deliver = len(children_need_delivery)

        if n_deliver == 0:
             return cost # All children are either served or have a sandwich ready at their table

        n_deliver_gf = len({c for c in children_need_delivery if self.child_allergies.get(c) == 'gluten'})
        n_deliver_any = len({c for c in children_need_delivery if self.child_allergies.get(c) == 'none'})

        # 6. Count Available Sandwiches (Not at Tables) and Ingredients
        ontray_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*") and f"(at {get_parts(fact)[2]} kitchen)" in state}
        made_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        notexist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}

        # Count ingredients in kitchen
        n_gf_bread = len({b for b in bread_gluten_map if bread_gluten_map[b] == 'none' and f"(at_kitchen_bread {b})" in state})
        n_gf_content = len({c for c in content_gluten_map if content_gluten_map[c] == 'none' and f"(at_kitchen_content {c})" in state})
        n_any_bread = len({b for b in bread_gluten_map if f"(at_kitchen_bread {b})" in state})
        n_any_content = len({c for c in content_gluten_map if f"(at_kitchen_content {c})" in state})
        n_notexist = len(notexist_sandwiches)

        # Categorize available sandwiches (not at tables) by state and safety type
        ontray_kitchen_gf = {s for s in ontray_kitchen_sandwiches if is_gf(s)}
        ontray_kitchen_any = ontray_kitchen_sandwiches - ontray_kitchen_gf # Non-GF on tray in kitchen

        made_kitchen_gf = {s for s in made_kitchen_sandwiches if is_gf(s)}
        made_kitchen_any = made_kitchen_sandwiches - made_kitchen_gf # Non-GF made in kitchen

        # Count available sources
        avail_ontray_kitchen_gf = len(ontray_kitchen_gf)
        avail_ontray_kitchen_any = len(ontray_kitchen_any)
        avail_made_kitchen_gf = len(made_kitchen_gf)
        avail_made_kitchen_any = len(made_kitchen_any)

        # Makeable counts: How many sandwiches *can* be made from available ingredients and notexist objects
        avail_makeable_gf = min(n_gf_bread, n_gf_content, n_notexist)
        # Total makeable (can be GF or Any)
        avail_makeable_any_incl_gf = min(n_any_bread, n_any_content, n_notexist)
        # Makeable that *must* be non-GF (if any)
        # avail_makeable_any_only = avail_makeable_any_incl_gf - avail_makeable_gf # This is tricky, better to use total makeable for Any needs

        # 7. & 8. Estimate Delivery Costs by Fulfilling Needs Greedily

        # Fulfill GF needs first using GF sources
        needed_gf = n_deliver_gf

        use_ontray_gf = min(needed_gf, avail_ontray_kitchen_gf)
        cost += use_ontray_gf * 2 # move + serve
        needed_gf -= use_ontray_gf
        ontray_gf_used = use_ontray_gf

        use_made_gf = min(needed_gf, avail_made_kitchen_gf)
        cost += use_made_gf * 3 # put + move + serve
        needed_gf -= use_made_gf
        made_gf_used = use_made_gf

        use_makeable_gf = min(needed_gf, avail_makeable_gf)
        cost += use_makeable_gf * 4 # make + put + move + serve
        needed_gf -= use_makeable_gf
        makeable_gf_used = use_makeable_gf

        # Fulfill Any needs using remaining sources (GF or Any)
        needed_any = n_deliver_any

        # Remaining on-tray kitchen sandwiches (Any or unused GF)
        avail_ontray_kitchen_remaining = avail_ontray_kitchen_any + (avail_ontray_kitchen_gf - ontray_gf_used)
        use_ontray_any = min(needed_any, avail_ontray_kitchen_remaining)
        cost += use_ontray_any * 2 # move + serve
        needed_any -= use_ontray_any

        # Remaining made kitchen sandwiches (Any or unused GF)
        avail_made_kitchen_remaining = avail_made_kitchen_any + (avail_made_kitchen_gf - made_gf_used)
        use_made_any = min(needed_any, avail_made_kitchen_remaining)
        cost += use_made_any * 3 # put + move + serve
        needed_any -= use_made_any

        # Remaining makeable sandwiches (Any or unused GF)
        # Total makeable available = avail_makeable_any_incl_gf
        # Total makeable GF used = makeable_gf_used
        avail_makeable_remaining = avail_makeable_any_incl_gf - makeable_gf_used
        use_makeable_any = min(needed_any, avail_makeable_remaining)
        cost += use_makeable_any * 4 # make + put + move + serve
        needed_any -= use_makeable_any

        # If needed_gf > 0 or needed_any > 0 here, it implies the problem is unsolvable
        # with the current resource model or the heuristic is underestimating resources.
        # For a greedy best-first search heuristic, returning a large number is appropriate.
        # However, assuming solvable problems, needed_gf and needed_any should be 0 here.
        # We can add a check and return float('inf') if needed_gf > 0 or needed_any > 0,
        # but for simplicity and typical solvable instances, we assume they are 0.

        return cost
