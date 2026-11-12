from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split a fact string into tokens."""
    return fact[1:-1].split()

class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Domain-dependent heuristic for the `childsnack` domain. Estimates the number
    of actions required to serve all children that must be served. The heuristic
    is designed to be informative (not necessarily admissible) and efficiently
    computable for guiding greedy best-first search.

    Assumptions
    -----------
    - The planner's internal state representation uses strings like
      "(predicate arg1 arg2)" and static facts are provided in task.static.
    - "waiting" and allergy predicates for children are static (they don't
      change during plan execution in these problems).
    - Trays, breads, contents, and sandwich objects exist as named objects in
      the task; trays can be moved freely to any place using a single
      'move_tray' action (no adjacency costs).
    - Making a no-gluten sandwich requires both a no-gluten bread and a
      no-gluten content (the make_sandwich_no_gluten action). Non-allergic
      children can be served with any sandwich (produced with make_sandwich).
    - The heuristic assumes actions have unit cost and counts actions:
      make_sandwich / make_sandwich_no_gluten (1),
      put_on_tray (1),
      move_tray (1),
      serve_sandwich / serve_sandwich_no_gluten (1).

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts static information:
    - The set of children that must be served and, for each child, their
      allergy status (allergic_gluten or not_allergic_gluten) and the place
      where they are waiting (waiting child place). These are taken from the
      static facts (task.static).
    - The set of bread/content identifiers that are marked as no_gluten in the
      static facts. (Whether a bread/content is presently at the kitchen is
      derived from the dynamic state.)

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. If the state already satisfies all goals (all required served facts are
       present), return 0.

    2. Gather dynamic resources from the state:
       - which children are already served,
       - trays and their current locations (from "(at tray place)"),
       - sandwiches that are currently on trays (ontray s t),
       - sandwiches that are currently at the kitchen (at_kitchen_sandwich s),
       - breads and contents currently at the kitchen and whether they are
         no-gluten (the no-gluten property is determined from static facts),
       - which sandwich names remain unused (presence of "(notexist s)").

    3. For each unserved child, compute the cheapest way to make a matching
       sandwich and deliver it, considering currently available resources.
       Possible (ordered-by-cost) delivery options for a child at place p:
       - Use an ontray sandwich with correct gluten property whose tray is
         already at p: cost 1 (serve).
       - Use an ontray sandwich with correct gluten property whose tray is
         elsewhere: cost 2 (move_tray + serve).
       - Use a sandwich that already exists in the kitchen: costs depend on
         whether a tray is at the kitchen and whether p == kitchen:
            - if tray at kitchen and p == kitchen: put_on_tray + serve = 2
            - if tray at kitchen and p != kitchen: put_on_tray + move_tray + serve = 3
            - if no tray at kitchen: need an extra move to bring a tray to kitchen:
              (move_tray to kitchen) + put_on_tray + (possible move_tray to p) + serve
              => 3 if p == kitchen else 4
       - Make a new sandwich (make_sandwich or make_sandwich_no_gluten) then
         follow the sequence for a kitchen sandwich. Make consumes one bread
         and one content from the kitchen (and for allergic children both must
         be no_gluten). Preference is given to preserve no-gluten resources for
         allergic children: non-allergic makes use of gluten breads/contents
         first when possible.
    4. The heuristic uses a greedy allocation: at each step it chooses the
       currently cheapest child to serve (given current resources), allocates
       the resources (consumes breads/contents, marks sandwiches/trays used),
       adds the computed cost for that child and repeats until all children
       are accounted for. This gives a fast and realistic estimate of the
       remaining number of actions.

    Rationale
    ---------
    This greedy resource-aware allocation balances accuracy and efficiency:
    it captures important resource competition (trays, kitchen sandwiches,
    gluten-free resources) without performing a full search. It returns 0
    for goal states, finite values for solvable states, and is inexpensive
    to compute per node.
    """

    def __init__(self, task):
        # Extract static facts for allergy and waiting info and no_gluten markings.
        static_facts = task.static
        self.children = set()
        self.child_allergy = {}  # child -> 'allergic' or 'not_allergic'
        self.child_waiting_place = {}  # child -> place

        # no_gluten markers for breads and contents (static)
        self.no_gluten_breads_static = set()
        self.no_gluten_contents_static = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "allergic_gluten":
                child = parts[1]
                self.children.add(child)
                self.child_allergy[child] = "allergic"
            elif pred == "not_allergic_gluten":
                child = parts[1]
                self.children.add(child)
                self.child_allergy[child] = "not_allergic"
            elif pred == "waiting":
                # (waiting child place)
                child, place = parts[1], parts[2]
                self.children.add(child)
                self.child_waiting_place[child] = place
            elif pred == "no_gluten_bread":
                bread = parts[1]
                self.no_gluten_breads_static.add(bread)
            elif pred == "no_gluten_content":
                content = parts[1]
                self.no_gluten_contents_static.add(content)
        # If some children have allergy info only in state (unlikely), they will
        # be handled in __call__ by checking state as fallback.

    def __call__(self, node):
        state = node.state

        # 1) Goal check: heuristic 0 if all goals already satisfied
        # Task.goals contains required served facts; the Heuristic base gives us
        # node.task not directly here, so we check goal facts by matching '(served child)'.
        # However, the planner calls heuristic only with a Node that has state,
        # and the problem's goals are in the task object available at initialization
        # only. To be safe we check whether each child that is expected to be served
        # has its served fact in the current state; if all such children are served,
        # return 0.
        # Determine goal-children as those present in self.child_waiting_place.
        goal_children = set(self.child_waiting_place.keys())
        served = {get_parts(f)[1] for f in state if get_parts(f)[0] == "served"}
        if goal_children <= served:
            return 0

        # Helper containers for dynamic resources
        trays_loc = {}  # tray -> place
        ontray = {}     # sandwich -> (tray, gluten_flag)   gluten_flag True if no_gluten_sandwich present
        at_kitchen_sand = {}  # sandwich -> gluten_flag
        notexist_sandwiches = set()
        breads_at_kitchen = set()
        contents_at_kitchen = set()

        # Fill dynamic info from state; also capture any allergy info present in state
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at":
                # could be "(at tray place)" or other at- facts
                # we only care about trays (predicates 'at' with first arg a tray)
                obj = parts[1]
                place = parts[2]
                # Very simple test: objects named 'tray' usually start with 'tray' in examples.
                if obj.startswith("tray"):
                    trays_loc[obj] = place
            elif pred == "ontray":
                # (ontray sandw t)
                sandw, tray = parts[1], parts[2]
                # determine gluten flag by searching for no_gluten_sandwich s in state
                # we'll mark gluten flag later after scanning all facts
                ontray[sandw] = (tray, False)
            elif pred == "no_gluten_sandwich":
                sandw = parts[1]
                # mark gluten flag if sandwich is ontray or at_kitchen (we might not have recorded yet)
                if sandw in ontray:
                    tray, _ = ontray[sandw]
                    ontray[sandw] = (tray, True)
                if sandw in at_kitchen_sand:
                    at_kitchen_sand[sandw] = True
            elif pred == "at_kitchen_sandwich":
                sandw = parts[1]
                at_kitchen_sand[sandw] = False  # gluten flag maybe updated later if present
            elif pred == "notexist":
                sandw = parts[1]
                notexist_sandwiches.add(sandw)
            elif pred == "at_kitchen_bread":
                bread = parts[1]
                breads_at_kitchen.add(bread)
            elif pred == "at_kitchen_content":
                content = parts[1]
                contents_at_kitchen.add(content)
            elif pred == "allergic_gluten" and parts[1] not in self.child_allergy:
                self.child_allergy[parts[1]] = "allergic"
                self.children.add(parts[1])
            elif pred == "not_allergic_gluten" and parts[1] not in self.child_allergy:
                self.child_allergy[parts[1]] = "not_allergic"
                self.children.add(parts[1])
            elif pred == "waiting" and parts[1] not in self.child_waiting_place:
                self.child_waiting_place[parts[1]] = parts[2]
                self.children.add(parts[1])

        # Ensure gluten flags for kitchen sandwiches could be set by separate facts
        # (no_gluten_sandwich was handled above). If still unknown, default to False.
        # (already default set)

        # Create bread/content pools separated by no_gluten property according to static info
        no_gluten_breads = set(b for b in breads_at_kitchen if b in self.no_gluten_breads_static)
        gluten_breads = set(breads_at_kitchen) - no_gluten_breads
        no_gluten_contents = set(c for c in contents_at_kitchen if c in self.no_gluten_contents_static)
        gluten_contents = set(contents_at_kitchen) - no_gluten_contents

        # Prepare unserved children list
        unserved_children = [c for c in self.child_waiting_place.keys() if c not in served]

        # If there are no trays known in the state, assume at least one tray exists
        # but not at kitchen — to keep heuristic finite we create a fake tray if needed.
        if not trays_loc:
            # create a single virtual tray named '__tray_virtual' placed at kitchen
            trays_loc["__tray_virtual"] = "kitchen"

        # For quick lookup of ontray sandwiches keyed by gluten flag and tray location
        # We'll treat ontray and at_kitchen_sand as resources that we can allocate.
        # Convert ontray dict to list entries for easier removal.
        ontray_list = []
        for s, (t, gluten_flag) in ontray.items():
            ontray_list.append({"sand": s, "tray": t, "gluten": gluten_flag})

        at_kitchen_list = []
        for s, gluten_flag in at_kitchen_sand.items():
            at_kitchen_list.append({"sand": s, "gluten": gluten_flag})

        # Represent trays as a dict we can update
        trays = dict(trays_loc)  # tray -> place

        # Convert notexist set to a list for counting/making sandwiches
        notexist_list = list(notexist_sandwiches)

        total_cost = 0

        # Greedy allocation loop: always pick the child with minimal immediate cost
        while unserved_children:
            best_child = None
            best_option = None
            best_cost = float("inf")

            # Evaluate for each child the minimal cost given current resources
            for child in unserved_children:
                place = self.child_waiting_place.get(child, None)
                allergy = self.child_allergy.get(child, "not_allergic")
                needs_no_gluten = (allergy == "allergic")

                # Evaluate options and compute minimal cost for this child
                # Option A: ontray at same place -> cost 1
                found_ontray_same = None
                for entry in ontray_list:
                    if entry["gluten"] or (not needs_no_gluten):
                        tray_place = trays.get(entry["tray"], None)
                        if tray_place == place:
                            found_ontray_same = entry
                            break
                if found_ontray_same is not None:
                    opt_cost = 1
                    opt_desc = ("ontray_same", found_ontray_same)
                    if opt_cost < best_cost:
                        best_cost = opt_cost
                        best_option = opt_desc
                        best_child = child
                    continue  # cheapest possible for this child

                # Option B: ontray elsewhere -> move_tray + serve = 2
                found_ontray_else = None
                for entry in ontray_list:
                    if entry["gluten"] or (not needs_no_gluten):
                        # any tray not at child's place qualifies
                        tray_place = trays.get(entry["tray"], None)
                        if tray_place is not None and tray_place != place:
                            found_ontray_else = entry
                            break
                if found_ontray_else is not None:
                    opt_cost = 2
                    opt_desc = ("ontray_else", found_ontray_else)
                    if opt_cost < best_cost:
                        best_cost = opt_cost
                        best_option = opt_desc
                        best_child = child
                    # don't continue: maybe at_kitchen option is cheaper (unlikely), so we check further

                # Option C: existing kitchen sandwich
                found_kitchen = None
                for entry in at_kitchen_list:
                    if entry["gluten"] or (not needs_no_gluten):
                        found_kitchen = entry
                        break
                if found_kitchen is not None:
                    # compute cost depending on tray at kitchen and child's place
                    # is there a tray at kitchen?
                    tray_at_kitchen = None
                    for t, p in trays.items():
                        if p == "kitchen":
                            tray_at_kitchen = t
                            break
                    if tray_at_kitchen is not None:
                        if place == "kitchen":
                            opt_cost = 2  # put_on_tray + serve
                        else:
                            opt_cost = 3  # put_on_tray + move_tray + serve
                        opt_desc = ("kitchen_existing", found_kitchen, tray_at_kitchen)
                    else:
                        # need to move some tray to kitchen first (1), then put, maybe move to place, serve
                        if place == "kitchen":
                            opt_cost = 3  # move_tray to kitchen + put + serve
                        else:
                            opt_cost = 4  # move to kitchen + put + move to place + serve
                        # pick any tray that we would move
                        any_tray = next(iter(trays.keys()))
                        opt_desc = ("kitchen_existing_no_tray_at_kitchen", found_kitchen, any_tray)
                    if opt_cost < best_cost:
                        best_cost = opt_cost
                        best_option = opt_desc
                        best_child = child

                # Option D: make a new sandwich (if sandwich names and bread/content resources available)
                # Determine feasibility and cost for a make followed by kitchen handling as above.
                can_make = False
                # Check availability of notexist sandwich names
                if notexist_list:
                    if needs_no_gluten:
                        # need one no-gluten bread and one no-gluten content
                        if no_gluten_breads and no_gluten_contents:
                            can_make = True
                    else:
                        # for non-allergic, prefer using gluten breads/contents to save no-gluten for allergic
                        if gluten_breads and gluten_contents:
                            can_make = True
                        else:
                            # fall back to using any bread and any content
                            if (no_gluten_breads or gluten_breads) and (no_gluten_contents or gluten_contents):
                                can_make = True
                if can_make:
                    # making is 1 action, then kitchen sandwich handling as above
                    # Re-evaluate kitchen-handling costs after make:
                    tray_at_kitchen = None
                    for t, p in trays.items():
                        if p == "kitchen":
                            tray_at_kitchen = t
                            break
                    if tray_at_kitchen is not None:
                        if place == "kitchen":
                            opt_cost = 1 + 1 + 1  # make + put + serve
                        else:
                            opt_cost = 1 + 1 + 1 + 1  # make + put + move + serve => 4
                    else:
                        # need to move tray to kitchen first
                        if place == "kitchen":
                            opt_cost = 1 + 1 + 1  # make + move_tray to kitchen + put + serve => 3
                        else:
                            opt_cost = 1 + 1 + 1 + 1 + 0  # make + move to kitchen + put + move to place + serve => 4
                            # some overlap with above; effectively 4
                    opt_desc = ("make_then_kitchen", needs_no_gluten)
                    if opt_cost < best_cost:
                        best_cost = opt_cost
                        best_option = opt_desc
                        best_child = child

                # If no option feasible, set a large but finite cost (heuristic remains finite)
                if best_option is None:
                    # cost to indicate many actions required, but finite:
                    # assume make+delivery cost of 6 (overestimate)
                    opt_cost = 6
                    opt_desc = ("fallback_make_assume", needs_no_gluten)
                    if opt_cost < best_cost:
                        best_cost = opt_cost
                        best_option = opt_desc
                        best_child = child

            # End iterating children: allocate resources for best_child using best_option
            if best_child is None:
                # safety break; shouldn't happen
                total_cost += 10
                break

            # Retrieve child details and allocate
            child = best_child
            place = self.child_waiting_place.get(child, None)
            allergy = self.child_allergy.get(child, "not_allergic")
            needs_no_gluten = (allergy == "allergic")

            # Apply allocation according to best_option
            opt = best_option
            if opt[0] == "ontray_same":
                entry = opt[1]
                # consume that ontray sandwich
                # remove from ontray_list
                for i, e in enumerate(ontray_list):
                    if e["sand"] == entry["sand"]:
                        del ontray_list[i]
                        break
                # the tray remains at the same place (serve doesn't move tray)
                # cost 1
                total_cost += 1
            elif opt[0] == "ontray_else":
                entry = opt[1]
                # move the tray to child's place then serve
                tray_name = entry["tray"]
                trays[tray_name] = place
                # consume ontray sandwich
                for i, e in enumerate(ontray_list):
                    if e["sand"] == entry["sand"]:
                        del ontray_list[i]
                        break
                total_cost += 2
            elif opt[0] == "kitchen_existing":
                kitchen_entry = opt[1]
                tray_name = opt[2]
                # Use that kitchen sandwich: perform put_on_tray (consumes at_kitchen_sandwich),
                # then possibly move_tray and serve.
                # Remove kitchen sandwich entry
                for i, e in enumerate(at_kitchen_list):
                    if e["sand"] == kitchen_entry["sand"]:
                        del at_kitchen_list[i]
                        break
                # perform put_on_tray -> sandwich becomes ontray at kitchen on chosen tray
                # Find the tray (tray_name) is at kitchen
                # Update ontray_list entry removal is immediate because we'll serve it now.
                if place == trays.get(tray_name, None):
                    # tray already at child's place (rare if place==kitchen)
                    # But since kitchen_entry assumed tray at kitchen, likely place==kitchen
                    # For place==kitchen: put_on_tray + serve
                    # After serve tray remains at kitchen (as child's place is kitchen)
                    total_cost += 2
                else:
                    # tray at kitchen, child at different place
                    # put_on_tray, move_tray, serve
                    trays[tray_name] = place  # after move_tray
                    total_cost += 3
            elif opt[0] == "kitchen_existing_no_tray_at_kitchen":
                kitchen_entry = opt[1]
                chosen_tray = opt[2]
                # Move chosen_tray to kitchen, put_on_tray, (move if needed), serve.
                # Update chosen_tray location
                trays[chosen_tray] = "kitchen"
                # Remove kitchen sandwich entry
                for i, e in enumerate(at_kitchen_list):
                    if e["sand"] == kitchen_entry["sand"]:
                        del at_kitchen_list[i]
                        break
                if place == "kitchen":
                    # move to kitchen + put + serve
                    total_cost += 3
                    # tray ends at kitchen (child place)
                    trays[chosen_tray] = "kitchen"
                else:
                    # move->put->move->serve
                    total_cost += 4
                    trays[chosen_tray] = place
            elif opt[0] == "make_then_kitchen":
                needs_no_gluten_flag = opt[1]
                # allocate a notexist sandwich name
                sand_name = notexist_list.pop() if notexist_list else None
                # consume bread and content from pools according to preference
                if needs_no_gluten_flag:
                    # allocate a no_gluten bread and no_gluten content
                    bread_used = next(iter(no_gluten_breads)) if no_gluten_breads else None
                    content_used = next(iter(no_gluten_contents)) if no_gluten_contents else None
                    if bread_used:
                        no_gluten_breads.remove(bread_used)
                    if content_used:
                        no_gluten_contents.remove(content_used)
                else:
                    # prefer gluten breads/contents
                    bread_used = None
                    content_used = None
                    if gluten_breads:
                        bread_used = next(iter(gluten_breads))
                        gluten_breads.remove(bread_used)
                    elif no_gluten_breads:
                        bread_used = next(iter(no_gluten_breads))
                        no_gluten_breads.remove(bread_used)
                    if gluten_contents:
                        content_used = next(iter(gluten_contents))
                        gluten_contents.remove(content_used)
                    elif no_gluten_contents:
                        content_used = next(iter(no_gluten_contents))
                        no_gluten_contents.remove(content_used)
                # After make, we treat it as a kitchen sandwich and then deliver as in kitchen_existing_no_tray_at_kitchen or kitchen_existing depending on tray presence
                # Find a tray at kitchen if any
                tray_at_kitchen = None
                for t, p in trays.items():
                    if p == "kitchen":
                        tray_at_kitchen = t
                        break
                if tray_at_kitchen is not None:
                    if place == "kitchen":
                        total_cost += 1 + 1 + 1  # make + put + serve
                        # tray remains at kitchen
                        trays[tray_at_kitchen] = "kitchen"
                    else:
                        total_cost += 1 + 1 + 1 + 1  # make + put + move + serve
                        trays[tray_at_kitchen] = place
                else:
                    # need to move some tray to kitchen first
                    some_tray = next(iter(trays.keys()))
                    if place == "kitchen":
                        total_cost += 1 + 1 + 1  # make + move_tray + put + serve
                        trays[some_tray] = "kitchen"
                    else:
                        total_cost += 1 + 1 + 1 + 1  # make + move_to_kitchen + put + move_to_place + serve
                        trays[some_tray] = place
                # sandwich name became used and served; nothing to add to at_kitchen_list or ontray_list
            elif opt[0] == "fallback_make_assume":
                # allocate generic estimated cost
                total_cost += 6
            else:
                # unknown option: add a conservative cost
                total_cost += 5

            # Mark this child as served in heuristic's bookkeeping
            unserved_children.remove(child)

        return total_cost