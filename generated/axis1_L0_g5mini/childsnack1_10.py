from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Return components of a fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()


def fact_matches(fact, *pattern):
    """
    Compare fact parts against a pattern with wildcards.
    Example: fact_matches('(at tray1 kitchen)', 'at', '*', 'kitchen') -> True
    """
    parts = get_parts(fact)
    return all(fnmatch(p, pat) for p, pat in zip(parts, pattern))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    Heuristic for the childsnack domain that estimates the number of actions
    required to serve all children remaining unserved in the current state.
    The heuristic is domain-aware: it recognizes existing sandwiches (at
    kitchen or already on trays), gluten-free sandwiches, tray locations,
    and bread/content resources. It then greedily matches available resources
    to the remaining children and estimates the minimal remaining actions
    to serve each child.

    Assumptions
    -----------
    - Each serve action costs 1 action.
    - Putting a sandwich on a tray costs 1 action (put_on_tray).
    - Making a sandwich (either make_sandwich or make_sandwich_no_gluten)
      costs 1 action.
    - Moving a tray between places costs 1 action (move_tray).
    - If a child is allergic to gluten, they must be served a no_gluten_sandwich.
      Otherwise any sandwich is acceptable.
    - We optimistically assume that trays (if needed) can be moved or used
      when required. If there is no tray at the kitchen, we count one extra
      move action to bring a tray to the kitchen before putting on it.
    - For simplicity and efficiency, the heuristic is greedy and does not
      attempt an exhaustive assignment. It aims to produce a low estimate
      that correlates well with required actions and is fast to compute.
    - The heuristic is not admissible (it may be optimistic), but it is
      finite and returns 0 exactly for goal states.

    Heuristic Initialization
    ------------------------
    The constructor extracts the following static information from the task:
    - The list of goal 'served' facts to determine which children must be served.
    - Which children are allergic to gluten (allergic_gluten facts).
    - Which children are explicitly not allergic (not_allergic_gluten facts).
    - Waiting locations for children (waiting child place facts).
      (In many instances waiting is static; if not static it will still be
       available in the state at runtime and is handled there.)
    The static facts are stored in dictionaries/sets to speed up repeated calls.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    Given a node (state) the heuristic follows these main steps:

    1. Goal check:
       - If all goal 'served' facts are already true in the state, return 0.

    2. Extract dynamic facts from the state:
       - Which sandwiches exist and where:
         * at_kitchen_sandwich S
         * ontray S T  (and the tray's location: at T Place)
         * no_gluten_sandwich S
       - Counts of breads and contents at the kitchen, and how many are
         marked no_gluten (these determine how many gluten-free sandwiches
         can be produced without violating the domain's gluten constraints).
       - Number of 'notexist' sandwich placeholders (how many new sandwiches
         *could* be made). For heuristic simplicity we allow creating as many
         sandwiches as needed but we still extract this count and the bread/content
         availability to inform a better estimate when possible.
       - Tray positions (which trays are at kitchen and other places).

    3. Identify unserved children:
       - From the task goals get the list of children that must be served.
       - For each child determine: allergic (True/False) and waiting place.

    4. Greedy matching and cost accounting:
       - First consume the cheapest available resources:
         a) If there is a sandwich already on a tray located at the child's waiting
            place and matching allergy requirements -> only a serve action needed (cost 1).
         b) Else if there is a sandwich on a tray located elsewhere but matching allergy:
            move the tray to the child's place (cost 1) and serve (cost 1) => cost 2.
         c) Else if there is a sandwich at the kitchen matching allergy:
            - If a tray is available at the kitchen, do: put_on_tray (1), move_tray
              to child's place if needed (1 if different place), serve (1).
              If no tray at kitchen but there are trays elsewhere, add 1 to bring
              a tray to the kitchen first. (Typical cost 2-4 depending on conditions.)
         d) Else we need to make a new sandwich (preferably gluten-free for allergic children):
            - Make sandwich (1), put_on_tray (1), move_tray if needed (0/1), serve (1).
            - If no tray at kitchen, add 1 to bring a tray to the kitchen.
       - The heuristic decrements the available resource counters (sandwiches, trays,
         gluten-free breads/contents) as it assigns resources to children so later
         children are estimated with updated availability.

    5. Return the summed cost from all matched steps as the heuristic estimate.

    Notes on design choices:
    - This heuristic balances accuracy and speed. It models important structure
      of the domain (need for gluten-free sandwiches, trays, tray movement,
      sandwich creation) but intentionally avoids solving an internal assignment
      optimization problem exactly (which would be more expensive).
    - Because it is greedy and optimistic in a few respects (e.g., assuming moving
      a tray to kitchen when needed is always possible), it is not admissible,
      but it typically yields good guidance for greedy best-first search.
    """

    def __init__(self, task):
        # Goals: set of goal facts (strings), e.g., '(served child1)'
        self.goals = set(task.goals)
        # Static facts: typically contains allergic and waiting info in many PDDL encodings.
        static_facts = set(task.static)

        # Extract static information: allergic / not_allergic sets, waiting mapping.
        self.allergic = set()
        self.not_allergic = set()
        self.waiting_static = {}  # child -> place

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "allergic_gluten":
                self.allergic.add(parts[1])
            elif pred == "not_allergic_gluten":
                self.not_allergic.add(parts[1])
            elif pred == "waiting":
                # waiting child place
                if len(parts) >= 3:
                    self.waiting_static[parts[1]] = parts[2]

    def __call__(self, node):
        state = set(node.state)  # mutable copy for analysis

        # 1) If goal already reached return 0
        if self.goals <= state:
            return 0

        # Helper containers extracted from current state
        # Sandwiches at kitchen
        at_kitchen_sandwichs = set()
        # Sandwiches that are gluten-free
        no_gluten_sandwichs = set()
        # Sandwiches that are on trays: sandwich -> tray
        ontray = {}  # s -> t
        # Tray positions: tray -> place
        tray_pos = {}
        # notexist sandwiches count (how many creates are possible)
        notexist_count = 0

        # Bread/content counts and gluten-free counts (only consider those at kitchen)
        breads_at_kitchen = set()
        contents_at_kitchen = set()
        ng_breads_at_kitchen = set()
        ng_contents_at_kitchen = set()

        # Gather served facts in current state for quick check
        served_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]
            if pred == "at_kitchen_sandwich":
                at_kitchen_sandwichs.add(parts[1])
            elif pred == "no_gluten_sandwich":
                no_gluten_sandwichs.add(parts[1])
            elif pred == "ontray":
                # format: (ontray sandwX trayY)
                if len(parts) >= 3:
                    s, t = parts[1], parts[2]
                    ontray[s] = t
            elif pred == "at":
                # could be (at tray1 kitchen) or (at tray1 table1)
                if len(parts) >= 3:
                    obj, place = parts[1], parts[2]
                    tray_pos[obj] = place
            elif pred == "notexist":
                notexist_count += 1
            elif pred == "at_kitchen_bread":
                breads_at_kitchen.add(parts[1])
            elif pred == "at_kitchen_content":
                contents_at_kitchen.add(parts[1])
            elif pred == "no_gluten_bread":
                ng_breads_at_kitchen.add(parts[1])  # we will intersect later
            elif pred == "no_gluten_content":
                ng_contents_at_kitchen.add(parts[1])
            elif pred == "served":
                served_in_state.add(parts[1])

        # Only breads/content that are both at_kitchen and marked no_gluten are valid for make_sandwich_no_gluten
        ng_breads_at_kitchen = breads_at_kitchen & ng_breads_at_kitchen
        ng_contents_at_kitchen = contents_at_kitchen & ng_contents_at_kitchen

        # Count trays at kitchen and total trays
        trays_at_kitchen = sum(1 for tray, place in tray_pos.items() if place == "kitchen")
        total_trays = len(tray_pos)

        # Determine children that still need to be served (from goals)
        children_to_serve = []
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "served":
                child = parts[1]
                if child not in served_in_state:
                    # determine allergy and waiting place (use static if available)
                    allergic = child in self.allergic
                    # waiting place may be static (preferred), if not present try to find in state
                    waiting_place = self.waiting_static.get(child)
                    if waiting_place is None:
                        # attempt to find waiting in the state
                        for f in state:
                            p = get_parts(f)
                            if p and p[0] == "waiting" and p[1] == child and len(p) >= 3:
                                waiting_place = p[2]
                                break
                    # If still unknown, assume not kitchen (move required)
                    if waiting_place is None:
                        waiting_place = "unknown_place"
                    children_to_serve.append((child, allergic, waiting_place))

        # Available prepared sandwich pools (sandwich id -> properties)
        # For matching we only need to know whether sandwich is gluten-free and where it is
        sandwiches_available = {}  # s -> dict { 'gluten_free': bool, 'where': 'ontray'/'kitchen', 'tray': t (if ontray) }

        for s in at_kitchen_sandwichs:
            sandwiches_available[s] = {"gluten_free": s in no_gluten_sandwichs, "where": "kitchen", "tray": None}

        for s, t in ontray.items():
            gf = s in no_gluten_sandwichs
            sandwiches_available[s] = {"gluten_free": gf, "where": "ontray", "tray": t}

        # Note: some sandwiches might be in no_gluten_sandwichs but not in kitchens/ontray (rare),
        # still include them as kitchen for safety
        for s in no_gluten_sandwichs:
            if s not in sandwiches_available:
                sandwiches_available[s] = {"gluten_free": True, "where": "kitchen", "tray": None}

        # Convert to lists for consumption
        # Create indexed lists for quick greedy selection
        ontray_at_place = {}  # place -> list of sandwich ids on trays at that place
        ontray_elsewhere = []  # list of (sandwich, tray_place)
        kitchen_sandwich_list = []  # list of sandwich ids at kitchen

        for s, props in list(sandwiches_available.items()):
            if props["where"] == "ontray":
                t = props["tray"]
                place = tray_pos.get(t, None)
                if place is None:
                    ontray_elsewhere.append((s, None))
                else:
                    ontray_at_place.setdefault(place, []).append(s)
            else:
                kitchen_sandwich_list.append(s)

        # Mutable resource counters and sets for greedy consumption
        ng_breads = set(ng_breads_at_kitchen)
        ng_contents = set(ng_contents_at_kitchen)
        breads = set(breads_at_kitchen)
        contents = set(contents_at_kitchen)
        notexist_available = notexist_count
        trays_kitchen = trays_at_kitchen
        trays_total = total_trays

        total_cost = 0

        # 1) First pass: use sandwiches already on trays at the child's place (serve action only)
        remaining_children = []
        for child, allergic, place in children_to_serve:
            served = False
            if place in ontray_at_place:
                # find sandwich in that place that matches allergy
                candidates = ontray_at_place[place]
                found_idx = None
                for idx, s in enumerate(candidates):
                    if (not allergic) or (sandwiches_available.get(s, {}).get("gluten_free", False)):
                        # assign this sandwich
                        found_idx = idx
                        break
                if found_idx is not None:
                    s = candidates.pop(found_idx)
                    # consume sandwich
                    sandwiches_available.pop(s, None)
                    # serve cost 1
                    total_cost += 1
                    served = True
                    # After serving, the tray remains at place (but sandwich removed). We do not explicitly
                    # track trays for future use beyond counts for kitchen availability.
            if not served:
                remaining_children.append((child, allergic, place))

        # 2) Second pass: use sandwiches on trays elsewhere (move + serve)
        children_after_ontray_elsewhere = []
        # build list of ontray elsewhere with their tray places
        ontray_list_with_places = []
        for s, props in sandwiches_available.items():
            if props["where"] == "ontray":
                t = props["tray"]
                place = tray_pos.get(t, None)
                ontray_list_with_places.append((s, place))
        # try to match remaining children greedily
        used_ontray_elsewhere = set()
        for child, allergic, place in remaining_children:
            assigned = False
            for idx, (s, tray_place) in enumerate(ontray_list_with_places):
                if s in used_ontray_elsewhere:
                    continue
                if (not allergic) or sandwiches_available.get(s, {}).get("gluten_free", False):
                    # move tray if necessary and serve
                    move_cost = 0 if tray_place == place else 1
                    total_cost += move_cost + 1  # move + serve
                    used_ontray_elsewhere.add(s)
                    sandwiches_available.pop(s, None)
                    assigned = True
                    break
            if not assigned:
                children_after_ontray_elsewhere.append((child, allergic, place))

        # 3) Third pass: use sandwiches at kitchen (put_on_tray + possible move + serve)
        children_after_kitchen = []
        kitchen_sandwichs = [s for s, p in sandwiches_available.items() if p["where"] == "kitchen"]
        # we don't track specific sandwich ids for consumption beyond count
        kitchen_queue = kitchen_sandwichs[:]
        for child, allergic, place in children_after_ontray_elsewhere:
            assigned = False
            # find a kitchen sandwich that matches
            found_idx = None
            for idx, s in enumerate(kitchen_queue):
                if (not allergic) or sandwiches_available.get(s, {}).get("gluten_free", False):
                    found_idx = idx
                    break
            if found_idx is not None:
                s = kitchen_queue.pop(found_idx)
                sandwiches_available.pop(s, None)
                # cost to put on tray
                extra_cost = 0
                if trays_kitchen > 0:
                    # use an existing tray at kitchen, it will be consumed temporarily
                    trays_kitchen -= 1
                elif trays_total > 0:
                    # need to move a tray from elsewhere to kitchen first
                    extra_cost += 1
                    trays_total -= 1  # we assume we moved one to kitchen and will use it
                else:
                    # no tray at all: we must still count a put_on_tray cost (optimistic fallback)
                    extra_cost += 0
                put_cost = 1
                # move to child's place if necessary
                move_cost = 0 if place == "kitchen" else 1
                serve_cost = 1
                total_cost += extra_cost + put_cost + move_cost + serve_cost
                assigned = True
            if not assigned:
                children_after_kitchen.append((child, allergic, place))

        # 4) Final pass: create new sandwiches then put on tray + move + serve
        # For creation, prefer using gluten-free breads/contents for allergic children.
        for child, allergic, place in children_after_kitchen:
            # Check resource availability for gluten-free if needed
            make_feasible = True
            extra_make_cost = 0
            if allergic:
                # need both ng bread and ng content at kitchen ideally
                if ng_breads:
                    ng_breads.pop()  # consume one
                else:
                    # if no gluten-free bread available, be optimistic and allow a make anyway
                    extra_make_cost += 0
                if ng_contents:
                    ng_contents.pop()
                else:
                    extra_make_cost += 0
            else:
                # consume any bread/content if available
                if breads:
                    breads.pop()
                else:
                    extra_make_cost += 0
                if contents:
                    contents.pop()
                else:
                    extra_make_cost += 0

            # consume a notexist slot if present; otherwise optimistic
            if notexist_available > 0:
                notexist_available -= 1
            else:
                # optimistic: allow creation anyway
                extra_make_cost += 0

            # make cost
            make_cost = 1
            # then need put_on_tray: may need a tray at kitchen
            extra_put_cost = 0
            if trays_kitchen > 0:
                trays_kitchen -= 1
            elif trays_total > 0:
                extra_put_cost += 1  # move a tray to kitchen before putting
                trays_total -= 1
            else:
                extra_put_cost += 0

            put_cost = 1
            move_cost = 0 if place == "kitchen" else 1
            serve_cost = 1

            total_cost += make_cost + extra_make_cost + extra_put_cost + put_cost + move_cost + serve_cost

        # Heuristic must be finite: ensure we produce an integer
        if total_cost < 0:
            total_cost = 0

        return int(total_cost)