from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Remove surrounding parentheses and split into tokens."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check whether a fact matches a pattern with possible '*' wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Summary
    -------
    A domain-dependent, non-admissible heuristic for the childsnack domain.
    The heuristic estimates the number of actions still required to serve all
    children that are not yet marked as (served ...). It reasons about:
      - existing sandwiches (either at the kitchen or on trays),
      - trays and their current locations,
      - available bread and content portions (including no-gluten resources),
      - and which children require no-gluten sandwiches (allergic_gluten).

    The heuristic returns 0 for goal states (all children served) and otherwise
    returns a finite integer estimate, preferring to re-use already available
    sandwiches and trays whenever possible.

    Assumptions
    -----------
    - Moving a tray between any two places takes 1 action (move_tray).
    - Putting a sandwich onto a tray (put_on_tray) costs 1 action.
    - Making a sandwich (make_sandwich or make_sandwich_no_gluten) costs 1 action.
    - Serving a child (serve_sandwich or serve_sandwich_no_gluten) costs 1 action.
    - There is no extra cost for selecting a particular sandwich object; sandwiches
      are fungible apart from the no-gluten property.
    - Making a no-gluten sandwich requires one no-gluten bread and one no-gluten
      content (both present at kitchen). Regular sandwiches consume any bread and
      any content available at kitchen.
    - The heuristic assumes direct tray moves (one action per move) and does not
      model sequences of moves between intermediate places — the domain's
      move_tray operator allows direct moves between places.

    Heuristic Initialization
    ------------------------
    In the constructor the heuristic extracts static information from the task:
      - Which children are allergic_gluten and which are not_allergic_gluten.
      - Waiting mapping child -> place (if present in static facts).
    These are used to determine for each child whether a no-gluten sandwich is
    required and to know the target location where the tray+sandwich must be
    located to serve that child.

    Step-By-Step Thinking for Computing Heuristic
    --------------------------------------------
    For a given state the heuristic does the following:

    1. Parse the state's facts to determine:
       - Which children are already served.
       - Locations of trays (mapping tray -> place and counts per place).
       - Sandwich facts:
         * which sandwiches currently exist as at_kitchen_sandwich,
         * which sandwiches are ontray (and on which tray/place),
         * which sandwiches are no_gluten_sandwich.
       - Counts of breads and contents at the kitchen, including counts of
         no-gluten breads and no-gluten contents.
       - Count of (notexist sandwich) facts to determine how many new sandwiches
         can still be created.

    2. Build the set of unserved children (children are discovered from the
       static info: allergic/not_allergic/waiting facts).

    3. Process unserved children greedily, prioritizing allergic children
       first (because they require no-gluten sandwiches). For each child the
       heuristic tries, in this order, to produce a low-cost way to reach a
       serve action:
         a) If there is already a sandwich on a tray at the child's place with
            the required gluten property, simply add 1 (serve) and mark that
            sandwich as used.
         b) Else if there is any sandwich on a tray elsewhere with the correct
            property, move that tray to the child's place (cost 1) and serve
            (cost 1) — total 2.
         c) Else if there is a matching sandwich at the kitchen:
            - If no tray is at the kitchen, move any tray to the kitchen (cost 1).
            - Put the sandwich on a tray (put_on_tray, cost 1).
            - If the child's place != kitchen, move the tray to the child's
              place (cost 1).
            - Serve (cost 1).
            The exact cost for this case is 1..4 depending on tray availability
            and whether the child is at kitchen.
            The heuristic also simulates the final location of the tray after
            serving (trays remain at the destination).
         d) Else (no existing sandwich usable): create a sandwich (make action,
            cost 1). If the child is allergic it tries to create a no-gluten
            sandwich consuming a no-gluten bread and a no-gluten content. For
            non-allergic children it will consume any available bread and content.
            After making, the sandwich ends up at the kitchen and the heuristic
            continues as in (c).
       After each assignment the heuristic updates its simulated resources
       (sandwich counts, bread/content supplies, trays' positions) so later
       children can reuse what was left and the estimate reflects resource
       competition.

    4. If at any point a child requires a no-gluten sandwich but there are no
       available no-gluten breads or contents and no existing no-gluten
       sandwiches, the heuristic assigns a large penalty (a high finite
       value) for that child to indicate difficulty (this makes the heuristic
       finite but large for unsolvable or resource-starved states).

    5. Sum costs for all unserved children and return the total as the heuristic
       value.

    The heuristic is intentionally greedy and approximative (not admissible),
    aiming to provide a good guidance for greedy best-first search by counting
    plausible minimum action sequences while modeling reuse of trays and
    sandwiches.
    """

    # penalty for impossible/unavailable required resources (keeps heuristic finite)
    RESOURCE_PENALTY = 1000

    def __init__(self, task):
        """Extract static information (allergies and waiting places) from task.static."""
        self.goals = task.goals
        static_facts = task.static

        # Children sets by allergy status
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "not_allergic_gluten", "*")
        }

        # Waiting mapping child -> place (may also appear in the state; static is preferred)
        self.waiting = {}
        for fact in static_facts:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1], get_parts(fact)[2]
                self.waiting[child] = place

        # Collect known children (from static info); at runtime we will also
        # consider children that appear in the state if needed.
        self.known_children = set(self.allergic_children) | set(self.not_allergic_children) | set(self.waiting.keys())

    def __call__(self, node):
        """Compute the action-count estimate to serve all children not yet served."""
        state = node.state

        # Quick goal check: if all goals satisfied, heuristic is zero
        if self.goals <= state:
            return 0

        # Parse state facts
        served_children = {
            get_parts(fact)[1] for fact in state if match(fact, "served", "*")
        }

        # Determine children to consider: union of known static children and any waiting facts in state
        children = set(self.known_children)
        for fact in state:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1], get_parts(fact)[2]
                children.add(child)
                # update waiting mapping for runtime facts
                if child not in self.waiting:
                    self.waiting[child] = place

        # Build tray location mapping and counts per place
        tray_location = {}
        trays_at_place = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = get_parts(fact)[1], get_parts(fact)[2]
                # We only track trays (their names are typed as tray in PDDL but not visible here),
                # but facts (at ?t ?p) include any 'at' facts; we assume trays are the ones that
                # also appear in other tray-related facts (e.g., ontray).
                # For safety, consider any object that also appears as a tray in ontray facts or goal static; however,
                # we will treat every 'at' object as potential tray and keep only those that appear later as trays.
                tray_location[obj] = place
                trays_at_place.setdefault(place, 0)
                trays_at_place[place] += 1

        # Sandwich-related collections
        ontray = {}          # sandwich -> tray
        ontray_by_place = {} # place -> list of sandwiches
        at_kitchen_sandwiches = set()  # sandwiches present at kitchen (at_kitchen_sandwich)
        no_gluten_sandwiches = set()   # sandwiches flagged no_gluten_sandwich
        notexist_sandwiches = set()    # available names that can still be made

        for fact in state:
            if match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1], get_parts(fact)[2]
                ontray[s] = t
                # determine tray location if known
                place = tray_location.get(t)
                if place is not None:
                    ontray_by_place.setdefault(place, []).append(s)
            elif match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                at_kitchen_sandwiches.add(s)
            elif match(fact, "no_gluten_sandwich", "*"):
                s = get_parts(fact)[1]
                no_gluten_sandwiches.add(s)
            elif match(fact, "notexist", "*"):
                s = get_parts(fact)[1]
                notexist_sandwiches.add(s)

        # Bread/content counts at kitchen
        at_kitchen_breads = set()
        at_kitchen_contents = set()
        no_gluten_breads = set()
        no_gluten_contents = set()

        for fact in state:
            if match(fact, "at_kitchen_bread", "*"):
                b = get_parts(fact)[1]
                at_kitchen_breads.add(b)
            elif match(fact, "at_kitchen_content", "*"):
                c = get_parts(fact)[1]
                at_kitchen_contents.add(c)
            elif match(fact, "no_gluten_bread", "*"):
                b = get_parts(fact)[1]
                no_gluten_breads.add(b)
            elif match(fact, "no_gluten_content", "*"):
                c = get_parts(fact)[1]
                no_gluten_contents.add(c)

        # Some no_gluten bread/content may also be at the kitchen; enforce intersection
        no_gluten_breads_at_kitchen = no_gluten_breads & at_kitchen_breads
        no_gluten_contents_at_kitchen = no_gluten_contents & at_kitchen_contents

        # For convenience build lists of sandwiches by location and property
        # Sandwich is considered 'available' if it's at kitchen or on a tray somewhere
        sandwiches_ontray = set(ontray.keys())
        sandwiches_at_kitchen = set(at_kitchen_sandwiches)
        # Existing usable sandwiches (anywhere) with gluten status
        existing_no_gluten_sandwichs_anywhere = {s for s in (sandwiches_ontray | sandwiches_at_kitchen) if s in no_gluten_sandwiches}
        existing_any_sandwichs_anywhere = (sandwiches_ontray | sandwiches_at_kitchen)

        # Count of trays at kitchen (may be zero)
        trays_at_kitchen = trays_at_place.get("kitchen", 0)

        # Initialize simulation-local mutable resources
        used_sandwiches = set()  # sandwiches consumed/assigned (we remove from available lists)
        local_ontray_by_place = {p: list(lst) for p, lst in ontray_by_place.items()}
        local_at_kitchen_sandwiches = set(at_kitchen_sandwiches)
        local_existing_no_gluten = set(existing_no_gluten_sandwichs_anywhere)
        local_existing_any = set(existing_any_sandwichs_anywhere)
        local_trays_at_place = dict(trays_at_place)  # place -> count
        local_tray_location = dict(tray_location)    # tray -> place (not fully tracked for new moves)
        local_notexist = set(notexist_sandwiches)
        local_at_kitchen_breads = set(at_kitchen_breads)
        local_at_kitchen_contents = set(at_kitchen_contents)
        local_no_gluten_breads_at_kitchen = set(no_gluten_breads_at_kitchen)
        local_no_gluten_contents_at_kitchen = set(no_gluten_contents_at_kitchen)

        # Build list of unserved children and order allergic first
        unserved_children = [c for c in children if c not in served_children]
        # Use waiting mapping to determine child's place; if missing, assume not far (kitchen)
        def child_place(child):
            return self.waiting.get(child, "kitchen")

        # Partition allergic and non-allergic (unknown children without allergy facts assumed not_allergic)
        allergic = [c for c in unserved_children if c in self.allergic_children]
        non_allergic = [c for c in unserved_children if c not in self.allergic_children]

        total_cost = 0

        # Helper functions to mutate local resources when using trays/sandwiches
        def consume_ontray_at_place(place, require_no_gluten):
            """If there is an ontray sandwich at place with required gluten, consume it and return True."""
            lst = local_ontray_by_place.get(place, [])
            for s in lst:
                if s in used_sandwiches:
                    continue
                if require_no_gluten and s not in local_existing_no_gluten:
                    continue
                # consume
                used_sandwiches.add(s)
                lst.remove(s)
                # mark sandwich as consumed in general sets
                local_existing_any.discard(s)
                local_existing_no_gluten.discard(s)
                return True
            return False

        def consume_ontray_anywhere(require_no_gluten):
            """If there is an ontray sandwich anywhere with required gluten, consume it and return its tray place (or None)."""
            # prefer trays already at kitchen? Not necessary, just find any
            for place, lst in list(local_ontray_by_place.items()):
                for s in list(lst):
                    if s in used_sandwiches:
                        continue
                    if require_no_gluten and s not in local_existing_no_gluten:
                        continue
                    # consume
                    used_sandwiches.add(s)
                    lst.remove(s)
                    local_existing_any.discard(s)
                    local_existing_no_gluten.discard(s)
                    return place
            return None

        def consume_at_kitchen_sandwich(require_no_gluten):
            """If there is a sandwich at kitchen, of required gluten property, consume it and return True."""
            for s in list(local_at_kitchen_sandwiches):
                if s in used_sandwiches:
                    continue
                if require_no_gluten and s not in local_existing_no_gluten:
                    continue
                # consume
                used_sandwiches.add(s)
                local_at_kitchen_sandwiches.remove(s)
                local_existing_any.discard(s)
                local_existing_no_gluten.discard(s)
                return True
            return False

        def make_sandwich_at_kitchen(require_no_gluten):
            """
            Attempt to make a sandwich at kitchen of the required property.
            Returns True if successful and updates local resources.
            """
            # need a notexist sandwich name to instantiate
            if not local_notexist:
                return False
            if require_no_gluten:
                # need both a no-gluten bread and content at kitchen
                if not local_no_gluten_breads_at_kitchen or not local_no_gluten_contents_at_kitchen:
                    return False
                # consume one bread and one content
                local_no_gluten_breads_at_kitchen.pop()
                local_no_gluten_contents_at_kitchen.pop()
                # also remove from general at_kitchen lists
                # (we stored only names, popping done)
            else:
                # need any bread and any content
                if not local_at_kitchen_breads or not local_at_kitchen_contents:
                    return False
                local_at_kitchen_breads.pop()
                local_at_kitchen_contents.pop()

                # if popped were also no-gluten ones, ensure intersection sets are updated (already popped above)
                local_no_gluten_breads_at_kitchen.discard(None)  # noop to keep linter calm

            # create sandwich at kitchen (materialize one notexist name)
            s = local_notexist.pop()
            local_at_kitchen_sandwiches.add(s)
            local_existing_any.add(s)
            if require_no_gluten:
                local_existing_no_gluten.add(s)
            return True

        # Function to ensure a tray is available at kitchen and to simulate moving it
        def ensure_tray_at_kitchen():
            """Ensure at least one tray is at the kitchen. Return cost incurred (0 or 1) and update local_trays."""
            nonlocal total_cost
            if local_trays_at_place.get("kitchen", 0) > 0:
                return 0
            # need to move any tray to kitchen (cost 1). Choose a tray from any other place if exists.
            # If there are no trays at all, we treat it as impossible in this domain (but planner usually has trays)
            # Find some place with trays
            for place, cnt in list(local_trays_at_place.items()):
                if place == "kitchen":
                    continue
                if cnt > 0:
                    local_trays_at_place[place] -= 1
                    if local_trays_at_place[place] == 0:
                        del local_trays_at_place[place]
                    local_trays_at_place["kitchen"] = local_trays_at_place.get("kitchen", 0) + 1
                    return 1
            # no trays anywhere, treat as impossible -> return penalty
            return None

        # Deliver a sandwich that is currently at kitchen to a child at child_p (assumes a sandwich is available at kitchen)
        def deliver_kitchen_sandwich_to(child_p):
            """
            Simulate the actions to put a kitchen sandwich onto a tray and move it to child's place and serve.
            Returns cost incurred or RESOURCE_PENALTY (None case handled by caller).
            """
            # ensure a tray is at kitchen
            res = ensure_tray_at_kitchen()
            if res is None:
                return None
            cost = 0 if res is None else res
            # after ensuring, we must have a tray at kitchen
            # put_on_tray: 1
            cost += 1
            # After put_on_tray, the used tray will be moved to child's place (unless child is at kitchen)
            # If child place is kitchen, we just serve (1)
            if child_p != "kitchen":
                # move_tray from kitchen to child place: 1
                cost += 1
                # simulate tray movement: decrement kitchen, increment child place
                local_trays_at_place["kitchen"] -= 1
                if local_trays_at_place["kitchen"] == 0:
                    del local_trays_at_place["kitchen"]
                local_trays_at_place[child_p] = local_trays_at_place.get(child_p, 0) + 1
            else:
                # child at kitchen: the tray remains at kitchen
                pass
            # serve action
            cost += 1
            return cost

        # Process children in order: allergic first then non-allergic
        for child in allergic + non_allergic:
            if child in served_children:
                continue
            need_no_gluten = child in self.allergic_children
            cp = child_place(child)

            # 1) Check for ontray at child's place with required gluten
            if consume_ontray_at_place(cp, need_no_gluten):
                total_cost += 1  # serve
                # tray remains at place (we removed sandwich from ontray)
                continue

            # 2) Check for ontray anywhere with required gluten -> move tray + serve
            place_with_ontray = consume_ontray_anywhere(need_no_gluten)
            if place_with_ontray is not None:
                # If tray was at same place would have matched earlier; so here moving is required if place differs
                move_cost = 0 if place_with_ontray == cp else 1
                total_cost += move_cost + 1  # move (maybe 0) and serve
                # simulate tray movement if moved
                if move_cost == 1:
                    # decrement source place, increment target
                    local_trays_at_place[place_with_ontray] = local_trays_at_place.get(place_with_ontray, 1) - 1
                    if local_trays_at_place.get(place_with_ontray, 0) <= 0:
                        local_trays_at_place.pop(place_with_ontray, None)
                    local_trays_at_place[cp] = local_trays_at_place.get(cp, 0) + 1
                # after serve tray remains at cp (we already updated)
                continue

            # 3) Check for sandwich at kitchen with required gluten
            if consume_at_kitchen_sandwich(need_no_gluten):
                # deliver from kitchen to child (may need to ensure tray at kitchen)
                cost = deliver_kitchen_sandwich_to(cp)
                if cost is None:
                    # no tray anywhere: heavy penalty
                    total_cost += self.RESOURCE_PENALTY
                else:
                    total_cost += cost
                continue

            # 4) No existing sandwich available: attempt to make one
            made = make_sandwich_at_kitchen(need_no_gluten)
            if made:
                total_cost += 1  # cost of make action
                # now sandwich is at kitchen; consume it and deliver
                # consume the newly created sandwich (it is in local_at_kitchen_sandwiches)
                # We will consume one kitchen sandwich (any)
                consumed = consume_at_kitchen_sandwich(need_no_gluten)
                if not consumed:
                    # This should not happen, but handle robustly
                    total_cost += self.RESOURCE_PENALTY
                    continue
                cost = deliver_kitchen_sandwich_to(cp)
                if cost is None:
                    total_cost += self.RESOURCE_PENALTY
                else:
                    total_cost += cost
                continue

            # 5) Could not make required sandwich due to missing resources.
            # If child is not allergic, try to make non-no-gluten sandwich as fallback (if the original attempt was for no_gluten)
            if not need_no_gluten:
                # Already tried to make regular (since need_no_gluten false), so here it truly failed.
                total_cost += self.RESOURCE_PENALTY
            else:
                # Child needed no_gluten but we couldn't make it; maybe there exists a regular sandwich we can use?
                # But allergy forbids using regular sandwich, so unsatisfiable unless an existing no-gluten sandwich is produced.
                total_cost += self.RESOURCE_PENALTY

        return int(total_cost)