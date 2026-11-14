from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing surrounding parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards (*).

    :param fact: A PDDL fact string, e.g. "(at_kitchen_bread bread2)"
    :param args: Pattern parts to match, e.g., "at_kitchen_bread", "*"
    :return: True if the fact matches the pattern, False otherwise
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnack domain.

    Summary:
    ----------
    Estimates the number of actions required to serve all children sandwiches at their respective places.
    The heuristic counts the minimal action costs to: make sandwiches considering gluten allergies, 
    place sandwiches on trays, move trays if necessary, and serve sandwiches to waiting children.
    It assumes the kitchen is the only place where sandwiches can be prepared and trays start in the kitchen.
    It considers gluten allergies to decide if a no-gluten sandwich is needed.
    The heuristic is zero only in goal states where all children are served.

    Assumptions:
    ------------
    - Each child needs exactly one served sandwich.
    - Each sandwich can only serve one child.
    - Bread and content portions must exist in the kitchen to make sandwiches.
    - No gluten sandwiches are needed for children allergic to gluten.
    - Trays start at the kitchen and can be moved to children's places.
    - Trays can hold multiple sandwiches.
    - Serving requires the tray to be at the child's place.
    - Actions have uniform cost of 1.
    - The heuristic does not need to be admissible but aims to be informative and efficiently computable.

    Heuristic Initialization:
    -------------------------
    - Extract the sets of children allergic and not allergic to gluten from static facts.
    - Extract the set of waiting children and their corresponding places.
    - Extract the goal children to be served.
    - Track no-gluten breads and content portions from static facts.
    - Track tray objects as those appearing in facts like "(at tray kitchen)" or "(at tray place)".
    - Extract from goals the children that must be served (usually all children).

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. Identify children not yet served by checking "served" facts.
    2. For each unserved child:
       a. Determine if the child is allergic to gluten or not.
       b. For allergic children, a no-gluten sandwich must be made.
       c. For not allergic children, any sandwich can be made.
       d. Check if a suitable sandwich already exists in the kitchen or on trays.
          - If present on tray at child's place and not yet served, just count serving as 1 action.
          - If present on tray but tray not at child's place, count 1 move_tray action plus serving.
          - If sandwich exists in kitchen but not on tray, count put_on_tray + potentially moving tray + serving.
          - Else, sandwich must be made from bread and content (make_sandwich or make_sandwich_no_gluten) plus tray actions and serving.
    3. For tray usage:
       - If no tray with the sandwich at child's place, the tray must be moved accordingly.
       - Tray moves are counted only once even if multiple sandwiches for children on same place share a tray.
    4. Sum all these minimal action counts for all unserved children.
    5. Return the total minimal action cost estimate as heuristic value.

    This heuristic balances sandwich preparation, tray logistics, and serving actions to approximate
    a tight estimate of remaining work while being efficiently computable.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract children allergic and not allergic to gluten
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "not_allergic_gluten", "*")
        }

        # Extract children waiting and their places
        self.waiting_children_places = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:3]
                self.waiting_children_places[child] = place

        # Extract no gluten breads and no gluten contents
        self.no_gluten_breads = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "no_gluten_bread", "*")
        }
        self.no_gluten_contents = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "no_gluten_content", "*")
        }

        # Extract trays from static or initial facts (any object appearing in at tray ...)
        self.tray_objects = set()
        for fact in self.static | task.initial_state:
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                # In domain trays are type tray, usually prefix e.g. tray1, but just store all that appear in "at" facts, we filter later
                if obj.startswith("tray"):
                    self.tray_objects.add(obj)

        # Extract goal children to be served
        self.goal_children = set()
        for fact in self.goals:
            if match(fact, "served", "*"):
                self.goal_children.add(get_parts(fact)[1])

    def __call__(self, node):
        state = node.state

        # Helper sets fetched from state
        served_children = {
            get_parts(f)[1]
            for f in state
            if match(f, "served", "*")
        }

        # Children not yet served but in goal
        unserved_children = self.goal_children - served_children

        # Track sandwiches and their locations
        kitchen_sandwiches = set(
            get_parts(f)[1]
            for f in state
            if match(f, "at_kitchen_sandwich", "*")
        )
        no_gluten_sandwiches = set(
            get_parts(f)[1]
            for f in state
            if match(f, "no_gluten_sandwich", "*")
        )

        # Sandwiches on trays and tray locations
        # Map tray -> place
        tray_location = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = get_parts(fact)[1:3]
                if obj in self.tray_objects:
                    tray_location[obj] = place
        # Map sandwich->tray
        sandwich_on_tray = {}
        for fact in state:
            if match(fact, "ontray", "*", "*"):
                sandwich, tray = get_parts(fact)[1:3]
                sandwich_on_tray[sandwich] = tray

        # Construct sandwich location info (place or kitchen)
        sandwich_location = {}
        for s in kitchen_sandwiches:
            sandwich_location[s] = "kitchen"
        for s, t in sandwich_on_tray.items():
            if t in tray_location:
                sandwich_location[s] = tray_location[t]

        # Availability of bread and content in kitchen
        kitchen_breads = {
            get_parts(f)[1]
            for f in state
            if match(f, "at_kitchen_bread", "*")
        }
        kitchen_contents = {
            get_parts(f)[1]
            for f in state
            if match(f, "at_kitchen_content", "*")
        }

        # Count available (not yet used) no-gluten breads and contents for making sandwiches
        # This heuristic is simplified and assumes that each sandwich requires 1 bread and 1 content.
        # For remaining sandwiches needed, count how many sandwiches can be made from kitchen supplies.
        breads_left_count = len(kitchen_breads)
        contents_left_count = len(kitchen_contents)

        # We assume tray can hold multiple sandwiches, so moving tray per place only counted once

        # Cost counters
        actions = 0

        # Track which trays need moving per place to avoid double counting tray moves
        trays_need_move_to_place = set()

        # For each unserved child, estimate minimal cost to serve
        for child in unserved_children:
            allergic = child in self.allergic_children
            place = self.waiting_children_places.get(child, None)
            if place is None:
                # No waiting place info, must count conservative cost (serve action)
                actions += 1
                continue

            # Find if there's a suitable sandwich already prepared for this child

            # Sandwich candidates for the child:
            # Allergic -> needs no_gluten_sandwich, Not allergic -> any sandwich
            # Find sandwiches in kitchen or on trays:

            # Sandwiches matching requirements
            if allergic:
                suitable_sandwiches = no_gluten_sandwiches
            else:
                # Any sandwich is suitable: kitchen_sandwiches includes no_gluten_sandwiches, so union
                suitable_sandwiches = kitchen_sandwiches

            # Sandwiches of required type that are ready (in kitchen or on trays)
            sandwiches_ready = {
                s for s in suitable_sandwiches if s in sandwich_location
            }

            # Try to find sandwich already on tray at child's place
            sandwich_on_tray_at_place = None
            sandwich_on_tray_elsewhere = None
            sandwich_in_kitchen = None
            # Keep track of tray for already placed sandwiches
            tray_for_this_sandwich = None

            for s in sandwiches_ready:
                loc = sandwich_location[s]
                if loc == place:
                    # Sandwich on tray at child's place
                    sandwich_on_tray_at_place = s
                    tray_for_this_sandwich = sandwich_on_tray.get(s, None)
                    break  # best case
            if sandwich_on_tray_at_place is None:
                # Try to find sandwich on tray elsewhere
                for s in sandwiches_ready:
                    loc = sandwich_location[s]
                    if loc != "kitchen" and loc != place:
                        sandwich_on_tray_elsewhere = s
                        tray_for_this_sandwich = sandwich_on_tray.get(s, None)
                        break
            if sandwich_on_tray_at_place is None and sandwich_on_tray_elsewhere is None:
                # Try to find sandwich in kitchen
                for s in sandwiches_ready:
                    if sandwich_location[s] == "kitchen":
                        sandwich_in_kitchen = s
                        break

            # Now compute minimal cost considering these cases

            if sandwich_on_tray_at_place is not None:
                # Only need to serve (1 action)
                actions += 1
                # This sandwich is taken, remove it from location to avoid double counting
                # but since heuristic is stateless, no need to update sets
                continue

            if sandwich_on_tray_elsewhere is not None and tray_for_this_sandwich is not None:
                # Need to move tray to child's place, then serve
                # Count 1 move_tray + 1 serve
                # Count move tray only once per tray per place:
                tray_place_pair = (tray_for_this_sandwich, place)
                if tray_place_pair not in trays_need_move_to_place:
                    actions += 1  # move_tray
                    trays_need_move_to_place.add(tray_place_pair)
                actions += 1  # serve
                continue

            if sandwich_in_kitchen is not None:
                # Need to put sandwich on tray, move tray if not at child's place, then serve
                # Find tray in kitchen or else pick any tray located in kitchen or place
                tray_in_kitchen = None
                for t in self.tray_objects:
                    if tray_location.get(t) == "kitchen":
                        tray_in_kitchen = t
                        break
                if tray_in_kitchen is None:
                    # No tray at kitchen, pick any tray (heuristic: just pick first known tray)
                    tray_in_kitchen = next(iter(self.tray_objects), None)
                actions += 1  # put_on_tray
                tray_place_pair = (tray_in_kitchen, place)
                if tray_location.get(tray_in_kitchen) != place:
                    if tray_place_pair not in trays_need_move_to_place:
                        actions += 1  # move_tray
                        trays_need_move_to_place.add(tray_place_pair)
                actions += 1  # serve
                continue

            # Need to make sandwich
            # Check if we have bread and content for the sandwich type

            # For no-gluten (allergic), must use no-gluten bread and content
            # For normal, any bread and content

            can_make_no_gluten = (len(self.no_gluten_breads & kitchen_breads) > 0 and
                                  len(self.no_gluten_contents & kitchen_contents) > 0)
            can_make_normal = (len(kitchen_breads) > 0 and len(kitchen_contents) > 0)

            if allergic and not can_make_no_gluten:
                # Cannot make required sandwich (no bread or content), heuristic arbitrarily large
                # but problem states finite heuristic for solvable states, so just count large cost
                actions += 10
                continue

            if not allergic and not can_make_normal:
                # Same as above for normal sandwich
                actions += 10
                continue

            # Make sandwich cost: make_sandwich or make_sandwich_no_gluten
            actions += 1

            # Remove one bread and one content from kitchen stocks (heuristic only)
            # To avoid double counting, decrease counts locally
            if allergic:
                # Remove no-gluten bread and content if available
                bread_used = next(iter(self.no_gluten_breads & kitchen_breads))
                kitchen_breads.remove(bread_used)
                content_used = next(iter(self.no_gluten_contents & kitchen_contents))
                kitchen_contents.remove(content_used)
            else:
                # Remove any bread and content
                # Prefer no-gluten ones if available (since they will also be used)
                if kitchen_breads & self.no_gluten_breads:
                    bread_used = next(iter(kitchen_breads & self.no_gluten_breads))
                else:
                    bread_used = next(iter(kitchen_breads))
                kitchen_breads.remove(bread_used)

                if kitchen_contents & self.no_gluten_contents:
                    content_used = next(iter(kitchen_contents & self.no_gluten_contents))
                else:
                    content_used = next(iter(kitchen_contents))
                kitchen_contents.remove(content_used)

            # Then put sandwich on tray
            # Find tray in kitchen or create new (heuristic assumes tray available)
            tray_in_kitchen = None
            for t in self.tray_objects:
                if tray_location.get(t) == "kitchen":
                    tray_in_kitchen = t
                    break
            if tray_in_kitchen is None:
                tray_in_kitchen = next(iter(self.tray_objects), None)
            actions += 1  # put_on_tray

            # Move tray to child's place if needed
            tray_place_pair = (tray_in_kitchen, place)
            if tray_location.get(tray_in_kitchen) != place:
                if tray_place_pair not in trays_need_move_to_place:
                    actions += 1  # move_tray
                    trays_need_move_to_place.add(tray_place_pair)

            # Then serve
            actions += 1

        return actions
