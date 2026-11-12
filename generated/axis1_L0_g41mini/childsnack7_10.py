from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at_kitchen_bread bread1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions approximately required to serve
    all waiting children their appropriate sandwiches, considering wheat/gluten allergies,
    sandwich creation, tray handling, and serving.

    Assumptions:
    - Each child must be served exactly one sandwich.
    - Gluten-allergic children require sandwiches made with gluten-free bread and content.
    - Sandwiches are created from one bread portion and one content portion.
    - Sandwiches must be placed on trays, and trays must be moved to the children's places before serving.
    - Actions can be done in any order to optimize serving multiple children; however, we estimate costs independently per child.
    - The kitchen contains bread portions, content portions, trays, and sandwiches (some may already be made).
    - The heuristic does not require being admissible but should guide search effectively.

    Heuristic Initialization:
    - Extract sets of children and determine their gluten-allergy status from static facts.
    - Extract static relationships such as which children are allergic.
    - Initialize counters for sandwiches, bread portions, content portions, trays, and sandwich states.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify children who are not yet served (goals not satisfied).
    2. For each waiting child not served, determine:
       - If a suitable sandwich (gluten-free if allergic, else any) is already on a tray at child's location → need only the serve action.
       - Else if suitable sandwich is on a tray but at kitchen → need to move the tray and serve.
       - Else if suitable sandwich is in kitchen but not on tray:
         - Count action to put sandwich on tray,
         - Move tray to child's place,
         - Serve.
       - Else (sandwich does not exist yet):
         - Need to make a sandwich with suitable bread/content,
         - Put sandwich on tray,
         - Move tray,
         - Serve.
    3. Bread and content portions availability is considered but the heuristic approximates without complex matching,
       assuming sufficient supply or counting lacks if obvious.
    4. If trays are insufficient for number of children needing sandwich-on-tray placement, would need multiple movement actions or reusing trays;
       heuristic assumes enough trays but accounts for number of trays available.
    5. Sum estimated minimal costs across children, considering actions per child independently with minimal sharing.
       Actions counted roughly as:
         - 1 for making sandwich (if needed),
         - 1 for putting sandwich on tray (if needed),
         - 1 for moving tray from kitchen to child (if needed),
         - 1 for serving.
    6. Returning zero if all children already served.
    """

    def __init__(self, task):
        """Extract static information about children, allergy, breads, contents, trays, and sandwiches."""
        self.goals = task.goals
        self.static = task.static

        # Extract children and allergy info from static facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.not_allergic_children.add(child)

    def __call__(self, node):
        state = node.state

        # Helper sets to track current objects and facts
        served_children = set()
        waiting_children = {}
        sandwiches_on_tray = {}  # sandwich -> tray
        tray_locations = {}      # tray -> place
        sandwiches_kitchen = set()
        sandwiches_no_gluten = set()
        breads_kitchen = set()
        breads_no_gluten = set()
        contents_kitchen = set()
        contents_no_gluten = set()
        trays_at_kitchen = set()
        sandwiches_notexist = set()
        sandwich_on_tray_pairs = set()

        # Extract facts from current state
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            pred = parts[0]

            if pred == "served":
                served_children.add(parts[1])
            elif pred == "waiting":
                # waiting child place
                child, place = parts[1], parts[2]
                waiting_children[child] = place
            elif pred == "ontray":
                sandwich, tray = parts[1], parts[2]
                sandwiches_on_tray[sandwich] = tray
                sandwich_on_tray_pairs.add((sandwich, tray))
            elif pred == "at":
                # (at tray place) or others
                obj, place = parts[1], parts[2]
                if obj.startswith("tray"):
                    tray_locations[obj] = place
            elif pred == "at_kitchen_sandwich":
                sandwiches_kitchen.add(parts[1])
            elif pred == "no_gluten_sandwich":
                sandwiches_no_gluten.add(parts[1])
            elif pred == "at_kitchen_bread":
                breads_kitchen.add(parts[1])
            elif pred == "no_gluten_bread":
                breads_no_gluten.add(parts[1])
            elif pred == "at_kitchen_content":
                contents_kitchen.add(parts[1])
            elif pred == "no_gluten_content":
                contents_no_gluten.add(parts[1])
            elif pred == "at_kitchen_sandwich":
                sandwiches_kitchen.add(parts[1])
            elif pred == "notexist":
                sandwiches_notexist.add(parts[1])
            elif pred == "at":
                # Could be tray location (already considered), ignore other
                pass
            elif pred == "at" and parts[1].startswith("tray"):
                tray_locations[parts[1]] = parts[2]

        # Collect all trays in kitchen and elsewhere
        for tray, place in tray_locations.items():
            if place == "kitchen":
                trays_at_kitchen.add(tray)

        # Remaining children to serve
        children_to_serve = set(waiting_children.keys()) - served_children
        if not children_to_serve:
            return 0  # Goal reached

        # Categorize available sandwiches by gluten properties
        # Also build sets of sandwiches on trays and in kitchen
        glutenfree_sandwiches = sandwiches_no_gluten.copy()
        gluten_sandwiches = sandwiches_kitchen - glutenfree_sandwiches

        # Count trays currently with sandwiches on them
        trays_with_sandwiches = set(sandwiches_on_tray.values())  # trays currently holding a sandwich

        # We want to estimate action counts per child roughly:
        # Actions: make_sandwich (if sandwich doesn't exist),
        # put_on_tray (if sandwich not on tray),
        # move_tray (if tray not at child's place),
        # serve_sandwich

        # To do this, for each child we find if suitable sandwich exists on tray
        # at child's location or kitchen, or exists but not on tray,
        # or doesn't exist yet.

        heuristic_cost = 0

        # For efficient lookups: map tray to sandwiches on it
        tray_to_sandwiches = {}
        for s, t in sandwiches_on_tray.items():
            tray_to_sandwiches.setdefault(t, set()).add(s)

        # Map sandwich to tray place if on tray
        sandwich_to_tray_place = {}
        for s, t in sandwiches_on_tray.items():
            place = tray_locations.get(t, None)
            if place is not None:
                sandwich_to_tray_place[s] = place

        # For each child to serve:
        for child in children_to_serve:
            gluten_free_needed = child in self.allergic_children

            # Find all suitable sandwiches for the child:
            # Suitable = gluten-free sandwiches if allergic, else any sandwich
            if gluten_free_needed:
                suitable_sandwiches = glutenfree_sandwiches
            else:
                # sandwiches without gluten or with gluten (all available sandwiches)
                suitable_sandwiches = sandwiches_kitchen.union(sandwiches_no_gluten)

            # Also consider sandwiches already on trays (could be out of kitchen)
            # Add on-tray sandwiches that are suitable (no gluten or gluten, depending on allergy)
            # (some sandwiches on trays might not be in at_kitchen_sandwich if moved)
            suitable_sandwiches = suitable_sandwiches.union(
                s for s in sandwiches_on_tray if (s in glutenfree_sandwiches if gluten_free_needed else True)
            )

            # Find if any suitable sandwich is already served or waiting
            # We check if a suitable sandwich is on tray at child's place:
            served_here = False
            on_tray_at_place = False
            on_tray_at_kitchen = False
            sandwich_exists_no_tray = False  # Sandwich exists but not on tray

            # Check for sandwiches on tray at child's place
            for s in suitable_sandwiches:
                tray = sandwiches_on_tray.get(s, None)
                if tray is not None:
                    place = tray_locations.get(tray, None)
                    if place == waiting_children[child]:
                        on_tray_at_place = True
                        break
                    elif place == "kitchen":
                        on_tray_at_kitchen = True
                else:
                    # sandwich exists but not on tray
                    if s in sandwiches_kitchen or s in sandwiches_no_gluten:
                        sandwich_exists_no_tray = True

            # If sandwich on tray at child's place: only serve needed
            if on_tray_at_place:
                heuristic_cost += 1  # serve_sandwich or serve_sandwich_no_gluten

            # Else if sandwich on tray but tray is at kitchen, need to move tray + serve
            elif on_tray_at_kitchen:
                heuristic_cost += 1  # move_tray from kitchen to child's place
                heuristic_cost += 1  # serve_sandwich/_no_gluten

            # Else if sandwich exists but not on tray, need to put sandwich on tray, move tray, serve
            elif sandwich_exists_no_tray:
                heuristic_cost += 1  # put_on_tray
                heuristic_cost += 1  # move_tray
                heuristic_cost += 1  # serve_sandwich/_no_gluten

            else:
                # Sandwich does not exist yet, need to make sandwich, put on tray, move tray, serve
                # Check if gluten-free sandwich must be made or normal
                heuristic_cost += 1  # make_sandwich_no_gluten or make_sandwich
                heuristic_cost += 1  # put_on_tray
                heuristic_cost += 1  # move_tray
                heuristic_cost += 1  # serve_sandwich/_no_gluten

        return heuristic_cost
