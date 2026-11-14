from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by stripping parentheses and splitting."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The fact as a string, e.g., "(at_kitchen_bread bread1)".
    - `args`: A sequence of strings or wildcards to match against each fact part.
    - Returns True if all parts match their pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary
    -------
    Estimates the number of actions required to serve all children with sandwiches
    according to their gluten allergy status.

    Assumptions
    -----------
    - Each child must be served exactly one sandwich.
    - Sandwiches can be normal or gluten-free, depending on child allergy.
    - Sandwiches are made in the kitchen from bread and content portions present there.
    - Gluten-free sandwiches require gluten-free bread and content.
    - Serving requires assembling sandwiches, placing them on trays, moving trays to the child's location, and serving.
    - Trays must be moved from kitchen to child's place if not already there.
    - Multiple sandwiches can be put on one tray, but each sandwich must be served to one child.
    - The heuristic does NOT have to be admissible but should be efficiently computable and informative.

    Heuristic Initialization
    ------------------------
    - Extract from static facts the allergy status of each child.
    - Extract gluten-free breads and contents from static facts.
    - Store initial information about children waiting positions.
    - Keep track of trays available to use.
    - Store goal children who need to be served.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify children who are not yet served.
       - Check all `(served ?child)` facts in the state.
    2. For each not served child, determine if gluten-free sandwich is needed:
       - If `(allergic_gluten child)`, child needs gluten-free sandwich.
       - Otherwise, normal sandwich is needed.
    3. Count how many gluten-free sandwiches and normal sandwiches are currently available at kitchen:
       - `(at_kitchen_sandwich ?s)` with `(no_gluten_sandwich ?s)` means gluten-free sandwich ready.
       - `(at_kitchen_sandwich ?s)` without that means normal sandwich ready.
    4. Count how many sandwiches are already on trays `(ontray ?s ?t)` and how many trays are at each place.
       - Sandwiches on trays can be served directly if the tray is at child's place.
    5. Estimate missing sandwiches to make for each category:
       - Gluten-free sandwiches needed minus gluten-free sandwiches available at kitchen or on trays.
       - Normal sandwiches needed minus normal sandwiches available at kitchen or on trays.
    6. Estimate moves required to get trays with sandwiches from kitchen to children's locations if trays not already where children wait.
    7. Summarize all parts into an estimated number of actions:
       - Making each sandwich roughly costs 1 action (`make_sandwich` or `make_sandwich_no_gluten`).
       - Putting a sandwich on tray costs 1 action.
       - Moving each tray from kitchen to child's location costs 1 action per tray move.
       - Serving a sandwich to a child costs 1 action.
    8. Consider that multiple sandwiches can be put on one tray, minimizing tray moves:
       - Count number of trays with sandwiches that are not yet at correct child's place to estimate moves.
       - Count minimal number of tray moves necessary to serve all children.
    9. Return the total estimated action count as the heuristic value.
    10. Return 0 if all children are served (goal reached).

    Note:
    - This heuristic is not guaranteed admissible but is designed to guide search efficiently.
    """

    def __init__(self, task):
        """
        Initialize by extracting allergy info, gluten-free breads and contents,
        trays, children, and waiting places from static and initial facts.
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # Extract allergy info: sets of allergic and not allergic children.
        self.allergic_children = set()
        self.not_allergic_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children.add(child)
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.not_allergic_children.add(child)

        # Extract gluten-free breads
        self.no_gluten_breads = {get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_bread", "*")}
        # Extract gluten-free contents
        self.no_gluten_contents = {get_parts(fact)[1] for fact in self.static if match(fact, "no_gluten_content", "*")}

        # Extract children who need to be served from goals: all children appearing in (served ?child)
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Extract trays from facts in initial state (we assume trays don't appear/disappear)
        self.trays = set()
        for fact in self.initial_state:
            if match(fact, "at", "*"):
                obj = get_parts(fact)[1]
                # We use this to find trays from objects present at some place
                # But better is from all objects of type tray? Here only way: infer trays from trays in initial state facts.
                # We do it dynamically in heuristic call by observing trays in state facts.

        # Extract waiting places of children from static (waiting child place) facts - static since child's place does not change in domain file.
        self.child_waiting_places = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                child, place = parts[1], parts[2]
                self.child_waiting_places[child] = place

    def __call__(self, node):
        state = node.state

        # 1. Identify served and unserved children from the state
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.goal_children - served_children

        if not unserved_children:
            # Goal reached
            return 0

        # 2. Classify unserved children by allergy
        gluten_allergic = unserved_children & self.allergic_children
        not_allergic = unserved_children & self.not_allergic_children

        # 3. Count sandwiches currently at kitchen categorized by gluten status
        kitchen_sandwiches = set(get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*"))
        gluten_free_sandwiches = set(get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*"))
        gluten_free_kitchen_sandwiches = kitchen_sandwiches & gluten_free_sandwiches
        normal_kitchen_sandwiches = kitchen_sandwiches - gluten_free_kitchen_sandwiches

        # 4. Count sandwiches currently on trays and their trays
        ontray_facts = [fact for fact in state if match(fact, "ontray", "*", "*")]
        sandwiches_on_trays = set()
        tray_to_sandwiches = {}
        sandwich_to_tray = {}
        trays_present = set()
        trays_location = {}

        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, place = get_parts(fact)[1], get_parts(fact)[2]
                trays_location[obj] = place

        for fact in ontray_facts:
            parts = get_parts(fact)
            sandwich = parts[1]
            tray = parts[2]
            sandwiches_on_trays.add(sandwich)
            sandwich_to_tray[sandwich] = tray
            if tray not in tray_to_sandwiches:
                tray_to_sandwiches[tray] = set()
            tray_to_sandwiches[tray].add(sandwich)
            trays_present.add(tray)

        # Among sandwiches on trays, identify gluten-free or normal
        gluten_free_on_tray = sandwiches_on_trays & gluten_free_sandwiches
        normal_on_tray = sandwiches_on_trays - gluten_free_on_tray

        # 5. Estimating sandwiches missing to fulfill gluten-free and normal sandwiches
        # Count total sandwiches ready (kitchen + on trays) per category
        gluten_free_ready = len(gluten_free_kitchen_sandwiches) + len(gluten_free_on_tray)
        normal_ready = len(normal_kitchen_sandwiches) + len(normal_on_tray)

        gluten_free_needed = len(gluten_allergic)
        normal_needed = len(not_allergic)

        gluten_free_missing = max(0, gluten_free_needed - gluten_free_ready)
        normal_missing = max(0, normal_needed - normal_ready)

        # 6. Estimate number of trays needed: each tray can hold multiple sandwiches.
        # We assume trays can hold multiple sandwiches, so number of trays needed at minimum = number of distinct places to deliver sandwiches.
        # Count how many trays are currently at each place
        tray_places = {}
        for tray in trays_present:
            place = trays_location.get(tray, None)
            if place is not None:
                tray_places.setdefault(place, set()).add(tray)

        # Count places where children are waiting and that require tray delivery of sandwiches
        children_places = {self.child_waiting_places[child] for child in unserved_children if child in self.child_waiting_places}

        # Count trays currently at children's places that have sandwiches to serve at that place
        # For simplification, we consider that each tray can serve sandwiches put on it
        trays_at_child_place = 0
        trays_to_move = 0

        # Identify trays that have sandwiches for children at the correct place and trays that need to be moved
        # For each tray, determine if it is at at least one child's place where unserved child is waiting
        # And if that tray has sandwiches to be served (assume all sandwiches on tray belong to children)
        # To simplify, assume all trays with sandwiches need to be moved to the children's places if not already there
        # We will approximate tray moves by number of distinct tray moves needed

        # Calculate set of trays that need move because tray is at kitchen but children are at other places
        # Trays with sandwiches at kitchen need to be moved to children's places
        trays_at_kitchen = {tray for tray in trays_present if trays_location.get(tray) == "kitchen"}
        trays_at_child_places = trays_present - trays_at_kitchen

        # Number of trays we have with sandwiches located at children's places
        trays_already_at_children = 0

        # Count how many covered children by trays already at child's places with at least one sandwich
        children_served_by_trays = set()
        for tray in trays_at_child_places:
            place = trays_location.get(tray)
            # If children wait at that place
            children_waiting_here = {c for c, pl in self.child_waiting_places.items() if pl == place and c in unserved_children}
            # Assuming tray sandwiches serve children waiting here
            # We count children served by trays at correct location
            children_served_by_trays.update(children_waiting_here)
            trays_already_at_children += 1

        # Children not served by trays in correct locations
        children_not_served_by_trays = unserved_children - children_served_by_trays

        # Estimate minimal number of trays needed for remaining children not served by trays in place
        # Because trays can carry multiple sandwiches, assume at least one new tray needed if there are any remaining children
        trays_needed_for_remaining = 1 if children_not_served_by_trays else 0

        # Estimate tray moves:
        # Trays at kitchen need to move to children's places, count these movements
        tray_moves = len(trays_at_kitchen)
        # Plus moves of any new tray(s) needed to serve remaining children not served by trays already at place
        tray_moves += trays_needed_for_remaining

        # 7. Estimate count of actions:

        # Making sandwiches: each missing sandwich costs one action
        make_actions = gluten_free_missing + normal_missing

        # Putting sandwiches on trays: each sandwich that is not yet on a tray costs one action
        # Count sandwiches at kitchen not on trays per category (these must be put on tray)
        gluten_free_kitchen_not_ontray = gluten_free_kitchen_sandwiches - gluten_free_on_tray
        normal_kitchen_not_ontray = normal_kitchen_sandwiches - normal_on_tray
        put_on_tray_actions = len(gluten_free_kitchen_not_ontray) + len(normal_kitchen_not_ontray)

        # Moving trays to children's places (where children wait): as counted above
        move_tray_actions = tray_moves

        # Serving sandwiches: each unserved child needs one serve action
        serve_actions = len(unserved_children)

        # Sum all
        total_estimate = make_actions + put_on_tray_actions + move_tray_actions + serve_actions

        # Return finite heuristic (always >=0)
        return total_estimate
