from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # This case should ideally not happen with facts from a PDDL parser,
        # but defensive programming might return [].
        # Assuming valid string facts for typical planning states.
        return fact[1:-1].split()
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
    This heuristic estimates the number of actions required to serve all
    unserved children. It counts the necessary 'serve', 'make_sandwich',
    'put_on_tray', and 'move_tray' actions based on the current state
    of children, sandwiches, and trays.

    # Assumptions
    - Each unserved child requires one sandwich and one 'serve' action.
    - Sandwiches must be made in the kitchen, put on a tray in the kitchen,
      and the tray moved to the child's location.
    - A tray can carry multiple sandwiches to the same location, but each
      location needing delivery requires at least one tray movement from
      the kitchen.
    - Sufficient bread, content, and 'notexist' sandwich objects are
      available to make needed sandwiches (relaxed assumption).
    - Sandwiches already on trays at the child's waiting location can be used
      immediately (after the 'serve' action).

    # Heuristic Initialization
    - Extracts the set of all children from the goal conditions.
    - Extracts allergy information for each child from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children that have not yet been served. The count of
       these children is the minimum number of 'serve' actions required.
       Add this count to the heuristic value. If no children are unserved,
       the goal is reached, and the heuristic is 0.
    2. Determine which unserved children are *not* currently reachable by
       a tray containing a suitable sandwich already at their waiting location.
       These children require a new sandwich delivery process starting from
       the kitchen.
    3. Count the number of sandwiches that need to be sourced from the kitchen
       (either existing kitchen stock or newly made) to satisfy the children
       identified in step 2. This count is equal to the number of children
       needing delivery.
    4. Count the number of 'make_sandwich' actions required. This is the number
       of sandwiches needed from the kitchen (step 3) minus the number of
       sandwiches already available in the kitchen ('at_kitchen_sandwich').
       Add this count to the heuristic value.
    5. Count the number of 'put_on_tray' actions required from the kitchen.
       This is the total number of sandwiches sourced from the kitchen
       (newly made plus existing kitchen stock). Add this count to the
       heuristic value.
    6. Count the number of distinct locations where children identified in
       step 2 are waiting. Each such location requires at least one 'move_tray'
       action from the kitchen. Add this count to the heuristic value.
    7. The total heuristic value is the sum of costs from steps 1, 4, 5, and 6.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        # Extract all children from the goal predicates
        self.all_children = {get_parts(goal)[1] for goal in task.goals if get_parts(goal) and get_parts(goal)[0] == 'served'}

        # Extract allergy information from static facts
        self.is_allergic = {}
        for fact_str in task.static:
            parts = get_parts(fact_str)
            if parts and parts[0] == 'allergic_gluten':
                self.is_allergic[parts[1]] = True
            elif parts and parts[0] == 'not_allergic_gluten':
                self.is_allergic[parts[1]] = False

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        """
        state = node.state
        h = 0

        # 1. Count unserved children (cost for serve actions)
        served_children = {c for fact in state if match(fact, "served", c)}
        unserved_children = self.all_children - served_children
        h += len(unserved_children)

        # If no children are unserved, the goal is reached
        if not unserved_children:
            return h

        # Extract relevant dynamic state information
        waiting_children = {c: p for fact in state if match(fact, "waiting", c, p)}
        ontray_sandwiches = {s: t for fact in state if match(fact, "ontray", s, t)}
        tray_locations = {t: p for fact in state if match(fact, "at", t, p)}
        gf_sandwiches = {s for fact in state if match(fact, "no_gluten_sandwich", s)}
        at_kitchen_sandwiches = {s for fact in state if match(fact, "at_kitchen_sandwich", s)}

        # Helper to check if a sandwich is suitable for a child
        def is_suitable(sandwich, child):
            s_is_gf = sandwich in gf_sandwiches
            c_is_allergic = self.is_allergic.get(child, False) # Default to False if info missing
            # Suitable if child is not allergic OR (child is allergic AND sandwich is gluten-free)
            return (not c_is_allergic) or (c_is_allergic and s_is_gf)

        # 2. Identify children who need a new delivery from the kitchen
        # Start assuming all unserved children need delivery
        children_need_delivery = set(unserved_children)
        used_ready_sandwiches = set() # Track sandwiches on trays at location that we've assigned
        used_ready_trays = set()      # Track trays at location that we've used

        # Iterate through unserved children to find if they are covered by existing ready sandwiches
        # Iterate over a copy because we might remove elements from children_need_delivery
        for child in list(children_need_delivery):
            child_loc = waiting_children.get(child)
            if child_loc is None:
                 # Child is not waiting at a known location - assume they need delivery
                 continue

            found_match = False
            # Look for a suitable, unused sandwich on a tray at the child's location
            for sandwich, tray in ontray_sandwiches.items():
                if sandwich in used_ready_sandwiches: continue
                if tray in used_ready_trays: continue
                if tray_locations.get(tray) == child_loc:
                    if is_suitable(sandwich, child):
                        used_ready_sandwiches.add(sandwich)
                        used_ready_trays.add(tray)
                        found_match = True
                        break # Found a match for this child

            if found_match:
                children_need_delivery.remove(child) # This child is covered by an existing ready sandwich

        n_children_need_delivery = len(children_need_delivery)

        # 3. Number of sandwiches needed from kitchen source = n_children_need_delivery

        # 4. Count make actions
        ak_total = len(at_kitchen_sandwiches)
        # Need to make sandwiches if the number of children needing delivery
        # exceeds the number of sandwiches already available in the kitchen.
        n_make = max(0, n_children_need_delivery - ak_total)
        h += n_make

        # 5. Count put_on_tray actions (for sandwiches sourced from kitchen)
        # Sandwiches sourced from the kitchen are the ones newly made (n_make)
        # plus the ones already at the kitchen (ak_total).
        n_put_on_tray = n_make + ak_total
        h += n_put_on_tray

        # 6. Count move_tray actions (for distinct locations needing delivery)
        locations_needing_delivery = set()
        for child in children_need_delivery:
             child_loc = waiting_children.get(child)
             if child_loc is not None:
                locations_needing_delivery.add(child_loc)

        h += len(locations_needing_delivery)

        return h
