from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or (len(parts) > len(args) and args[-1] != '*'):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children, plus the estimated actions needed
    to get suitable sandwiches on trays to the children's locations. The estimation
    considers the state of required resources (sandwiches, trays) relative to the
    children's needs and locations.

    # Assumptions
    - The goal is to serve all children who are initially in a 'waiting' state.
    - Each child needs one suitable sandwich delivered on a tray to their location.
    - A gluten-allergic child requires a gluten-free sandwich.
    - A non-allergic child can accept any sandwich.
    - Actions have a cost of 1.
    - Resources (bread, content, trays, sandwich names) are generally sufficient to solve the problem,
      although the heuristic does not explicitly check for resource depletion leading to unsolvability.

    # Heuristic Initialization
    - Extracts information about which children are initially waiting and at which locations.
    - Extracts information about which children are allergic to gluten.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated by summing up the estimated costs for different stages
    of satisfying the goal (serving all children):

    1.  **Cost for Serving:** Count the number of children who are currently in a 'waiting' state
        but are not yet 'served'. Each such child requires a final 'serve' action. Add this count
        to the heuristic. If this count is zero, the goal is reached, and the heuristic is 0.

    2.  **Identify Uncovered Needs:** Determine which unserved children are *not* currently
        covered by a suitable sandwich already present on a tray at their waiting location.
        Each such child represents an "unmet delivery need" at their location. Count the total
        number of such unmet needs (`num_deliveries_needed`).

    3.  **Cost for Tray Movements to Locations:** Identify the distinct locations where
        uncovered children are waiting (`needed_locations`). Count the number of trays
        that are already present at these needed locations (`trays_at_needed_locations`).
        The minimum number of 'move_tray' actions required to bring trays to cover the
        remaining needed locations is `max(0, |needed_locations| - |trays_at_needed_locations|)`.
        Add this count to the heuristic.

    4.  **Cost for Putting Sandwiches on Trays:** Each of the `num_deliveries_needed` requires
        a suitable sandwich to be placed on a tray. Count the number of suitable sandwiches
        that are *already* on trays but *not* at one of the `needed_locations`
        (`avail_st_elsewhere`). These sandwiches are already on a tray and only need a
        'move_tray' action (accounted for in step 3 if the tray moves to a needed location).
        The remaining `max(0, num_deliveries_needed - avail_st_elsewhere)` deliveries require
        a sandwich that is currently `at_kitchen_sandwich` or needs to be made. These
        sandwiches will require a 'put_on_tray' action at the kitchen. Add this count
        to the heuristic.

    5.  **Cost for Making Sandwiches:** Some deliveries might require sandwiches that do not
        yet exist. Count the number of suitable sandwiches available *not* on trays at needed
        locations (`avail_suitable_not_at_needed_tray`), which includes sandwiches `at_kitchen_sandwich`
        and `ontray` elsewhere. The number of deliveries that cannot be satisfied by these
        existing sandwiches is `max(0, num_deliveries_needed - avail_suitable_not_at_needed_tray)`.
        Each of these requires a 'make_sandwich' action. Add this count to the heuristic.

    The total heuristic is the sum of costs from steps 1, 3, 4, and 5.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children.
        """
        self.goals = task.goals
        static_facts = task.static

        self.waiting_children_info = {} # {child: location}
        self.allergic_children_info = {} # {child: bool}

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                child, loc = parts[1], parts[2]
                self.waiting_children_info[child] = loc
            elif parts[0] == "allergic_gluten":
                self.allergic_children_info[parts[1]] = True
            elif parts[0] == "not_allergic_gluten":
                self.allergic_children_info[parts[1]] = False

    def is_suitable_for_child(self, sandwich, child, state):
        """Check if a sandwich is suitable for a specific child."""
        child_is_allergic = self.allergic_children_info.get(child, False)
        is_gf_sandwich = "(no_gluten_sandwich " + sandwich + ")" in state
        return (child_is_allergic and is_gf_sandwich) or (not child_is_allergic)

    def is_suitable_for_any_uncovered(self, sandwich, uncovered_children_set, state):
        """Check if a sandwich is suitable for at least one child in the uncovered set."""
        for child in uncovered_children_set:
            if self.is_suitable_for_child(sandwich, child, state):
                return True
        return False


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        heuristic = 0

        # 1. Cost for Serving
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "?c")}
        unserved_children = {c for c in self.waiting_children_info if c not in served_children}

        if not unserved_children:
            return 0 # Goal reached

        heuristic += len(unserved_children) # Cost for serve actions

        # 2. Identify Uncovered Needs (children not covered by suitable sandwich on tray at location)
        uncovered_children_at_location = set()
        for child in unserved_children:
            child_loc = self.waiting_children_info[child]
            is_covered_at_location = False
            for fact in state:
                if match(fact, "ontray", "?s", "?t"):
                    s = get_parts(fact)[1]
                    t = get_parts(fact)[2]
                    if self.is_suitable_for_child(s, child, state):
                        # Check if tray is at child's location
                        if "(at " + t + " " + child_loc + ")" in state:
                            is_covered_at_location = True
                            break # Found a suitable sandwich on a tray at the location

            if not is_covered_at_location:
                uncovered_children_at_location.add(child)

        num_deliveries_needed = len(uncovered_children_at_location)

        if num_deliveries_needed > 0:
            # Costs for getting sandwich-tray units to needed locations

            needed_locations = {self.waiting_children_info[c] for c in uncovered_children_at_location}

            # 3. Cost for Tray Movements to Locations
            trays_at_needed_locations = {get_parts(fact)[1] for fact in state if match(fact, "at", "?t", "?p") and get_parts(fact)[2] in needed_locations}
            heuristic += max(0, len(needed_locations) - len(trays_at_needed_locations)) # move_tray cost

            # 4. Cost for Putting Sandwiches on Trays & 5. Cost for Making Sandwiches
            # Count suitable sandwiches available *not* on trays at needed locations.
            # These are sources for the deliveries.

            avail_st_elsewhere = 0 # Suitable sandwiches on trays elsewhere
            used_trays_for_st_elsewhere = set() # Prevent double counting trays

            for fact in state:
                if match(fact, "ontray", "?s", "?t"):
                    s = get_parts(fact)[1]
                    t = get_parts(fact)[2]
                    # Check if sandwich is suitable for *any* uncovered child
                    if self.is_suitable_for_any_uncovered(s, uncovered_children_at_location, state):
                        # Check if tray is NOT at a needed location
                        tray_loc = None
                        for tray_loc_fact in state:
                            if match(tray_loc_fact, "at", t, "?p_prime"):
                                tray_loc = get_parts(tray_loc_fact)[2]
                                break
                        if tray_loc is not None and tray_loc not in needed_locations and t not in used_trays_for_st_elsewhere:
                            avail_st_elsewhere += 1
                            used_trays_for_st_elsewhere.add(t)

            avail_s_kitchen = 0 # Suitable sandwiches at kitchen
            for fact in state:
                if match(fact, "at_kitchen_sandwich", "?s"):
                    s = get_parts(fact)[1]
                    # Check if sandwich is suitable for *any* uncovered child
                    if self.is_suitable_for_any_uncovered(s, uncovered_children_at_location, state):
                         avail_s_kitchen += 1

            avail_suitable_not_at_needed_tray = avail_st_elsewhere + avail_s_kitchen

            # Number of deliveries that need a 'make' action
            deliveries_needing_new_s = max(0, num_deliveries_needed - avail_suitable_not_at_needed_tray)
            heuristic += deliveries_needing_new_s # make cost

            # Number of deliveries that need a 'put_on_tray' action
            # These are deliveries using sandwiches that are NOT already on trays elsewhere (i.e., at kitchen or need making)
            deliveries_needing_put_on_tray = max(0, num_deliveries_needed - avail_st_elsewhere)
            heuristic += deliveries_needing_put_on_tray # put_on_tray cost


        return heuristic
