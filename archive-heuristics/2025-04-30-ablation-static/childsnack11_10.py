from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Basic check for arity if no wildcard is used in args
    if len(parts) != len(args) and '*' not in args:
         return False
    # Check if each part matches the corresponding arg pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are waiting and need to be served according to the goal state.
    It counts the number of unserved children and adds estimates for the necessary
    intermediate steps: making suitable sandwiches (gluten-free for allergic
    children), putting kitchen sandwiches onto trays, and moving trays to the
    locations where children are waiting.

    # Assumptions
    - The goal is to satisfy all `(served ?c)` predicates for children `?c`
      specified in the problem's goal state.
    - Sufficient ingredients (bread and content) and trays exist in the problem
      instance to serve all children. The heuristic does not check ingredient
      availability beyond counting how many sandwiches *need* to be made.
    - Children remain waiting at their initial locations until served.
    - Sandwiches are either gluten-free or regular. Allergic children require
      gluten-free sandwiches. Non-allergic children can receive any sandwich.
    - The heuristic assumes that any unserved child listed in the goal is
      currently in a `(waiting ?c ?p)` state for some place `?p`.

    # Heuristic Initialization
    - Identify all children and their allergy status (`allergic_gluten`,
      `not_allergic_gluten`) from the static facts.
    - Store the set of children who need to be served based on the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Unserved Children:
       - Determine the set of children who are in the goal state but are not
         currently marked as `(served ?c)` in the current state. Let the count
         be `N_unserved`.
       - Separate these unserved children into allergic and non-allergic groups
         based on the static facts. Count the number of unserved allergic
         children, `N_allergic_unserved`.

    2. Count Existing Sandwiches:
       - Identify all sandwich objects that exist in the current state (i.e.,
         are not marked with `(notexist ?s)`).
       - Categorize these existing sandwiches into gluten-free (`no_gluten_sandwich`)
         and regular types based on state facts. Count them as `N_existing_gf`
         and `N_existing_reg`.
       - Count the number of existing sandwiches that are currently located
         `at_kitchen_sandwich`. Let this be `N_at_kitchen_sandwich`.

    3. Estimate Sandwiches to Make:
       - Calculate the minimum number of additional gluten-free sandwiches needed
         to satisfy all unserved allergic children: `needed_gf = max(0, N_allergic_unserved - N_existing_gf)`.
       - Calculate the minimum number of additional regular sandwiches needed to
         satisfy the remaining unserved non-allergic children: `needed_reg = max(0, (N_unserved - N_allergic_unserved) - N_existing_reg)`.
       - The estimated number of `make_sandwich_no_gluten` actions is `needed_gf`.
       - The estimated number of `make_sandwich` actions is `needed_reg`.
       - Total `make` actions = `needed_gf + needed_reg`. (This assumes sufficient ingredients are available).

    4. Estimate Sandwiches to Put on Tray:
       - Any sandwich that is currently `at_kitchen_sandwich` needs a `put_on_tray` action.
       - Any sandwich that is estimated to be made in step 3 will also initially
         be `at_kitchen_sandwich` and will subsequently need a `put_on_tray` action.
       - The estimated number of `put_on_tray` actions is `N_at_kitchen_sandwich + needed_gf + needed_reg`.

    5. Estimate Trays to Move:
       - Identify the set of distinct places where the unserved children are
         currently waiting (`waiting ?c ?p`).
       - Identify the set of distinct places where trays are currently located
         (`at ?t ?p`).
       - Count the number of waiting places that do not currently have a tray.
         Each such place requires at least one `move_tray` action to bring a
         tray there. Let this be `N_move_tray`.

    6. Estimate Serve Actions:
       - Each unserved child requires one `serve_sandwich` or `serve_sandwich_no_gluten` action.
       - The estimated number of `serve` actions is `N_unserved`.

    7. Sum Costs:
       - The total heuristic value is the sum of the estimated actions from steps 3, 4, 5, and 6.
       - Heuristic = `(make actions)` + `(put actions)` + `(move actions)` + `(serve actions)`
       - Heuristic = `(needed_gf + needed_reg)` + `(N_at_kitchen_sandwich + needed_gf + needed_reg)` + `N_move_tray` + `N_unserved`.
       - This simplifies to: Heuristic = `N_unserved` + `N_move_tray` + `N_at_kitchen_sandwich` + `2 * (needed_gf + needed_reg)`.

    This heuristic is not admissible because it sums action counts for different
    stages/resources independently and doesn't account for shared actions
    optimally (e.g., one tray move can serve multiple children at the same
    location, one `put_on_tray` can enable serving multiple children if the
    tray goes to the right place, one `make` action provides a sandwich that
    can be used for one child). However, it provides a reasonable lower bound
    on actions needed for each stage and combines them, aiming to guide a
    greedy search effectively by prioritizing states where more progress has
    been made across these stages.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about children
        and the goal.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all children and their allergy status from static facts
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")
        }
        self.not_allergic_children = {
            get_parts(fact)[1] for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")
        }
        # self.all_children = self.allergic_children | self.not_allergic_children # Not strictly needed for heuristic calculation

        # Store goal children (those who need to be served)
        self.goal_children = {
            get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")
        }

        # We cannot reliably get all sandwich objects from __init__ static facts alone.
        # We will infer existing sandwiches in __call__ based on state facts.


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify Unserved Children
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.goal_children - served_children

        if not unserved_children:
            return 0 # Goal reached

        N_unserved = len(unserved_children)
        N_allergic_unserved = len(unserved_children.intersection(self.allergic_children))
        # N_reg_unserved = N_unserved - N_allergic_unserved # Number of unserved non-allergic children

        # Get waiting locations for unserved children
        waiting_places = set() # {place}
        # Iterate through unserved children and find their waiting place in the state
        for child in unserved_children:
            # Find the fact (waiting child place) in the state
            found_waiting_place = False
            for fact in state:
                if match(fact, "waiting", child, "*"):
                    place = get_parts(fact)[2]
                    waiting_places.add(place)
                    found_waiting_place = True
                    break # Found the waiting place for this child, move to the next child
            # If an unserved child is not found in a 'waiting' fact, the problem might be ill-formed
            # or the state is invalid according to domain rules. We proceed assuming valid states.
            # If found_waiting_place is False, this child's waiting place is unknown,
            # which might lead to an underestimation of N_move_tray if that place needed a tray.
            # Assuming valid states where unserved goal children are waiting.


        # 2. Count Existing Sandwiches
        # Infer all sandwich objects present in the current state facts
        all_sandwiches_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts[0] in ['at_kitchen_sandwich', 'ontray', 'notexist', 'no_gluten_sandwich']:
                  if len(parts) > 1:
                      all_sandwiches_in_state.add(parts[1])

        # Filter out those that explicitly do not exist
        not_exist_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "notexist", "*")}
        existing_sandwiches = all_sandwiches_in_state - not_exist_sandwiches # These are sandwiches that have been made

        # Categorize existing sandwiches
        gf_sandwich_facts = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}
        existing_gf_sandwiches = existing_sandwiches.intersection(gf_sandwich_facts)
        existing_reg_sandwiches = existing_sandwiches - existing_gf_sandwiches

        N_existing_gf = len(existing_gf_sandwiches)
        N_existing_reg = len(existing_reg_sandwiches)

        # Count sandwiches at kitchen
        at_kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        N_at_kitchen_sandwich = len(at_kitchen_sandwiches)


        # 3. Estimate Sandwiches to Make
        # We need enough GF sandwiches for all allergic unserved children.
        # We need enough total sandwiches (GF or regular) for all unserved children.
        # Sandwiches needed = N_unserved total, with N_allergic_unserved being GF.

        # Number of GF sandwiches we still need to make
        needed_gf = max(0, N_allergic_unserved - N_existing_gf)

        # Number of regular sandwiches we still need to make.
        # This is the total sandwiches needed minus existing ones (both GF and Reg)
        # minus the new GF ones we are making.
        needed_reg = max(0, N_unserved - N_existing_gf - N_existing_reg - needed_gf)
        # Note: needed_reg could also be calculated as max(0, (N_unserved - N_allergic_unserved) - N_existing_reg)
        # if we assume GF sandwiches can only serve allergic children and regular only non-allergic.
        # However, regular sandwiches *can* serve non-allergic children. The total needed is N_unserved.
        # Let's use the simpler count: total needed - existing total = N_unserved - (N_existing_gf + N_existing_reg).
        # But we must ensure enough GF for allergic.
        # Let's stick to the original logic: needed_gf for allergic, needed_reg for the rest.
        needed_reg = max(0, (N_unserved - N_allergic_unserved) - N_existing_reg)


        # We assume sufficient ingredients for solvable problems.
        N_make_gf = needed_gf
        N_make_reg = needed_reg
        # total_make_actions = N_make_gf + N_make_reg # Not used directly in simplified formula


        # 4. Estimate Sandwiches to Put on Tray
        # Number of sandwiches currently in kitchen that need put_on_tray.
        # Sandwiches made (needed_gf + needed_reg) will also appear in the kitchen first.
        # total_put_actions = N_at_kitchen_sandwich + N_make_gf + N_make_reg # Not used directly in simplified formula


        # 5. Estimate Trays to Move
        tray_locations = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}

        # Places where children are waiting that do not have a tray
        places_needing_tray = waiting_places - tray_locations
        N_move_tray = len(places_needing_tray)


        # 6. Estimate Serve Actions
        N_serve_actions = N_unserved


        # 7. Sum Costs (Simplified Formula)
        # Heuristic = N_unserved (serve)
        #           + N_move_tray (move)
        #           + N_at_kitchen_sandwich (put existing kitchen sandwiches)
        #           + 2 * (needed_gf + needed_reg) (make + put for new sandwiches)
        heuristic_value = N_unserved + N_move_tray + N_at_kitchen_sandwich + 2 * (needed_gf + needed_reg)

        return heuristic_value

