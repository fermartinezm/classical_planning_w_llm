from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It sums up the estimated costs for four main stages: making
    sandwiches, putting sandwiches on trays, moving trays to children's locations,
    and serving the children. It is non-admissible and designed for greedy
    best-first search.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Allergic children require gluten-free sandwiches. Non-allergic children
      can accept any sandwich (including gluten-free).
    - Sandwiches must be made, put on a tray, the tray moved to the child's
      location, and then served.
    - Tray capacity is effectively infinite for heuristic calculation purposes.
    - Multiple children at the same location can be served by a single tray
      moved to that location.
    - The heuristic counts the number of "items" or "tasks" needed at each
      stage of the process pipeline (make -> put -> move -> serve).

    # Heuristic Initialization
    - Extracts the initial waiting location and allergy status for each child
      from the initial state. This information is static throughout the problem.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for four stages:

    1.  **Cost to Serve:** Each unserved child requires one 'serve' action.
        This cost is simply the total number of unserved children.
        `Cost_serve = N_unserved`

    2.  **Cost to Move Trays:** A tray needs to be present at each location
        where unserved children are waiting. If a location with unserved
        children does not currently have a tray, a tray must be moved there.
        This cost is the number of distinct places with unserved children
        that do not currently have a tray.
        `Cost_move_tray = |places_needing_tray|`

    3.  **Cost to Put on Tray:** Sandwiches must be on trays to be moved and
        served. This cost estimates how many sandwiches still need to transition
        from 'not on tray' to 'on tray' to meet the total demand of unserved
        children. It is calculated as the maximum of 0 and the difference
        between the total number of unserved children (total sandwiches needed)
        and the number of sandwiches already on trays.
        `Cost_put_on_tray = max(0, N_unserved - N_ontray)`

    4.  **Cost to Make Sandwiches:** Enough suitable sandwiches must be made
        to satisfy the demand of all unserved children. This cost estimates
        how many sandwiches still need to be made, considering allergy
        constraints and already made sandwiches.
        - Calculate the number of gluten-free (GF) sandwiches strictly required
          for allergic children that are not yet made.
          `Needed_make_gf = max(0, N_unserved_allergic - N_gf_made)`
        - Calculate the number of sandwiches needed for non-allergic children
          that cannot be met by already made non-GF sandwiches or by excess
          GF sandwiches (i.e., GF sandwiches made beyond the needs of allergic
          children). This remaining demand must be met by making more sandwiches
          (either GF or non-GF).
          `Available_for_non_allergic = N_any_made + max(0, N_gf_made - N_unserved_allergic)`
          `Remaining_non_allergic_demand = max(0, N_unserved_non_allergic - Available_for_non_allergic)`
          `Needed_make_any_or_gf = Remaining_non_allergic_demand`
        - The total cost to make sandwiches is the sum of GF sandwiches that
          must be made and the additional sandwiches needed for non-allergic
          children.
        `Cost_make = Needed_make_gf + Needed_make_any_or_gf`

    The total heuristic value is the sum of these four costs:
    `h = Cost_serve + Cost_move_tray + Cost_put_on_tray + Cost_make`
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static child information.
        """
        # The goal conditions are stored in task.goals, but we identify unserved
        # children by checking the 'served' predicate in the state against
        # the initial 'waiting' children.
        self.goals = task.goals

        # Extract initial child information: location and allergy status
        self.child_info = {} # {child_name: {'location': place, 'allergic': bool}}
        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "waiting":
                child, place = parts[1], parts[2]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['location'] = place
            elif predicate == "allergic_gluten":
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = True
            elif predicate == "not_allergic_gluten":
                child = parts[1]
                if child not in self.child_info:
                    self.child_info[child] = {}
                self.child_info[child]['allergic'] = False

        # Ensure all children found have allergy status and location (problems should be well-formed)
        # This loop is mainly for robustness against malformed initial states
        children_to_remove = []
        for child, info in self.child_info.items():
             if 'allergic' not in info:
                 # This shouldn't happen in valid PDDL, but handle defensively
                 # print(f"Warning: Allergy status not found for child {child}. Assuming not allergic.")
                 info['allergic'] = False
             if 'location' not in info:
                 # This shouldn't happen either (waiting children must have a location)
                 # print(f"Warning: Initial waiting location not found for child {child}. Skipping.")
                 children_to_remove.append(child)

        for child in children_to_remove:
            del self.child_info[child]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved children and their needs
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children_info = {
            c: info for c, info in self.child_info.items() if c not in served_children
        }

        N_unserved = len(unserved_children_info)

        # If no children are unserved, the goal is reached.
        if N_unserved == 0:
            return 0

        N_unserved_allergic = sum(1 for info in unserved_children_info.values() if info.get('allergic', False))
        N_unserved_non_allergic = N_unserved - N_unserved_allergic

        # 2. Calculate Cost to Serve
        Cost_serve = N_unserved

        # 3. Calculate Cost to Move Trays
        places_with_unserved = {info['location'] for info in unserved_children_info.values()}
        tray_locations = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")
        }
        places_with_trays = set(tray_locations.values())
        places_needing_tray = places_with_unserved - places_with_trays
        Cost_move_tray = len(places_needing_tray)

        # 4. Calculate Cost to Put on Tray
        sandwiches_on_trays = {
            get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")
        }
        N_ontray = len(sandwiches_on_trays)
        Cost_put_on_tray = max(0, N_unserved - N_ontray)

        # 5. Calculate Cost to Make Sandwiches
        sandwiches_in_kitchen = {
            get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")
        }
        made_sandwiches = sandwiches_in_kitchen.union(sandwiches_on_trays)

        gf_sandwiches = {
            get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")
        }

        N_gf_made = len(made_sandwiches.intersection(gf_sandwiches))
        N_any_made = len(made_sandwiches) - N_gf_made # Includes non-GF made sandwiches

        # Number of GF sandwiches we *must* make for allergic children
        Needed_make_gf = max(0, N_unserved_allergic - N_gf_made)

        # Number of additional sandwiches (can be GF or Any) needed for non-allergic children
        # after using available non-GF and any excess GF sandwiches.
        # Excess GF = max(0, N_gf_made - N_unserved_allergic)
        # Available for non-allergic = N_any_made + Excess GF
        # Remaining non-allergic demand = max(0, N_unserved_non_allergic - Available for non-allergic)
        Available_for_non_allergic = N_any_made + max(0, N_gf_made - N_unserved_allergic)
        Remaining_non_allergic_demand = max(0, N_unserved_non_allergic - Available_for_non_allergic)
        Needed_make_any_or_gf = Remaining_non_allergic_demand

        Cost_make = Needed_make_gf + Needed_make_any_or_gf

        # Total heuristic is the sum of costs for each stage
        total_cost = Cost_serve + Cost_move_tray + Cost_put_on_tray + Cost_make

        return total_cost
