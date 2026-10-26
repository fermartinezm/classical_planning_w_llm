# Need to import Heuristic and Task from wherever they are defined
# Assuming they are in a 'heuristics.heuristic_base' and 'task' module respectively
from heuristics.heuristic_base import Heuristic
from task import Task # Task is used in __init__
import math # Import math for infinity

# Helper function to parse fact string
def parse_fact(fact_str):
    """Parses a PDDL fact string into predicate and arguments."""
    # Remove leading/trailing parens and split by space
    parts = fact_str[1:-1].split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
        This heuristic estimates the number of actions required to reach the
        goal state (all children served) by summing up the estimated costs
        of satisfying the demand at different stages of the snack delivery
        pipeline: making sandwiches, putting them on trays, moving trays
        to children's locations, and finally serving the children. It counts
        the number of items/visits needed at each stage transition based on
        the current state and the total demand (unserved children).

    Assumptions:
        - All actions have a cost of 1.
        - Resources (ingredients, sandwich objects, trays) are fixed and
          available as specified in the initial state and static facts.
        - The heuristic uses a simplified view of tray movements, counting
          "tray-visits" needed at places with waiting children rather than
          optimal tray routing.

    Heuristic Initialization:
        In the constructor (__init__), the heuristic processes the static
        facts and initial state facts provided in the Task object to build
        internal data structures. This includes identifying:
        - All children and their allergy status.
        - The waiting place for each child.
        - Which bread and content portions are gluten-free.
        - All possible places in the domain.
        - The total number of trays available in the problem.

    Step-By-Step Thinking for Computing Heuristic:
        The heuristic function (__call__) takes the current state (a frozenset
        of facts) as input and computes the estimated cost as follows:

        1.  **Parse State:** Iterate through the facts in the current state
            to determine:
            - Which children have been served.
            - The counts of available bread and content portions (regular and
              gluten-free) in the kitchen.
            - The counts of available sandwiches (regular and gluten-free)
              in the kitchen and on trays.
            - The count of unused sandwich objects.
            - The location of each tray.
            - The gluten-free status of sandwiches that exist.

        2.  **Identify Unserved Children and Needs:** Determine which children
            are not yet served based on the goal facts and the current state.
            Count the total number of unserved allergic children (needing
            gluten-free sandwiches) and non-allergic children (needing any
            sandwich). This gives the total demand for sandwiches of each type.

        3.  **Calculate Heuristic Components:** Compute four additive components:
            a.  **H_serve:** The number of unserved children. This is the minimum
                number of `serve_sandwich` actions required.
            b.  **H_make:** The number of sandwiches that still need to be made
                to satisfy the total demand (unserved children), considering
                sandwiches already made (in kitchen or on trays). This is
                `max(0, Total_Sandwiches_Needed - Sandwiches_Already_Made)`.
                Total sandwiches needed is the sum of unserved allergic and
                non-allergic children. Sandwiches already made are those
                currently `at_kitchen_sandwich` or `ontray`.
            c.  **H_put:** The number of sandwiches that need to be put on trays
                to satisfy the total demand, considering sandwiches already
                on trays. This is `max(0, Total_Sandwiches_Needed - Sandwiches_Already_On_Trays)`.
            d.  **H_move_visits:** An estimate of tray movement cost. For each
                place where unserved children are waiting, count how many
                more "tray-visits" are needed than the number of trays
                currently present at that place, based on the number of
                waiting children. This is calculated as
                `sum(max(0, Children_Waiting_At_Place_p - Trays_At_Place_p) for all places p)`.

        4.  **Check for Unsolvability:** Before returning the value, check if the
            total demand for sandwiches (of each type, considering non-allergic
            flexibility) exceeds the total potential supply (available + makable).
            Also check if the total number of sandwiches needed exceeds the total
            number of trays available in the domain. If any of these conditions
            are met, the heuristic returns `float('inf')`.

        5.  **Sum Components:** The final heuristic value is the sum of the
            four components: `H_serve + H_put + H_make + H_move_visits`.

        The heuristic returns 0 if and only if all children are served (goal state),
        as all components will be zero in that case.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # --- Initialize static information ---
        self.is_ng_bread = {}
        self.is_ng_content = {}
        self.is_allergic = {} # True for allergic, False for not_allergic
        self.waiting_place = {}
        self.all_children = set()
        self.all_places = {'kitchen'} # Add constant kitchen
        self.total_trays = 0

        # Process static facts
        for fact_str in self.static_facts:
            predicate, args = parse_fact(fact_str)
            if predicate == 'waiting':
                child, place = args
                self.waiting_place[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif predicate == 'allergic_gluten':
                child = args[0]
                self.is_allergic[child] = True
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten':
                child = args[0]
                self.is_allergic[child] = False
                self.all_children.add(child)
            elif predicate == 'no_gluten_bread':
                self.is_ng_bread[args[0]] = True
            elif predicate == 'no_gluten_content':
                self.is_ng_content[args[0]] = True

        # Process initial state facts to get total trays and initial places
        for fact_str in self.initial_state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'at':
                # Assuming 'at' facts in initial state only refer to trays
                tray, place = args
                self.total_trays += 1
                self.all_places.add(place)

        # Ensure all places from waiting facts are in self.all_places
        for place in self.waiting_place.values():
            self.all_places.add(place)


    def __call__(self, node):
        state = node.state

        # --- Parse State Information ---
        served_children = set()
        available_bread_ng = 0
        available_bread_reg = 0
        available_content_ng = 0
        available_content_reg = 0
        available_sandwich_ng_kitchen = 0
        available_sandwich_reg_kitchen = 0
        available_sandwich_ng_ontray = 0
        available_sandwich_reg_ontray = 0
        unused_sandwiches = 0
        trays_at = {p: 0 for p in self.all_places}
        sandwich_is_ng = {}

        # First pass to get basic counts and sandwich NG status
        state_at_kitchen_sandwich = []
        state_ontray = []

        for fact_str in state:
            predicate, args = parse_fact(fact_str)
            if predicate == 'served':
                served_children.add(args[0])
            elif predicate == 'at_kitchen_bread':
                if self.is_ng_bread.get(args[0], False):
                    available_bread_ng += 1
                else:
                    available_bread_reg += 1
            elif predicate == 'at_kitchen_content':
                if self.is_ng_content.get(args[0], False):
                    available_content_ng += 1
                else:
                    available_content_reg += 1
            elif predicate == 'at_kitchen_sandwich':
                state_at_kitchen_sandwich.append(args[0]) # Store sandwich name
            elif predicate == 'ontray':
                state_ontray.append((args[0], args[1])) # Store (sandwich, tray)
            elif predicate == 'at':
                trays_at[args[1]] = trays_at.get(args[1], 0) + 1 # Count trays at places
            elif predicate == 'notexist':
                unused_sandwiches += 1
            elif predicate == 'no_gluten_sandwich':
                sandwich_is_ng[args[0]] = True

        # Second pass to count sandwiches by type and location using NG status
        for s in state_at_kitchen_sandwich:
            if sandwich_is_ng.get(s, False):
                available_sandwich_ng_kitchen += 1
            else:
                available_sandwich_reg_kitchen += 1

        for s, t in state_ontray:
            if sandwich_is_ng.get(s, False):
                available_sandwich_ng_ontray += 1
            else:
                available_sandwich_reg_ontray += 1


        # --- Calculate Heuristic Components ---

        # H_serve: Number of unserved children
        unserved_children_list = [c for c in self.all_children if c not in served_children]
        h_serve = len(unserved_children_list)

        # If goal reached, heuristic is 0
        if h_serve == 0:
            return 0

        # Determine total sandwich needs
        needed_ng = sum(1 for c in unserved_children_list if self.is_allergic.get(c, False))
        needed_reg = sum(1 for c in unserved_children_list if not self.is_allergic.get(c, False))
        needed_total = needed_ng + needed_reg

        # Calculate makable sandwiches (potential supply from ingredients)
        ingredients_ng = min(available_bread_ng, available_content_ng)
        ingredients_reg = min(available_bread_reg, available_content_reg)
        # Prioritize NG sandwiches for unused objects
        makable_ng_possible = min(ingredients_ng, unused_sandwiches)
        makable_reg_possible = min(ingredients_reg, max(0, unused_sandwiches - makable_ng_possible))

        # H_make: Sandwiches needing to be made
        sandwiches_already_made = available_sandwich_ng_kitchen + available_sandwich_reg_kitchen + available_sandwich_ng_ontray + available_sandwich_reg_ontray
        h_make = max(0, needed_total - sandwiches_already_made)

        # H_put: Sandwiches needing to be put on trays
        sandwiches_already_on_trays = available_sandwich_ng_ontray + available_sandwich_reg_ontray
        h_put = max(0, needed_total - sandwiches_already_on_trays)

        # H_move_visits: Tray movements needed
        children_waiting_at = {p: 0 for p in self.all_places}
        for child in unserved_children_list:
            place = self.waiting_place.get(child)
            if place: # Child should always have a waiting place if not served and problem is well-formed
                children_waiting_at[place] = children_waiting_at.get(place, 0) + 1

        h_move_visits = 0
        for place in self.all_places:
            needed_visits = children_waiting_at.get(place, 0)
            available_trays = trays_at.get(place, 0)
            h_move_visits += max(0, needed_visits - available_trays)

        # Total heuristic is the sum of components
        heuristic_value = h_serve + h_make + h_put + h_move_visits

        # --- Check for Unsolvability ---
        # Total potential sandwiches = already made + makable
        total_potential_ng = available_sandwich_ng_kitchen + available_sandwich_ng_ontray + makable_ng_possible
        total_potential_reg = available_sandwich_reg_kitchen + available_sandwich_reg_ontray + makable_reg_possible

        # Check if enough NG sandwiches can ever be produced/are available for allergic children
        if needed_ng > total_potential_ng:
            return math.inf

        # Check if enough total sandwiches can ever be produced/are available for non-allergic children
        # Non-allergic can use remaining potential NG or potential Reg
        remaining_potential_ng = total_potential_ng - needed_ng
        available_for_non_allergic = remaining_potential_ng + total_potential_reg
        if needed_reg > available_for_non_allergic:
            return math.inf

        # Check if enough trays exist for the total number of sandwiches needed
        if needed_total > self.total_trays:
            return math.inf

        return heuristic_value
