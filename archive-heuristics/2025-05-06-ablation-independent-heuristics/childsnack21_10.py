# Need to import the base Heuristic class
from heuristics.heuristic_base import Heuristic
# Need Task class for type hinting in docstring
from task import Task

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to reach the goal state
    by summing up the estimated number of 'make_sandwich', 'put_on_tray',
    'move_tray', and 'serve_sandwich' actions needed to serve all unserved
    children. It accounts for gluten-free requirements and available
    sandwiches/trays at different stages of the serving process.

    Assumptions:
    - The heuristic is non-negative and is 0 if and only if the goal state is reached.
    - The heuristic value is finite for solvable states.
    - Enough bread, content, and sandwich objects exist in the initial state
      to serve all children, so we do not need to check resource availability
      beyond counting objects in the current state.
    - The heuristic does not need to be admissible.

    Heuristic Initialization:
    In the constructor, the heuristic pre-processes the static facts from the
    task definition. It extracts and stores:
    - The allergy status for each child (allergic_gluten or not_allergic_gluten).
    - The waiting place for each child.
    - The set of gluten-free bread and content portions (although not strictly
      used in the current heuristic calculation, this information is available
      in static facts).
    - The set of all children that need to be served (derived from the goal facts).

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic calculates the sum of four components,
    each estimating the number of actions needed at a specific stage of
    serving a child:

    1.  h_serve: The number of 'serve_sandwich' actions needed. This is simply
        the count of children who have not yet been served according to the
        current state.

    2.  h_move_tray: The minimum number of 'move_tray' actions needed. This is
        estimated by counting the number of distinct places where unserved
        children are waiting, but where no tray is currently located. Each such
        place requires at least one tray movement to become serviceable.

    3.  h_put_on_tray: The number of 'put_on_tray' actions needed. This estimates
        how many sandwiches need to transition from being "made" (either in the
        kitchen or just created) to being "on a tray". It is calculated by
        finding the deficit between the total number of suitable sandwiches
        required for unserved children and the number of suitable sandwiches
        already present on trays. Gluten-free requirements are respected, and
        any surplus of made gluten-free sandwiches is considered available to
        meet regular sandwich needs.

    4.  h_make_sandwich: The number of 'make_sandwich' actions needed. This
        estimates how many sandwiches need to transition from the 'notexist'
        state to being "made". It is calculated by finding the deficit between
        the total number of suitable sandwiches required for unserved children
        and the number of suitable sandwiches that have already been made
         (either in the kitchen or on trays). Gluten-free requirements are
        respected, and any surplus of made gluten-free sandwiches is considered
        available to meet regular sandwich needs.

    The total heuristic value is the sum of h_serve + h_move_tray + h_put_on_tray + h_make_sandwich.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic by pre-processing static task information.

        Args:
            task: The planning task object.
        """
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Pre-process static facts
        self.child_allergy = {}  # child_name -> True (allergic) or False
        self.child_place = {}    # child_name -> place_name
        self.gf_bread = set()    # set of gluten-free bread names
        self.gf_content = set()  # set of gluten-free content names
        self.all_children = set() # set of all children in the problem

        for fact_str in self.static:
            pred, args = self._parse_fact(fact_str)
            if pred == 'allergic_gluten':
                self.child_allergy[args[0]] = True
            elif pred == 'not_allergic_gluten':
                self.child_allergy[args[0]] = False
            elif pred == 'waiting':
                # Assuming each child waits at exactly one place
                self.child_place[args[0]] = args[1]
            elif pred == 'no_gluten_bread':
                self.gf_bread.add(args[0])
            elif pred == 'no_gluten_content':
                self.gf_content.add(args[0])

        # Get all children from the goal facts (assuming goal is always (served childX))
        for goal_fact in self.goals:
             pred, args = self._parse_fact(goal_fact)
             if pred == 'served' and args:
                 self.all_children.add(args[0])


    def _parse_fact(self, fact_str):
        """Helper method to parse a PDDL fact string."""
        # Remove surrounding parentheses and split by space
        parts = fact_str.strip('()').split()
        if not parts: # Handle empty string case if necessary
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def __call__(self, node) -> int:
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            The estimated number of actions to reach the goal.
        """
        state = node.state # state is a frozenset

        # 1. Identify unserved children
        served_children = set()
        # Need to iterate through state to find served children
        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'served' and args:
                served_children.add(args[0])

        unserved_children = self.all_children - served_children
        num_unserved_children = len(unserved_children)

        # Heuristic is 0 if all children are served
        if num_unserved_children == 0:
            return 0

        # 2. Categorize unserved children by allergy
        unserved_allergic = {c for c in unserved_children if self.child_allergy.get(c, False)}
        unserved_not_allergic = unserved_children - unserved_allergic
        num_unserved_gf = len(unserved_allergic)
        num_unserved_reg = len(unserved_not_allergic)

        # 3. Count sandwiches at different stages and types
        # Need to know which *made* sandwiches are GF first
        no_gluten_sandwiches_made = set()
        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'no_gluten_sandwich' and args:
                no_gluten_sandwiches_made.add(args[0])

        at_kitchen_sandwich_gf = set()
        at_kitchen_sandwich_reg = set()
        ontray_sandwich_gf = set()
        ontray_sandwich_reg = set()
        trays_at_place = {} # place -> set of trays

        for fact_str in state:
            pred, args = self._parse_fact(fact_str)
            if pred == 'at_kitchen_sandwich' and args:
                s = args[0]
                if s in no_gluten_sandwiches_made:
                    at_kitchen_sandwich_gf.add(s)
                else:
                    at_kitchen_sandwich_reg.add(s)
            elif pred == 'ontray' and len(args) == 2:
                s, t = args
                if s in no_gluten_sandwiches_made:
                    ontray_sandwich_gf.add(s)
                else:
                    ontray_sandwich_reg.add(s)
            elif pred == 'at' and len(args) == 2:
                t, p = args
                if p not in trays_at_place:
                    trays_at_place[p] = set()
                trays_at_place[p].add(t)
            # We don't strictly need counts of bread/content/notexist for this heuristic calculation
            # but they could be used for a more complex version or solvability check.

        num_sandwiches_made_gf = len(at_kitchen_sandwich_gf) + len(ontray_sandwich_gf)
        num_sandwiches_made_reg = len(at_kitchen_sandwich_reg) + len(ontray_sandwich_reg)
        num_sandwiches_ontray_gf = len(ontray_sandwich_gf)
        num_sandwiches_ontray_reg = len(ontray_sandwich_reg)

        # 4. Identify places needing trays
        # Ensure child_place lookup is safe in case of unexpected static facts
        places_with_unserved_children = {self.child_place[c] for c in unserved_children if c in self.child_place}
        places_with_trays = set(trays_at_place.keys())

        # 5. Calculate costs for each stage
        cost_serve = num_unserved_children

        # Cost to move trays to places that need them
        # Count places with unserved children that have no tray
        places_needing_tray = places_with_unserved_children - places_with_trays
        cost_move_tray = len(places_needing_tray)


        # Cost to put sandwiches on trays
        # Need num_unserved_gf GF sandwiches total. Have num_sandwiches_ontray_gf on trays.
        needed_to_put_gf = max(0, num_unserved_gf - num_sandwiches_ontray_gf)
        # Need num_unserved_reg Reg sandwiches total. Have num_sandwiches_ontray_reg on trays.
        # Surplus GF sandwiches on trays can cover Reg needs.
        gf_ontray_surplus = max(0, num_sandwiches_ontray_gf - num_unserved_gf)
        needed_to_put_reg = max(0, num_unserved_reg - (num_sandwiches_ontray_reg + gf_ontray_surplus))
        cost_put_on_tray = needed_to_put_gf + needed_to_put_reg

        # Cost to make sandwiches
        # Need num_unserved_gf GF sandwiches total. Have num_sandwiches_made_gf made.
        needed_to_make_gf = max(0, num_unserved_gf - num_sandwiches_made_gf)
        # Need num_unserved_reg Reg sandwiches total. Have num_sandwiches_made_reg made.
        # Surplus GF sandwiches made can cover Reg needs.
        gf_made_surplus = max(0, num_sandwiches_made_gf - num_unserved_gf)
        needed_to_make_reg = max(0, num_unserved_reg - (num_sandwiches_made_reg + gf_made_surplus))
        cost_make_sandwich = needed_to_make_gf + needed_to_make_reg

        # Total heuristic is the sum of costs for each stage
        h_value = cost_serve + cost_move_tray + cost_put_on_tray + cost_make_sandwich

        return h_value
