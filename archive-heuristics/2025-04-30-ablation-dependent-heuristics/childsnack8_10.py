# Required imports
import collections

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into a predicate and a list of objects.
    e.g., '(served child1)' -> ('served', ['child1'])
    e.g., '(waiting child1 table1)' -> ('waiting', ['child1', 'table1'])
    """
    # Remove leading '(' and trailing ')'
    fact_string = fact_string[1:-1]
    parts = fact_string.split()
    predicate = parts[0]
    objects = parts[1:]
    return predicate, objects

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served) by summing up estimated costs for different
    stages of the process: making sandwiches, putting sandwiches on trays,
    moving trays to children's locations, and serving children. It counts
    the number of items (sandwiches, trays, children) that need to
    transition through these stages based on the current state and the total
    demand from unserved children.

    Assumptions:
    - Assumes sufficient bread, content, and sandwich objects are available
      to make any required sandwiches. Resource constraints are not strictly
      modeled beyond counting the need to 'make'.
    - Assumes trays can be moved between any two locations in one action.
    - Assumes a tray at a location can be used to serve any child waiting
      at that location, provided it carries the appropriate sandwich type.
    - The heuristic is not admissible, designed for greedy best-first search.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition
    to build lookup structures for:
    - Identifying allergic and non-allergic children.
    - Mapping children to their waiting locations.
    - Identifying gluten-free bread and content types (though not used in the
      final heuristic calculation, it's good practice to parse static info).

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify all children who are not yet served based on the current state.
    2.  Determine the total number of gluten-free (GF) sandwiches and any-type
        sandwiches required to serve all unserved children, based on their
        allergy status.
    3.  Count the number of GF and regular sandwiches currently available
        (either in the kitchen or on any tray).
    4.  Calculate `cost_make`: The number of sandwiches (GF and Any) that
        still need to be created from ingredients to meet the total demand
        not covered by currently available sandwiches. This is a lower bound
        on the number of `make_sandwich` actions.
    5.  Calculate `cost_put_on_tray`: The number of sandwiches currently in
        the kitchen that are needed to meet the total demand not covered by
        sandwiches already on trays. This is a lower bound on the number of
        `put_on_tray` actions.
    6.  Identify all locations where unserved children are waiting.
    7.  Identify all locations where trays are currently present.
    8.  Calculate `cost_move_tray`: The number of locations with unserved
        children that do not currently have a tray. Each such location will
        likely require at least one `move_tray` action to bring a tray there.
    9.  Calculate `cost_serve`: The total number of unserved children. Each
        unserved child will require a final `serve_sandwich` action.
    10. The total heuristic value is the sum of `cost_make`, `cost_put_on_tray`,
        `cost_move_tray`, and `cost_serve`.
    11. If the goal state is reached (no children are unserved), the heuristic is 0.
    """

    def __init__(self, task):
        """
        Initializes the heuristic with static information from the task.

        Args:
            task: The planning task object.
        """
        self.all_children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children_loc = {} # {child_name: place_name}
        self.gf_bread_types = set() # Not strictly used in heuristic calculation, but parsed
        self.gf_content_types = set() # Not strictly used in heuristic calculation, but parsed
        self.all_places = set() # Places where children are waiting

        # Parse static facts
        for fact_string in task.static:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                child = objects[0]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten':
                child = objects[0]
                self.not_allergic_children.add(child)
                self.all_children.add(child)
            elif predicate == 'waiting':
                child, place = objects
                self.waiting_children_loc[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif predicate == 'no_gluten_bread':
                 self.gf_bread_types.add(objects[0])
            elif predicate == 'no_gluten_content':
                 self.gf_content_types.add(objects[0])

        # Ensure all children mentioned in waiting are in the sets
        for child in self.waiting_children_loc:
             self.all_children.add(child)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal.
        """
        # Parse dynamic facts from the current state
        served_children = set()
        kitchen_sandwiches = set() # Sandwiches at_kitchen_sandwich
        ontray_sandwiches = set() # Sandwiches on any tray
        tray_locations = {} # {tray: place}
        gf_sandwiches_state = set() # GF status from state (made sandwiches)

        for fact_string in state:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'served':
                served_children.add(objects[0])
            elif predicate == 'at_kitchen_sandwich':
                kitchen_sandwiches.add(objects[0])
            elif predicate == 'ontray':
                sandwich, tray = objects
                ontray_sandwiches.add(sandwich)
                # We don't need ontray_mapping for this heuristic calculation
            elif predicate == 'at':
                tray, place = objects
                tray_locations[tray] = place
            elif predicate == 'no_gluten_sandwich':
                 gf_sandwiches_state.add(objects[0])

        # --- Heuristic Calculation ---

        # 1. Unserved children count (cost_serve)
        unserved_children = self.all_children - served_children
        num_unserved = len(unserved_children)

        # If no children are unserved, the goal is reached
        if num_unserved == 0:
            return 0

        # 2. Sandwich needs
        needed_gf_total = len({c for c in unserved_children if c in self.allergic_children})
        needed_any_total = len({c for c in unserved_children if c in self.not_allergic_children})
        needed_total = needed_gf_total + needed_any_total

        # 3. Available sandwiches
        # GF status of sandwiches on trays or in kitchen
        avail_gf_ontray = {s for s in ontray_sandwiches if s in gf_sandwiches_state}
        # avail_reg_ontray = ontray_sandwiches - avail_gf_ontray # Not explicitly needed
        avail_gf_kitchen = {s for s in kitchen_sandwiches if s in gf_sandwiches_state}
        avail_reg_kitchen = kitchen_sandwiches - avail_gf_kitchen

        # 4. Cost to make sandwiches
        # Need GF sandwiches not covered by available GF sandwiches (ontray or kitchen)
        make_gf = max(0, needed_gf_total - (len(avail_gf_ontray) + len(avail_gf_kitchen)))
        # Need Any sandwiches not covered by available Reg sandwiches (ontray or kitchen)
        # PLUS surplus available GF sandwiches (ontray or kitchen)
        avail_any_for_non_allergic = (len(ontray_sandwiches) - len(avail_gf_ontray)) + (len(kitchen_sandwiches) - len(avail_gf_kitchen)) + max(0, (len(avail_gf_ontray) + len(avail_gf_kitchen)) - needed_gf_total)
        make_any = max(0, needed_any_total - avail_any_for_non_allergic)
        cost_make = make_gf + make_any

        # 5. Cost to put sandwiches on trays
        # Sandwiches currently in kitchen that are needed.
        # Total sandwiches needed from kitchen or make = max(0, needed_total - sandwiches_already_on_trays)
        sandwiches_already_on_trays = len(ontray_sandwiches)
        needed_from_kitchen_or_make = max(0, needed_total - sandwiches_already_on_trays)
        # Number of kitchen sandwiches that need to be put on tray is the minimum of
        # available kitchen sandwiches and the number needed from kitchen/make.
        cost_put_on_tray = min(len(kitchen_sandwiches), needed_from_kitchen_or_make)


        # 6. Cost to move trays
        # Locations with unserved children
        locations_with_unserved = {self.waiting_children_loc[c] for c in unserved_children}
        # Locations with trays
        locations_with_trays = set(tray_locations.values())
        # Locations needing a tray that don't have one
        locations_needing_tray_delivery = locations_with_unserved - locations_with_trays
        cost_move_tray = len(locations_needing_tray_delivery)

        # 7. Cost to serve children (already calculated as num_unserved)
        cost_serve = num_unserved

        # Total heuristic is the sum of costs for each stage
        heuristic_value = cost_make + cost_put_on_tray + cost_move_tray + cost_serve

        return heuristic_value

    def is_goal(self, state):
        """
        Checks if the goal state is reached.
        Goal: All children are served.
        """
        served_children = set()
        for fact_string in state:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'served':
                served_children.add(objects[0])

        # Goal is reached if the set of served children is equal to the set of all children
        return served_children == self.all_children

