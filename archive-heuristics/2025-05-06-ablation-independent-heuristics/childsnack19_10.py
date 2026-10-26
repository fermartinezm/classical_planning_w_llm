from heuristics.heuristic_base import Heuristic
from task import Task # Import Task for type hinting and accessing task attributes

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into predicate and objects."""
    # Remove parentheses
    cleaned_fact = fact_string.strip()[1:-1]
    # Split by space
    parts = cleaned_fact.split()
    predicate = parts[0]
    objects = parts[1:] if len(parts) > 1 else []
    return predicate, objects

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach the goal
    state (all children served) by summing up the estimated costs of moving
    items (sandwiches, trays) through the necessary stages of preparation
    and delivery, plus the final serving action for each child. It counts
    the number of items/locations that need to transition through key steps:
    making sandwiches, ensuring sandwiches are on trays, moving trays to children's
    locations, and finally serving the children.

    Assumptions:
    - Sufficient bread, content, and 'notexist' sandwich objects are available
      in the problem definition to make all required sandwiches. The heuristic
      does not penalize for resource scarcity beyond the counts of already
      made sandwiches.
    - Sufficient tray objects exist in the problem definition to be moved
      as needed.
    - Children's locations and allergies are static.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition.
    It extracts and stores:
    - Which children are allergic or not allergic in the `is_allergic` dictionary.
    - The waiting location for each child in the `child_location` dictionary.
    - Which bread and content portions are gluten-free in `is_gf_bread` and `is_gf_content`
      dictionaries (though these are not directly used in the current heuristic calculation).
    It also identifies all children involved in the task based on goals and static facts
    and stores them in the `all_children` set, defaulting children not specified
    as allergic in static facts to non-allergic.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state, the heuristic is computed as follows:
    1.  Identify all children that have not yet been served by checking the goal
        facts (`(served ?c)`) against the current state facts.
    2.  If all children are served, the heuristic is 0.
    3.  Count the number of unserved children (`N_unserved`), splitting the count into
        allergic (`N_allergic_unserved`) and non-allergic (`N_reg_unserved`) categories
        using the pre-calculated static allergy information.
    4.  Identify the set of unique locations (excluding the 'kitchen') where
        unserved children are waiting, using the pre-calculated static location
        information. Store the count as `N_locations_needing_trays`.
    5.  Iterate through the current state facts to count:
        - Sandwiches currently in the kitchen (`at_kitchen_sandwich`), splitting
          by gluten-free status (checked by looking for the `no_gluten_sandwich` fact
          in the current state). Store counts as `N_gf_kitchen` and `N_reg_kitchen`.
        - Sandwiches currently on trays (`ontray`), splitting by gluten-free status.
          Store counts as `N_gf_ontray` and `N_reg_ontray`.
        - Trays currently in the kitchen (`at ?t kitchen`). Store count as `N_trays_kitchen`.
        - Trays currently located at any of the locations identified in step 4.
          Store the count of distinct trays as `N_trays_at_needed_locations`.
    6.  Calculate the number of gluten-free and regular sandwiches that still
        need to be *made*. This is the number of unserved allergic/non-allergic
        children minus the number of already made (kitchen + ontray) gluten-free/regular
        sandwiches, clamped at zero. This gives the estimated number of
        'make_sandwich' actions needed (`Cost_make`).
    7.  Calculate the number of sandwiches that ultimately need to end up on a tray
        to serve the unserved children, minus those already on trays. This represents
        the number of 'put_on_tray' actions needed (`Cost_put_on_tray`).
    8.  Calculate the number of tray movements needed to get trays to the
        locations where unserved children are waiting. This is the number of
        unique locations needing trays minus the number of trays already at
        those locations, clamped at zero. This gives the estimated number of
        'move_tray' actions to locations (`Cost_move_to_locations`).
    9.  The number of 'serve' actions needed is simply the number of unserved
        children (`Cost_serve`).
    10. The total heuristic value is the sum of the estimated costs for these
        sequential stages: `Cost_make + Cost_put_on_tray + Cost_move_to_locations + Cost_serve`.
    """

    def __init__(self, task: Task):
        """
        Initializes the heuristic with static information from the task.

        Args:
            task: An instance of the Task class containing problem definition.
        """
        super().__init__()
        self.goals = task.goals
        self.static = task.static

        # Store static information
        self.is_allergic = {}
        self.child_location = {}
        self.is_gf_bread = {}
        self.is_gf_content = {}
        self.all_children = set()

        # Parse static facts
        for fact_string in self.static:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'allergic_gluten':
                child_name = objects[0]
                self.is_allergic[child_name] = True
                self.all_children.add(child_name)
            elif predicate == 'not_allergic_gluten':
                child_name = objects[0]
                self.is_allergic[child_name] = False
                self.all_children.add(child_name)
            elif predicate == 'waiting':
                child_name, place_name = objects
                self.child_location[child_name] = place_name
                self.all_children.add(child_name)
            elif predicate == 'no_gluten_bread':
                bread_name = objects[0]
                self.is_gf_bread[bread_name] = True
            elif predicate == 'no_gluten_content':
                content_name = objects[0]
                self.is_gf_content[content_name] = True

        # Ensure all children from goals are considered, even if not in static (unlikely in this domain)
        for goal_string in self.goals:
             predicate, objects = parse_fact(goal_string)
             if predicate == 'served':
                 child_name = objects[0]
                 self.all_children.add(child_name)
                 # Default to not allergic if allergy not specified in static
                 if child_name not in self.is_allergic:
                     self.is_allergic[child_name] = False


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.

        Args:
            node: The search node containing the state (frozenset of fact strings).

        Returns:
            The estimated number of actions to reach the goal (integer).
        """
        state = node.state

        # --- Count items/needs from state and static info ---

        unserved_children = set()
        N_allergic_unserved = 0
        N_reg_unserved = 0
        Needed_locations = set() # Locations (not kitchen) with unserved children

        # Get unserved children and their locations/allergy
        for child in self.all_children:
            if f'(served {child})' not in state:
                unserved_children.add(child)
                child_loc = self.child_location.get(child)
                # Add location to needed locations if not kitchen and location is known
                if child_loc and child_loc != 'kitchen':
                     Needed_locations.add(child_loc)

                # Count allergic vs non-allergic unserved children
                if self.is_allergic.get(child, False): # Default to not allergic if info missing
                    N_allergic_unserved += 1
                else:
                    N_reg_unserved += 1

        N_unserved = len(unserved_children)

        # Heuristic is 0 if all children are served
        if N_unserved == 0:
            return 0

        N_locations_needing_trays = len(Needed_locations)

        N_gf_kitchen = 0
        N_reg_kitchen = 0
        N_gf_ontray = 0
        N_reg_ontray = 0
        N_trays_kitchen = 0
        trays_at_needed_locations_set = set() # Use a set to count distinct trays

        # Helper to check if a sandwich is GF based on state fact
        def is_sandwich_gf(sandwich_name, current_state):
             return f'(no_gluten_sandwich {sandwich_name})' in current_state

        # Count sandwiches and trays from the current state
        for fact_string in state:
            predicate, objects = parse_fact(fact_string)
            if predicate == 'at_kitchen_sandwich':
                s_name = objects[0]
                if is_sandwich_gf(s_name, state):
                    N_gf_kitchen += 1
                else:
                    N_reg_kitchen += 1
            elif predicate == 'ontray':
                s_name, t_name = objects
                if is_sandwich_gf(s_name, state):
                    N_gf_ontray += 1
                else:
                    N_reg_ontray += 1
            elif predicate == 'at':
                t_name, p_name = objects
                if p_name == 'kitchen':
                    N_trays_kitchen += 1
                elif p_name in Needed_locations:
                    trays_at_needed_locations_set.add(t_name)

        N_trays_at_needed_locations = len(trays_at_needed_locations_set)


        # --- Calculate heuristic components (action counts) ---

        # A. Sandwiches to Make: Number of sandwiches that are needed but not yet made.
        # Total made sandwiches available (kitchen + ontray)
        N_avail_gf = N_gf_kitchen + N_gf_ontray
        N_avail_reg = N_reg_kitchen + N_reg_ontray
        # Sandwiches of each type that still need to be made
        M_gf = max(0, N_allergic_unserved - N_avail_gf)
        M_reg = max(0, N_reg_unserved - N_avail_reg)
        Cost_make = M_gf + M_reg

        # B. Sandwiches to Put on Tray: Number of sandwiches that are needed but not yet on a tray.
        # Total needed sandwiches = N_allergic_unserved + N_reg_unserved
        # Total sandwiches already on trays = N_gf_ontray + N_reg_ontray
        # Sandwiches that still need to get onto a tray = max(0, Total needed - Total on trays)
        Cost_put_on_tray = max(0, (N_allergic_unserved + N_reg_unserved) - (N_gf_ontray + N_reg_ontray))

        # C. Move trays to locations: Number of locations with unserved children
        # that do not currently have a tray.
        Cost_move_to_locations = max(0, N_locations_needing_trays - N_trays_at_needed_locations)

        # D. Serve sandwiches: Number of unserved children.
        Cost_serve = N_unserved

        # Total heuristic is the sum of these estimated action counts.
        # This heuristic counts the number of items/stages that need to be processed:
        # - Sandwiches needing making.
        # - Sandwiches needing putting on tray (those not already on trays).
        # - Locations needing a tray delivered.
        # - Children needing serving.
        h = Cost_make + Cost_put_on_tray + Cost_move_to_locations + Cost_serve

        return h
