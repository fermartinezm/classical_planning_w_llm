import collections

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string into predicate and arguments.
    e.g., '(at tray1 kitchen)' -> ('at', ['tray1', 'kitchen'])
    """
    # Removes surrounding brackets and splits by space
    parts = fact_string.strip("()").split()
    predicate = parts[0]
    args = parts[1:]
    return predicate, args


class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the "distance" of each unserved child from being served.
    The distance is measured by how many stages of the sandwich delivery
    pipeline (make -> put on tray -> move tray -> serve) the child's required
    sandwich still needs to go through. It is an additive heuristic that counts
    missing prerequisites for each unserved child independently, which may
    overestimate the true cost but provides a reasonable estimate for greedy
    best-first search.

    Assumptions:
    - The heuristic is designed for greedy best-first search and does not need
      to be admissible.
    - The problem instance is solvable. The heuristic does not explicitly check
      for resource dead-ends (e.g., not enough bread/content/notexist items)
      when estimating the 'make' stage cost, assuming that if a sandwich is
      needed at that stage, it can eventually be made.
    - The cost of moving a tray between any two places is 1 action.
    - Shared resources (sandwiches, trays) are not explicitly modeled for
      optimal allocation in the heuristic calculation; instead, it counts
      missing prerequisites per child, which can lead to overestimation but
      simplifies computation.

    Heuristic Initialization:
    The constructor processes the static facts from the task description.
    It identifies all objects of each type (child, sandwich, bread-portion,
    content-portion, tray, place) by parsing initial state and static facts.
    It stores static relationships such as which children are allergic to gluten,
    where each child is waiting, and which bread/content items are gluten-free.
    This pre-processing allows for quick lookups during the heuristic
    computation for each state.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  Identify all children who have not yet been served (`(served ?c)` is false).
    2.  If all children are served, the state is a goal state, and the heuristic is 0.
    3.  Initialize the heuristic value `h` with the total number of unserved children.
        This represents the minimum number of `serve` actions required.
    4.  Initialize counters for children needing further steps:
        `children_need_delivery = 0` (need sandwich on tray at their location)
        `children_need_put = 0` (need sandwich on tray anywhere)
        `children_need_make = 0` (need sandwich made or moved to kitchen)
    5.  For each unserved child:
        a.  Determine the child's waiting location (`p`) and allergy status (allergic/not allergic) using the pre-processed static information.
        b.  Check if a suitable sandwich (`s`) is currently on a tray (`t`) that is located *at* the child's waiting place (`p`). A sandwich is suitable if it meets the child's gluten requirements (any sandwich for non-allergic, gluten-free for allergic).
        c.  If no such suitable sandwich is found on a tray at the child's location, increment `children_need_delivery`. This child needs a sandwich delivered.
        d.  If the child needs delivery (step 5c was true), check if a suitable sandwich (`s`) is currently on a tray (`t`) *anywhere* (not necessarily at `p`).
        e.  If no such suitable sandwich is found on a tray anywhere, increment `children_need_put`. This child needs a sandwich put on a tray.
        f.  If the child needs put (step 5e was true), check if a suitable sandwich (`s`) is currently `at_kitchen_sandwich`.
        g.  If no such suitable sandwich is found `at_kitchen_sandwich`, increment `children_need_make`. This child needs a sandwich made (or moved to the kitchen if it exists elsewhere, but the heuristic simplifies this to 'make').
    6.  Add `children_need_delivery`, `children_need_put`, and `children_need_make` to the heuristic value `h`.
    7.  Return the final value of `h`.

    This heuristic is monotonic across the defined stages (served -> on_tray_at_loc -> on_tray_anywhere -> kitchen/made), providing a clear gradient towards the goal.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Args:
            task: The planning task object (instance of the Task class).
        """
        self.static_facts = task.static

        # Dictionaries mapping predicate name and arg index to type for parsing objects
        arg_types = {
            'at_kitchen_bread': {0: 'bread-portion'},
            'at_kitchen_content': {0: 'content-portion'},
            'at_kitchen_sandwich': {0: 'sandwich'},
            'no_gluten_bread': {0: 'bread-portion'},
            'no_gluten_content': {0: 'content-portion'},
            'ontray': {0: 'sandwich', 1: 'tray'},
            'no_gluten_sandwich': {0: 'sandwich'},
            'allergic_gluten': {0: 'child'},
            'not_allergic_gluten': {0: 'child'},
            'served': {0: 'child'},
            'waiting': {0: 'child', 1: 'place'},
            'at': {0: 'tray', 1: 'place'},
            'notexist': {0: 'sandwich'},
        }

        # Sets to store all objects of each type
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_bread = set()
        self.all_content = set()
        self.all_trays = set()
        self.all_places = set()
        self.all_places.add('kitchen') # kitchen is a constant place

        object_sets = {
            'child': self.all_children,
            'sandwich': self.all_sandwiches,
            'bread-portion': self.all_bread,
            'content-portion': self.all_content,
            'tray': self.all_trays,
            'place': self.all_places,
        }

        # Process initial state and static facts to find all objects
        # We process initial state as well because not all objects might appear in static facts
        for fact_string in task.initial_state | task.static:
            predicate, args = parse_fact(fact_string)
            if predicate in arg_types:
                for i, arg in enumerate(args):
                    if i in arg_types[predicate]:
                        obj_type = arg_types[predicate][i]
                        if obj_type in object_sets:
                            object_sets[obj_type].add(arg)

        # Convert sets to frozensets for immutability
        self.all_children = frozenset(self.all_children)
        self.all_sandwiches = frozenset(self.all_sandwiches)
        self.all_bread = frozenset(self.all_bread)
        self.all_content = frozenset(self.all_content)
        self.all_trays = frozenset(self.all_trays)
        self.all_places = frozenset(self.all_places)

        # Store specific static relationships
        self.allergic_children = {c for c in self.all_children if '(allergic_gluten ' + c + ')' in self.static_facts}
        self.waiting_map = {}
        for fact_string in self.static_facts:
            predicate, args = parse_fact(fact_string)
            if predicate == 'waiting':
                child, place = args
                self.waiting_map[child] = place

        # Store static gluten-free info (though not directly used in this simple heuristic, good practice)
        self.no_gluten_bread_set_static = {b for b in self.all_bread if '(no_gluten_bread ' + b + ')' in self.static_facts}
        self.no_gluten_content_set_static = {c for c in self.all_content if '(no_gluten_content ' + c + ')' in self.static_facts}


    def is_suitable(self, s_name, child_name, no_gluten_sandwich_set_state):
        """
        Checks if a sandwich is suitable for a given child based on gluten requirements.

        Args:
            s_name: The name of the sandwich.
            child_name: The name of the child.
            no_gluten_sandwich_set_state: Set of gluten-free sandwich names in the current state.

        Returns:
            True if the sandwich is suitable, False otherwise.
        """
        is_allergic = child_name in self.allergic_children

        if is_allergic:
            # Allergic child needs a gluten-free sandwich
            return s_name in no_gluten_sandwich_set_state
        else:
            # Non-allergic child can have any sandwich
            return True


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: A frozenset of strings representing the current state facts.

        Returns:
            An integer heuristic value.
        """
        # Extract dynamic information from the current state
        served_children = set()
        at_kitchen_sandwich_set = set()
        ontray_map = {} # {sandwich: tray}
        at_map = {} # {tray: place}
        no_gluten_sandwich_set_state = set()

        for fact_string in state:
            predicate, args = parse_fact(fact_string)
            if predicate == 'served': served_children.add(args[0])
            elif predicate == 'at_kitchen_sandwich': at_kitchen_sandwich_set.add(args[0])
            elif predicate == 'ontray': ontray_map[args[0]] = args[1]
            elif predicate == 'at': at_map[args[0]] = args[1]
            elif predicate == 'no_gluten_sandwich': no_gluten_sandwich_set_state.add(args[0])

        # Identify unserved children
        unserved_children = set(self.all_children) - served_children

        # If all children are served, goal reached
        if not unserved_children:
            return 0

        # Base heuristic: Number of children still needing the final 'serve' action
        h = len(unserved_children)

        # Counters for children needing earlier stages of the pipeline
        children_need_delivery = 0 # Need sandwich on tray at their location
        children_need_put = 0      # Need sandwich on tray anywhere
        children_need_make = 0     # Need sandwich made or moved to kitchen

        # Evaluate each unserved child's needs
        for child in unserved_children:
            child_loc = self.waiting_map[child]

            # Check 1: Does the child have a suitable sandwich on a tray AT their location?
            has_ontray_at_loc_suitable = False
            for s_name, t_name in ontray_map.items():
                # Check if the tray is at the child's location
                if t_name in at_map and at_map[t_name] == child_loc:
                    # Check if the sandwich is suitable for the child
                    if self.is_suitable(s_name, child, no_gluten_sandwich_set_state):
                         has_ontray_at_loc_suitable = True
                         break # Found a suitable sandwich at the location for this child

            # If not ready for delivery, increment counter and check earlier stages
            if not has_ontray_at_loc_suitable:
                children_need_delivery += 1

                # Check 2: Does the child have a suitable sandwich ON TRAY anywhere?
                has_ontray_anywhere_suitable = False
                for s_name in ontray_map.keys():
                    # Check if the sandwich is suitable for the child
                    if self.is_suitable(s_name, child, no_gluten_sandwich_set_state):
                        has_ontray_anywhere_suitable = True
                        break # Found a suitable sandwich on a tray anywhere for this child

                # If not ready for put_on_tray, increment counter and check earlier stages
                if not has_ontray_anywhere_suitable:
                    children_need_put += 1

                    # Check 3: Does the child have a suitable sandwich AT KITCHEN?
                    has_kitchen_suitable = False
                    for s_name in at_kitchen_sandwich_set:
                        # Check if the sandwich is suitable for the child
                        if self.is_suitable(s_name, child, no_gluten_sandwich_set_state):
                            has_kitchen_suitable = True
                            break # Found a suitable sandwich in the kitchen for this child

                    # If not ready from kitchen, it needs to be made (or moved to kitchen)
                    # We simplify this to needing the 'make' step
                    if not has_kitchen_suitable:
                        children_need_make += 1 # Assume makeable if needed

        # Add the costs for the earlier stages needed by children
        h += children_need_delivery
        h += children_need_put
        h += children_need_make

        return h

