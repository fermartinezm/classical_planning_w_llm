# Required imports
# frozenset and set are built-in types, no import needed.

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
        This heuristic estimates the number of actions required to reach the goal
        state (all children served) by summing up the estimated costs of
        sequential stages: making needed sandwiches, putting them on trays,
        moving trays to children's locations, and finally serving the children.
        It is not admissible but aims to guide a greedy best-first search
        efficiently by prioritizing states that are closer to satisfying
        the prerequisites for serving children.

    Assumptions:
        - The problem instance is solvable. This implies that there are enough
          ingredients, sandwich objects, and trays in the initial state to
          eventually serve all children. The heuristic does not explicitly
          check for resource sufficiency beyond counting deficits for actions.
        - Each action has a cost of 1.

    Heuristic Initialization:
        The constructor processes the static facts from the task definition
        to identify:
        - Which children are allergic to gluten.
        - Which children are not allergic to gluten.
        - Which children are waiting and at which locations.
        - Which bread and content portions are gluten-free.
        This information is stored for efficient access during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state, the heuristic calculates the sum of estimated costs for
        the following stages:

        1.  Cost to Make Sandwiches (h_make):
            - Count the number of unserved children who need a gluten-free sandwich (N_gf_needed).
            - Count the number of unserved children who need a regular sandwich (N_reg_needed).
            - Count the total number of gluten-free sandwiches already made (in kitchen or on tray) (N_gf_available_total).
            - Count the total number of regular sandwiches already made (in kitchen or on tray) (N_reg_available_total).
            - The number of gluten-free sandwiches that still need to be made is max(0, N_gf_needed - N_gf_available_total).
            - The number of regular sandwiches that still need to be made is max(0, N_reg_needed - N_reg_available_total).
            - h_make is the sum of these two counts (each make action creates one sandwich).

        2.  Cost to Put Sandwiches on Trays (h_put_on_tray):
            - Count the number of gluten-free sandwiches already on trays (N_gf_available_ontray).
            - Count the number of regular sandwiches already on trays (N_reg_available_ontray).
            - The number of additional gluten-free sandwiches that need to be put on trays is max(0, N_gf_needed - N_gf_available_ontray).
            - The number of additional regular sandwiches that need to be put on trays is max(0, N_reg_needed - N_reg_available_ontray).
            - h_put_on_tray is the sum of these two counts (each put_on_tray action moves one sandwich).

        3.  Cost to Move Tray to Kitchen (h_move_tray_to_kitchen):
            - If h_put_on_tray > 0 (meaning sandwiches need to be put on trays) AND there are no trays currently at the kitchen, then one move_tray action is needed to bring a tray to the kitchen.
            - h_move_tray_to_kitchen is 1 in this case, otherwise 0.

        4.  Cost to Move Trays to Children's Locations (h_move_trays_to_places):
            - Identify the set of unique locations where unserved children are waiting (Places_needed).
            - Count the number of these locations that currently have at least one tray (N_places_covered_by_trays).
            - The number of additional locations that need a tray moved there is max(0, |Places_needed| - N_places_covered_by_trays).
            - h_move_trays_to_places is this count (each move_tray action can potentially cover one new needed location).

        5.  Cost to Serve Children (h_serve):
            - Count the total number of unserved children (N_unserved).
            - h_serve is equal to N_unserved (each serve action serves one child).

        The total heuristic value is the sum:
        h = h_make + h_put_on_tray + h_move_tray_to_kitchen + h_move_trays_to_places + h_serve.

        If the state is a goal state (all children served), N_unserved is 0, which makes all components 0, resulting in a heuristic of 0.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static facts.

        Args:
            task: The planning task object containing static facts, initial state, goals, etc.
        """
        self.static_facts = task.static

        # Pre-process static facts
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.waiting_children = {}  # child -> place
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.all_children = set() # Keep track of all children mentioned in waiting facts

        for fact in self.static_facts:
            # Extract predicate and arguments, ignoring surrounding brackets
            parts = fact.strip('()').split()
            if not parts: # Skip empty strings if any
                continue
            pred = parts[0]
            if pred == 'allergic_gluten':
                if len(parts) > 1: self.allergic_children.add(parts[1])
            elif pred == 'not_allergic_gluten':
                 if len(parts) > 1: self.not_allergic_children.add(parts[1])
            elif pred == 'waiting':
                if len(parts) > 2:
                    child = parts[1]
                    place = parts[2]
                    self.waiting_children[child] = place
                    self.all_children.add(child)
            elif pred == 'no_gluten_bread':
                 if len(parts) > 1: self.no_gluten_bread.add(parts[1])
            elif pred == 'no_gluten_content':
                 if len(parts) > 1: self.no_gluten_content.add(parts[1])

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: A frozenset of facts representing the current state.

        Returns:
            An integer heuristic value.
        """
        # Parse state facts
        served_children = set()
        at_kitchen_sandwich = set()
        ontray_sandwich_tray = set() # Store (sandwich, tray) tuples
        no_gluten_sandwich = set() # Need this from state as sandwiches are created
        tray_location = {} # tray -> place

        for fact in state:
            # Extract predicate and arguments, ignoring surrounding brackets
            parts = fact.strip('()').split()
            if not parts: # Skip empty strings if any
                continue
            pred = parts[0]
            if pred == 'served':
                if len(parts) > 1: served_children.add(parts[1])
            elif pred == 'at_kitchen_sandwich':
                if len(parts) > 1: at_kitchen_sandwich.add(parts[1])
            elif pred == 'ontray':
                if len(parts) > 2: ontray_sandwich_tray.add((parts[1], parts[2]))
            elif pred == 'no_gluten_sandwich':
                 if len(parts) > 1: no_gluten_sandwich.add(parts[1])
            elif pred == 'at':
                if len(parts) > 2:
                    tray = parts[1]
                    place = parts[2]
                    tray_location[tray] = place

        # 1. Unserved Children and Needs
        # Only consider children who were initially waiting according to static facts
        unserved_children = {c for c in self.all_children if c not in served_children}
        N_unserved = len(unserved_children)

        # Goal reached
        if N_unserved == 0:
            return 0

        N_gf_needed = len([c for c in unserved_children if c in self.allergic_children])
        N_reg_needed = len([c for c in unserved_children if c in self.not_allergic_children])

        # 2. Available Sandwiches (made)
        gf_kitchen_count = len([s for s in at_kitchen_sandwich if s in no_gluten_sandwich])
        reg_kitchen_count = len([s for s in at_kitchen_sandwich if s not in no_gluten_sandwich])
        gf_ontray_count = len([s for s, t in ontray_sandwich_tray if s in no_gluten_sandwich])
        reg_ontray_count = len([s for s, t in ontray_sandwich_tray if s not in no_gluten_sandwich])

        N_gf_available_total = gf_kitchen_count + gf_ontray_count
        N_reg_available_total = reg_kitchen_count + reg_ontray_count

        # 3. Calculate Heuristic Components

        # h_make: Cost to make sandwiches
        h_make = max(0, N_gf_needed - N_gf_available_total) + max(0, N_reg_needed - N_reg_available_total)

        # h_put_on_tray: Cost to put sandwiches on trays from kitchen
        # These are the sandwiches needed on trays minus those already on trays.
        # We assume they will come from the kitchen (either existing or just made).
        N_put_on_tray = max(0, N_gf_needed - N_gf_available_ontray) + max(0, N_reg_needed - N_reg_available_ontray)
        h_put_on_tray = N_put_on_tray

        # h_move_tray_to_kitchen: Cost to move a tray to kitchen if needed for put_on_tray
        N_trays_kitchen = len([t for t, p in tray_location.items() if p == 'kitchen'])
        h_move_tray_to_kitchen = 0
        if h_put_on_tray > 0 and N_trays_kitchen == 0:
            h_move_tray_to_kitchen = 1

        # h_move_trays_to_places: Cost to move trays to children's locations
        # Only consider places where unserved children are waiting
        places_with_unserved_children = {self.waiting_children[c] for c in unserved_children if c in self.waiting_children}
        N_places_needed = len(places_with_unserved_children)

        needed_places_covered_by_trays = {p for t, p in tray_location.items() if p in places_with_unserved_children}
        N_places_covered_by_trays = len(needed_places_covered_by_trays)

        h_move_trays_to_places = max(0, N_places_needed - N_places_covered_by_trays)

        # h_serve: Cost to serve children
        h_serve = N_unserved

        # Total heuristic
        total_heuristic = h_make + h_put_on_tray + h_move_tray_to_kitchen + h_move_trays_to_places + h_serve

        return total_heuristic
