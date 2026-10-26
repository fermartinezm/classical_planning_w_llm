from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It breaks down the process into stages: making sandwiches, putting them on trays,
    moving trays to the children's locations, and finally serving the sandwiches.
    It sums the number of children/sandwiches that need to pass through each stage,
    plus penalties for resource shortages (ingredients, notexist sandwiches) that
    prevent sandwich making.

    # Assumptions
    - Each required action (make, put_on_tray, move_tray, serve) contributes 1 to the heuristic cost.
    - Resource limitations (ingredients, notexist sandwich objects) add cost if they prevent
      the necessary sandwiches from being made.
    - Tray capacity is effectively infinite for heuristic calculation purposes.
    - A single tray move can deliver all necessary sandwiches to one location.

    # Heuristic Initialization
    The heuristic extracts static information from the task definition:
    - Which children are allergic to gluten.
    - Which children are not allergic to gluten.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    This information is used to determine sandwich suitability and ingredient types.
    It also counts the total number of allergic and non-allergic children in the problem
    to determine the total number of GF and regular sandwiches ultimately required.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is calculated as follows:

    1.  **Base Cost (Serve Actions):** Count the total number of children who are currently `waiting` and not `served`. Each waiting child requires a final `serve_sandwich` action. Add this count to the heuristic.

    2.  **Cost for Tray Movement:** Count the number of waiting children who *cannot* be served by a suitable sandwich already present on a tray at their location. Each such child implicitly requires a sandwich to be moved to their location on a tray. Add this count to the heuristic.

    3.  **Cost for Putting on Tray:** Count the number of waiting children whose required suitable sandwich is *not* currently on any tray (it must be `at_kitchen_sandwich` or `notexist`). Each such child implicitly requires a `put_on_tray` action. Add this count to the heuristic.

    4.  **Cost for Making Sandwich:** Count the number of waiting children whose required suitable sandwich is *not* currently made (it must be `notexist`). Each such child implicitly requires a `make_sandwich` action. Add this count to the heuristic.

    5.  **Cost for Ingredient Shortage:** Calculate the total number of gluten-free and regular sandwiches that still need to be made to satisfy all children in the problem. Determine how many of these required sandwiches *cannot* be made due to a shortage of gluten-free or regular bread/content pairs at the kitchen. Add this deficit count to the heuristic.

    6.  **Cost for Notexist Sandwich Shortage:** Calculate the total number of sandwiches that still need to be made. Determine how many of these required sandwiches *cannot* be made because there are not enough `notexist` sandwich objects available. Add this deficit count to the heuristic.

    The total heuristic value is the sum of costs from steps 1 through 6.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about allergies and gluten-free items.
        """
        self.goals = task.goals # Not directly used in this heuristic calculation, but stored.

        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_bread_types = set()
        self.no_gluten_content_types = set()

        # Extract static information from the task's static facts
        for fact in task.static:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif predicate == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread":
                self.no_gluten_bread_types.add(parts[1])
            elif predicate == "no_gluten_content":
                self.no_gluten_content_types.add(parts[1])

        # Total count of allergic/non-allergic children in the problem
        # This assumes all children are listed in static facts with their allergy status.
        self.total_allergic_children = len(self.allergic_children)
        self.total_not_allergic_children = len(self.not_allergic_children)


    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # --- Parse the current state ---
        waiting_children_by_place = {} # {place: list of child names}
        # served_children = set() # Not needed for heuristic calculation
        sandwiches_at_kitchen = set() # set of sandwich names
        sandwiches_on_trays = set() # set of sandwich names
        sandwich_to_tray = {} # {sandwich name: tray name}
        tray_locations = {} # {tray name: place name}
        gluten_free_sandwiches_made = set() # set of sandwich names
        bread_at_kitchen = {} # {bread name: is_gluten_free boolean}
        content_at_kitchen = {} # {content name: is_gluten_free boolean}
        notexist_sandwiches = set() # set of sandwich names

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "waiting":
                child, place = parts[1], parts[2]
                waiting_children_by_place.setdefault(place, []).append(child)
            # elif predicate == "served":
            #     served_children.add(parts[1]) # Not needed for heuristic
            elif predicate == "at_kitchen_sandwich":
                sandwiches_at_kitchen.add(parts[1])
            elif predicate == "ontray":
                s, t = parts[1], parts[2]
                sandwiches_on_trays.add(s)
                sandwich_to_tray[s] = t
            elif predicate == "at" and len(parts) == 3: # e.g., (at tray1 kitchen), (at tray1 table1)
                 obj, place = parts[1], parts[2]
                 # Assuming objects with 'at' predicate are trays in this domain
                 tray_locations[obj] = place
            elif predicate == "no_gluten_sandwich":
                gluten_free_sandwiches_made.add(parts[1])
            elif predicate == "at_kitchen_bread":
                bread = parts[1]
                bread_at_kitchen[bread] = (bread in self.no_gluten_bread_types)
            elif predicate == "at_kitchen_content":
                content = parts[1]
                content_at_kitchen[content] = (content in self.no_gluten_content_types)
            elif predicate == "notexist":
                notexist_sandwiches.add(parts[1])

        # --- Calculate counts for heuristic ---

        num_waiting = sum(len(children) for children in waiting_children_by_place.values())

        # If no children are waiting, the goal is reached.
        if num_waiting == 0:
            return 0

        # Calculate num_children_ready_to_serve
        # This is the number of children who can be served by suitable sandwiches currently on trays at their location.
        num_children_ready_to_serve = 0
        for place, children in waiting_children_by_place.items():
            needed_gf_at_P = sum(1 for child in children if child in self.allergic_children)
            needed_reg_at_P = sum(1 for child in children if child not in self.allergic_children)

            available_gf_on_tray_at_P = 0
            available_reg_on_tray_at_P = 0
            # Count suitable sandwiches already on trays at this specific place
            for s_ontray in sandwiches_on_trays:
                t_ontray = sandwich_to_tray.get(s_ontray)
                p_ontray = tray_locations.get(t_ontray)
                if p_ontray == place:
                    if s_ontray in gluten_free_sandwiches_made:
                        available_gf_on_tray_at_P += 1
                    else:
                        available_reg_on_tray_at_P += 1

            # How many children at this place can be served by the available sandwiches?
            served_by_gf_at_P = min(needed_gf_at_P, available_gf_on_tray_at_P)
            remaining_gf_at_P = available_gf_on_tray_at_P - served_by_gf_at_P
            served_by_reg_at_P = min(needed_reg_at_P, available_reg_on_tray_at_P + remaining_gf_at_P)

            num_children_ready_to_serve += served_by_gf_at_P + served_by_reg_at_P

        # Calculate total sandwiches on trays
        total_sandwiches_on_trays = len(sandwiches_on_trays)

        # Calculate total sandwiches made (at kitchen or on trays)
        all_made_sandwiches = sandwiches_at_kitchen | sandwiches_on_trays
        total_sandwiches_made = len(all_made_sandwiches)

        # --- Compute the heuristic value ---
        h = 0

        # Cost 1: Serve action for each waiting child
        h += num_waiting

        # Cost 2: Tray move needed for children whose sandwich isn't at their location
        # This is the number of children who need a sandwich delivered.
        h += max(0, num_waiting - num_children_ready_to_serve)

        # Cost 3: Put on tray needed for children whose sandwich isn't on any tray
        # This is the number of sandwiches that need to be put on a tray.
        h += max(0, num_waiting - total_sandwiches_on_trays)

        # Cost 4: Make needed for children whose sandwich isn't made
        # This is the number of sandwiches that need to be made.
        sandwiches_to_make = max(0, num_waiting - total_sandwiches_made)
        h += sandwiches_to_make

        # Cost 5: Ingredient shortage for making sandwiches
        # Calculate how many GF/Reg sandwiches are still needed in total for all children in the problem.
        # This is based on the total number of allergic/non-allergic children from static facts.
        needed_gf_total = self.total_allergic_children
        made_gf_total = len(gluten_free_sandwiches_made)
        to_make_gf = max(0, needed_gf_total - made_gf_total)

        needed_reg_total = self.total_not_allergic_children
        made_reg_total = total_sandwiches_made - made_gf_total # Total made minus GF made
        to_make_reg = max(0, needed_reg_total - made_reg_total)
        # Adjust reg needed if GF surplus exists (GF sandwiches can satisfy reg needs)
        to_make_reg_net = max(0, to_make_reg - max(0, made_gf_total - needed_gf_total))

        # Count available ingredient pairs at kitchen
        num_gf_bread_kitchen = sum(1 for b, is_gf in bread_at_kitchen.items() if is_gf)
        num_gf_content_kitchen = sum(1 for c, is_gf in content_at_kitchen.items() if is_gf)
        gf_ingredient_pairs_available = min(num_gf_bread_kitchen, num_gf_content_kitchen)

        num_reg_bread_kitchen = sum(1 for b, is_gf in bread_at_kitchen.items() if not is_gf)
        num_reg_content_kitchen = sum(1 for c, is_gf in content_at_kitchen.items() if not is_gf)
        reg_ingredient_pairs_available = min(num_reg_bread_kitchen, num_reg_content_kitchen)

        # Cost related to missing ingredient pairs for sandwiches that need to be made
        # This counts how many sandwiches *cannot* be made due to ingredient shortage.
        h += max(0, to_make_gf - gf_ingredient_pairs_available)
        h += max(0, to_make_reg_net - reg_ingredient_pairs_available)

        # Cost 6: Notexist sandwich shortage for making sandwiches
        # This counts how many sandwiches *cannot* be made because there are not enough notexist objects.
        num_notexist = len(notexist_sandwiches)
        h += max(0, sandwiches_to_make - num_notexist)

        return h
