# Ensure all necessary imports are here
from heuristics.heuristic_base import Heuristic
from task import Task
import math
from collections import defaultdict

# Helper function to parse fact strings like '(predicate obj1 obj2)'
def parse_fact_objects(fact_string):
    """Removes parentheses and splits the fact string into predicate and arguments."""
    parts = fact_string[1:-1].split()
    if not parts:
        return None, [] # Handle empty fact string case, though unlikely in PDDL
    return parts[0], parts[1:]

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the cost to reach the goal (all children served)
    by summing the estimated costs for each unserved child. The cost for
    serving a child depends on the current state of a suitable sandwich
    needed for that child, prioritizing sandwiches that are closer to being
    ready for serving (e.g., already on a tray at the child's location are
    cheaper than sandwiches that need to be made). The heuristic counts the
    number of unserved children requiring a sandwich from different states
    and assigns a cost based on the minimum actions needed to get a sandwich
    from that state to the child and serve it.

    Assumptions:
    - Sufficient trays are available in the kitchen for 'put_on_tray' actions
      if needed.
    - Ingredients are consumed when making sandwiches, but the heuristic
      only checks the initial counts of ingredients and 'notexist' sandwiches
      to estimate the maximum number of new sandwiches that can be made.
    - Any tray can be moved to any location.
    - The cost calculation assumes a sequence of actions (make, put_on_tray,
      move_tray, serve) and sums the required steps based on the starting
      point of the sandwich resource.

    Heuristic Initialization:
    The constructor parses the static facts from the task description to
    identify:
    - Children who are allergic to gluten.
    - Children who are not allergic to gluten.
    - Bread portions that are no-gluten.
    - Content portions that are no-gluten.
    - The waiting location for each child.
    - The set of all children that need to be served (goal children).

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify all children that are not yet served based on the current state
       and the goal. If all children are served, the heuristic is 0.
    2. Categorize the unserved children into those needing a gluten-free
       sandwich (allergic) and those needing any sandwich (not allergic).
    3. Count the available suitable sandwiches in the current state, categorized
       by their state and type (gluten-free or regular):
       - On trays at any child's waiting location (Cost 1: serve).
       - On trays elsewhere (not at any child's waiting location) or on trays
         at the kitchen (Cost 2: move + serve).
       - As sandwiches in the kitchen ('at_kitchen_sandwich') (Cost 3: put + move + serve).
       - As 'notexist' sandwiches (potential new sandwiches), considering
         available ingredients (Cost 4: make + put + move + serve).
    4. Calculate the maximum number of new gluten-free and regular sandwiches
       that can be made based on available 'notexist' sandwiches and ingredients
       in the kitchen.
    5. Initialize the total heuristic cost to 0.
    6. Initialize the remaining number of gluten-free and regular sandwiches
       needed to the counts of unserved allergic and non-allergic children,
       respectively.
    7. Iterate through the available sandwich resources in increasing order of
       estimated cost (Cost 1 to Cost 4).
    8. For each cost category, determine how many remaining gluten-free and
       regular needs can be fulfilled by the available resources in that category.
       Prioritize fulfilling gluten-free needs with gluten-free sandwiches.
       Allow excess gluten-free sandwiches to fulfill regular needs.
    9. Add the cost (count * cost_per_unit) to the total heuristic and update
       the remaining needs.
    10. After processing all categories, if there are still remaining needs,
        the state is likely unsolvable, return infinity.
    11. Otherwise, return the total calculated cost.
    """
    def __init__(self, task):
        super().__init__()
        self.goal_children = set()
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.child_location = {}
        self.all_places = set() # Keep track of all places mentioned

        # Parse static facts
        for fact_string in task.static:
            pred, args = parse_fact_objects(fact_string)
            if pred == 'allergic_gluten' and args:
                self.allergic_children.add(args[0])
            elif pred == 'not_allergic_gluten' and args:
                self.not_allergic_children.add(args[0])
            elif pred == 'no_gluten_bread' and args:
                self.no_gluten_breads.add(args[0])
            elif pred == 'no_gluten_content' and args:
                self.no_gluten_contents.add(args[0])
            elif pred == 'waiting' and len(args) > 1:
                self.child_location[args[0]] = args[1]
                self.all_places.add(args[1])
            elif pred == 'at' and len(args) > 1: # Trays can be at places in static init
                 self.all_places.add(args[1])

        # Add kitchen to places if not already there
        self.all_places.add('kitchen')

        # Parse goal facts
        # Assuming goal is a conjunction of (served ?c)
        if isinstance(task.goals, frozenset):
             for goal_fact_string in task.goals:
                 pred, args = parse_fact_objects(goal_fact_string)
                 if pred == 'served' and args:
                     self.goal_children.add(args[0])
        # Handle case where goal might be a single fact string
        elif isinstance(task.goals, str):
             pred, args = parse_fact_objects(task.goals)
             if pred == 'served' and args:
                 self.goal_children.add(args[0])


    def __call__(self, node):
        state = node.state

        # Parse state facts
        served_children = set()
        at_kitchen_breads_state = set()
        at_kitchen_contents_state = set()
        at_kitchen_sandwiches_state = set()
        ontray_map = {} # sandwich -> tray
        no_gluten_sandwiches_state = set()
        tray_location = {} # tray -> place
        notexist_sandwiches_state = set()

        for fact_string in state:
            pred, args = parse_fact_objects(fact_string)
            if pred == 'served' and args:
                served_children.add(args[0])
            elif pred == 'at_kitchen_bread' and args:
                at_kitchen_breads_state.add(args[0])
            elif pred == 'at_kitchen_content' and args:
                at_kitchen_contents_state.add(args[0])
            elif pred == 'at_kitchen_sandwich' and args:
                at_kitchen_sandwiches_state.add(args[0])
            elif pred == 'ontray' and len(args) > 1:
                ontray_map[args[0]] = args[1]
            elif pred == 'no_gluten_sandwich' and args:
                no_gluten_sandwiches_state.add(args[0])
            elif pred == 'at' and len(args) > 1:
                tray_location[args[0]] = args[1]
            elif pred == 'notexist' and args:
                notexist_sandwiches_state.add(args[0])

        unserved_children = self.goal_children - served_children

        if not unserved_children:
            return 0

        # Count unserved children by allergy type
        n_gluten_unserved = len([c for c in unserved_children if c in self.allergic_children])
        n_regular_unserved = len([c for c in unserved_children if c in self.not_allergic_children])

        # --- Count available suitable sandwiches by state and type ---

        # State 5: Ontray at any child's waiting location (Cost 1: serve)
        ontray_at_child_location_gluten = 0
        ontray_at_child_location_regular = 0
        child_waiting_places = set(self.child_location.values())

        for s, t in ontray_map.items():
            loc = tray_location.get(t)
            if loc in child_waiting_places:
                 is_gluten = s in no_gluten_sandwiches_state
                 if is_gluten:
                     ontray_at_child_location_gluten += 1
                 else:
                     ontray_at_child_location_regular += 1

        avail_c1_gluten = ontray_at_child_location_gluten
        avail_c1_regular = ontray_at_child_location_regular

        # State 4 & 3: Ontray elsewhere (not child location) or kitchen (Cost 2: move + serve)
        ontray_elsewhere_kitchen_gluten = 0
        ontray_elsewhere_kitchen_regular = 0

        for s, t in ontray_map.items():
             loc = tray_location.get(t)
             # Count sandwiches on trays NOT at a child's waiting location
             if loc is not None and loc not in child_waiting_places:
                 is_gluten = s in no_gluten_sandwiches_state
                 if is_gluten:
                     ontray_elsewhere_kitchen_gluten += 1
                 else:
                     ontray_elsewhere_kitchen_regular += 1

        avail_c2_gluten = ontray_elsewhere_kitchen_gluten
        avail_c2_regular = ontray_elsewhere_kitchen_regular


        # State 2: Kitchen sandwich (at_kitchen_sandwich) (Cost 3: put + move + serve)
        kitchen_sandwich_gluten = 0
        kitchen_sandwich_regular = 0
        for s in at_kitchen_sandwiches_state:
            is_gluten = s in no_gluten_sandwiches_state
            if is_gluten:
                kitchen_sandwich_gluten += 1
            else:
                kitchen_sandwich_regular += 1

        avail_c3_gluten = kitchen_sandwich_gluten
        avail_c3_regular = kitchen_sandwich_regular

        # State 1: Notexist (Cost 4: make + put + move + serve)
        notexist_sandwiches_count = len(notexist_sandwiches_state)
        avail_gluten_bread_count = len([b for b in at_kitchen_breads_state if b in self.no_gluten_breads])
        avail_gluten_content_count = len([c for c in at_kitchen_contents_state if c in self.no_gluten_contents])
        avail_regular_bread_count = len([b for b in at_kitchen_breads_state if b not in self.no_gluten_breads])
        avail_regular_content_count = len([c for c in at_kitchen_contents_state if c not in self.no_gluten_contents])

        # Max makeable considering ingredients and notexist sandwiches
        max_makeable_gluten = min(notexist_sandwiches_count, avail_gluten_bread_count, avail_gluten_content_count)

        # Remaining ingredients after making max_makeable_gluten
        rem_notexist = notexist_sandwiches_count - max_makeable_gluten
        rem_gluten_bread = avail_gluten_bread_count - max_makeable_gluten
        rem_gluten_content = avail_gluten_content_count - max_makeable_gluten
        rem_regular_bread = avail_regular_bread_count
        rem_regular_content = avail_regular_content_count

        # Regular sandwiches can use regular ingredients OR remaining gluten ingredients
        avail_bread_for_regular = rem_regular_bread + rem_gluten_bread
        avail_content_for_regular = rem_regular_content + rem_gluten_content

        max_makeable_regular = min(rem_notexist, avail_bread_for_regular, avail_content_for_regular)

        avail_c4_gluten = max_makeable_gluten
        avail_c4_regular = max_makeable_regular

        # --- Consume resources by cost category ---

        remaining_gluten_needs = n_gluten_unserved
        remaining_regular_needs = n_regular_unserved
        total_cost = 0

        # Cost 1: Ontray at child location
        served_gluten_c1 = min(remaining_gluten_needs, avail_c1_gluten)
        remaining_gluten_needs -= served_gluten_c1
        served_regular_c1 = min(remaining_regular_needs, avail_c1_regular + max(0, avail_c1_gluten - served_gluten_c1)) # Use excess gluten for regular
        remaining_regular_needs -= served_regular_c1
        total_cost += served_gluten_c1 * 1 + served_regular_c1 * 1

        # Cost 2: Ontray elsewhere or kitchen
        served_gluten_c2 = min(remaining_gluten_needs, avail_c2_gluten)
        remaining_gluten_needs -= served_gluten_c2
        served_regular_c2 = min(remaining_regular_needs, avail_c2_regular + max(0, avail_c2_gluten - served_gluten_c2)) # Use excess gluten for regular
        remaining_regular_needs -= served_regular_c2
        total_cost += served_gluten_c2 * 2 + served_regular_c2 * 2

        # Cost 3: Kitchen sandwich
        served_gluten_c3 = min(remaining_gluten_needs, avail_c3_gluten)
        remaining_gluten_needs -= served_gluten_c3
        served_regular_c3 = min(remaining_regular_needs, avail_c3_regular + max(0, avail_c3_gluten - served_gluten_c3)) # Use excess gluten for regular
        remaining_regular_needs -= served_regular_c3
        total_cost += served_gluten_c3 * 3 + served_regular_c3 * 3

        # Cost 4: New sandwich
        served_gluten_c4 = min(remaining_gluten_needs, avail_c4_gluten)
        remaining_gluten_needs -= served_gluten_c4
        served_regular_c4 = min(remaining_regular_needs, avail_c4_regular + max(0, avail_c4_gluten - served_gluten_c4)) # Use excess gluten for regular
        remaining_regular_needs -= served_regular_c4
        total_cost += served_gluten_c4 * 4 + served_regular_c4 * 4

        # If any needs remain, it's unsolvable from this state
        if remaining_gluten_needs > 0 or remaining_regular_needs > 0:
            return math.inf
        else:
            return total_cost
