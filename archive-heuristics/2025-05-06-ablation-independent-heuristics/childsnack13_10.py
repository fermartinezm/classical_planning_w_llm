from heuristics.heuristic_base import Heuristic
from task import Task
from collections import defaultdict

# Helper function outside the class
def parse_fact_objects(fact_string):
    """Helper to parse objects from a PDDL fact string."""
    # Example: '(at tray1 kitchen)' -> ['tray1', 'kitchen']
    # Example: '(served child1)' -> ['child1']
    parts = fact_string[1:-1].split()
    return parts[1:] # Skip the predicate name

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    Estimates the number of actions required to serve all unserved children.
    It does this by greedily assigning unserved children to the cheapest
    available method of service (sandwich already at location, sandwich on
    tray at kitchen, sandwich loose in kitchen, or sandwich needing to be made),
    while tracking and consuming available resources (sandwiches, ingredients,
    notexist objects, trays). Children requiring gluten-free sandwiches are
    prioritized. The heuristic value is the sum of costs for the assigned
    stages. If any child cannot be served with available resources, the
    heuristic returns infinity.

    Assumptions:
    - The domain actions have costs implicitly defined by the stages:
        - Stage 1 (Serve at location): 1 action
        - Stage 2 (Move from kitchen + Serve): 2 actions
        - Stage 3 (Put on tray + Move + Serve): 3 actions
        - Stage 4 (Make + Put + Move + Serve): 4 actions
    - A child needs one sandwich.
    - Making a sandwich consumes one bread, one content, and one notexist object.
    - Putting a sandwich on a tray requires a tray.
    - Moving a tray requires a tray.
    - Serving a sandwich on a tray requires the tray to be at the child's location.
    - Resource counts (ingredients, notexist, trays, sandwiches) are the primary bottleneck.
    - The heuristic simulates resource consumption for each child served in the heuristic calculation.
    - Prioritizing gluten-free needs and cheaper stages is a good greedy strategy.
    - Tray availability for stages 2, 3, and 4 is simplified to a total count of available trays.

    Heuristic Initialization:
    - Parses static facts from the task to identify allergic/non-allergic children
      and the static gluten-free status of bread and content portions.

    Step-By-Step Thinking for Computing Heuristic:
    1. Check if the goal state is reached (all children served). If yes, return 0.
    2. Parse the current state to count initial available resources:
       - Bread (GF, Non-GF) in kitchen.
       - Content (GF, Non-GF) in kitchen.
       - Notexist sandwich objects.
       - Sandwiches (GF, Non-GF) loose in kitchen.
       - Sandwiches (GF, Non-GF) on trays at kitchen.
       - Sandwiches (GF, Non-GF) on trays at each child's waiting location.
       - Total trays.
    3. Initialize mutable counts of available resources based on the initial counts.
    4. Identify unserved children from the goal facts and their required sandwich type (GF or Any) and waiting location from the current state.
    5. Sort unserved children, prioritizing those needing gluten-free sandwiches.
    6. Initialize heuristic value `h = 0` and a list of children still needing service in the heuristic calculation.
    7. Iterate through the children needing service, attempting to assign them to the cheapest possible stage based on *current* available resources (which are depleted as children are assigned):
       a. **Stage 1 (Cost 1):** Check for a suitable sandwich on a tray at the child's location (prioritizing non-GF for 'any' need). If available, assign it, add 1 to `h`, mark the child as served in the list, and consume the sandwich resource.
       b. **Stage 2 (Cost 2):** If Stage 1 failed, check for a suitable sandwich on a tray at the kitchen (prioritizing non-GF for 'any' need). If available and a tray is available for the implied move, assign it, add 2 to `h`, mark the child as served, consume the sandwich resource, and consume one tray capacity.
       c. **Stage 3 (Cost 3):** If Stage 2 failed, check for a suitable sandwich loose in the kitchen (prioritizing non-GF for 'any' need). If available and a tray is available for the implied put+move, assign it, add 3 to `h`, mark the child as served, consume the sandwich resource, and consume one tray capacity.
       d. **Stage 4 (Cost 4):** If Stage 3 failed, check if ingredients (bread, content), a notexist object, and a tray are available to make a suitable sandwich (prioritizing non-GF ingredients for 'any' need). If available, assign it, add 4 to `h`, mark the child as served, consume the required ingredients, notexist object, and one tray capacity.
    8. After iterating through all children and stages, check if any children remain unserved in the list. If yes, return `float('inf)`.
    9. If all children are accounted for, return the total heuristic value `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.goals = task.goals # Goal facts, e.g., {'(served child1)', '(served child2)'}

        # Parse static facts once
        self.allergic_children = {obj for fact in task.static if fact.startswith('(allergic_gluten ') for obj in parse_fact_objects(fact)}
        self.non_allergic_children = {obj for fact in task.static if fact.startswith('(not_allergic_gluten ') for obj in parse_fact_objects(fact)}
        self.no_gluten_bread_static = {obj for fact in task.static if fact.startswith('(no_gluten_bread ') for obj in parse_fact_objects(fact)}
        self.no_gluten_content_static = {obj for fact in task.static if fact.startswith('(no_gluten_content ') for obj in parse_fact_objects(fact)}

    def __call__(self, node):
        state = node.state

        # 1. Check if goal is reached
        if self.task.goal_reached(state):
             return 0

        # Parse state facts
        waiting_children_state = {parse_fact_objects(fact)[0]: parse_fact_objects(fact)[1] for fact in state if fact.startswith('(waiting ')} # Child -> Place
        at_kitchen_bread_state = {obj for fact in state if fact.startswith('(at_kitchen_bread ') for obj in parse_fact_objects(fact)}
        at_kitchen_content_state = {obj for fact in state if fact.startswith('(at_kitchen_content ') for obj in parse_fact_objects(fact)}
        at_kitchen_sandwich_state = {obj for fact in state if fact.startswith('(at_kitchen_sandwich ') for obj in parse_fact_objects(fact)}
        ontray_state = {(parse_fact_objects(fact)[0], parse_fact_objects(fact)[1]) for fact in state if fact.startswith('(ontray ')} # (Sandwich, Tray)
        no_gluten_sandwich_state = {obj for fact in state if fact.startswith('(no_gluten_sandwich ') for obj in parse_fact_objects(fact)}
        at_state = {parse_fact_objects(fact)[0]: parse_fact_objects(fact)[1] for fact in state if fact.startswith('(at ')} # Tray -> Place
        notexist_state = {obj for fact in state if fact.startswith('(notexist ') for obj in parse_fact_objects(fact)}

        # 2. Count initial available resources
        avail_gf_bread = len([b for b in at_kitchen_bread_state if b in self.no_gluten_bread_static])
        avail_any_bread = len(at_kitchen_bread_state)
        avail_nongf_bread = avail_any_bread - avail_gf_bread # Assuming any bread is either GF or non-GF
        avail_gf_content = len([c for c in at_kitchen_content_state if c in self.no_gluten_content_static])
        avail_any_content = len(at_kitchen_content_state)
        avail_nongf_content = avail_any_content - avail_gf_content # Assuming any content is either GF or non-GF
        avail_notexist = len(notexist_state)
        total_trays = len(at_state)

        # Available sandwiches by type and location/state (use lists for mutable consumption)
        current_gf_ontray_at_place = defaultdict(list)
        current_nongf_ontray_at_place = defaultdict(list)
        current_gf_ontray_at_kitchen = []
        current_nongf_ontray_at_kitchen = []
        current_gf_kitchen_loose = []
        current_nongf_kitchen_loose = []

        for s, t in ontray_state:
            place = at_state.get(t)
            is_gf = s in no_gluten_sandwich_state
            if place == 'kitchen':
                if is_gf: current_gf_ontray_at_kitchen.append((s, t))
                else: current_nongf_ontray_at_kitchen.append((s, t))
            elif place is not None:
                if is_gf: current_gf_ontray_at_place[place].append((s, t))
                else: current_nongf_ontray_at_place[place].append((s, t))

        for s in at_kitchen_sandwich_state:
            is_gf = s in no_gluten_sandwich_state
            if is_gf: current_gf_kitchen_loose.append(s)
            else: current_nongf_kitchen_loose.append(s)

        # 3. Initialize mutable counts of available resources
        current_gf_bread = avail_gf_bread
        current_nongf_bread = avail_nongf_bread
        current_gf_content = avail_gf_content
        current_nongf_content = avail_nongf_content
        current_notexist = avail_notexist
        current_trays_available_for_put_move = total_trays # Simplified tray availability

        # 4. Identify unserved children and their needs/location
        unserved_children_info = [] # list of (child, place, req_type)
        # Iterate through children mentioned in the goal
        goal_children = {parse_fact_objects(g)[0] for g in self.goals if g.startswith('(served ')}

        for child in goal_children:
            if f'(served {child})' not in state: # Check if served in current state
                 # Find their waiting location in the current state
                 place = waiting_children_state.get(child)
                 if place is not None: # Child must be waiting somewhere
                     req_type = 'gf' if child in self.allergic_children else 'any'
                     unserved_children_info.append((child, place, req_type))
                 # else: Child is not waiting? This shouldn't happen based on domain structure.

        # 5. Sort unserved children (GF first)
        unserved_children_info.sort(key=lambda x: x[2], reverse=True) # 'gf' comes after 'any'

        # 6. Initialize heuristic and tracking
        h = 0
        children_to_serve = list(unserved_children_info) # Use a mutable list, mark served by setting to None

        # 7. Greedily assign children to stages

        # Stage 1 (Cost 1: Serve at location)
        for i in range(len(children_to_serve)):
            c, p, req_type = children_to_serve[i]
            if c is None: continue # Already served in heuristic

            served_now = False
            if req_type == 'gf':
                if current_gf_ontray_at_place.get(p, []):
                    current_gf_ontray_at_place[p].pop(0) # Consume one GF
                    h += 1
                    children_to_serve[i] = (None, None, None)
                    served_now = True
            else: # req_type == 'any'
                # Prioritize non-GF if available
                if current_nongf_ontray_at_place.get(p, []):
                    current_nongf_ontray_at_place[p].pop(0) # Consume one non-GF
                    h += 1
                    children_to_serve[i] = (None, None, None)
                    served_now = True
                # If no non-GF, use GF if available
                elif current_gf_ontray_at_place.get(p, []):
                    current_gf_ontray_at_place[p].pop(0) # Consume one GF
                    h += 1
                    children_to_serve[i] = (None, None, None)
                    served_now = True

        # Stage 2 (Cost 2: Move from kitchen + Serve)
        for i in range(len(children_to_serve)):
            c, p, req_type = children_to_serve[i]
            if c is None: continue # Already served in heuristic

            served_now = False
            if current_trays_available_for_put_move > 0:
                if req_type == 'gf':
                    if current_gf_ontray_at_kitchen:
                        current_gf_ontray_at_kitchen.pop(0) # Consume one GF
                        current_trays_available_for_put_move -= 1
                        h += 2
                        children_to_serve[i] = (None, None, None)
                        served_now = True
                else: # req_type == 'any'
                    # Prioritize non-GF
                    if current_nongf_ontray_at_kitchen:
                        current_nongf_ontray_at_kitchen.pop(0) # Consume one non-GF
                        current_trays_available_for_put_move -= 1
                        h += 2
                        children_to_serve[i] = (None, None, None)
                        served_now = True
                    # If no non-GF, use GF
                    elif current_gf_ontray_at_kitchen:
                        current_gf_ontray_at_kitchen.pop(0) # Consume one GF
                        current_trays_available_for_put_move -= 1
                        h += 2
                        children_to_serve[i] = (None, None, None)
                        served_now = True

        # Stage 3 (Cost 3: Put + Move + Serve)
        for i in range(len(children_to_serve)):
            c, p, req_type = children_to_serve[i]
            if c is None: continue # Already served in heuristic

            served_now = False
            if current_trays_available_for_put_move > 0:
                if req_type == 'gf':
                    if current_gf_kitchen_loose:
                        current_gf_kitchen_loose.pop(0) # Consume one GF
                        current_trays_available_for_put_move -= 1
                        h += 3
                        children_to_serve[i] = (None, None, None)
                        served_now = True
                else: # req_type == 'any'
                    # Prioritize non-GF
                    if current_nongf_kitchen_loose:
                        current_nongf_kitchen_loose.pop(0) # Consume one non-GF
                        current_trays_available_for_put_move -= 1
                        h += 3
                        children_to_serve[i] = (None, None, None)
                        served_now = True
                    # If no non-GF, use GF
                    elif current_gf_kitchen_loose:
                        current_gf_kitchen_loose.pop(0) # Consume one GF
                        current_trays_available_for_put_move -= 1
                        h += 3
                        children_to_serve[i] = (None, None, None)
                        served_now = True

        # Stage 4 (Cost 4: Make + Put + Move + Serve)
        for i in range(len(children_to_serve)):
            c, p, req_type = children_to_serve[i]
            if c is None: continue # Already served in heuristic

            served_now = False
            # Need notexist and a tray
            if current_notexist > 0 and current_trays_available_for_put_move > 0:
                can_make = False

                if req_type == 'gf':
                    if current_gf_bread > 0 and current_gf_content > 0:
                        current_gf_bread -= 1
                        current_gf_content -= 1
                        can_make = True
                else: # req_type == 'any'
                    # Try non-GF ingredients first
                    if current_nongf_bread > 0 and current_nongf_content > 0:
                         current_nongf_bread -= 1
                         current_nongf_content -= 1
                         can_make = True
                    # If non-GF not available, try GF ingredients
                    elif current_gf_bread > 0 and current_gf_content > 0:
                         current_gf_bread -= 1
                         current_gf_content -= 1
                         can_make = True

                if can_make:
                     # Ingredients were available, now consume notexist and tray
                     current_notexist -= 1
                     current_trays_available_for_put_move -= 1
                     h += 4
                     children_to_serve[i] = (None, None, None)
                     served_now = True

        # 8. Check if any children are still unserved
        for c, p, req_type in children_to_serve:
            if c is not None:
                # This child could not be served with available resources in this greedy assignment.
                return float('inf')

        # 9. Return total heuristic value
        return h
