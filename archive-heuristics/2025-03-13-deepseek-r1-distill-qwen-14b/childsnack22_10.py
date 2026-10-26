from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children by:
    1. Making necessary sandwiches.
    2. Moving trays to the required places.
    3. Serving the sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A sandwich can be gluten-free or regular, depending on the child's allergy.
    - If a child is allergic, their sandwich must be gluten-free.
    - If a child is not allergic, any sandwich type is acceptable.
    - Bread and content portions are available in the kitchen.
    - Trays start in the kitchen and can be moved to other places.

    # Heuristic Initialization
    - Extract static facts about children's allergies, available gluten-free ingredients, and initial waiting locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If not, proceed.
    2. Check if a suitable sandwich exists:
       - If the child is allergic, ensure the sandwich is gluten-free.
       - If the child is not allergic, any sandwich type is acceptable.
    3. If the sandwich does not exist, calculate actions to make it:
       - 1 action for making a regular sandwich.
       - 1 additional action if making a gluten-free sandwich (due to different preconditions).
    4. Check the tray's location:
       - If the tray is not at the child's waiting place, calculate actions to move it there.
    5. Calculate the serving action:
       - 1 action to serve the sandwich.
    6. Sum the actions for all unserved children, ensuring not to double-count tray movements when multiple children are at the same place.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information from the task."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information into dictionaries
        self.allergic_children = set()
        self.waiting_locations = {}
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        self.tray_initial_location = None

        for fact in static_facts:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten '):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(waiting '):
                child = parts[1]
                place = parts[2]
                self.waiting_locations[child] = place
            elif fact.startswith('(no_gluten_bread '):
                self.gluten_free_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content '):
                self.gluten_free_contents.add(parts[1])
            elif fact.startswith('(at tray'):
                self.tray_initial_location = parts[1]

    def __call__(self, node):
        """Compute the estimated number of actions to reach the goal state."""
        state = node.state
        total_actions = 0

        # Track which children are already served
        served_children = set()
        for fact in state:
            if fact.startswith('(served '):
                served_children.add(fact.split()[1])

        # For each child, determine if they need a sandwich and calculate actions
        for child in self.waiting_locations:
            if child in served_children:
                continue  # Skip already served children

            # Determine if the child is allergic
            if child in self.allergic_children:
                needs_gluten_free = True
            else:
                needs_gluten_free = False

            # Check if a suitable sandwich exists
            has_suitable_sandwich = False
            for fact in state:
                if fact.startswith('(no_gluten_sandwich '):
                    sandw = fact.split()[1]
                    if (fact in state and
                        (needs_gluten_free or not needs_gluten_free) and
                        (not needs_gluten_free or needs_gluten_free)):
                        has_suitable_sandwich = True
                        break
                elif fact.startswith('(at_kitchen_sandwich '):
                    sandw = fact.split()[1]
                    if (not needs_gluten_free and
                        '(no_gluten_sandwich ' + sandw + ')' not in state):
                        has_suitable_sandwich = True
                        break

            if not has_suitable_sandwich:
                # Need to make a suitable sandwich
                if needs_gluten_free:
                    # Make a gluten-free sandwich
                    total_actions += 1  # make_sandwich_no_gluten
                else:
                    # Make a regular sandwich
                    total_actions += 1  # make_sandwich

            # Check if the tray is at the child's waiting place
            tray_at_location = False
            for fact in state:
                if fact.startswith('(ontray '):
                    sandw = fact.split()[1]
                    tray = fact.split()[2]
                    for other_fact in state:
                        if other_fact.startswith('(at ' + tray + ' '):
                            if other_fact.split()[2] == self.waiting_locations[child]:
                                tray_at_location = True
                                break
                    if tray_at_location:
                        break

            if not tray_at_location:
                # Move the tray to the child's location
                # Assume moving a tray takes 1 action (move_tray)
                total_actions += 1

            # Serve the sandwich
            total_actions += 1  # serve_sandwich or serve_sandwich_no_gluten

        return total_actions
