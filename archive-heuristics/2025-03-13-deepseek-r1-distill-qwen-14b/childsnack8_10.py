from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their required sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A sandwich can be either regular or no-gluten, depending on the child's allergy status.
    - Bread and content portions are available in the kitchen and can be used to make sandwiches.
    - Trays start in the kitchen and can be moved to the child's location.

    # Heuristic Initialization
    - Extract static facts about children's allergies, available no-gluten ingredients, and waiting locations.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of children who are not yet served.
    2. For each child, determine if they require a no-gluten sandwich.
    3. Check if the required sandwich type (no-gluten or regular) is already available.
    4. If not available, estimate the number of actions needed to make the required sandwich(es).
    5. Estimate the number of actions needed to move the tray(s) to the child's waiting location.
    6. Sum the actions needed for making sandwiches, moving trays, and serving children.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about the problem."""
        self.goals = task.goals
        static_facts = task.static

        # Extract information about children's allergies and waiting locations
        self.allergic_children = set()
        self.waiting_locations = {}
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.not_allergic_children = set()

        for fact in static_facts:
            parts = fact[1:-1].split()
            if fact.startswith('(allergic_gluten'):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(not_allergic_gluten'):
                self.not_allergic_children.add(parts[1])
            elif fact.startswith('(waiting'):
                child, loc = parts[1], parts[2]
                self.waiting_locations[child] = loc
            elif fact.startswith('(no_gluten_bread'):
                self.no_gluten_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content'):
                self.no_gluten_contents.add(parts[1])

    def __call__(self, node):
        """Estimate the number of actions needed to serve all children."""
        state = node.state

        # Extract current information from the state
        current_sandwiches = set()
        current_trays = set()
        current_served = set()
        current_at_kitchen_bread = set()
        current_at_kitchen_content = set()
        current_at_kitchen_sandwich = set()
        current_on_tray = set()

        for fact in state:
            if fact.startswith('(served'):
                current_served.add(fact.split()[1])
            elif fact.startswith('(at_kitchen_bread'):
                current_at_kitchen_bread.add(fact.split()[1])
            elif fact.startswith('(at_kitchen_content'):
                current_at_kitchen_content.add(fact.split()[1])
            elif fact.startswith('(at_kitchen_sandwich'):
                current_at_kitchen_sandwich.add(fact.split()[1])
            elif fact.startswith('(ontray'):
                sandwich, tray = fact.split()[1], fact.split()[2]
                current_on_tray.add((sandwich, tray))
            elif fact.startswith('(notexist'):
                pass  # Not relevant for current heuristic

        # Count number of children not yet served
        unserved_children = set(self.waiting_locations.keys()) - current_served
        num_unserved = len(unserved_children)

        if num_unserved == 0:
            return 0

        total_actions = 0

        # For each unserved child, determine if they need a no-gluten sandwich
        for child in unserved_children:
            needs_no_gluten = child in self.allergic_children

            # Check if a suitable sandwich already exists on a tray
            suitable_sandwich = None
            for (s, t), loc in self.waiting_locations.items():
                if s == child and (t, loc) in current_on_tray:
                    suitable_sandwich = s
                    break

            if suitable_sandwich is not None:
                continue  # No action needed for this child

            # Determine if a suitable sandwich can be made
            if needs_no_gluten:
                # Check if no-gluten ingredients are available
                has_bread = any(b in current_at_kitchen_bread for b in self.no_gluten_breads)
                has_content = any(c in current_at_kitchen_content for c in self.no_gluten_contents)
                if has_bread and has_content:
                    # Can make a no-gluten sandwich
                    total_actions += 1  # Make sandwich
                else:
                    # Need to acquire no-gluten ingredients (not modeled in this heuristic)
                    total_actions += 2  # Estimate for acquiring ingredients + making
            else:
                # Regular sandwich can be made if any bread and content are available
                if current_at_kitchen_bread and current_at_kitchen_content:
                    total_actions += 1  # Make sandwich
                else:
                    # Need to acquire regular ingredients (not modeled)
                    total_actions += 2  # Estimate for acquiring ingredients + making

            # Estimate tray movement and serving
            tray_needed = True
            for t in ['tray1', 'tray2', 'tray3']:
                if f'(at {t} kitchen)' in state:
                    tray_needed = False
                    break
            if tray_needed:
                total_actions += 1  # Move tray to child's location

            total_actions += 1  # Serve the sandwich

        return total_actions
