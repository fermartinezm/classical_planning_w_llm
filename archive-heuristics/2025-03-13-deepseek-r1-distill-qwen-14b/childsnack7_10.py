from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A sandwich can be either regular or no-gluten, depending on the child's allergy status.
    - Each sandwich requires one bread and one content portion.
    - Trays can hold multiple sandwiches but must be moved to the child's location before serving.

    # Heuristic Initialization
    - Extract static facts about children's allergies, available bread and content portions, and tray positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If served, no actions are needed.
    2. For unserved children, determine if they are allergic to gluten to decide the required sandwich type.
    3. Check if the required sandwich type is already made and on the correct tray at the child's location.
    4. If the sandwich is not yet made, estimate the actions needed:
       a. Make the sandwich using available bread and content.
       b. Put the sandwich on a tray.
       c. Move the tray to the child's location if it's not already there.
    5. Sum the estimated actions for all children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts about the problem."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.allergic_children = set()
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.tray_positions = dict()

        for fact in static_facts:
            if fact.startswith('(allergic_gluten '):
                child = fact.split()[1]
                self.allergic_children.add(child)
            elif fact.startswith('(no_gluten_bread '):
                bread = fact.split()[1]
                self.no_gluten_bread.add(bread)
            elif fact.startswith('(no_gluten_content '):
                content = fact.split()[1]
                self.no_gluten_content.add(content)
            elif fact.startswith('(at tray'):
                tray = fact.split()[1]
                pos = ' '.join(fact.split()[3:-1])
                self.tray_positions[tray] = pos

    def __call__(self, node):
        """Compute the estimated number of actions to serve all children."""
        state = node.state
        current_facts = set(state)

        heuristic_cost = 0

        # Track used resources
        used_breads = set()
        used_contents = set()
        used_trays = set()

        # Process each child
        for goal in self.goals:
            if not goal.startswith('(served '):
                continue
            child = goal.split()[1]

            # Skip if child is already served
            if f'(served {child})' in current_facts:
                continue

            # Determine if child is allergic
            if child in self.allergic_children:
                required_sandwich = 'no_gluten'
            else:
                required_sandwich = 'regular'

            # Check if the required sandwich is already made and on the tray
            sandwich_made = False
            sandwich_on_tray = False
            tray_at_location = False

            # Search for existing sandwich
            for fact in current_facts:
                if fact.startswith('(at_kitchen_sandwich '):
                    s = fact.split()[1]
                    if required_sandwich == 'no_gluten' and not (f'(no_gluten_sandwich {s})' in current_facts):
                        continue
                    if required_sandwich == 'regular' and f'(no_gluten_sandwich {s})' in current_facts:
                        continue
                    sandwich_made = True
                    sandwich = s
                    break

            if sandwich_made:
                # Check if sandwich is on a tray
                for fact in current_facts:
                    if fact.startswith('(ontray '):
                        s, t = fact.split()[1], fact.split()[2]
                        if s == sandwich:
                            sandwich_on_tray = True
                            tray = t
                            break

            if sandwich_on_tray:
                # Check if tray is already at the child's location
                tray_pos = self.tray_positions.get(tray, None)
                child_pos = None
                for fact in current_facts:
                    if fact.startswith('(waiting ' + child + ' '):
                        child_pos = ' '.join(fact.split()[2:-1])
                        break
                if tray_pos == child_pos:
                    tray_at_location = True

            if not sandwich_made:
                # Need to make the sandwich
                heuristic_cost += 1  # make_sandwich or make_sandwich_no_gluten
                # Deduct used resources
                if required_sandwich == 'no_gluten':
                    used_breads.add('bread' + str(len(used_breads) + 1))
                    used_contents.add('content' + str(len(used_contents) + 1))
                else:
                    # Use any available bread and content
                    pass  # Resources are assumed to be sufficient

            if not sandwich_on_tray:
                # Need to put sandwich on tray
                heuristic_cost += 1  # put_on_tray

            if not tray_at_location:
                # Need to move tray to child's location
                heuristic_cost += 2  # move_tray (from kitchen to child's place)

        return heuristic_cost
