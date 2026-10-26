from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches, considering the need for gluten-free options and the current state of the kitchen.

    # Assumptions:
    - Each child needs exactly one sandwich.
    - A sandwich requires one bread and one content portion.
    - Some children require gluten-free sandwiches, which must be made with specific ingredients.
    - Sandwiches can be placed on trays and moved between places.

    # Heuristic Initialization
    - Extract static facts about children's allergies, available gluten-free ingredients, and tray positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Count the number of children who are not yet served.
    2. For each unserved child:
       a. Check if a sandwich for the child already exists.
       b. If not, determine if the necessary ingredients are available.
       c. Calculate the actions needed to make the sandwich, place it on a tray, and serve it.
    3. Sum the actions for all unserved children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and goal conditions."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information into useful data structures
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        self.tray_positions = set()

        for fact in static_facts:
            if fact.startswith('(allergic_gluten '):
                child = fact.split()[1]
                self.allergic_children.add(child)
            elif fact.startswith('(not_allergic_gluten '):
                child = fact.split()[1]
                self.not_allergic_children.add(child)
            elif fact.startswith('(no_gluten_bread '):
                bread = fact.split()[1]
                self.gluten_free_breads.add(bread)
            elif fact.startswith('(no_gluten_content '):
                content = fact.split()[1]
                self.gluten_free_contents.add(content)
            elif fact.startswith('(at tray'):
                tray = fact.split()[1]
                pos = ' '.join(fact.split()[2:-1])
                self.tray_positions.add((tray, pos))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        goal_served = set()
        total_actions = 0

        # Extract current state information
        current_sandwiches = set()
        current_trays = set()
        current_served = set()
        bread_available = set()
        content_available = set()
        tray_locations = {}

        for fact in state:
            if fact.startswith('(served '):
                child = fact.split()[1]
                current_served.add(child)
            elif fact.startswith('(at_kitchen_sandwich '):
                sandwich = fact.split()[1]
                current_sandwiches.add(sandwich)
            elif fact.startswith('(ontray '):
                sandwich, tray = fact.split()[1], fact.split()[2]
                tray_locations[tray] = 'kitchen'  # since ontray implies tray is in kitchen unless moved
            elif fact.startswith('(at_kitchen_bread '):
                bread = fact.split()[1]
                bread_available.add(bread)
            elif fact.startswith('(at_kitchen_content '):
                content = fact.split()[1]
                content_available.add(content)
            elif fact.startswith('(notexist '):
                sandwich = fact.split()[1]
                pass  # sandwich does not exist yet

        # Count number of children not yet served
        unserved_children = set()
        for goal in self.goals:
            if goal.startswith('(served '):
                child = goal.split()[1]
                if child not in current_served:
                    unserved_children.add(child)

        # For each unserved child, determine the required actions
        for child in unserved_children:
            # Check if sandwich already exists for this child
            sandwich_exists = any(sandwich.startswith(child) for sandwich in current_sandwiches)
            if not sandwich_exists:
                # Need to make a new sandwich
                total_actions += 1  # make_sandwich or make_sandwich_no_gluten

                # Check if bread and content are available
                if child in self.allergic_children:
                    # Must use gluten-free ingredients
                    has_bread = any(bread in self.gluten_free_breads and bread in bread_available for bread in self.gluten_free_breads)
                    has_content = any(content in self.gluten_free_contents and content in content_available for content in self.gluten_free_contents)
                    if has_bread and has_content:
                        pass  # ingredients are available
                    else:
                        # Need to get gluten-free ingredients, but assuming they are already in kitchen
                        pass  # no additional actions needed as they are assumed available
                else:
                    # Can use any bread and content
                    has_bread = len(bread_available) > 0
                    has_content = len(content_available) > 0
                    if has_bread and has_content:
                        pass  # ingredients are available
                    else:
                        # Need to get more ingredients, but assuming they are available
                        pass  # no additional actions needed

            # Check if sandwich is on a tray
            sandwich = next((s for s in current_sandwiches if s.startswith(child)), None)
            if sandwich:
                # Check if the sandwich is on a tray
                tray = None
                for fact in state:
                    if fact.startswith('(ontray ' + sandwich + ' '):
                        tray = fact.split()[2]
                        break
                if tray:
                    # Check if the tray is already at the child's location
                    for fact in state:
                        if fact.startswith('(at tray' + tray + ' '):
                            pos = ' '.join(fact.split()[2:-1])
                            if pos == 'table' + child[-1]:  # assuming child1 waits at table1, etc.
                                # Tray is already at the correct place
                                pass
                            else:
                                # Need to move the tray
                                total_actions += 2  # move_tray action
                else:
                    # Need to put the sandwich on a tray
                    total_actions += 1  # put_on_tray action
            else:
                # Sandwich needs to be made and put on tray
                total_actions += 2  # make and put_on_tray

            # Serve the sandwich
            total_actions += 1  # serve_sandwich or serve_sandwich_no_gluten

        return total_actions
