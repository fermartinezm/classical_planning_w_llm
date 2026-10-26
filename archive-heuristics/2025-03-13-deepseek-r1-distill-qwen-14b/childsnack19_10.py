from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - Sandwiches can be made with or without gluten based on the child's allergy.
    - A sandwich must be on a tray before it can be served.
    - Trays start in the kitchen and must be moved to the child's location.

    # Heuristic Initialization
    - Extract static facts about gluten-free ingredients and children's allergies.
    - Map each child to their target tray based on the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served.
    2. If not served, determine if their sandwich is ready and on the correct tray.
    3. If the sandwich isn't made yet:
       a. Estimate actions to make the sandwich (considering gluten constraints).
       b. Estimate actions to move the sandwich to the tray.
    4. If the tray isn't at the child's location:
       a. Estimate actions to move the tray.
    5. Sum the estimated actions for all children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic with static facts and goal information."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()
        self.allergic_children = set()
        self.tray_initial_positions = set()
        self.child_to_tray = dict()

        # Parse static facts
        for fact in static_facts:
            parts = fact[1:-1].split()
            if fact.startswith('(no_gluten_bread '):
                self.gluten_free_breads.add(parts[1])
            elif fact.startswith('(no_gluten_content '):
                self.gluten_free_contents.add(parts[1])
            elif fact.startswith('(allergic_gluten '):
                self.allergic_children.add(parts[1])
            elif fact.startswith('(at tray'):
                self.tray_initial_positions.add(parts[1])
        
        # Map each child to their target tray based on the goal
        for goal in self.goals:
            if goal.startswith('(served '):
                child = goal.split()[1]
                # Find the tray associated with this child in the initial state
                for fact in task.initial_state:
                    if fact.startswith('(waiting ' + child + ' '):
                        place = fact.split()[2]
                        # Find the tray at this place
                        for tray_fact in task.initial_state:
                            if tray_fact.startswith('(at tray') and ' ' + place + ')' in tray_fact:
                                tray = tray_fact.split()[1]
                                self.child_to_tray[child] = tray
                                break

    def __call__(self, node):
        """Compute the estimated number of actions to serve all children."""
        state = node.state
        total_cost = 0

        # Track current state information
        current_sandwiches = {}
        current_trays = {}
        served_children = set()
        tray_positions = {}

        for fact in state:
            if fact.startswith('(served '):
                served_children.add(fact.split()[1])
            elif fact.startswith('(ontray '):
                sandwich = fact.split()[1]
                tray = fact.split()[2]
                current_sandwiches[sandwich] = 'on_tray'
                current_trays[tray] = 'occupied'
            elif fact.startswith('(at tray'):
                tray = fact.split()[1]
                position = fact.split()[2]
                tray_positions[tray] = position

        # For each child, check if they need a sandwich and calculate required actions
        for goal in self.goals:
            if goal.startswith('(served '):
                child = goal.split()[1]
                if child in served_children:
                    continue  # Already served

                # Determine if the child is allergic to gluten
                is_allergic = child in self.allergic_children

                # Check if the sandwich is already made and on the tray
                found_sandwich = None
                for sandwich_fact in state:
                    if sandwich_fact.startswith('(at_kitchen_sandwich '):
                        s = sandwich_fact.split()[1]
                        if s not in current_sandwiches:
                            found_sandwich = s
                            break
                
                if found_sandwich is None:
                    # Need to make a new sandwich
                    total_cost += 1  # Make sandwich action

                    # Check if we need gluten-free ingredients
                    if is_allergic:
                        # Find available gluten-free bread and content
                        bread = None
                        content = None
                        for fact in state:
                            if fact.startswith('(at_kitchen_bread ') and fact.split()[1] in self.gluten_free_breads:
                                bread = fact.split()[1]
                                break
                        for fact in state:
                            if fact.startswith('(at_kitchen_content ') and fact.split()[1] in self.gluten_free_contents:
                                content = fact.split()[1]
                                break
                        # No available gluten-free ingredients would make this state unsolvable, but we assume solvable instances
                    else:
                        # Find any available bread and content
                        bread = next(fact.split()[1] for fact in state if fact.startswith('(at_kitchen_bread '))
                        content = next(fact.split()[1] for fact in state if fact.startswith('(at_kitchen_content '))
                    
                    # No need to track resources as we assume availability

                # Move sandwich to tray if not already on tray
                target_tray = self.child_to_tray[child]
                if target_tray not in tray_positions or tray_positions[target_tray] != 'kitchen':
                    # Move tray to kitchen if not already there
                    total_cost += 1  # move_tray action
                    tray_positions[target_tray] = 'kitchen'
                
                # Put sandwich on tray
                total_cost += 1  # put_on_tray action

                # Serve the sandwich
                total_cost += 1  # serve_sandwich action

        return total_cost
