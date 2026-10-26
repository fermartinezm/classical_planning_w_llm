from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

def match(fact, pattern):
    """Check if a fact matches a pattern using wildcards."""
    parts = get_parts(fact)
    pattern_parts = pattern.split()
    if len(parts) != len(pattern_parts):
        return False
    return all(fnmatch(part, pat) for part, pat in zip(parts, pattern_parts))

class childsnack1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    Estimates the number of actions needed to serve all children by considering:
    - Making required sandwiches (gluten-free if needed)
    - Placing sandwiches on trays in the kitchen
    - Moving trays to children's locations
    - Serving each child

    # Assumptions
    - All required bread and content portions are available in the kitchen.
    - Each sandwich serves one child and is removed from the tray upon serving.
    - Trays can be moved between locations as needed.

    # Heuristic Initialization
    - Extracts static information about allergic children and their waiting locations.
    - Identifies all children in the problem.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify served and unserved children in the current state.
    2. For each unserved child:
        a. Check if a suitable sandwich is already on a tray at their location.
        b. If yes, count one action (serve).
        c. If no, count actions for making, placing, moving, and serving (4 actions).
    3. If no trays are in the kitchen and sandwiches are needed, add one action to move a tray to the kitchen.
    4. Sum all actions for the heuristic value.
    """

    def __init__(self, task):
        """Extract static information about children and their locations."""
        self.all_children = set()
        self.allergic_children = set()
        self.child_location = {}

        # Process static facts to determine allergies and waiting locations
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                self.allergic_children.add(child)
                self.all_children.add(child)
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                self.all_children.add(child)
            elif parts[0] == 'waiting':
                child = parts[1]
                location = parts[2]
                self.child_location[child] = location

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        served = set()
        required_sandwiches = 0
        served_available = 0

        # Determine served children
        for fact in state:
            if match(fact, 'served *'):
                child = get_parts(fact)[1]
                served.add(child)

        # Process each unserved child
        for child in self.all_children:
            if child in served:
                continue
            location = self.child_location[child]
            allergic = child in self.allergic_children

            # Check trays at the child's location
            trays_at_location = set()
            for fact in state:
                if match(fact, f'at * {location}'):
                    tray = get_parts(fact)[1]
                    trays_at_location.add(tray)

            # Check for suitable sandwiches on these trays
            suitable = False
            for fact in state:
                if match(fact, 'ontray * *'):
                    parts = get_parts(fact)
                    sandwich, tray = parts[1], parts[2]
                    if tray in trays_at_location:
                        if allergic:
                            if f'(no_gluten_sandwich {sandwich})' in state:
                                suitable = True
                                break
                        else:
                            suitable = True
                            break
            if suitable:
                served_available += 1
            else:
                required_sandwiches += 1

        # Check if trays are present in the kitchen
        trays_in_kitchen = sum(1 for fact in state if match(fact, 'at * kitchen'))
        initial_move = 1 if (required_sandwiches > 0 and trays_in_kitchen == 0) else 0

        # Calculate total heuristic estimate
        return (required_sandwiches * 4) + served_available + initial_move
