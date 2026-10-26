from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their appropriate sandwiches.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - A sandwich can be served only if it is on a tray at the child's waiting place.
    - A sandwich must be made before it can be put on a tray.
    - If a child is allergic to gluten, they must receive a no-gluten sandwich.

    # Heuristic Initialization
    - Extract static facts about children's allergies, available no-gluten portions, and tray positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each child, check if they are already served. If yes, no actions are needed.
    2. For each unserved child:
       a. Determine if they are waiting at a place and need a sandwich.
       b. Check if the appropriate sandwich type (gluten-free or regular) is available on a tray at their place.
       c. If the sandwich is already on the tray, only one action (serving) is needed.
       d. If the sandwich is not on the tray but is already made, add actions to put it on the tray and serve.
       e. If the sandwich is not made yet, calculate the actions to make it, put it on the tray, and serve.
    3. Sum the actions needed for all children to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information into data structures
        self.allergic_children = set()
        self.no_gluten_breads = set()
        self.no_gluten_contents = set()
        self.tray_initial_positions = dict()

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_bread", "*"):
                self.no_gluten_breads.add(get_parts(fact)[1])
            elif match(fact, "no_gluten_content", "*"):
                self.no_gluten_contents.add(get_parts(fact)[1])
            elif match(fact, "at", "tray*", "*"):
                tray, pos = get_parts(fact)[1], get_parts(fact)[2]
                self.tray_initial_positions[tray] = pos

    def __call__(self, node):
        """Estimate the minimum number of actions to serve all children."""
        state = node.state

        def get_parts(fact):
            """Extract components of a PDDL fact."""
            return fact[1:-1].split()

        def match(fact, *args):
            """Check if a fact matches a given pattern."""
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        state_facts = set(state)
        total_actions = 0

        # Find all children that need to be served
        children_to_serve = []
        for fact in state_facts:
            if match(fact, "waiting", "*", "*"):
                child = get_parts(fact)[1]
                children_to_serve.append(child)

        # For each child, determine the required actions
        for child in children_to_serve:
            # Check if already served
            if any(match(f, "served", child) for f in state_facts):
                continue

            # Determine if the child is allergic
            if child in self.allergic_children:
                required_sandwich = "no_gluten_sandwich"
            else:
                required_sandwich = "at_kitchen_sandwich"

            # Check if the required sandwich exists and is on a tray at the child's place
            child_place = None
            for fact in state_facts:
                if match(fact, "waiting", child, "*"):
                    child_place = get_parts(fact)[2]
                    break

            # Find all sandwiches that are no_gluten if needed
            available_sandwiches = []
            for fact in state_facts:
                if match(fact, "no_gluten_sandwich", "*"):
                    available_sandwiches.append(fact)
                elif match(fact, "at_kitchen_sandwich", "*"):
                    available_sandwiches.append(fact)

            # Check if a suitable sandwich is already on a tray at the child's place
            sandwich_on_tray = None
            tray_at_child_place = None
            for fact in state_facts:
                if match(fact, "ontray", "*", "*"):
                    s, t = get_parts(fact)[1], get_parts(fact)[2]
                    if match(f"at {t} {child_place}", "*"):
                        tray_at_child_place = t
                        sandwich_on_tray = s
                        break

            # If no suitable sandwich is found, check if one can be made
            if not sandwich_on_tray:
                # Check if all components for a suitable sandwich are available
                bread_available = any(match(f, "at_kitchen_bread", "*") for f in state_facts)
                content_available = any(match(f, "at_kitchen_content", "*") for f in state_facts)
                if child in self.allergic_children:
                    bread_available = any(match(f, "no_gluten_bread", "*") for f in state_facts)
                    content_available = any(match(f, "no_gluten_content", "*") for f in state_facts)
                if bread_available and content_available:
                    # Actions needed: make sandwich (1), put on tray (1), move tray if necessary (variable)
                    total_actions += 2  # make and put on tray
                    # Find a tray not already at the child's place
                    available_trays = [t for t in self.tray_initial_positions.keys() if t != tray_at_child_place]
                    if available_trays:
                        tray = available_trays[0]
                        total_actions += 2  # move tray to child's place and serve
                    else:
                        total_actions += 1  # assume tray is already moved
                else:
                    # Need to get bread and content, but this is beyond basic heuristic estimation
                    total_actions += 4  # rough estimate for making and moving
            else:
                # Sandwich is already on tray, just serve
                total_actions += 1

        return total_actions
