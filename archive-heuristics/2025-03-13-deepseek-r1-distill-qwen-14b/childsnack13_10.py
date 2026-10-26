from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children their sandwiches. It considers the need for gluten-free sandwiches for allergic children, the availability of ingredients, and the status of sandwiches and trays.

    # Assumptions:
    - Each child must be served exactly one sandwich.
    - Allergic children require a no-gluten sandwich.
    - A sandwich can be made if the required bread and content portions are available.
    - A sandwich must be on a tray to be served.
    - Trays start in the kitchen and must be moved to the child's location if needed.

    # Heuristic Initialization
    - Extract static facts about children's allergies, available gluten-free ingredients, and initial tray positions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of children who have not been served.
    2. For each unserved child:
       a. Determine if they are allergic to gluten.
       b. Check if a no-gluten sandwich is available if they are allergic.
       c. Check if a regular sandwich is available if they are not allergic.
    3. For each required sandwich that is not yet made:
       a. Check if the necessary bread and content portions are available.
       b. If not, estimate the actions needed to obtain them.
    4. For each sandwich that is made but not on a tray:
       a. Estimate the actions needed to put it on a tray.
    5. For each tray that is not in the kitchen:
       a. Estimate the actions needed to move it back to the kitchen.
    6. Sum all the estimated actions to get the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts about the problem."""
        self.goals = task.goals
        static_facts = task.static

        # Extract static information into data structures
        self.allergic_children = {}
        self.no_gluten_bread = set()
        self.no_gluten_content = set()
        self.tray_positions = {}

        for fact in static_facts:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergic_children[child] = False
            elif match(fact, "no_gluten_bread", "*"):
                bread = get_parts(fact)[1]
                self.no_gluten_bread.add(bread)
            elif match(fact, "no_gluten_content", "*"):
                content = get_parts(fact)[1]
                self.no_gluten_content.add(content)
            elif match(fact, "at", "tray*", "*"):
                parts = get_parts(fact)
                tray = parts[1]
                position = parts[2]
                self.tray_positions[tray] = position

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        def get_parts(fact):
            """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
            return fact[1:-1].split()

        def match(fact, *args):
            """
            Check if a PDDL fact matches a given pattern.

            - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
            - `args`: The expected pattern (wildcards `*` allowed).
            - Returns `True` if the fact matches the pattern, else `False`.
            """
            parts = get_parts(fact)
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Extract current state information
        current_sandwiches = {}
        current_trays = {}
        current_served = set()
        current_bread = set()
        current_content = set()
        current_available_sandwiches = set()

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                sandwich = get_parts(fact)[1]
                current_sandwiches[sandwich] = "kitchen"
            elif match(fact, "ontray", "*", "*"):
                sandwich = get_parts(fact)[1]
                tray = get_parts(fact)[2]
                current_sandwiches[sandwich] = tray
            elif match(fact, "served", "*"):
                child = get_parts(fact)[1]
                current_served.add(child)
            elif match(fact, "at_kitchen_bread", "*"):
                bread = get_parts(fact)[1]
                current_bread.add(bread)
            elif match(fact, "at_kitchen_content", "*"):
                content = get_parts(fact)[1]
                current_content.add(content)
            elif match(fact, "notexist", "*"):
                sandwich = get_parts(fact)[1]
                current_available_sandwiches.add(sandwich)

        # Count the number of children who need to be served
        children = [child for child in get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")]
        unserved_children = set(children) - current_served

        if not unserved_children:
            return 0

        total_actions = 0

        # For each unserved child, determine the required sandwich and actions
        for child in unserved_children:
            is_allergic = self.allergic_children.get(child, False)
            needs_gluten_free = is_allergic

            # Check if a suitable sandwich is already available
            found_sandwich = None
            for sandwich in current_available_sandwiches:
                if needs_gluten_free and (get_parts(f"no_gluten_sandwich {sandwich}")[1] == "yes"):
                    found_sandwich = sandwich
                    break
                elif not needs_gluten_free:
                    found_sandwich = sandwich
                    break

            if not found_sandwich:
                # Need to make a new sandwich
                total_actions += 1  # make_sandwich or make_sandwich_no_gluten

                # Check if bread and content are available
                if needs_gluten_free:
                    if not (self.no_gluten_bread and self.no_gluten_content):
                        total_actions += 2  # get bread and content
                else:
                    if not (current_bread and current_content):
                        total_actions += 2  # get bread and content

            # Check if the sandwich is on a tray
            if found_sandwich not in current_sandwiches or current_sandwiches[found_sandwich] != "kitchen":
                total_actions += 1  # put_on_tray

            # Move tray if necessary
            tray = None
            for fact in state:
                if match(fact, "ontray", found_sandwich, "*"):
                    tray = get_parts(fact)[2]
                    break
            if tray and self.tray_positions.get(tray, "kitchen") != "kitchen":
                total_actions += 2  # move tray to kitchen and back

            # Serve the sandwich
            total_actions += 1  # serve_sandwich or serve_sandwich_no_gluten

        return total_actions
