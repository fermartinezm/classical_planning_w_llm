# Need to ensure Heuristic base class is available.
# Assuming it's in a file structure like:
# project_root/
# |- heuristics/
#    |- __init__.py
#    |- heuristic_base.py
#    |- childsnack_heuristic.py # This file

# The import statement should reflect this structure.
# from heuristics.heuristic_base import Heuristic

# Assuming the environment provides the Heuristic base class directly or via a simple import.
# Based on the example code provided (`from heuristics.heuristic_base import Heuristic`),
# this import seems correct for the expected environment.

# from fnmatch import fnmatch # Not strictly needed for the current get_parts logic
from heuristics.heuristic_base import Heuristic # Assuming this path is correct

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential leading/trailing whitespace and empty facts
    fact = fact.strip()
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children (each needing a 'serve' action)
    and adds an estimated cost (make, put, move) for each unserved child
    whose needs cannot be met by a suitable sandwich already on a tray at their location.

    # Assumptions
    - Each unserved child requires a 'serve' action.
    - If a suitable sandwich is not already on a tray at the child's location,
      it requires approximately 3 additional actions: make sandwich, put on tray,
      and move tray to the child's location.
    - Resource availability (bread, content, sandwich objects, trays) is not
      strictly modeled beyond checking for existing sandwiches on trays.
    - Tray movements from the kitchen to any child's location take 1 action.
    - A single suitable sandwich on a tray at a location can potentially satisfy
      the need of one unserved child at that location.

    # Heuristic Initialization
    - Extracts static information about children (allergy status, waiting place)
      from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children defined in the static facts.
    2. Determine which children are currently unserved by checking the state.
    3. If all children are served, the heuristic is 0.
    4. Initialize the heuristic value with the total number of unserved children
       (representing the minimum number of 'serve' actions required).
    5. For each unserved child, check if a suitable sandwich is already available
       on a tray located at the child's waiting place.
       - A sandwich is suitable if it's gluten-free for an allergic child,
         or any sandwich for a non-allergic child.
       - Availability requires the sandwich to be 'ontray' and the tray to be 'at'
         the child's waiting place in the current state.
    6. Count the number of unserved children for whom no such suitable sandwich
       is currently available on a tray at their location.
    7. For each of these children, add an estimated cost of 3 actions
       (make sandwich, put sandwich on tray, move tray to location) to the heuristic value.
    8. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information.
        """
        # self.goals = task.goals # Not strictly needed for this heuristic logic, but can be stored.
        static_facts = task.static

        self.child_allergy = {}  # child_name -> True if allergic, False otherwise
        self.child_place = {}    # child_name -> place_name
        self.all_children = set() # set of all child names

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            if predicate == 'allergic_gluten' and len(parts) == 2:
                child = parts[1]
                self.child_allergy[child] = True
                self.all_children.add(child)
            elif predicate == 'not_allergic_gluten' and len(parts) == 2:
                child = parts[1]
                self.child_allergy[child] = False
                self.all_children.add(child)
            elif predicate == 'waiting' and len(parts) == 3:
                child, place = parts[1], parts[2]
                self.child_place[child] = place
            # Other static facts like no_gluten_bread, no_gluten_content are not
            # directly used in this heuristic's calculation logic.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all children.
        """
        state = node.state

        # 1. Identify unserved children
        unserved_children = {
            child for child in self.all_children
            if '(served ' + child + ')' not in state
        }

        # If all children are served, goal reached, heuristic is 0.
        if not unserved_children:
            return 0

        # 4. Initialize heuristic with the number of serve actions needed
        # This is a lower bound as each unserved child needs at least one serve action.
        h = len(unserved_children)

        # Build lookup structures from the current state for efficiency
        sandwiches_on_trays = {} # sandwich -> tray
        tray_locations = {}      # tray -> place
        gf_sandwiches_in_state = set() # set of gluten-free sandwiches currently existing

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == 'ontray' and len(parts) == 3:
                sandwich, tray = parts[1], parts[2]
                sandwiches_on_trays[sandwich] = tray
            elif predicate == 'at' and len(parts) == 3:
                 # Check if the object is a tray by its name convention or type if available
                 # Assuming tray names start with 'tray' based on examples
                 obj, place = parts[1], parts[2]
                 if obj.startswith('tray'): # Simple check based on naming convention
                     tray_locations[obj] = place
            elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                 sandwich = parts[1]
                 gf_sandwiches_in_state.add(sandwich)
            # Facts like at_kitchen_bread, at_kitchen_content, at_kitchen_sandwich, notexist
            # are not directly used in this heuristic's calculation logic, as we
            # approximate the cost of making/putting/moving if a sandwich isn't
            # already ready at the location.

        # 6. Count children needing a new sandwich delivery process
        # This counts children whose needs are NOT met by a suitable sandwich
        # already present on a tray at their waiting location.
        num_not_ready_at_location = 0

        for child in unserved_children:
            child_place = self.child_place.get(child) # Get place from static info
            is_allergic = self.child_allergy.get(child, False) # Get allergy from static info

            # This case should ideally not happen in valid PDDL instances,
            # but handle defensively. If child info is missing, assume they
            # need the full delivery cycle.
            if child_place is None:
                 num_not_ready_at_location += 1
                 continue # Skip to next child

            is_ready = False
            # Check if any suitable sandwich is on a tray at the child's location
            for sandwich, tray in sandwiches_on_trays.items():
                if tray_locations.get(tray) == child_place:
                    # Tray is at the child's location
                    is_suitable = False
                    if is_allergic:
                        # Allergic child needs a gluten-free sandwich
                        if sandwich in gf_sandwiches_in_state:
                            is_suitable = True
                    else:
                        # Non-allergic child can have any sandwich
                        is_suitable = True

                    if is_suitable:
                        is_ready = True
                        break # Found a suitable sandwich on a tray at the location for this child

            if not is_ready:
                num_not_ready_at_location += 1

        # 7. Add estimated cost for children needing full delivery
        # Each such child needs make, put, move, serve. Serve is already counted in h.
        # So add 3 actions (make, put, move) per child not ready at location.
        h += num_not_ready_at_location * 3

        # 8. Return total heuristic value
        return h
