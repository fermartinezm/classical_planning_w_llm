from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    This heuristic estimates the number of actions required to serve all children
    by summing up the estimated steps needed for each unserved child independently.

    The estimated steps for an unserved child are based on the current state
    of a suitable sandwich for them:
    - 1 action (serve) if a suitable sandwich is already on a tray at their location.
    - 2 actions (move tray, serve) if a suitable sandwich is on a tray elsewhere.
    - 3 actions (put on tray, move tray, serve) if a suitable sandwich is in the kitchen.
    - 4 actions (make sandwich, put on tray, move tray, serve) if no suitable
      sandwich exists yet.

    This heuristic is non-admissible as it sums costs independently per child
    and simplifies resource/tray sharing.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting child information (allergy, waiting place)
        from static facts and identifying goal children.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract child allergy status and waiting places from static facts
        # {child_name: {'allergic': bool, 'waiting_place': place}}
        self.child_info = {}

        # First pass: Extract allergy and waiting info from static facts
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'allergic_gluten':
                child = parts[1]
                if child not in self.child_info: self.child_info[child] = {}
                self.child_info[child]['allergic'] = True
            elif parts[0] == 'not_allergic_gluten':
                child = parts[1]
                if child not in self.child_info: self.child_info[child] = {}
                self.child_info[child]['allergic'] = False
            elif parts[0] == 'waiting':
                 child, place = parts[1:]
                 if child not in self.child_info: self.child_info[child] = {}
                 self.child_info[child]['waiting_place'] = place

        # Identify goal children from goals
        self.goal_children = set()
        for goal in self.goals:
             parts = get_parts(goal)
             if parts[0] == 'served':
                 child = parts[1]
                 self.goal_children.add(child)
                 # Ensure goal children are in child_info even if only in goals (less likely)
                 if child not in self.child_info:
                      self.child_info[child] = {}


    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.
        Estimates the minimal number of required actions to reach a goal state.
        """
        state = node.state
        total_cost = 0

        # Identify unserved children from the goal children set
        unserved_children = set()
        for child in self.goal_children:
            if "(served " + child + ")" not in state:
                unserved_children.add(child)

        # If all goal children are served, heuristic is 0
        if not unserved_children:
            return 0

        # Calculate cost for each unserved child
        for child in unserved_children:
            # Each unserved child needs at least 1 action (serve)
            child_cost = 1

            # Get child's waiting place and allergy status
            child_info = self.child_info.get(child, {})
            child_place = child_info.get('waiting_place')
            # Default to False if allergy info is missing (shouldn't happen for goal children)
            child_is_allergic = child_info.get('allergic', False)

            # If waiting place is unknown, this child cannot be served in a standard way.
            # Assuming valid problems where waiting place is always known for goal children.
            if child_place is None:
                 # This indicates a problem definition issue if a goal child has no waiting place in static.
                 # In a real planner, this might warrant returning infinity.
                 # For this exercise, we proceed assuming child_place is valid for goal children.
                 pass


            # Check if a suitable sandwich is already on a tray at the child's place
            suitable_ontray_at_place = False
            suitable_ontray_anywhere = False
            suitable_kitchen = False

            # Iterate through state facts to find sandwich/tray status
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    s, t = get_parts(fact)[1:]
                    # Check if this sandwich is suitable for the current child
                    if self.is_suitable_sandwich(s, child, state):
                        suitable_ontray_anywhere = True # Found one on a tray somewhere
                        # Check if the tray is at the child's waiting place
                        if child_place is not None and "(at " + t + " " + child_place + ")" in state:
                            suitable_ontray_at_place = True # Found one on a tray at the right place
                            break # Found the best case for this child's sandwich/tray state

            if suitable_ontray_at_place:
                pass # Tray is at the right place, no move cost (cost remains 1)
            else:
                child_cost += 1 # Need to move tray (cost becomes 2)

                if suitable_ontray_anywhere:
                    pass # Sandwich is on a tray, no put_on_tray cost (cost remains 2)
                else:
                    child_cost += 1 # Need to put on tray (cost becomes 3)

                    # Check if a suitable sandwich is in the kitchen
                    for fact in state:
                        if match(fact, "at_kitchen_sandwich", "*"):
                            s = get_parts(fact)[1]
                            # Check if this sandwich is suitable for the current child
                            if self.is_suitable_sandwich(s, child, state):
                                suitable_kitchen = True
                                break # Found one in the kitchen

                    if suitable_kitchen:
                        pass # Sandwich is in the kitchen, no make cost (cost remains 3)
                    else:
                        child_cost += 1 # Need to make sandwich (cost becomes 4)
                        # Resource availability (bread, content, notexist) is not explicitly checked here
                        # for simplicity and efficiency, assuming resources are eventually available
                        # in a solvable problem.

            total_cost += child_cost

        return total_cost

    def is_suitable_sandwich(self, sandwich_obj, child_obj, state):
        """
        Checks if a sandwich is suitable for a child based on allergy status.
        Requires state to check if the sandwich is gluten-free.
        Requires self.child_info (from static facts) to check child allergy.
        """
        # Check if the sandwich is gluten-free
        # (no_gluten_sandwich s) is an effect of make_sandwich_no_gluten
        sandwich_is_ng = "(no_gluten_sandwich " + sandwich_obj + ")" in state

        # Check if the child is allergic (info is in self.child_info, extracted from static)
        child_is_allergic = self.child_info.get(child_obj, {}).get('allergic', False)

        if child_is_allergic:
            return sandwich_is_ng # Allergic child needs GF
        else:
            return True # Non-allergic child can have any sandwich

