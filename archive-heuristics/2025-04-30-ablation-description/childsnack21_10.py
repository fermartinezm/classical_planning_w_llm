from fnmatch import fnmatch
# Assuming Heuristic base class is available at this path
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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

    # Summary
    This heuristic estimates the cost to reach the goal (all children served)
    by summing an estimated minimum number of actions required for each unserved child.
    The estimate for each child is based on the current state of availability and
    location of a suitable sandwich for them.

    # Assumptions
    - Actions have a conceptual unit cost for heuristic estimation purposes.
    - The sequence of steps to serve a child is: Make Sandwich -> Put on Tray -> Move Tray -> Serve.
    - The heuristic assigns a cost based on which step is the next required one for a child.
    - If necessary ingredients for a suitable sandwich are unavailable and no such sandwich exists,
      the state is considered unsolvable for that child, contributing infinity to the heuristic.
    - Gluten allergies must be respected.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - The waiting place for each child.
    - Which children are allergic to gluten.
    - Which bread and content portions are gluten-free (needed to check ingredient availability).

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is part of the goal (i.e., needs to be served):
    1. Check if the child is already served in the current state. If yes, the cost for this child is 0.
    2. If the child is not served, determine their waiting place P and if they are allergic.
    3. Check if there is *any* suitable sandwich S on *any* tray T that is currently *at* place P.
       - A sandwich S is suitable for an allergic child if it is gluten-free.
       - A sandwich S is suitable for a non-allergic child if it is any sandwich.
       - If such a sandwich and tray exist at the child's location, the child is "ready to be served". The estimated cost for this child is 1 (representing the 'serve' action).
    4. If the child is not "ready to be served" at their location, check if there is *any* suitable sandwich S on *any* tray T that is currently *in the kitchen*.
       - If yes, the estimated cost for this child is 2 (representing 'move_tray' + 'serve').
    5. If no suitable sandwich is on a tray (either at the location or in the kitchen), check if there is *any* suitable sandwich S currently *in the kitchen* (not on a tray).
       - If yes, the estimated cost for this child is 3 (representing 'put_on_tray' + 'move_tray' + 'serve').
    6. If no suitable sandwich exists anywhere (neither on a tray nor in the kitchen), a new one must be made. Check if the necessary ingredients (bread and content portions in the kitchen, respecting gluten requirements) are available to make a suitable sandwich.
       - If ingredients are available, the estimated cost for this child is 4 (representing 'make_sandwich' + 'put_on_tray' + 'move_tray' + 'serve').
       - If ingredients are *not* available, the state is likely unsolvable for this child, and the estimated cost is infinity.
    7. The total heuristic value is the sum of the estimated costs for all children in the goal list.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Goal children (those who need to be served).
        - Waiting places for children.
        - Allergy status of children.
        - Gluten-free status of bread and content portions.
        """
        self.goals = task.goals  # Goal conditions, used to identify children to be served
        self.static = task.static  # Static facts

        # Extract children who need to be served from the goals
        self.goal_children = {get_parts(goal)[1] for goal in self.goals if match(goal, "served", "*")}

        # Extract static information
        self.waiting_places = {}  # child -> place
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.static_ng_bread = set()
        self.static_ng_content = set()

        for fact in self.static:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts
            predicate = parts[0]
            if predicate == "waiting" and len(parts) == 3:
                child, place = parts[1], parts[2]
                self.waiting_places[child] = place
            elif predicate == "allergic_gluten" and len(parts) == 2:
                self.allergic_children.add(parts[1])
            elif predicate == "not_allergic_gluten" and len(parts) == 2:
                self.not_allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread" and len(parts) == 2:
                self.static_ng_bread.add(parts[1])
            elif predicate == "no_gluten_content" and len(parts) == 2:
                self.static_ng_content.add(parts[1])

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all children in the goal.
        """
        state = node.state  # Current world state

        # Parse state facts for quick lookup
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        
        # Map trays to their current locations
        tray_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if match(fact, "at", "*", "*") and parts[1].startswith("tray"):
                 tray_locations[parts[1]] = parts[2]

        # Map sandwiches to the trays they are on
        sandwich_on_tray = {}
        for fact in state:
             parts = get_parts(fact)
             if match(fact, "ontray", "*", "*"):
                 sandwich_on_tray[parts[1]] = parts[2]

        # Set of sandwiches currently in the kitchen (not on a tray)
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}

        # Set of gluten-free sandwiches
        ng_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, "no_gluten_sandwich", "*")}

        # Set of bread and content portions currently in the kitchen
        kitchen_bread = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_bread", "*")}
        kitchen_content = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_content", "*")}

        total_heuristic = 0

        # Iterate through each child that needs to be served according to the goal
        for child in self.goal_children:
            # If child is already served, no cost needed for this child
            if child in served_children:
                continue

            # Child is unserved, calculate their estimated cost
            place = self.waiting_places.get(child)
            if place is None:
                 # This child is in the goal but not waiting anywhere? Should not happen in valid problems.
                 # Treat as unsolvable for safety.
                 return float('inf')

            is_allergic = child in self.allergic_children

            child_cost = 0 # Placeholder, will be assigned 1, 2, 3, 4, or inf

            # Check if a suitable sandwich is already at the child's location on a tray
            is_covered_at_location = False
            for sandwich, tray in sandwich_on_tray.items():
                if tray_locations.get(tray) == place:
                    # Check if the sandwich is suitable for the child
                    is_suitable = (sandwich in ng_sandwiches) if is_allergic else True
                    if is_suitable:
                        is_covered_at_location = True
                        break # Found a covering sandwich/tray at location

            if is_covered_at_location:
                child_cost = 1 # Needs 1 action: serve

            else: # Not covered at location, check other possibilities
                # Check if a suitable sandwich is on a tray in the kitchen
                is_covered_on_tray_kitchen = False
                for sandwich, tray in sandwich_on_tray.items():
                    if tray_locations.get(tray) == "kitchen":
                        is_suitable = (sandwich in ng_sandwiches) if is_allergic else True
                        if is_suitable:
                            is_covered_on_tray_kitchen = True
                            break # Found a covering sandwich/tray in kitchen

                if is_covered_on_tray_kitchen:
                    child_cost = 2 # Needs 2 actions: move_tray + serve

                else: # Not on a tray anywhere, check if in kitchen
                    is_covered_in_kitchen = False
                    for sandwich in kitchen_sandwiches:
                        is_suitable = (sandwich in ng_sandwiches) if is_allergic else True
                        if is_suitable:
                            is_covered_in_kitchen = True
                            break # Found a suitable sandwich in kitchen

                    if is_covered_in_kitchen:
                        child_cost = 3 # Needs 3 actions: put_on_tray + move_tray + serve

                    else: # No suitable sandwich exists or is available in the kitchen/on trays
                        # Need to make a sandwich
                        ingredients_available = False
                        if is_allergic:
                            # Need NG bread and NG content in kitchen
                            has_ng_bread_kitchen = any(b in kitchen_bread for b in self.static_ng_bread)
                            has_ng_content_kitchen = any(c in kitchen_content for c in self.static_ng_content)
                            ingredients_available = has_ng_bread_kitchen and has_ng_content_kitchen
                        else:
                            # Need any bread and any content in kitchen
                            has_any_bread_kitchen = bool(kitchen_bread) # Check if set is not empty
                            has_any_content_kitchen = bool(kitchen_content) # Check if set is not empty
                            ingredients_available = has_any_bread_kitchen and has_any_content_kitchen

                        if ingredients_available:
                            child_cost = 4 # Needs 4 actions: make_sandwich + put_on_tray + move_tray + serve
                        else:
                            # Cannot make the required sandwich and none exists
                            child_cost = float('inf') # Unsolvable for this child

            # Add the estimated cost for this child to the total heuristic
            total_heuristic += child_cost

        return total_heuristic
