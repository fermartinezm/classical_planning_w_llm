from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a valid match
    if len(parts) != len(args):
        return False
    # Use fnmatch for wildcard matching
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children (each needing a serve action),
    estimates the number of sandwiches that need to be made, put on trays,
    and the number of locations needing a tray delivery.

    # Assumptions
    - Each unserved child requires one suitable sandwich.
    - Sandwiches must be made (if necessary), put on a tray, and the tray moved to the child's location before serving.
    - A tray can hold multiple sandwiches and serve multiple children at the same location.
    - Ingredient availability and sandwich slots limit the number of new sandwiches that can be made.
    - The cost of getting ingredients to the kitchen is not considered (they start there).
    - The cost of moving trays between non-kitchen locations is simplified (only count move to a location needing a tray).
    - Any available ingredient of the correct type can be used with any other available ingredient of the correct type.
    - Any available sandwich slot can be used for any new sandwich.
    - Any available tray can be used for any sandwich and moved to any location.

    # Heuristic Initialization
    - Extract static facts about child allergies and waiting locations.
    - Identify all children who are goals (i.e., need to be served).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Check Goal State:** Identify the set of children that need to be served (from the task goal). Check if all of them have the `(served ?c)` predicate true in the current state. If so, the heuristic is 0.
    2.  **Count Unserved Children:** Count the number of children from the goal set that are not yet served. This count is a lower bound on the number of `serve` actions required and is added to the heuristic.
    3.  **Identify Sandwich Needs:** For the unserved children, count how many require a gluten-free sandwich (allergic) and how many require a regular sandwich (not allergic).
    4.  **Count Available Sandwiches and Ingredients:** Count existing sandwiches (on trays or in kitchen) by type (gluten-free/regular). Count available bread and content portions in the kitchen by type, and count available sandwich slots (`notexist`).
    5.  **Estimate Sandwiches to Make:** Calculate how many needed sandwiches (of each type) cannot be met by existing ones. Determine how many of these can actually be made based on available ingredients and sandwich slots. Add the cost for the `make_sandwich` or `make_sandwich_no_gluten` actions for the sandwiches that *can* be made.
    6.  **Estimate Sandwiches Needing 'Put on Tray':** Count the number of needed sandwiches that are not already on trays. These sandwiches must be moved from the kitchen onto a tray. This includes sandwiches that were initially in the kitchen and those that are newly made. Add the cost for the `put_on_tray` actions for these.
    7.  **Identify Locations Needing Trays:** Find all unique locations where unserved children are waiting. Check which of these locations do not currently have any tray present. Add the cost for the `move_tray` action for each such location. This counts one move per location needing a tray, regardless of how many sandwiches or children are destined for that location.
    8.  **Sum Costs:** The total heuristic value is the sum of the costs calculated in steps 2, 5, 6, and 7.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - The set of children that need to be served (from the goal).
        - Static facts about child allergies and waiting locations.
        """
        self.goal_children = set()
        for goal in task.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "served":
                self.goal_children.add(parts[1])

        self.child_allergy = {} # child -> True if allergic, False otherwise
        self.child_location = {} # child -> place
        # self.all_children = set() # Not strictly needed for heuristic calculation

        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == "allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = True
                # self.all_children.add(child)
            elif parts[0] == "not_allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = False
                # self.all_children.add(child)
            elif parts[0] == "waiting":
                child, place = parts[1], parts[2]
                self.child_location[child] = place
                # self.all_children.add(child)

        # Ensure all children in goals have allergy/location info, default if missing
        # (Although domain structure implies this info exists for goal children)
        for child in self.goal_children:
             if child not in self.child_allergy:
                 # Default to not allergic if info missing - might need domain analysis
                 # Based on example 2, all children in goals have allergy/waiting info
                 pass # Assume info exists from static facts as per examples
             if child not in self.child_location:
                 pass # Assume info exists from static facts as per examples


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Check Goal State
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_goal_children = self.goal_children - served_children

        if not unserved_goal_children:
            return 0 # Goal reached

        h = 0

        # 2. Count Unserved Children (for serve actions)
        num_unserved = len(unserved_goal_children)
        h += num_unserved

        # 3. Identify Sandwich Needs
        needed_gf = 0
        needed_regular = 0
        waiting_children_locations = set()

        for child in unserved_goal_children:
            # Use .get() with default False for safety, though static facts should cover goal children
            if self.child_allergy.get(child, False):
                needed_gf += 1
            else:
                needed_regular += 1
            # Track locations of unserved children
            if child in self.child_location: # Should always be true for goal children
                 waiting_children_locations.add(self.child_location[child])


        # 4. Count Available Sandwiches and Ingredients
        avail_gf_ontray = 0
        avail_regular_ontray = 0
        avail_gf_kitchen = 0
        avail_regular_kitchen = 0

        sandwich_is_gf = {} # {sandwich: True/False}

        # First pass to determine which sandwiches are GF
        for fact in state:
             if match(fact, "no_gluten_sandwich", "*"):
                 sandwich_is_gf[get_parts(fact)[1]] = True

        # Second pass to count sandwiches by location and type
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == "ontray":
                s = parts[1]
                if sandwich_is_gf.get(s, False): # Default to False if not explicitly GF
                    avail_gf_ontray += 1
                else:
                    avail_regular_ontray += 1
            elif parts[0] == "at_kitchen_sandwich":
                s = parts[1]
                if sandwich_is_gf.get(s, False): # Default to False if not explicitly GF
                    avail_gf_kitchen += 1
                else:
                    avail_regular_kitchen += 1

        avail_gf_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and match(fact, "no_gluten_bread", get_parts(fact)[1]))
        avail_gf_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and match(fact, "no_gluten_content", get_parts(fact)[1]))
        avail_regular_bread = sum(1 for fact in state if match(fact, "at_kitchen_bread", "*") and not match(fact, "no_gluten_bread", get_parts(fact)[1]))
        avail_regular_content = sum(1 for fact in state if match(fact, "at_kitchen_content", "*") and not match(fact, "no_gluten_content", get_parts(fact)[1]))
        avail_slots = sum(1 for fact in state if match(fact, "notexist", "*"))

        # 5. Estimate Sandwiches to Make
        total_avail_gf = avail_gf_kitchen + avail_gf_ontray
        total_avail_regular = avail_regular_kitchen + avail_regular_ontray

        needed_from_make_gf = max(0, needed_gf - total_avail_gf)
        needed_from_make_regular = max(0, needed_regular - total_avail_regular)

        # How many can we actually make based on ingredients and slots?
        can_make_gf_ingr = min(avail_gf_bread, avail_gf_content)
        made_gf = min(needed_from_make_gf, avail_slots, can_make_gf_ingr)
        h += made_gf * 1 # Cost for make_sandwich_no_gluten

        remaining_slots = avail_slots - made_gf
        remaining_gf_bread = avail_gf_bread - made_gf
        remaining_gf_content = avail_gf_content - made_gf

        # Regular sandwiches can use remaining GF ingredients if needed
        can_make_regular_ingr = min(avail_regular_bread + remaining_gf_bread, avail_regular_content + remaining_gf_content)
        made_regular = min(needed_from_make_regular, remaining_slots, can_make_regular_ingr)
        h += made_regular * 1 # Cost for make_sandwich

        # 6. Estimate Sandwiches Needing 'Put on Tray'
        # These are needed sandwiches that are not already on trays.
        # They must either be moved from the kitchen stock or are newly made.
        needed_not_on_trays_gf = max(0, needed_gf - avail_gf_ontray)
        needed_not_on_trays_regular = max(0, needed_regular - avail_regular_ontray)

        h += (needed_not_on_trays_gf + needed_not_on_trays_regular) * 1 # Cost for put_on_tray

        # 7. Identify Locations Needing Trays
        locations_with_trays = {get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*")}
        locations_needing_tray_move = waiting_children_locations - locations_with_trays

        h += len(locations_needing_tray_move) * 1 # Cost for move_tray

        return h
