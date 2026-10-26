from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Helper function to split a PDDL fact string into its predicate and arguments."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args for a meaningful match
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the total number of actions required to reach a goal state
    by summing the minimum estimated actions needed to serve each unserved child
    independently. It considers the current state of suitable sandwiches (location,
    existence) and ingredient availability, calculating the steps needed to get a
    sandwich to the child's location for serving.

    Assumptions:
    - All actions have a cost of 1.
    - Resource contention (e.g., multiple children needing the same sandwich, tray,
      or ingredients) is ignored. The heuristic calculates the cost for each child
      as if resources were available for them independently (a form of relaxation).
    - Assumes solvable problems; states where a child cannot be served due to
      missing ingredients (and no makeable sandwiches) are assigned a fixed penalty.
    - Assumes sandwich objects exist to be "made" (i.e., there are `notexist`
      predicates for potential sandwiches).

    Heuristic Initialization:
    The constructor precomputes static information from the task:
    - Which children are allergic to gluten and which are not.
    - The waiting location for each child.
    - Which bread portions are gluten-free.
    - Which content portions are gluten-free.
    This information is stored in sets and dictionaries for quick lookup during
    heuristic calculation.

    Step-By-Step Thinking for Computing Heuristic:
    1. Initialize the total heuristic value to 0.
    2. Identify all children who are currently waiting (`waiting ?c ?p`) but not yet served (`served ?c`).
    3. If there are no unserved children, the state is a goal state, and the heuristic is 0.
    4. Count the available gluten-free and total bread/content portions in the kitchen. Check if a tray is currently in the kitchen.
    5. Iterate through each unserved child:
        a. Determine if the child requires a gluten-free sandwich based on static facts.
        b. Calculate the minimum number of steps required *before* the final 'serve' action can be performed for this child. This is done by checking for the "closest" available suitable sandwich or ingredients:
            - **State 4 (0 steps before serve):** Check if a suitable sandwich is already on a tray at the child's waiting location. If yes, 0 steps are needed before serving.
            - **State 3 (1 step before serve):** If not in State 4, check if a suitable sandwich is on a tray located *elsewhere* (not at the child's location). If yes, 1 step (move tray) is needed before serving.
            - **State 2 (2 or 3 steps before serve):** If not in State 3 or 4, check if a suitable sandwich is in the kitchen. If yes, 2 steps (put on tray, move tray) are needed. If there is no tray currently in the kitchen, an extra step (move tray to kitchen) is needed, making it 3 steps before serving.
            - **State 1 (3 or 4 steps before serve):** If not in State 2, 3, or 4, check if suitable ingredients are available in the kitchen to make a suitable sandwich. If yes, 3 steps (make sandwich, put on tray, move tray) are needed. If there is no tray currently in the kitchen, an extra step (move tray to kitchen) is needed, making it 4 steps before serving.
            - **State 0 (Penalty):** If none of the above conditions are met (i.e., no suitable sandwich exists anywhere and cannot be made due to missing ingredients), assign a fixed penalty (e.g., 5 steps) representing a difficult or blocked state.
        c. Add the calculated minimum steps before serve *plus 1* (for the final 'serve' action itself) to the total heuristic value.
    6. Return the total accumulated heuristic value.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.child_waiting_location = {}
        self.allergic_children = set()
        self.not_allergic_children = set()
        self.gf_bread_types = set()
        self.gf_content_types = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                self.child_waiting_location[parts[1]] = parts[2]
            elif parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif parts[0] == "not_allergic_gluten":
                self.not_allergic_children.add(parts[1])
            elif parts[0] == "no_gluten_bread":
                self.gf_bread_types.add(parts[1])
            elif parts[0] == "no_gluten_content":
                self.gf_content_types.add(parts[1])

    def is_suitable(self, sandwich, child_name, state):
        """Checks if a sandwich is suitable for a child based on allergies."""
        needs_gf = child_name in self.allergic_children
        if needs_gf:
            # Allergic child needs a gluten-free sandwich
            return f"(no_gluten_sandwich {sandwich})" in state
        else:
            # Non-allergic child can eat any sandwich that exists
            # We just need to confirm the sandwich object exists in some state predicate
            # (at_kitchen_sandwich s) or (ontray s t) implies existence.
            # The check is implicit in how we find 's' in the state facts.
            return True

    def count_gf_bread(self, state):
        """Counts gluten-free bread portions in the kitchen."""
        count = 0
        for fact in state:
            if match(fact, "at_kitchen_bread", "?b"):
                b = get_parts(fact)[1]
                if b in self.gf_bread_types:
                    count += 1
        return count

    def count_total_bread(self, state):
        """Counts all bread portions in the kitchen."""
        count = 0
        for fact in state:
            if match(fact, "at_kitchen_bread", "?b"):
                count += 1
        return count

    def count_gf_content(self, state):
        """Counts gluten-free content portions in the kitchen."""
        count = 0
        for fact in state:
            if match(fact, "at_kitchen_content", "?c"):
                c = get_parts(fact)[1]
                if c in self.gf_content_types:
                    count += 1
        return count

    def count_total_content(self, state):
        """Counts all content portions in the kitchen."""
        count = 0
        for fact in state:
            if match(fact, "at_kitchen_content", "?c"):
                count += 1
        return count

    def has_tray_in_kitchen(self, state):
        """Checks if there is any tray located in the kitchen."""
        for fact in state:
            if match(fact, "at", "?t", "kitchen"):
                return True
        return False

    def __call__(self, node):
        state = node.state
        heuristic = 0

        # Identify children who are waiting but not served
        children_needing_serve = {
            c for c, p in self.child_waiting_location.items()
            if f"(served {c})" not in state
        }

        # If all children are served, the goal is reached
        if not children_needing_serve:
            return 0

        # Get counts of available ingredients and check for tray in kitchen
        gf_bread_count = self.count_gf_bread(state)
        total_bread_count = self.count_total_bread(state)
        gf_content_count = self.count_gf_content(state)
        total_content_count = self.count_total_content(state)
        has_kitchen_tray = self.has_tray_in_kitchen(state)

        # Iterate through each unserved child to estimate their individual cost
        for child in children_needing_serve:
            p = self.child_waiting_location[child]
            needs_gf = child in self.allergic_children

            min_steps_before_serve = float('inf')

            # Check State 4: Suitable sandwich on tray at child's location (0 steps before serve)
            found_state4 = False
            for fact in state:
                if match(fact, "ontray", "?s", "?t"):
                    s = get_parts(fact)[1]
                    t = get_parts(fact)[2]
                    if self.is_suitable(s, child, state):
                        if f"(at {t} {p})" in state:
                            min_steps_before_serve = 0
                            found_state4 = True
                            break # Found the best case for this child
            if found_state4:
                heuristic += (min_steps_before_serve + 1) # +1 for the serve action
                continue # Move to the next child

            # Check State 3: Suitable sandwich on tray elsewhere (1 step before serve: move tray)
            found_state3 = False
            if min_steps_before_serve > 0: # Only check if State 4 wasn't found
                for fact in state:
                    if match(fact, "ontray", "?s", "?t"):
                        s = get_parts(fact)[1]
                        t = get_parts(fact)[2]
                        if self.is_suitable(s, child, state):
                            # Check if tray t is anywhere *not* at p
                            tray_at_p = False
                            tray_loc = None
                            for fact_at in state:
                                if match(fact_at, "at", t, "?loc"):
                                    tray_loc = get_parts(fact_at)[2]
                                    if tray_loc == p:
                                        tray_at_p = True
                                    break # Found tray location
                            # If tray is on a tray (checked by ontray) and its location is known and not at p
                            if not tray_at_p and tray_loc is not None:
                                 min_steps_before_serve = min(min_steps_before_serve, 1)
                                 found_state3 = True
                                 # Don't break, keep searching for potentially better options (State 4 already handled)

            # Check State 2: Suitable sandwich in kitchen (2 or 3 steps before serve)
            found_state2 = False
            if min_steps_before_serve > 1: # Only check if State 3 or 4 wasn't found
                for fact in state:
                    if match(fact, "at_kitchen_sandwich", "?s"):
                        s = get_parts(fact)[1]
                        if self.is_suitable(s, child, state):
                            # Needs put_on_tray (1) + move_tray (1) = 2 steps before serve
                            # If no tray in kitchen, need move tray to kitchen (1) first = 3 steps before serve
                            steps = 2 if has_kitchen_tray else 3
                            min_steps_before_serve = min(min_steps_before_serve, steps)
                            found_state2 = True
                            # Don't break, keep searching for potentially better options

            # Check State 1: Can make suitable sandwich (3 or 4 steps before serve)
            can_make = False
            if needs_gf:
                if gf_bread_count > 0 and gf_content_count > 0:
                    can_make = True
            else: # needs regular
                # Can use any bread/content
                if total_bread_count > 0 and total_content_count > 0:
                     can_make = True

            if min_steps_before_serve > 2 and can_make: # Only check if State 2, 3, 4 wasn't found
                # Needs make (1) + put_on_tray (1) + move_tray (1) = 3 steps before serve
                # If no tray in kitchen, need move tray to kitchen (1) first = 4 steps before serve
                steps = 3 if has_kitchen_tray else 4
                min_steps_before_serve = min(min_steps_before_serve, steps)

            # State 0: Stuck (ingredients missing or no makeable sandwich)
            # If min_steps_before_serve is still infinity, it means no suitable sandwich
            # exists anywhere and cannot be made with available ingredients.
            if min_steps_before_serve == float('inf'):
                 # Assign a fixed penalty cost
                 min_steps_before_serve = 5 # Penalty value

            # Add the cost for this child (steps before serve + serve action)
            heuristic += (min_steps_before_serve + 1)

        return heuristic
