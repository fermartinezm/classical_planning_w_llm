from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Use zip to compare parts and args element by element, stopping at the shortest sequence.
    # fnmatch handles the wildcard matching.
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions needed to serve all children
    by summing up the estimated steps required for each unserved child independently.
    It counts the missing stages in the process: making a suitable sandwich,
    putting it on a tray, moving the tray to the child's location, and finally serving.

    # Assumptions
    - Each unserved child requires a suitable sandwich.
    - Resources (bread, content) are assumed to be available in the kitchen if a sandwich object exists but hasn't been made (`notexist`).
    - Trays can be moved between any places.
    - A child waits at a fixed location throughout the problem (based on static facts).
    - Allergy status of children is static.
    - A non-allergic child can be served any sandwich. An allergic child must be served a gluten-free sandwich.

    # Heuristic Initialization
    - Extracts the goal conditions (`served` facts).
    - Extracts static facts:
        - Mapping of children to their waiting places (`waiting` facts).
        - Allergy status for each child (`allergic_gluten`, `not_allergic_gluten` facts).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs for each child that needs to be served
    according to the goal state but is not yet served in the current state.

    For each unserved child `c` waiting at place `p`:
    1.  **Serve Action:** Add 1 to the cost (for the final `serve_sandwich` action). This is the minimum cost if all preconditions for serving are met.
    2.  **Move Tray:** Check if *any* tray is currently located at the child's waiting place `p`. If no tray is at `p`, add 1 to the cost (for a `move_tray` action).
    3.  **Put on Tray:** Check if a suitable sandwich `s` for child `c` is already on *any* tray `t` that is currently located at place `p`. If no such suitable sandwich exists on a tray *at `p`*, add 1 to the cost (for a `put_on_tray` action). A sandwich `s` is suitable for child `c` if:
        - Child `c` is allergic to gluten AND sandwich `s` is `no_gluten_sandwich`.
        - Child `c` is NOT allergic to gluten (any sandwich `s` is suitable).
    4.  **Make Sandwich:** Check if a suitable sandwich `s` for child `c` has already been made (i.e., it exists either `at_kitchen_sandwich` or `ontray` anywhere). If no suitable sandwich exists anywhere, add 1 to the cost (for a `make_sandwich` or `make_sandwich_no_gluten` action).

    The total heuristic value is the sum of these costs for all unserved children.
    The heuristic is 0 if and only if all goal children are served.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals

        # Extract static information: child waiting places and allergy status
        self.child_waiting_place = {}
        self.child_allergy = {} # True if allergic, False otherwise

        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == "waiting":
                child, place = parts[1:]
                self.child_waiting_place[child] = place
            elif parts[0] == "allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = True
            elif parts[0] == "not_allergic_gluten":
                child = parts[1]
                self.child_allergy[child] = False

        # Note: Assumes all children mentioned in goals or waiting facts
        # have their allergy status defined in static facts.

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_heuristic_cost = 0

        # Identify children that need serving
        children_to_serve = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                child = get_parts(goal)[1]
                if goal not in state:
                    children_to_serve.add(child)

        # Calculate cost for each unserved child
        for child in children_to_serve:
            child_cost = 0

            # Step 1: Serve action (always needed if not served)
            child_cost += 1

            # Get the child's waiting place (should be in static facts)
            place = self.child_waiting_place.get(child)
            if place is None:
                 # This child's location is unknown, cannot estimate cost.
                 # In a real scenario, this might indicate an unsolvable state
                 # or a problem definition issue. For heuristic, we can skip
                 # or return infinity. Skipping assumes valid problems.
                 continue

            # Step 2: Move tray
            # Check if any tray is at the child's waiting place
            tray_at_place_p = any(match(fact, "at", "*", place) for fact in state)
            if not tray_at_place_p:
                child_cost += 1

            # Step 3: Put on tray
            # Check if a suitable sandwich is on a tray at the child's waiting place
            suitable_sandwich_on_tray_at_p = False
            for fact in state:
                if match(fact, "ontray", "*", "*"):
                    s, t = get_parts(fact)[1:]
                    # Check if the tray is at the child's place
                    if f"(at {t} {place})" in state:
                        # Check if the sandwich is suitable for the child
                        if self._is_suitable_sandwich(s, child, state):
                            suitable_sandwich_on_tray_at_p = True
                            break # Found one suitable sandwich on a tray at the right place

            if not suitable_sandwich_on_tray_at_p:
                child_cost += 1

            # Step 4: Make sandwich
            # Check if a suitable sandwich has been made anywhere (kitchen or on any tray)
            suitable_sandwich_available_somewhere = False
            # Check in kitchen
            for fact in state:
                if match(fact, "at_kitchen_sandwich", "*"):
                    s = get_parts(fact)[1]
                    if self._is_suitable_sandwich(s, child, state):
                        suitable_sandwich_available_somewhere = True
                        break
            # If not found in kitchen, check on any tray
            if not suitable_sandwich_available_somewhere:
                 for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        s, t = get_parts(fact)[1:]
                        if self._is_suitable_sandwich(s, child, state):
                            suitable_sandwich_available_somewhere = True
                            break

            if not suitable_sandwich_available_somewhere:
                child_cost += 1

            total_heuristic_cost += child_cost

        return total_heuristic_cost

    def _is_suitable_sandwich(self, sandwich, child, state):
        """Helper to check if a sandwich is suitable for a child based on allergy."""
        is_no_gluten_s = f"(no_gluten_sandwich {sandwich})" in state
        # Get allergy status, default to False if child not found (shouldn't happen in valid problems)
        is_allergic_c = self.child_allergy.get(child, False)

        if is_allergic_c:
            # Allergic child needs a gluten-free sandwich
            return is_no_gluten_s
        else:
            # Non-allergic child can have any sandwich (gluten or non-gluten)
            return True

# Note: The 'Heuristic' base class is assumed to be imported from
# 'heuristics.heuristic_base'. If running this code standalone,
# you would need to provide a definition for the base class.
# Example dummy base class:
# class Heuristic:
#     def __init__(self, task): pass
#     def __call__(self, node): pass
