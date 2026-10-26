from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle potential non-string inputs or malformed facts gracefully
        # Depending on how the state is represented, this might need adjustment.
        # Assuming standard PDDL fact string representation.
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing up the estimated
    effort required for each unserved child. The effort for a child is estimated
    based on how many steps away a suitable sandwich is from being served to them.

    # Assumptions
    - The goal is to serve a specific set of children.
    - Children's allergy status and initial waiting locations are static.
    - All potential sandwich objects are defined in the initial state using the
      `notexist` predicate.
    - Ingredients (bread, content) are assumed to be available in the kitchen
      if a sandwich needs to be made (this is a simplification).
    - Trays are assumed to be available when needed in the kitchen (this is a simplification).

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the goal state.
    - Extracts the initial waiting location for each child from the initial state.
    - Extracts the allergy status for each child from the static facts.
    - Extracts the names of all potential sandwich objects from the initial state
      (those initially marked with `notexist`).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of costs calculated for each child that is
    required to be served in the goal but is not yet served in the current state.

    For each unserved goal child `c` waiting at location `p_c`:
    The cost for this child is calculated based on the following stages, adding 1
    for each stage that has not yet been reached for a suitable sandwich:

    1.  **Serving Action:** A `serve_sandwich` action is needed. Cost starts at 1.
    2.  **Sandwich on Tray at Location:** Is there a suitable sandwich `s` on a tray `t`
        (`(ontray s t)`) where the tray `t` is at the child's location `p_c`
        (`(at t p_c)`)?
        - If NO: Add 1 to the cost (represents the need for a `move_tray` action or
          getting the sandwich/tray to the location).
    3.  **Sandwich on Tray Anywhere:** Is there a suitable sandwich `s` on *any* tray `t`
        (`(ontray s t)`)? (This stage is only checked if stage 2 was not met).
        - If NO: Add 1 to the cost (represents the need for a `put_on_tray` action or
          getting the sandwich onto a tray).
    4.  **Sandwich Exists:** Has a suitable sandwich `s` been made? (i.e., the fact
        `(notexist s)` is NOT true for a suitable sandwich object `s`). (This stage
        is only checked if stage 3 was not met).
        - If NO: Add 1 to the cost (represents the need for a `make_sandwich` action).

    A sandwich `s` is considered "suitable" for child `c` if:
    - Child `c` is allergic to gluten AND `(no_gluten_sandwich s)` is true in the state.
    - Child `c` is NOT allergic to gluten AND `(no_gluten_sandwich s)` is false in the state.

    The total heuristic is the sum of these costs for all unserved goal children.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state # Need initial state to find all sandwich objects

        # Identify the set of children that must be served (from goal)
        self.goal_children = {get_parts(g)[1] for g in self.goals if match(g, "served", "*")}

        # Identify the initial waiting location for each child
        # Assuming waiting facts are in the initial state and are where serving must happen
        self.initial_waiting_locations = {
            get_parts(f)[1]: get_parts(f)[2]
            for f in self.initial_state # Use initial_state to get waiting locations
            if match(f, "waiting", "*", "*")
        }

        # Identify which children are allergic from static facts
        self.allergic_children = {
            get_parts(f)[1]
            for f in self.static
            if match(f, "allergic_gluten", "*")
        }

        # Identify all possible sandwich objects from the initial state (those that initially don't exist)
        self.all_sandwich_objects = {
            get_parts(f)[1]
            for f in self.initial_state
            if match(f, "notexist", "*")
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # Iterate through all children that need to be served according to the goal
        for child in self.goal_children:
            # If the child is already served, they contribute 0 to the heuristic
            if f"(served {child})" not in state:
                # This child needs to be served. Add base cost for the serve action.
                cost_for_child = 1

                # Determine the child's waiting location
                child_location = self.initial_waiting_locations.get(child)
                if child_location is None:
                     # This shouldn't happen in valid problems where goal children are waiting
                     # Handle defensively, maybe skip or add a large penalty?
                     # For simplicity, let's assume valid problems have waiting facts for goal children.
                     # If not found, this child cannot be served, potentially infinite cost, but GBFS needs finite.
                     # Let's just skip this child or add a fixed large cost if location is unknown.
                     # Assuming location is always known from initial state for goal children.
                     pass # Location must be known from initial_waiting_locations

                needs_gf = child in self.allergic_children

                # Check Stage 2: Is a suitable sandwich on a tray at the child's location?
                suitable_at_location = False
                for fact in state:
                    if match(fact, "ontray", "*", "*"):
                        s, t = get_parts(fact)[1:]
                        # Check if the tray is at the child's location
                        if f"(at {t} {child_location})" in state:
                            # Check if the sandwich is suitable (GF if needed, non-GF otherwise)
                            is_gf_sandwich = f"(no_gluten_sandwich {s})" in state
                            if (needs_gf and is_gf_sandwich) or (not needs_gf and not is_gf_sandwich):
                                suitable_at_location = True
                                break # Found a suitable sandwich at the location

                if not suitable_at_location:
                    # Stage 2 not met. Add cost for getting it there (e.g., move_tray).
                    cost_for_child += 1

                    # Check Stage 3: Is a suitable sandwich on a tray anywhere?
                    suitable_on_tray = False
                    for fact in state:
                         if match(fact, "ontray", "*", "*"):
                            s, t = get_parts(fact)[1:]
                            # Check if the sandwich is suitable (GF if needed, non-GF otherwise)
                            is_gf_sandwich = f"(no_gluten_sandwich {s})" in state
                            if (needs_gf and is_gf_sandwich) or (not needs_gf and not is_gf_sandwich):
                                suitable_on_tray = True
                                break # Found a suitable sandwich on a tray anywhere

                    if not suitable_on_tray:
                        # Stage 3 not met. Add cost for putting it on a tray (e.g., put_on_tray).
                        cost_for_child += 1

                        # Check Stage 4: Does a suitable sandwich exist (has it been made)?
                        suitable_exists = False
                        # Iterate through all possible sandwich objects defined in the problem
                        for s_obj in self.all_sandwich_objects:
                            # A sandwich exists if it's NOT marked as notexist in the current state
                            if f"(notexist {s_obj})" not in state:
                                # Check if this existing sandwich is suitable
                                is_gf_sandwich = f"(no_gluten_sandwich {s_obj})" in state
                                if (needs_gf and is_gf_sandwich) or (not needs_gf and not is_gf_sandwich):
                                    suitable_exists = True
                                    break # Found a suitable sandwich that exists

                        if not suitable_exists:
                             # Stage 4 not met. Add cost for making the sandwich (e.g., make_sandwich).
                             cost_for_child += 1

                # Add the calculated cost for this unserved child to the total heuristic
                h += cost_for_child

        return h

