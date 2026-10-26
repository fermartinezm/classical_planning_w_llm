from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class is available

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove surrounding parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve each
    unserved child independently, summing up the estimated costs. For each
    unserved child, it counts the number of "missing steps" or "missing
    state components" required to reach the served state: having a suitable
    sandwich available, having that sandwich on a tray, having that tray
    at the child's table, and finally performing the serve action.

    # Assumptions
    - Each unserved child needs a dedicated serving process (making a sandwich,
      putting it on a tray, moving the tray, serving). Resource sharing (like
      trays or sandwiches) is not explicitly modeled in the cost calculation
      for different children, leading to a non-admissible heuristic.
    - Suitable ingredients are assumed to be available in the kitchen if a
      sandwich needs to be made.
    - Trays are assumed to be available in the kitchen if a sandwich needs
      to be put on a tray.
    - Moving a tray between any two places (kitchen or tables) takes 1 action.
    - Making a sandwich takes 1 action.
    - Putting a sandwich on a tray takes 1 action.
    - Serving a child takes 1 action.
    - A sandwich is suitable for an allergic child only if it is a
      no-gluten sandwich. Any sandwich is suitable for a non-allergic child.

    # Heuristic Initialization
    - Identify all children that are part of the goal (need to be served).
    - Map each child to the table where they are waiting.
    - Identify which children are allergic to gluten.
    - Collect the names of all possible sandwich objects, tray objects, and
      table objects defined in the task.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is computed as follows:
    1. Initialize the total heuristic cost to 0.
    2. Iterate through each child `c` that is in the goal list (identified during initialization).
    3. If child `c` is already in the `(served c)` state in the current state, the cost for this
       child is 0, so continue to the next child.
    4. If child `c` is not served:
       a. Determine the table `table_c` where child `c` is waiting (from initialization data) and whether
          child `c` is allergic to gluten (`needs_gf`, from initialization data).
       b. Initialize the cost for child `c` (`cost_c`) to 0.
       c. **Component 1: Suitable sandwich exists?** Check if there is *any*
          sandwich object `s` that is currently represented in the state
          (e.g., `(at_kitchen_sandwich s)`, `(ontray s t)`, `(has_bread s b)`,
          `(has_content s c)`, `(no_gluten_sandwich s)`) and is suitable for child `c`
          (i.e., if `needs_gf`, `(no_gluten_sandwich s)` must be true in the state).
          If no such suitable sandwich is found in the state, add 1 to `cost_c`
          (representing the 'make sandwich' action).
       d. **Component 2: Suitable sandwich on tray?** Check if there is *any*
          suitable sandwich `s` (from step 4c, if one exists or is assumed to
          be made) that is currently `(ontray s t)` for some tray `t` in the state. If not,
          add 1 to `cost_c` (representing the 'put on tray' action).
       e. **Component 3: Tray with suitable sandwich at table?** Check if there
          is *any* tray `t` (from step 4d, if one exists or is assumed to have
          the sandwich) that is currently `(at t table_c)` in the state. If not,
          add 1 to `cost_c` (representing the 'move tray' action).
       f. **Component 4: Serve action needed?** Add 1 to `cost_c` (representing
          the final 'serve' action).
       g. Add `cost_c` to the total heuristic cost.
    5. Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information, static facts,
        and object names.
        """
        self.goals = task.goals
        self.static = task.static

        # Identify children that need to be served (from goals)
        self.goal_children = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == "served"}

        # Map children to their waiting places (from static facts)
        self.waiting_places = {get_parts(s)[1]: get_parts(s)[2] for s in self.static if get_parts(s) and get_parts(s)[0] == "waiting"}

        # Identify allergic children (from static facts)
        self.allergic_children = {get_parts(s)[1] for s in self.static if get_parts(s) and get_parts(s)[0] == "allergic_gluten"}

        # Collect all object names of relevant types from task facts
        # Assuming task.facts contains grounded type facts like (child child1)
        self.all_children = {get_parts(f)[1] for f in task.facts if get_parts(f) and get_parts(f)[0] == "child"}
        self.all_trays = {get_parts(f)[1] for f in task.facts if get_parts(f) and get_parts(f)[0] == "tray"}
        self.all_sandwiches = {get_parts(f)[1] for f in task.facts if get_parts(f) and get_parts(f)[0] == "sandwich"}
        # Assuming places include 'kitchen' and tables. We only care about tables for serving.
        self.all_tables = {get_parts(f)[1] for f in task.facts if get_parts(f) and get_parts(f)[0] == "place" and get_parts(f)[1] != "kitchen"}


    def is_suitable(self, sandwich, needs_gf, state):
        """
        Checks if a given sandwich is suitable for a child based on allergy.

        - `sandwich`: The name of the sandwich object (string).
        - `needs_gf`: Boolean, True if the child is allergic to gluten.
        - `state`: The current state (frozenset of facts).

        Returns True if suitable, False otherwise.
        """
        if not needs_gf:
            # Non-allergic children can eat any sandwich.
            return True
        else:
            # Allergic children need a no-gluten sandwich.
            # Check if the fact (no_gluten_sandwich sandwich) is in the state.
            return f"(no_gluten_sandwich {sandwich})" in state

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all goal children.
        """
        state = node.state
        total_cost = 0

        # Collect sandwiches that are currently known in the state
        # (i.e., mentioned in facts that indicate their existence/properties, excluding notexist)
        known_sandwiches_in_state = set()
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] in ["at_kitchen_sandwich", "ontray", "has_bread", "has_content", "no_gluten_sandwich"]:
                 if len(parts) > 1:
                     known_sandwiches_in_state.add(parts[1])


        for child in self.goal_children:
            # If the child is already served, no cost is added for this child.
            if f"(served {child})" in state:
                continue

            # Child is not served, calculate cost for this child.
            cost_c = 0

            # Get the table where the child is waiting
            table_c = self.waiting_places.get(child)
            # Assuming valid problem structure where goal children are waiting.
            # If not, this would indicate an unsolvable state or problem error.
            if table_c is None:
                 # Should not happen in valid problems, but handle defensively.
                 # Returning a very high cost or infinity indicates difficulty/impossibility.
                 # For this problem, let's assume valid inputs.
                 continue # Skip this child if waiting place is unknown


            # Determine if the child needs a gluten-free sandwich
            needs_gf = child in self.allergic_children

            # Component 1: Suitable sandwich exists?
            # Check if any known sandwich in the state is suitable for this child.
            suitable_sandwiches_available = {
                s for s in known_sandwiches_in_state
                if self.is_suitable(s, needs_gf, state)
            }

            if not suitable_sandwiches_available:
                cost_c += 1 # Need to make one

            # Component 2: Suitable sandwich on tray?
            # Check if any suitable sandwich (real or assumed from step 1) is on a tray.
            # We check against all sandwiches that *could* be suitable and are on trays.
            # This is where non-admissibility comes in - we don't track *which* suitable sandwich.
            has_suitable_on_tray = False
            # Only check if a suitable sandwich exists or is assumed creatable (cost_c includes make if needed)
            if suitable_sandwiches_available or cost_c >= 1: # If we added cost for making, assume one can be put on tray
                 has_suitable_on_tray = any(
                     match(fact, "ontray", s, "*")
                     for fact in state
                     for s in suitable_sandwiches_available # Check existing suitable ones
                 )
                 # If no existing suitable sandwich is on a tray, but one can be made,
                 # we assume it *can* be put on a tray. The cost is added if *none* are on trays.
                 # The check `if not has_suitable_on_tray:` after this block handles the cost.


            if not has_suitable_on_tray:
                cost_c += 1 # Need to put one on a tray

            # Component 3: Tray with suitable sandwich at table?
            # Check if any tray that *could* hold a suitable sandwich is at the child's table.
            has_tray_at_table = False
            # Only check if a suitable sandwich on a tray exists or is assumed (cost_c includes put if needed)
            if has_suitable_on_tray or cost_c >= 2: # If we added cost for put, assume one can be moved
                 # Find trays that currently have *any* suitable sandwich on them in the state
                 trays_with_suitable_sandwich_in_state = {
                     get_parts(fact)[2] for fact in state
                     if match(fact, "ontray", "*", "*") and self.is_suitable(get_parts(fact)[1], needs_gf, state)
                 }

                 # Check if any of these trays are at the child's table
                 has_tray_at_table = any(
                     match(fact, "at", t, table_c)
                     for fact in state
                     for t in trays_with_suitable_sandwich_in_state
                 )

            if not has_tray_at_table:
                cost_c += 1 # Need to move tray

            # Component 4: Serve action needed?
            cost_c += 1 # Need to perform the final serve action

            total_cost += cost_c

        return total_cost

