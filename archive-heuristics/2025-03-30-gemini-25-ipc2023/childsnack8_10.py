import sys
import os
from fnmatch import fnmatch

# Ensure the heuristic_base module can be found
# This might need adjustment based on the project structure
# Assuming heuristic_base.py is in a directory named 'heuristics'
# adjacent to the current script or in the Python path.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Adjust the path if necessary, e.g., if run from a different directory
    # This is a simple attempt to handle potential import issues
    # You might need a more robust path management solution depending on setup
    # For example, adding the parent directory to sys.path
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(current_dir)
    # sys.path.append(parent_dir)
    try:
        from heuristics.heuristic_base import Heuristic
    except ImportError:
        # If still not found, provide a dummy class for structure
        print("Warning: Heuristic base class not found. Using dummy base class.", file=sys.stderr)
        class Heuristic:
            def __init__(self, task): pass
            def __call__(self, node): return 0


# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    if not fact or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed facts
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern. Wildcards (*) are allowed in the pattern.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the pattern length
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the ChildSnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    specified in the goal. It calculates the cost for each unserved child
    individually and sums these costs. The cost for a single child considers
    the steps needed: making a sandwich (if necessary), putting it on a tray,
    moving the tray to the child's location, and serving. It accounts for
    existing sandwiches and tray locations to estimate the remaining actions.

    # Assumptions
    - The heuristic assumes that ingredients (bread, content) are available when needed
      to make a sandwich. It doesn't track ingredient counts.
    - It assumes any available tray can be used. Tray objects are identified by
      checking if their name contains 'tray'. This might need adjustment if naming
      conventions differ. A more robust implementation would use type information
      from the task object if available.
    - It estimates tray movement costs simply: 1 action to move a tray to the kitchen
      if none are there, and 1 action to move a tray from the kitchen to the child's
      location if needed. It doesn't optimize tray routes or sharing. It assumes at
      least one tray exists if movement is required but none are at the kitchen.
    - The cost is calculated independently for each child and summed, which might
      overestimate the total cost as actions like moving a tray could potentially
      help serve multiple children at the same location. This is acceptable for a
      non-admissible heuristic aiming for guidance in greedy search.

    # Heuristic Initialization
    - Stores goal children: Which children need to be served eventually based on `(served ?c)` goals.
    - Stores static information parsed from `task.static`:
        - Child allergies (`allergic_gluten`, `not_allergic_gluten`) mapped to `child -> bool`.
        - Child waiting locations (`waiting`) mapped to `child -> place`.
        - Gluten-free status of initial bread/content (stored but not currently used in heuristic logic).
    - Validates that all goal children have necessary static information (allergy, location).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify Unserved Children: Determine which children listed in the goal (`self.goal_children`) are not present in the `(served ?c)` facts of the current state. If all goal children are served, the heuristic value is 0.
    2. Initialize Total Cost: Set `total_cost = 0`.
    3. Parse Current State: Extract dynamic information from `node.state`:
        - Locations of all trays (`tray -> place`). Identified by checking `(at ?obj ?place)` where `?obj` name contains 'tray'.
        - Set of sandwiches currently at the kitchen (`(sandwich, is_gluten_free)`). Gluten-free status determined by checking for `(no_gluten_sandwich ?s)`.
        - Dictionary of sandwiches currently on trays (`sandwich -> (tray, is_gluten_free)`).
    4. Iterate Through Unserved Children: For each `child` in the set of unserved children:
        a. Get Child Info: Retrieve allergy status (`needs_gluten_free`) and target location (`target_place`) for the child.
        b. Calculate Cost for Child `c` (`child_cost`):
            i. Check if Ready: Is a suitable sandwich `s` (matching gluten requirement) already on a tray `t` located at `target_place`?
               - If yes: `child_cost = 1` (for the `serve` action).
            ii. Check if On Tray Elsewhere: If not ready, is a suitable sandwich `s` on *any* tray `t` which is currently *not* at `target_place`?
               - If yes: `child_cost = 1 (serve) + 1 (move tray t) = 2`.
            iii. Check if At Kitchen: If not on any tray, is a suitable sandwich `s` at the kitchen?
               - If yes: Base cost is `child_cost = 1 (serve) + 1 (put_on_tray) = 2`.
                 - Add movement costs:
                   - Check if *any* tray is currently at the 'kitchen'. If not, add `+1` (action: move some tray to kitchen).
                   - Check if `target_place` is 'kitchen'. If not, add `+1` (action: move tray from kitchen to `target_place`).
                 - Total `child_cost` becomes 2, 3, or 4.
            iv. Need to Make: If no suitable sandwich exists anywhere (not on tray, not at kitchen):
               - Base cost is `child_cost = 1 (serve) + 1 (put_on_tray) + 1 (make_sandwich) = 3`.
               - Add movement costs using the same logic as step iii:
                 - If no tray is at the kitchen, add `+1`.
                 - If `target_place` is not 'kitchen', add `+1`.
               - Total `child_cost` becomes 3, 4, or 5.
        c. Add Child Cost to Total: `total_cost += child_cost`.
    5. Return Total Cost: The final `total_cost` is the heuristic estimate for the state.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class if needed
        self.goals = task.goals
        static_facts = task.static

        self.goal_children = set()
        for goal in self.goals:
            # Example goal: "(served child1)"
            if match(goal, "served", "*"):
                self.goal_children.add(get_parts(goal)[1])

        self.child_allergies = {} # child -> is_allergic (bool)
        self.child_locations = {} # child -> place
        # These are stored but not used by the current heuristic logic
        # self.is_no_gluten_bread = set() # bread
        # self.is_no_gluten_content = set() # content

        for fact in static_facts:
            # Use match for safer parsing based on predicate structure
            if match(fact, "allergic_gluten", "?c"):
                self.child_allergies[get_parts(fact)[1]] = True
            elif match(fact, "not_allergic_gluten", "?c"):
                self.child_allergies[get_parts(fact)[1]] = False
            elif match(fact, "waiting", "?c", "?p"):
                parts = get_parts(fact)
                self.child_locations[parts[1]] = parts[2]
            # Example for storing ingredient info if needed later:
            # elif match(fact, "no_gluten_bread", "?b"):
            #      self.is_no_gluten_bread.add(get_parts(fact)[1])
            # elif match(fact, "no_gluten_content", "?c"):
            #      self.is_no_gluten_content.add(get_parts(fact)[1])

        # Validation: Ensure all goal children have required static info
        for child in self.goal_children:
            if child not in self.child_allergies:
                # This indicates an issue with the PDDL instance or static fact parsing
                # Defaulting, but a stricter approach might raise an error.
                print(f"Warning: Static allergy info missing for goal child {child}. Assuming not allergic.", file=sys.stderr)
                self.child_allergies[child] = False
            if child not in self.child_locations:
                 # This is critical, cannot proceed without knowing where the child is.
                 raise ValueError(f"Error: Static waiting location missing for goal child {child}")


    def __call__(self, node):
        state = node.state

        # 1. Find served children in the current state
        served_children = set()
        for fact in state:
            if match(fact, "served", "?c"):
                served_children.add(get_parts(fact)[1])

        # 2. Identify unserved goal children
        unserved_children = self.goal_children - served_children

        # Goal reached state
        if not unserved_children:
            return 0

        # 3. Parse current state for dynamic info
        sandwiches_at_kitchen = set() # Elements are (sandwich_name, is_gluten_free)
        sandwiches_on_trays = {} # Key: sandwich_name, Value: (tray_name, is_gluten_free)
        tray_locations = {} # Key: tray_name, Value: place_name
        is_sandwich_gf = {} # Key: sandwich_name, Value: bool

        # First pass: find tray locations and gluten-free sandwiches
        for fact in state:
            if match(fact, "at", "?t", "?p"):
                 # Assume ?t is a tray if its name contains 'tray'.
                 # This is a heuristic assumption based on common naming.
                 obj_name = get_parts(fact)[1]
                 # A simple check, adjust if naming conventions differ.
                 # Consider checking against known tray objects if available from task
                 if 'tray' in obj_name:
                     tray_locations[obj_name] = get_parts(fact)[2]
            elif match(fact, "no_gluten_sandwich", "?s"):
                 is_sandwich_gf[get_parts(fact)[1]] = True

        # Second pass: determine sandwich locations and types using gf status
        for fact in state:
             if match(fact, "at_kitchen_sandwich", "?s"):
                 sandwich = get_parts(fact)[1]
                 # Default to False if not found in is_sandwich_gf dict
                 is_gf = is_sandwich_gf.get(sandwich, False)
                 sandwiches_at_kitchen.add((sandwich, is_gf))
             elif match(fact, "ontray", "?s", "?t"):
                 parts = get_parts(fact)
                 sandwich = parts[1]
                 tray = parts[2]
                 is_gf = is_sandwich_gf.get(sandwich, False)
                 sandwiches_on_trays[sandwich] = (tray, is_gf)

        # 4. Calculate cost for each unserved child
        total_cost = 0

        for child in unserved_children:
            child_cost = 0
            needs_gluten_free = self.child_allergies[child] # Get allergy status (checked in init)
            target_place = self.child_locations[child] # Get target location (checked in init)

            # i. Check if Ready: suitable sandwich on tray at target place
            found_ready = False
            for sandwich, (tray, is_gf) in sandwiches_on_trays.items():
                if tray_locations.get(tray) == target_place:
                    # Check suitability (matches gluten need)
                    if (needs_gluten_free and is_gf) or (not needs_gluten_free):
                        found_ready = True
                        break # Found one suitable sandwich ready to serve

            if found_ready:
                child_cost = 1 # Cost is just the 'serve' action
            else:
                # ii. Check if On Tray Elsewhere: suitable sandwich on any tray NOT at target place
                found_on_tray_elsewhere = False
                for sandwich, (tray, is_gf) in sandwiches_on_trays.items():
                     if (needs_gluten_free and is_gf) or (not needs_gluten_free):
                         # Check if tray exists in locations and is NOT at target
                         current_tray_loc = tray_locations.get(tray)
                         if current_tray_loc is not None and current_tray_loc != target_place:
                             found_on_tray_elsewhere = True
                             break # Found one suitable sandwich on another tray

                if found_on_tray_elsewhere:
                    child_cost = 1 # serve action
                    child_cost += 1 # move tray action (Estimate = 1)
                else:
                    # iii. Check if At Kitchen: suitable sandwich at kitchen
                    found_at_kitchen = False
                    for sandwich, is_gf in sandwiches_at_kitchen:
                        if (needs_gluten_free and is_gf) or (not needs_gluten_free):
                            found_at_kitchen = True
                            break # Found one suitable sandwich at the kitchen

                    if found_at_kitchen:
                        # Base cost for serving a sandwich from the kitchen
                        child_cost = 1 # serve action
                        child_cost += 1 # put_on_tray action (Base = 2)

                        # Add estimated movement costs
                        tray_at_kitchen = any(loc == 'kitchen' for loc in tray_locations.values())
                        target_place_is_kitchen = (target_place == 'kitchen')

                        # Cost to get a tray to the kitchen (if none is there)
                        if not tray_at_kitchen:
                            # Assumes at least one tray exists somewhere to be moved.
                            # If no trays exist at all, this state might be unsolvable for this path.
                            # Heuristic estimates cost 1 assuming a tray can be brought.
                            if tray_locations: # Check if any tray exists at all
                                child_cost += 1 # move other -> kitchen action estimate
                            else:
                                # No trays exist - technically infinite cost?
                                # Return a large number or stick to +1? Stick to +1 for simplicity.
                                child_cost += 1

                        # Cost to move the tray from kitchen to target (if target isn't kitchen)
                        if not target_place_is_kitchen:
                             child_cost += 1 # move kitchen -> target_place action estimate
                        # Total cost: 2, 3, or 4

                    else:
                        # iv. Need to Make Sandwich: No suitable sandwich exists anywhere
                        # Base cost for making, putting, and serving
                        child_cost = 1 # serve action
                        child_cost += 1 # put_on_tray action
                        child_cost += 1 # make_sandwich action (Base = 3)

                        # Add estimated movement costs (same logic as when found at kitchen)
                        tray_at_kitchen = any(loc == 'kitchen' for loc in tray_locations.values())
                        target_place_is_kitchen = (target_place == 'kitchen')

                        if not tray_at_kitchen:
                            if tray_locations:
                                child_cost += 1 # move other -> kitchen action estimate
                            else:
                                child_cost += 1 # Assume cost 1 even if no trays exist

                        if not target_place_is_kitchen:
                            child_cost += 1 # move kitchen -> target_place action estimate
                        # Total cost: 3, 4, or 5

            total_cost += child_cost

        return total_cost
