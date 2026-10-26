import os
import sys
from fnmatch import fnmatch

# Try to import the Heuristic base class from the expected location.
# If the import fails, define a dummy base class for standalone testing
# or basic structural validation.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            self.task = task
            # print("Using dummy Heuristic base class.") # Optional warning
        def __call__(self, node):
            raise NotImplementedError("Heuristic calculation not implemented in dummy class.")


# Helper function to extract parts of a PDDL fact string like '(predicate obj1 obj2)'
def get_parts(fact):
    """
    Extracts the components (predicate and arguments) of a PDDL fact string.

    Args:
        fact (str): The PDDL fact string (e.g., '(at tray1 kitchen)').

    Returns:
        list: A list of strings representing the components (e.g., ['at', 'tray1', 'kitchen']),
              or an empty list if the fact string is malformed.
    """
    if not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        # Return empty list for malformed or non-standard fact strings
        return []
    # Remove parentheses and split by whitespace
    return fact.strip()[1:-1].split()

# Helper function to match parsed fact parts against a pattern with wildcards
def match(fact_parts, *pattern):
    """
    Checks if the parts of a parsed fact match a given pattern using fnmatch wildcards.

    Args:
        fact_parts (list): A list of strings from get_parts().
        *pattern: A sequence of strings representing the pattern, where '*' acts as a wildcard.

    Returns:
        bool: True if the fact parts match the pattern, False otherwise.
    """
    # Check if the number of parts matches the pattern length
    if len(fact_parts) != len(pattern):
        return False
    # Use fnmatch to compare each part against the corresponding pattern element
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL Childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state.
    It calculates this estimate by summing the minimum estimated costs required to
    serve each child specified in the goal conditions but not yet served in the
    current state. The cost for serving an individual child is determined by analyzing
    the current availability and location of suitable sandwiches and trays, assuming
    a standard sequence of actions (make, put on tray, move tray, serve) if necessary.

    # Assumptions
    - Ingredient Availability: The heuristic assumes that required ingredients
      (bread portions, content portions) and available sandwich objects
      (those satisfying `(notexist ?s)`) are always sufficient to make a new
      sandwich when needed. It does not perform resource checking for ingredients.
    - Tray Availability: It assumes at least one tray object exists within the problem.
    - Independent Costs: The cost calculation for each child is performed independently.
      This means shared actions, such as moving a single tray carrying sandwiches for
      multiple children, might be counted multiple times in the total heuristic value.
      This overestimation is acceptable for a non-admissible heuristic designed to
      guide greedy best-first search.
    - Object Identification: It assumes tray objects can be identified (e.g., by name
      prefix 'tray*' as seen in examples). A more robust implementation might use
      PDDL types if available.
    - 'kitchen' Constant: Assumes 'kitchen' is a unique, constant place object.

    # Heuristic Initialization
    - The constructor (`__init__`) processes the task's static information (`task.static`)
      and goal specification (`task.goals`).
    - It stores:
        - A mapping from each child to their fixed waiting location (`child_locations`).
        - A set of children who are allergic to gluten (`allergic_children`).
        - A set of children who need to be served to satisfy the goal (`goal_children`).

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize Heuristic Value: Start with a total heuristic value `h = 0`.
    2. Identify Unserved Children: Determine the set of children present in `goal_children`
       but not yet marked as `(served ?c)` in the current `node.state`.
    3. Calculate Cost per Unserved Child: For each unserved child `c`:
        a. Retrieve `c`'s waiting place (`p_child`) and allergy status from the stored static info.
        b. Estimate the minimum actions (`cost_c`) needed to serve `c` based on the current state:
            i. **Case 1: Ready to Serve (Cost=1):**
               Is a suitable sandwich `s` (gluten-free if `c` is allergic) already on a tray `t`
               that is currently at `p_child`? If yes, `cost_c = 1` (only the 'serve' action remains).
            ii. **Case 2: Sandwich on Tray Elsewhere (Cost=2):**
                If not Case 1, is a suitable sandwich `s` on *any* tray `t`, but `t` is currently
                at a location `p_tray` different from `p_child`? If yes, `cost_c = 2`
                (1 'move_tray' action + 1 'serve' action).
            iii. **Case 3: Sandwich at Kitchen (Cost=2/3 or 3/4):**
                 If not Cases 1 or 2, is a suitable sandwich `s` available at the kitchen
                 (`at_kitchen_sandwich s`)?
                 - If yes, check if any tray is currently at the kitchen (`at t kitchen`).
                 - If a tray is at the kitchen: `cost_c = 1 ('put_on_tray') + move_cost + 1 ('serve')`.
                   `move_cost` is 0 if `p_child` is 'kitchen', 1 otherwise. (Total cost: 2 or 3).
                 - If no tray is at the kitchen: `cost_c = 1 ('move_tray' to kitchen) + 1 ('put_on_tray') + move_cost + 1 ('serve')`.
                   (Total cost: 3 or 4).
            iv. **Case 4: Need to Make Sandwich (Cost=3/4 or 4/5):**
                 If none of the above conditions are met, assume a new sandwich must be made.
                 - Check if any tray is currently at the kitchen.
                 - If a tray is at the kitchen: `cost_c = 1 ('make') + 1 ('put_on_tray') + move_cost + 1 ('serve')`.
                   (Total cost: 3 or 4).
                 - If no tray is at the kitchen: `cost_c = 1 ('make') + 1 ('move_tray' to kitchen) + 1 ('put_on_tray') + move_cost + 1 ('serve')`.
                   (Total cost: 4 or 5).
        c. Add `cost_c` to the total heuristic value `h`.
    4. Return Total Value: The final sum `h` is the heuristic estimate. A value of 0 indicates
       that all goal children are served in the current state.
    """

    def __init__(self, task):
        """Initializes the heuristic by extracting static information from the task."""
        super().__init__(task) # Initialize base class if it requires it
        self.goals = task.goals
        static_facts = task.static

        self.child_locations = {} # Map: child -> waiting place
        self.allergic_children = set() # Set of children with (allergic_gluten ?c)
        self.goal_children = set() # Set of children required to be served by the goal

        # Parse static facts to populate initial information
        for fact_str in static_facts:
            parts = get_parts(fact_str)
            if not parts: continue # Skip if fact string is malformed

            # Store child waiting locations
            if match(parts, "waiting", "*", "*"):
                child, place = parts[1], parts[2]
                self.child_locations[child] = place
            # Store which children are allergic
            elif match(parts, "allergic_gluten", "*"):
                self.allergic_children.add(parts[1])
            # Note: We infer non-allergic status by absence from allergic_children set

        # Parse goal facts to identify target children
        for goal_str in self.goals:
             parts = get_parts(goal_str)
             if not parts: continue
             # Identify children that need the (served ?c) predicate to be true
             if match(parts, "served", "*"):
                 self.goal_children.add(parts[1])

        # Optional validation: Check if all goal children have known locations.
        # This helps catch potential issues in PDDL problem definitions.
        # for child in self.goal_children:
        #     if child not in self.child_locations:
        #         print(f"Warning: Goal child {child} is missing a 'waiting' predicate in static facts.")


    def __call__(self, node):
        """Computes the heuristic value for the given state node."""
        state = node.state # The current state (a frozenset of fact strings)
        heuristic_value = 0

        # --- 1. Parse the Current State ---
        # Sets and dictionaries to store dynamic information from the current state
        served_children_in_state = set()
        sandwiches_at_kitchen = {} # Map: sandwich_name -> is_gluten_free (bool)
        sandwiches_on_tray = {} # Map: sandwich_name -> (tray_name, is_gluten_free)
        tray_locations = {} # Map: tray_name -> location_name
        gluten_free_sandwiches = set() # Set of sandwiches with (no_gluten_sandwich ?s)

        for fact_str in state:
            parts = get_parts(fact_str)
            if not parts: continue # Skip malformed facts

            # Identify children already served in this state
            if match(parts, "served", "*"):
                served_children_in_state.add(parts[1])

            # Identify sandwiches currently at the kitchen
            elif match(parts, "at_kitchen_sandwich", "*"):
                s = parts[1]
                # Initialize as not gluten-free; will be updated later if needed
                sandwiches_at_kitchen[s] = False

            # Identify sandwiches currently on a tray
            elif match(parts, "ontray", "*", "*"):
                s, t = parts[1], parts[2]
                # Initialize as not gluten-free; will be updated later
                sandwiches_on_tray[s] = (t, False)

            # Identify which sandwiches are marked as gluten-free in this state
            elif match(parts, "no_gluten_sandwich", "*"):
                 gluten_free_sandwiches.add(parts[1])

            # Identify current locations of trays (simple assumption: objects named 'tray*')
            elif match(parts, "at", "*", "*"):
                obj, loc = parts[1], parts[2]
                # This identification method is basic; relies on naming convention.
                if obj.startswith("tray"):
                    tray_locations[obj] = loc

        # Update the gluten-free status for sandwiches based on the parsed state
        for s in sandwiches_at_kitchen:
            if s in gluten_free_sandwiches:
                sandwiches_at_kitchen[s] = True # Mark as gluten-free
        # Need to update the tuple value for sandwiches on trays
        updated_sandwiches_on_tray = {}
        for s, (t, _) in sandwiches_on_tray.items():
            is_gf = s in gluten_free_sandwiches
            updated_sandwiches_on_tray[s] = (t, is_gf)
        sandwiches_on_tray = updated_sandwiches_on_tray # Replace with updated map

        # Determine if any known tray is currently located at the kitchen
        is_tray_at_kitchen = any(loc == 'kitchen' for loc in tray_locations.values())

        # --- 2. Calculate Heuristic Cost for Each Unserved Goal Child ---
        for child in self.goal_children:
            # Only calculate cost if the child is not already served
            if child not in served_children_in_state:
                child_loc = self.child_locations.get(child)
                # If child's location is unknown (problem definition issue), skip calculation
                if child_loc is None:
                    continue

                is_allergic = child in self.allergic_children
                cost_c = 0 # Estimated cost for serving this child
                found_case = 0 # Flag to track which case applies (1-4)

                # Check Case 1: Is a suitable sandwich ready on a tray at the child's location?
                for s, (t, is_gf) in sandwiches_on_tray.items():
                    # Check if the tray 't' is at the child's location 'child_loc'
                    if tray_locations.get(t) == child_loc:
                        # Check if the sandwich 's' is suitable (GF if needed, any otherwise)
                        if (is_allergic and is_gf) or (not is_allergic):
                            found_case = 1
                            break # Found a sandwich ready to be served
                if found_case == 1:
                    cost_c = 1 # Only the 'serve' action is needed

                else:
                    # Check Case 2: Is a suitable sandwich on a tray, but elsewhere?
                    for s, (t, is_gf) in sandwiches_on_tray.items():
                        tray_loc = tray_locations.get(t)
                        # Check if tray 't' exists and is *not* at the child's location
                        if tray_loc is not None and tray_loc != child_loc:
                            # Check if sandwich 's' is suitable
                            if (is_allergic and is_gf) or (not is_allergic):
                                found_case = 2
                                break # Found sandwich needing tray move + serve
                    if found_case == 2:
                        cost_c = 2 # Cost = 1 (move_tray) + 1 (serve)

                    else:
                        # Check Case 3: Is a suitable sandwich ready at the kitchen?
                        suitable_sandwich_at_kitchen = False
                        for s, is_gf in sandwiches_at_kitchen.items():
                            if (is_allergic and is_gf) or (not is_allergic):
                                suitable_sandwich_at_kitchen = True
                                break # Found a suitable sandwich at the kitchen

                        if suitable_sandwich_at_kitchen:
                            found_case = 3
                            # Cost depends on whether a tray needs moving to/from kitchen
                            move_to_child_cost = 0 if child_loc == 'kitchen' else 1
                            if is_tray_at_kitchen:
                                # Tray is already at kitchen
                                cost_c = 1 + move_to_child_cost + 1 # put_on_tray + move_tray + serve
                            else:
                                # Need to bring a tray to the kitchen first
                                cost_c = 1 + 1 + move_to_child_cost + 1 # move_tray_to_kitchen + put_on_tray + move_tray_to_child + serve
                        else:
                            # Case 4: No suitable sandwich exists; assume we need to make one.
                            found_case = 4
                            move_to_child_cost = 0 if child_loc == 'kitchen' else 1
                            if is_tray_at_kitchen:
                                # Tray is already at kitchen
                                cost_c = 1 + 1 + move_to_child_cost + 1 # make + put_on_tray + move_tray + serve
                            else:
                                # Need to bring a tray to the kitchen first
                                cost_c = 1 + 1 + 1 + move_to_child_cost + 1 # make + move_tray_to_kitchen + put_on_tray + move_tray_to_child + serve

                # Add the calculated cost for this child to the total heuristic value
                heuristic_value += cost_c

        # --- 3. Return Total Heuristic Value ---
        # The sum represents the estimated actions needed to serve all remaining goal children.
        return heuristic_value
