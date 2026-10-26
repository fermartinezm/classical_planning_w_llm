import sys
import re
from fnmatch import fnmatch

# Try to import the base class Heuristic.
# If running standalone or in a different environment, provide a dummy class.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class 'heuristics.heuristic_base.Heuristic' not found. Using a dummy base class.", file=sys.stderr)
    class Heuristic:
        """Dummy base class for Heuristic."""
        def __init__(self, task):
            # Store goals to allow checking for goal state in __call__
            self.goals = task.goals
        def __call__(self, node):
            raise NotImplementedError("Heuristic logic not implemented in dummy base class.")

# Helper functions for parsing PDDL facts represented as strings
def get_parts(fact: str):
    """
    Extracts the predicate and arguments from a PDDL fact string.
    Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    Handles potential extra whitespace.
    Returns an empty list if the fact is malformed or empty after stripping parentheses.
    """
    # Remove leading/trailing whitespace
    fact = fact.strip()
    # Remove surrounding parentheses if present
    if fact.startswith("(") and fact.endswith(")"):
        fact = fact[1:-1]
    # Split by one or more whitespace characters; filter out empty strings
    parts = re.split(r'\s+', fact)
    return [part for part in parts if part]

def match(fact: str, *args):
    """
    Checks if a PDDL fact string matches a given pattern.
    Uses fnmatch to allow wildcard '*' matching for arguments.
    Example: match("(at tray1 kitchen)", "at", "*", "kitchen") -> True
             match("(at tray1 table1)", "at", "tray1", "kitchen") -> False
    """
    parts = get_parts(fact)
    # Check if the number of parts in the fact matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check each part against the corresponding pattern argument using fnmatch
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL ChildSnacks domain.

    # Summary
    Estimates the number of actions required to serve all children specified in the goal.
    It calculates the minimum estimated actions for each unserved child independently
    and sums these costs. The estimate considers the current state of sandwiches
    (location, gluten status) and trays. It is designed for Greedy Best-First Search
    and does not need to be admissible.

    # Assumptions
    - The primary goal is to achieve `(served c)` for all goal children `c`. Other goal
      types are ignored by this heuristic calculation but considered for the final goal check.
    - Each unserved child requires a sequence of actions: potentially make sandwich,
      put on tray, move tray, serve.
    - The heuristic simplifies by assuming resources like ingredients and sandwich 'slots'
      (`notexist s`) are available when needed for making a sandwich (Path 4).
    - It assumes a tray can always be moved to the kitchen if needed for the 'put_on_tray' action.
      The cost estimation for Path 3 implicitly assumes this availability or movement cost.
    - The heuristic sums the minimum estimated cost for each child individually,
      ignoring potential resource conflicts (e.g., one sandwich cannot serve two children)
      and potential optimizations (e.g., one tray movement serving multiple children
      at the same location). This makes it non-admissible but faster to compute.
    - The kitchen location is identified by the constant 'kitchen'.

    # Heuristic Initialization
    - Stores the set of children `c` that need to be served according to the goal `(served c)`.
    - Stores static information about each goal child: their waiting location `(waiting c p)` and
      whether they require a gluten-free sandwich (derived from `(allergic_gluten c)`).
    - Identifies the 'kitchen' constant place.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current state satisfies all goal conditions (`task.goals`). If yes, return 0.
    2.  Identify all children `c` that are goal conditions `(served c)` but are not
        currently served in the state `(served c) not in state`.
    3.  Initialize total heuristic value `h = 0`.
    4.  Parse the current state `node.state` to efficiently access:
        - Locations of all trays `(at t p)`.
        - Sandwiches currently on trays `(ontray s t)`.
        - Sandwiches currently at the kitchen `(at_kitchen_sandwich s)`.
        - Which sandwiches are gluten-free `(no_gluten_sandwich s)`.
    5.  For each unserved goal child `c`:
        a. Retrieve the child's waiting place `p` and whether they need a gluten-free sandwich (`needs_gf`)
           from the pre-computed static information stored during initialization.
        b. Estimate the minimum actions required (`cost_c`) by checking possible scenarios
           in increasing order of cost:
           i.   **Path 1 (Cost 1): Serve.** Is there a suitable sandwich `s` (correct `gf` status)
                already on a tray `t` located at the child's place `p`? If yes, `cost_c = 1`.
           ii.  **Path 2 (Cost 2): Move Tray + Serve.** If Path 1 fails, is there a suitable sandwich `s` on a
                tray `t`, but the tray is at a different location `p'` (and the tray location is known)?
                If yes, `cost_c = 2` (1 move + 1 serve).
           iii. **Path 3 (Cost 3): Put on Tray + Move + Serve.** If Paths 1 & 2 fail, is there a suitable sandwich `s`
                at the kitchen AND is there *any* tray currently at the kitchen? If yes, `cost_c = 3`
                (1 put + 1 move + 1 serve).
           iv.  **Path 4 (Cost 4): Make + Put + Move + Serve.** If Paths 1, 2 & 3 fail, assume
                we need to make a new sandwich from scratch. `cost_c = 4`
                (1 make + 1 put + 1 move + 1 serve). This assumes ingredients and a
                `notexist` sandwich slot are available.
        c. Add the determined `cost_c` for child `c` to the total heuristic value `h`.
    6.  After iterating through all unserved children, if `h` is 0 but the state is not a goal state
        (this could happen if goals included non-`served` predicates which are unmet, or if the
        goal list of children was empty), return 1 to ensure search does not terminate prematurely.
        Otherwise, return the calculated `h`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic. Extracts goal children and their static properties
        (waiting place, gluten allergy) from the task definition.
        """
        # Initialize the base class, which might store task goals
        super().__init__(task)
        # Store static facts for easy access during initialization
        self.static_facts = task.static

        # Define the constant kitchen place name based on the domain file
        self.kitchen = "kitchen"

        # --- Extract Goal Children ---
        self.goal_children = set()
        for goal in self.goals:
            # Use match helper for robust parsing of goal facts
            if match(goal, "served", "*"):
                parts = get_parts(goal)
                # Ensure the fact has the expected structure (predicate + 1 argument)
                if len(parts) == 2:
                    self.goal_children.add(parts[1]) # Add the child name

        # --- Extract Static Information for Goal Children ---
        self.child_waiting_place = {} # Map: child -> place
        self.child_needs_gluten_free = {} # Map: child -> bool
        allergic_children = set() # Temporary set to store allergic children

        # First pass over static facts: identify all allergic children
        for fact in self.static_facts:
             if match(fact, "allergic_gluten", "*"):
                parts = get_parts(fact)
                if len(parts) == 2:
                    allergic_children.add(parts[1])

        # Second pass over static facts: find waiting places for goal children
        # and determine their gluten-free requirement based on allergy status.
        for fact in self.static_facts:
            if match(fact, "waiting", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3:
                    child, place = parts[1], parts[2]
                    # Only store information for children who are part of the goal
                    if child in self.goal_children:
                        self.child_waiting_place[child] = place
                        # Check if this goal child was found in the allergic set
                        self.child_needs_gluten_free[child] = (child in allergic_children)

        # --- Sanity Check (Optional) ---
        # Verify that all goal children have associated static information.
        for child in self.goal_children:
            if child not in self.child_waiting_place:
                 # This indicates a potential inconsistency in the PDDL instance file
                 print(f"Warning: Goal child '{child}' is missing static '(waiting ...)' info.", file=sys.stderr)
            # child_needs_gluten_free is derived, so covered by the waiting_place check


    def __call__(self, node):
        """
        Computes the heuristic value (estimated actions) for the given state node.
        """
        state = node.state

        # --- Goal Check ---
        # If the current state satisfies all goal conditions, heuristic value is 0.
        # Assumes self.goals is available from the base class or stored in __init__.
        if self.goals <= state:
            return 0

        h = 0 # Initialize heuristic value

        # --- State Parsing ---
        # Efficiently parse the current state to extract relevant information
        served_children = set()         # Set of children already served
        tray_locations = {}             # Map: tray -> current place
        sandwiches_on_trays = {}        # Map: sandwich -> tray it's on
        sandwiches_at_kitchen = set()   # Set of sandwiches at the kitchen
        is_gluten_free = set()          # Set of sandwiches marked as gluten-free

        for fact in state:
            # Use try-except for robustness against unexpected fact formats
            try:
                parts = get_parts(fact)
                predicate = parts[0] if parts else None # Get predicate if parts is not empty

                # Use elif for efficiency as a fact matches only one pattern
                if predicate == "served" and len(parts) == 2:
                    served_children.add(parts[1])
                elif predicate == "at" and len(parts) == 3:
                    # Assuming the format is (at <object> <place>)
                    # We are interested in tray locations: (at <tray> <place>)
                    # We need a way to distinguish trays if other objects use 'at'.
                    # Assuming the first argument is the tray based on 'move_tray'.
                    # This might need refinement if other objects can be 'at' places.
                    # For now, assume parts[1] is a potential tray.
                    tray_locations[parts[1]] = parts[2]
                elif predicate == "ontray" and len(parts) == 3:
                    sandwiches_on_trays[parts[1]] = parts[2] # Map: sandwich -> tray
                elif predicate == "at_kitchen_sandwich" and len(parts) == 2:
                    sandwiches_at_kitchen.add(parts[1])
                elif predicate == "no_gluten_sandwich" and len(parts) == 2:
                    is_gluten_free.add(parts[1])
            except IndexError:
                # Log or ignore malformed facts encountered in the state
                # print(f"Warning: Malformed fact encountered: {fact}", file=sys.stderr)
                continue # Skip this fact and proceed with the next one

        # --- Heuristic Calculation ---
        # Iterate through each child required in the goal state
        for child in self.goal_children:
            # Skip children who are already served in the current state
            if child in served_children:
                continue

            # Check if static info (waiting place) is available for this goal child.
            # If not, it indicates an issue, assign max cost and log warning.
            if child not in self.child_waiting_place:
                 # print(f"Warning: Missing static info for goal child '{child}'. Assigning max cost.", file=sys.stderr)
                 h += 4 # Assign default maximum cost for this child
                 continue

            # Retrieve pre-calculated static info for the child
            place = self.child_waiting_place[child]
            needs_gf = self.child_needs_gluten_free.get(child, False) # Default to False if missing

            # --- Estimate Cost for this Child ---
            cost_c = 4  # Default cost: Make(1) + Put(1) + Move(1) + Serve(1)

            # Check Path 1 (Serve): Cost 1
            path1_found = False
            for s, t in sandwiches_on_trays.items(): # Iterate through sandwiches on trays
                s_is_gf = s in is_gluten_free
                s_is_suitable = (needs_gf and s_is_gf) or (not needs_gf)
                # Check if sandwich is suitable AND the tray 't' is at the child's 'place'
                if s_is_suitable and tray_locations.get(t) == place:
                    cost_c = 1 # Only 1 action needed: serve
                    path1_found = True
                    break # Found the cheapest path for this child
            if path1_found:
                h += cost_c
                continue # Move to the next child

            # Check Path 2 (Move, Serve): Cost 2
            path2_found = False
            for s, t in sandwiches_on_trays.items():
                s_is_gf = s in is_gluten_free
                s_is_suitable = (needs_gf and s_is_gf) or (not needs_gf)
                tray_loc = tray_locations.get(t) # Get the location of tray 't'
                # Check if sandwich is suitable AND tray 't' exists AND is NOT at the child's 'place'
                if s_is_suitable and tray_loc is not None and tray_loc != place:
                    cost_c = 2 # Actions needed: move_tray(1) + serve(1)
                    path2_found = True
                    break # Found the cheapest path for this child
            if path2_found:
                h += cost_c
                continue # Move to the next child

            # Check Path 3 (Put, Move, Serve): Cost 3
            path3_found = False
            # This path requires a tray to be present at the kitchen
            tray_at_kitchen = any(loc == self.kitchen for loc in tray_locations.values())
            if tray_at_kitchen:
                # Check if a suitable sandwich exists at the kitchen
                for s in sandwiches_at_kitchen:
                    s_is_gf = s in is_gluten_free
                    s_is_suitable = (needs_gf and s_is_gf) or (not needs_gf)
                    if s_is_suitable:
                        cost_c = 3 # Actions: put_on_tray(1) + move_tray(1) + serve(1)
                        path3_found = True
                        break # Found the cheapest path for this child
            if path3_found:
                h += cost_c
                continue # Move to the next child

            # Path 4 (Make, Put, Move, Serve): Cost 4
            # If none of the cheaper paths were found, assign the default cost of 4.
            h += cost_c

        # --- Final Adjustment ---
        # If the heuristic calculation resulted in 0, but the state is not actually
        # a goal state (e.g., other non-'served' goals exist and are unmet),
        # return 1 to prevent the search from stopping prematurely.
        if h == 0 and not (self.goals <= state):
             return 1
        else:
             # Return the calculated heuristic value (h will be 0 if it is a goal state)
             return h
