# Import necessary libraries and base class
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists in the specified path

# Helper function to parse PDDL facts
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes surrounding parentheses and splits the string by spaces.

    Args:
        fact (str): A PDDL fact string, e.g., "(at tray1 kitchen)".

    Returns:
        list: A list containing the predicate and its arguments,
              e.g., ['at', 'tray1', 'kitchen']. Returns an empty list
              if the fact is invalid or empty after stripping parentheses.
    """
    # Ensure fact is a string and has content inside parentheses
    if isinstance(fact, str) and len(fact) > 2 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    return []

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL ChildSnack domain.

    # Summary
    This heuristic estimates the total number of actions required to reach the goal state
    (serving all specified children) from the current state. It calculates this by summing
    the estimated minimum remaining actions needed for each child who is not yet served.
    The estimate for an individual child is determined by finding the "most advanced"
    suitable sandwich available for that child in the current state and counting the
    actions needed from that point. Suitability considers the child's potential gluten allergy.

    # Assumptions
    - The heuristic assumes that necessary resources (ingredients like bread/content,
      unused sandwich objects) are available if a new sandwich needs to be made.
      It does not check for the existence of `(at_kitchen_bread)`, `(at_kitchen_content)`,
      or `(notexist ?s)` when estimating the cost of making a sandwich.
    - It assumes a tray is available at the kitchen or can be brought there without extra
      cost counted towards the specific child when a sandwich needs to be put on a tray
      from the kitchen.
    - It assumes that moving a tray from its current location to the child's target
      location requires exactly one `move_tray` action, regardless of the distance or
      intermediate locations.
    - The heuristic calculates the cost for each child independently. It does not model
      resource contention (e.g., two children needing the same single gluten-free sandwich
      or the same tray simultaneously). This means it might underestimate the true cost
      in resource-constrained scenarios, but serves as a guiding estimate.

    # Heuristic Initialization (`__init__`)
    - The constructor parses the static facts provided in the task description.
    - It builds data structures to store:
        - `child_allergy`: A dictionary mapping each child to `True` if they are allergic
          to gluten (`allergic_gluten`) and `False` otherwise (`not_allergic_gluten`).
        - `child_location`: A dictionary mapping each child to the place where they are
          waiting (`waiting`).
    - It identifies the set of `goal_children` that must be served based on the `(served ?c)`
      predicates in the task's goal description.
    - It performs basic validation to ensure that allergy and location information is
      available for all children mentioned in the goal. If information is missing (which
      suggests an ill-formed problem description), it prints a warning and assigns
      default values (not allergic, dummy location) to allow the heuristic calculation
      to proceed, though potentially less accurately.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1. Initialize the total heuristic estimate `h = 0`.
    2. Parse the current `state` (a set of fact strings) provided in the `node` argument:
        - Identify which children are already served (`served_children`).
        - Find sandwiches currently at the kitchen (`kitchen_sandwiches`).
        - Track which sandwiches are on which trays (`ontray_sandwiches`).
        - Record the current location of each tray (`tray_locations`).
        - Note which existing sandwiches are gluten-free (`is_gf_sandwich`).
    3. Determine the set of `unserved_children` by subtracting `served_children` from the
       set of `goal_children` identified during initialization.
    4. If `unserved_children` is empty, the current state is a goal state (or satisfies the
       'served' part of the goal). Return `h = 0`.
    5. Iterate through each `child` in `unserved_children`:
        a. Retrieve the child's target location (`target_loc`) and whether they need a
           gluten-free sandwich (`needs_gf`) from the pre-computed static information.
           Handle potential missing info using the defaults set in `__init__`.
        b. If the `target_loc` is unknown (due to missing static facts), add a maximum
           cost of 4 to `h` for this child and continue to the next child.
        c. Initialize the minimum estimated cost for this child, `min_cost_for_c`, to 4.
           This represents the default cost assuming the sandwich needs to be made from
           scratch (1 make + 1 put + 1 move + 1 serve).
        d. Iterate through all sandwiches that currently exist (`all_existing_sandwiches =
           kitchen_sandwiches | ontray_sandwiches.keys()`):
            i. Check if the current sandwich `s` is suitable for the child:
               - If `needs_gf` is `True`, `s` must be in `is_gf_sandwich`.
               - If `needs_gf` is `False`, any `s` is suitable.
            ii. If `s` is suitable, determine its current stage and update `min_cost_for_c`
                if this sandwich offers a cheaper path to serving the child:
                - **Stage 4 (Ready to Serve):** If `s` is on a tray `t` AND the tray `t`'s
                  current location (`tray_locations.get(t)`) is equal to `target_loc`.
                  Set `min_cost_for_c = 1`. Since this is the lowest possible cost,
                  break the inner loop (over sandwiches) for this child.
                  *Estimated Actions: 1 (serve)*
                - **Stage 3 (On Tray, Needs Move):** If `s` is on a tray `t` BUT the tray's
                  location is known and is NOT `target_loc`, OR if the tray's location
                  is unknown (meaning it can't be at the target).
                  Update `min_cost_for_c = min(min_cost_for_c, 2)`.
                  *Estimated Actions: 2 (move, serve)*
                - **Stage 2 (At Kitchen):** If `s` is in `kitchen_sandwiches`.
                  Update `min_cost_for_c = min(min_cost_for_c, 3)`.
                  *Estimated Actions: 3 (put, move, serve)*
                - **Stage 1 (Needs Making):** If no suitable existing sandwich is found
                  in any of the above stages after checking all sandwiches, `min_cost_for_c`
                  remains at its initial value of 4.
                  *Estimated Actions: 4 (make, put, move, serve)*
        e. Add the final calculated `min_cost_for_c` for the current child to the total
           heuristic value `h`.
    6. After iterating through all unserved children, return the total accumulated cost `h`.
    """

    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.child_allergy = {} # child -> bool (True if allergic)
        self.child_location = {} # child -> place
        self.goal_children = set() # Children that must be served in the goal

        # Parse static facts to populate allergy and location info
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "allergic_gluten" and args:
                self.child_allergy[args[0]] = True
            elif predicate == "not_allergic_gluten" and args:
                self.child_allergy[args[0]] = False
            elif predicate == "waiting" and len(args) == 2:
                child, place = args
                self.child_location[child] = place

        # Identify children required to be served by the goal
        for goal in self.goals:
             parts = get_parts(goal)
             if not parts: continue

             predicate = parts[0]
             args = parts[1:]
             if predicate == "served" and args:
                 self.goal_children.add(args[0])

        # --- Validation and Default Setting ---
        # Ensure all children mentioned in the goal have the necessary static info.
        # If not, print a warning and set defaults to avoid errors during heuristic calculation.
        for child in self.goal_children:
            if child not in self.child_location:
                 print(f"Warning: Heuristic init: Goal child '{child}' missing 'waiting' location in static facts. Assigning dummy location.")
                 # Assign a dummy location; heuristic value might be less accurate for this child.
                 self.child_location[child] = "unknown_location_heuristic_default"
            if child not in self.child_allergy:
                 print(f"Warning: Heuristic init: Goal child '{child}' missing allergy info (allergic_gluten/not_allergic_gluten) in static facts. Assuming not allergic.")
                 # Assume not allergic as a default.
                 self.child_allergy[child] = False


    def __call__(self, node):
        """
        Calculates the heuristic value for a given state node.

        Args:
            node: A state node object containing the current state (`node.state`).

        Returns:
            int: The estimated cost (number of actions) to reach the goal.
        """
        state = node.state
        h = 0 # Initialize heuristic value

        # --- Parse Current State ---
        served_children = set()
        kitchen_sandwiches = set()
        ontray_sandwiches = {} # sandwich -> tray map
        tray_locations = {} # tray -> place map
        is_gf_sandwich = set() # set of gluten-free sandwiches

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]
            args = parts[1:]

            if predicate == "served" and args:
                served_children.add(args[0])
            elif predicate == "at_kitchen_sandwich" and args:
                kitchen_sandwiches.add(args[0])
            elif predicate == "ontray" and len(args) == 2:
                sandwich, tray = args
                ontray_sandwiches[sandwich] = tray
            elif predicate == "at" and len(args) == 2:
                # The domain defines 'at' predicate only for trays: (at ?t - tray ?p - place)
                # So, the first argument must be a tray.
                tray, place = args
                tray_locations[tray] = place
            elif predicate == "no_gluten_sandwich" and args:
                is_gf_sandwich.add(args[0])

        # --- Identify Unserved Children ---
        unserved_children = self.goal_children - served_children

        # If all goal children are served, heuristic is 0
        if not unserved_children:
            return 0

        # Combine all sandwiches that currently exist
        all_existing_sandwiches = kitchen_sandwiches.union(ontray_sandwiches.keys())

        # --- Calculate Cost for Each Unserved Child ---
        for child in unserved_children:
            # Retrieve static info safely using get() with defaults established in __init__
            needs_gf = self.child_allergy.get(child, False) # Default to False if somehow missing
            target_loc = self.child_location.get(child, "unknown_location_heuristic_default")

            # If location is unknown (due to missing static fact), assign max cost
            if target_loc == "unknown_location_heuristic_default":
                 h += 4 # Assign default max cost
                 continue

            # --- Determine Minimum Cost for this Child ---
            # Default cost assumes making from scratch: make(1)+put(1)+move(1)+serve(1) = 4
            min_cost_for_c = 4

            found_stage_4 = False # Flag to optimize inner loop: stop if cost 1 is found

            for s in all_existing_sandwiches:
                # 1. Check if sandwich 's' is suitable for the child's allergy needs
                is_suitable = (not needs_gf) or (s in is_gf_sandwich)
                if not is_suitable:
                    continue # Skip this sandwich if it doesn't meet allergy requirements

                # 2. Check the stage of this suitable sandwich 's'
                if s in ontray_sandwiches:
                    tray = ontray_sandwiches[s]
                    current_tray_loc = tray_locations.get(tray) # Use get() for safety if tray location is somehow unknown

                    if current_tray_loc == target_loc:
                        # Stage 4: Ready to serve (sandwich on tray at target location)
                        min_cost_for_c = 1 # Cost: serve(1)
                        found_stage_4 = True
                        break # Found the best possible state (cost 1), no need to check other sandwiches for this child
                    else:
                        # Stage 3: On tray, needs move (location known but not target, or location unknown)
                        min_cost_for_c = min(min_cost_for_c, 2) # Cost: move(1) + serve(1)

                elif s in kitchen_sandwiches:
                    # Stage 2: At kitchen (needs put, move, serve)
                    min_cost_for_c = min(min_cost_for_c, 3) # Cost: put(1) + move(1) + serve(1)

            # Add the minimum cost found for this child to the total heuristic value
            h += min_cost_for_c

        return h

