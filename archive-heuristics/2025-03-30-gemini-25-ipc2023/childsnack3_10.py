import math
from fnmatch import fnmatch
# Assuming the Heuristic base class is available in this path
from heuristics.heuristic_base import Heuristic
# Assuming the Task class definition is available if needed for type hints,
# but the heuristic itself only uses the task object passed to __init__
# from planning_task import Task


# Helper functions defined outside the class for potential reuse
def get_parts(fact):
    """
    Extract the components of a PDDL fact string.
    Removes the surrounding parentheses and splits the string by spaces.

    Args:
        fact (str): A PDDL fact string, e.g., "(at tray1 kitchen)".

    Returns:
        list[str]: A list of strings representing the parts of the fact,
                   e.g., ["at", "tray1", "kitchen"]. Returns empty list
                   for invalid input like empty string or None.
    """
    if not fact or len(fact) < 2:
        return []
    # Remove parentheses and split
    return fact[1:-1].split()

def match(fact_parts, *pattern):
    """
    Check if the parts of a PDDL fact match a given pattern.
    Allows '*' as a wildcard in the pattern.

    Args:
        fact_parts (list[str]): The pre-split parts of a PDDL fact.
        *pattern (str): A sequence of strings forming the pattern to match against.
                        Can include '*' as a wildcard for any single part.

    Returns:
        bool: True if the fact parts match the pattern, False otherwise.
    """
    # Check if the number of parts matches the pattern length
    if len(fact_parts) != len(pattern):
        return False
    # Check each part against the corresponding pattern element
    return all(fnmatch(part, arg) for part, arg in zip(fact_parts, pattern))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the number of actions required to reach a state
    where all goal children are served (`(served ?c)` is true for all children `?c`
    specified in the goal). It calculates this estimate by summing the estimated
    minimum number of actions needed for four sub-tasks:
    1. Making the required sandwiches (`cost_make`).
    2. Putting sandwiches onto trays (`cost_put`).
    3. Serving the sandwiches to the children (`cost_serve`).
    4. Moving the trays between locations (`cost_move`).
    The heuristic considers resource limitations like the availability of
    ingredients (gluten-free and regular), sandwich 'slots' (`notexist`), and trays.

    # Assumptions
    - Each child specified in the goal needs exactly one sandwich to be served.
    - Gluten-free sandwiches, if available beyond the needs of allergic children,
      can be used to serve non-allergic children.
    - Trays are assumed to have unlimited capacity for holding sandwiches, although
      the `put_on_tray` and `serve_sandwich` actions handle only one sandwich at a time.
      The heuristic focuses on the number of actions, not tray capacity limits.
    - The heuristic aims for informative guidance for Greedy Best-First Search and
      does not guarantee admissibility (i.e., it might overestimate the true cost).
      It prioritizes efficient computability and estimation accuracy over admissibility.

    # Heuristic Initialization (`__init__`)
    - Stores the goal conditions (`task.goals`) for later checking.
    - Parses static facts (`task.static`) provided by the task object to build
      efficient data structures for quick lookups during heuristic evaluation:
        - `child_to_place`: A dictionary mapping each child object to their static waiting place.
        - `child_is_allergic`: A dictionary mapping each child object to their allergy status (True if allergic to gluten, False otherwise).
        - `is_gf_bread`: A set containing all bread portions that are gluten-free.
        - `is_gf_content`: A set containing all content portions that are gluten-free.
    - Initializes sets to keep track of all known objects of different types
      (children, trays, bread portions, content portions, sandwiches, places),
      discovered primarily from static facts and potentially updated during state evaluation.
    - Identifies the set of children that need to be served to satisfy the goal (`goal_children`).
    - Defines a value representing infinity (`float('inf')`) to return for states
      detected as unsolvable due to resource constraints.

    # Step-By-Step Thinking for Computing Heuristic (`__call__`)
    1.  **Parse Current State:** Iterate through the facts in the current state (`node.state`). Extract dynamic information relevant to the heuristic calculation:
        - Which children are already served.
        - Which bread and content portions are currently in the kitchen.
        - Which sandwiches exist, where they are (kitchen or on a tray), and whether they are gluten-free.
        - Which trays exist and their current locations.
        - How many 'slots' for creating new sandwiches are available (count of `(notexist ?s)` facts).
        - Dynamically discover objects (trays, sandwiches, places) present in the state if they weren't mentioned in static facts.
    2.  **Identify Unserved Children:** Determine the set of `goal_children` that are not yet in the `served_children` set based on the current state. If this set is empty, the goal is potentially met; verify against `self.goals` and return 0 if true, or 1 otherwise (as a safeguard for non-goal terminal states).
    3.  **Calculate `cost_make` (Estimate for Making Sandwiches):**
        a. Count the number of unserved allergic children (`N_allergic`) and unserved non-allergic children (`N_non_allergic`).
        b. Count the number of existing gluten-free sandwiches (`S_gf_existing`) and regular sandwiches (`S_reg_existing`) currently present anywhere (in the kitchen or on any tray).
        c. Calculate how many *new* gluten-free sandwiches seem necessary: `needed_gf = max(0, N_allergic - S_gf_existing)`.
        d. Calculate how many *new* regular sandwiches seem necessary. This considers that leftover existing gluten-free sandwiches (`available_gf_for_reg = max(0, S_gf_existing - N_allergic)`) could potentially serve non-allergic children: `needed_reg = max(0, N_non_allergic - (S_reg_existing + available_gf_for_reg))`.
        e. **Resource Check:** Verify if the kitchen has sufficient resources to create these `needed_gf` and `needed_reg` sandwiches:
            - Check if enough pairs of gluten-free bread and content are available for `needed_gf`.
            - Check if enough pairs of *any* remaining bread and content (including leftover gluten-free) are available for `needed_reg`.
            - Check if enough `(notexist ?s)` facts (sandwich slots) are available for the total number of new sandwiches (`needed_gf + needed_reg`).
            - If any resource check fails, return `self.infinity` because the required sandwiches cannot be made from this state.
        f. `cost_make` = `needed_gf + needed_reg`. This estimates the number of `make_sandwich` or `make_sandwich_no_gluten` actions required.
    4.  **Calculate `cost_serve` (Estimate for Serving Sandwiches):**
        a. Each unserved child requires one serving action (`serve_sandwich` or `serve_sandwich_no_gluten`) eventually.
        b. `cost_serve` = total number of unserved children.
    5.  **Calculate `cost_put` (Estimate for Putting Sandwiches on Tray):**
        a. Estimate how many serving actions must be fulfilled by sandwiches that are *not* currently on a tray. These sandwiches must come from the kitchen (either existing there or newly made) and therefore require a `put_on_tray` action to get them onto a tray before serving.
        b. Count the total number of sandwiches currently on any tray (`sandwiches_currently_on_trays_count`).
        c. `cost_put = max(0, cost_serve - sandwiches_currently_on_trays_count)`. This estimates the number of `put_on_tray` actions needed.
    6.  **Calculate `cost_move` (Estimate for Moving Trays):**
        a. **Resource Check:** If tray actions (`cost_put > 0`) or service at non-kitchen locations are needed, but no trays exist in the current state (`known_trays` is empty), return `self.infinity`.
        b. Identify the set of unique non-kitchen places where unserved children are waiting (`places_to_visit`).
        c. Identify the set of places currently occupied by any tray (`places_with_trays`).
        d. Estimate moves needed to visit target serving locations: Add 1 `move_tray` action for each place in `places_to_visit` that does not currently have any tray present.
        e. Estimate move needed for loading sandwiches: If `cost_put > 0` (indicating sandwiches need loading from the kitchen) and no tray is currently at the kitchen (`'kitchen'` is not in `places_with_trays`), and at least one tray exists elsewhere, add 1 `move_tray` action. This estimates the cost of bringing a tray back to the kitchen if necessary for loading.
        f. `cost_move` is the sum of these estimated move actions.
    7.  **Total Heuristic Value:**
        a. The final heuristic estimate `h` is the sum of the costs calculated for each sub-task: `h = cost_make + cost_put + cost_serve + cost_move`.
        b. Return `h`.
    """
    def __init__(self, task):
        self.goals = frozenset(task.goals) # Store goals efficiently
        self.static_facts = frozenset(task.static) # Store static facts efficiently

        # --- Pre-process static information ---
        self.child_to_place = {}
        self.child_is_allergic = {}
        self.is_gf_bread = set()
        self.is_gf_content = set()
        self.all_children = set()
        self.all_trays = set() # Will be populated dynamically too
        self.all_bread = set()
        self.all_content = set()
        self.all_sandwiches = set() # Will be populated dynamically too
        self.all_places = {'kitchen'} # kitchen is a constant place

        for fact in self.static_facts:
            parts = get_parts(fact)
            # Using match helper for clarity and robustness
            if match(parts, "waiting", "?c", "?p"):
                child, place = parts[1], parts[2]
                self.child_to_place[child] = place
                self.all_children.add(child)
                self.all_places.add(place)
            elif match(parts, "allergic_gluten", "?c"):
                child = parts[1]
                self.child_is_allergic[child] = True
                self.all_children.add(child)
            elif match(parts, "not_allergic_gluten", "?c"):
                child = parts[1]
                self.child_is_allergic[child] = False
                self.all_children.add(child)
            elif match(parts, "no_gluten_bread", "?b"):
                bread = parts[1]
                self.is_gf_bread.add(bread)
                self.all_bread.add(bread)
            elif match(parts, "no_gluten_content", "?c"):
                content = parts[1]
                self.is_gf_content.add(content)
                self.all_content.add(content)
            # Add elif for other static predicates if needed for object discovery

        # Identify goal children from the goal specification
        self.goal_children = set()
        for goal_fact in self.goals:
             parts = get_parts(goal_fact)
             if match(parts, "served", "?c"):
                 child = parts[1]
                 self.goal_children.add(child)
                 # Ensure child object is known
                 self.all_children.add(child)

        # Define a representation for infinity
        self.infinity = float('inf')


    def __call__(self, node):
        """
        Calculate the heuristic value for the given state node.

        Args:
            node: A node object containing the state (.state) to evaluate.
                  The state is expected to be a frozenset of PDDL fact strings.

        Returns:
            float: The estimated cost (number of actions) to reach a goal state.
                   Returns float('inf') if the state is deemed unsolvable.
        """
        state = node.state

        # --- Parse current dynamic state ---
        served_children = set()
        kitchen_bread = set()
        kitchen_content = set()
        kitchen_sandwiches = set()
        sandwiches_on_tray = {} # map {sandwich: tray}
        tray_locations = {} # map {tray: place}
        is_gf_sandwich = set()
        available_sandwich_slots = 0
        current_trays = set()
        current_sandwiches = set()

        # Discover objects and facts present in the current state
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            predicate = parts[0]

            # Match facts relevant to the heuristic calculation
            if match(parts, "served", "?c"):
                served_children.add(parts[1])
            elif match(parts, "at_kitchen_bread", "?b"):
                kitchen_bread.add(parts[1])
            elif match(parts, "at_kitchen_content", "?c"):
                kitchen_content.add(parts[1])
            elif match(parts, "at_kitchen_sandwich", "?s"):
                sandwich = parts[1]
                kitchen_sandwiches.add(sandwich)
                current_sandwiches.add(sandwich)
            elif match(parts, "ontray", "?s", "?t"):
                sandwich, tray = parts[1], parts[2]
                sandwiches_on_tray[sandwich] = tray
                current_sandwiches.add(sandwich)
                current_trays.add(tray)
            elif match(parts, "at", "?t", "?p"):
                tray, place = parts[1], parts[2]
                tray_locations[tray] = place
                current_trays.add(tray)
                self.all_places.add(place) # Discover places dynamically
            elif match(parts, "no_gluten_sandwich", "?s"):
                sandwich = parts[1]
                is_gf_sandwich.add(sandwich)
                current_sandwiches.add(sandwich) # Ensure sandwich is known
            elif match(parts, "notexist", "?s"):
                available_sandwich_slots += 1
                # Discover potential sandwich names if not seen elsewhere
                # self.all_sandwiches.add(parts[1]) # Not strictly needed for calculation

        # --- Identify unserved children ---
        unserved_children = self.goal_children - served_children
        if not unserved_children:
            # If the set of unserved goal children is empty, check if goal is truly met
            # The check `self.goals <= state` verifies if all goal facts hold true in the current state.
            return 0 if self.goals <= state else 1 # Return 0 for goal, 1 for dead-ends that look like goals

        # --- Calculate cost_make ---
        unserved_allergic = {c for c in unserved_children if self.child_is_allergic.get(c, False)}
        unserved_non_allergic = unserved_children - unserved_allergic

        existing_sandwiches = kitchen_sandwiches | set(sandwiches_on_tray.keys())
        existing_gf_sandwiches = {s for s in existing_sandwiches if s in is_gf_sandwich}
        existing_reg_sandwiches = existing_sandwiches - existing_gf_sandwiches

        needed_gf = max(0, len(unserved_allergic) - len(existing_gf_sandwiches))
        available_gf_for_reg = max(0, len(existing_gf_sandwiches) - len(unserved_allergic))
        needed_reg = max(0, len(unserved_non_allergic) - (len(existing_reg_sandwiches) + available_gf_for_reg))

        # Resource Check for Making Sandwiches
        gf_bread_avail = {b for b in kitchen_bread if b in self.is_gf_bread}
        gf_content_avail = {c for c in kitchen_content if c in self.is_gf_content}
        if needed_gf > min(len(gf_bread_avail), len(gf_content_avail)):
            return self.infinity # Not enough GF ingredients

        # Check total ingredients for regular sandwiches (can use non-GF or leftover GF)
        non_gf_bread_avail = kitchen_bread - gf_bread_avail
        non_gf_content_avail = kitchen_content - gf_content_avail
        # Calculate ingredients remaining after making needed_gf sandwiches
        remaining_gf_bread = len(gf_bread_avail) - needed_gf
        remaining_gf_content = len(gf_content_avail) - needed_gf
        # Total bread/content available for the needed_reg sandwiches
        bread_for_reg = len(non_gf_bread_avail) + remaining_gf_bread
        content_for_reg = len(non_gf_content_avail) + remaining_gf_content
        if needed_reg > min(bread_for_reg, content_for_reg):
             return self.infinity # Not enough total ingredients for regular

        # Check sandwich slots ('notexist' facts)
        if needed_gf + needed_reg > available_sandwich_slots:
            return self.infinity # Not enough sandwich slots

        cost_make = needed_gf + needed_reg

        # --- Calculate cost_serve ---
        cost_serve = len(unserved_children)

        # --- Calculate cost_put ---
        sandwiches_currently_on_trays_count = len(sandwiches_on_tray)
        # Estimate actions needed to get required sandwiches onto trays
        cost_put = max(0, cost_serve - sandwiches_currently_on_trays_count)

        # --- Calculate cost_move ---
        cost_move = 0
        # Identify non-kitchen places where service is needed
        places_to_visit = set()
        for child in unserved_children:
            place = self.child_to_place.get(child)
            # Only consider places different from kitchen for move calculation
            if place and place != 'kitchen':
                places_to_visit.add(place)

        places_with_trays = set(tray_locations.values())
        known_trays_in_state = current_trays # Use trays found in this specific state

        # Check if trays are needed but none exist in the current state
        if (cost_put > 0 or places_to_visit) and not known_trays_in_state:
             # If we need to put things on a tray or move to a place, but no trays exist now
             return self.infinity

        # Estimate moves to target locations not currently having a tray
        cost_move += len(places_to_visit - places_with_trays)

        # Estimate move to kitchen for loading if needed and no tray is there
        if cost_put > 0 and 'kitchen' not in places_with_trays and known_trays_in_state:
            # Need to load, no tray at kitchen, but trays exist elsewhere
            cost_move += 1

        # --- Total Heuristic Value ---
        h = cost_make + cost_put + cost_serve + cost_move

        # Final safety check: if calculation yields 0 but state is not goal, return 1.
        # This prevents the search from terminating prematurely if h=0 calculation has a flaw
        # for a non-goal state. This should ideally not be triggered if logic is correct.
        if h == 0 and not self.goals <= state:
             return 1

        return h
