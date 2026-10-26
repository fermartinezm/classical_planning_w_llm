from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic: # Assuming Heuristic base class is available elsewhere and this class will inherit from it or match its interface
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the remaining effort by summing up the minimum estimated actions
    required for each unserved child to receive a suitable sandwich. It considers the current
    location and state of suitable sandwiches (on tray at location, on tray elsewhere,
    at kitchen, or needs to be made).

    # Assumptions
    - Each unserved child needs one suitable sandwich.
    - A suitable sandwich is gluten-free for allergic children and any sandwich for others.
    - The minimum actions to get a sandwich to a child depend on the sandwich's current location:
        - On a tray at the child's location: 1 action (serve).
        - On a tray elsewhere: 2 actions (move tray, serve).
        - At the kitchen: 3 actions (put on tray, move tray, serve).
        - Needs to be made: 4 actions (make, put on tray, move tray, serve).
    - It is possible to make a suitable sandwich if there is a 'notexist' fact for any sandwich object
      AND appropriate ingredients (any bread/content for non-allergic, gluten-free bread/content for allergic)
      are available at the kitchen.
    - Trays are assumed to be available at the kitchen when needed for 'put_on_tray'.
    - If a child cannot be served and a suitable sandwich cannot be made based on the above checks,
      a large penalty is added, assuming the state is far from a solution or potentially unsolvable
      from this point (though the initial problem is assumed solvable).

    # Heuristic Initialization
    - Identify all children that need to be served from the goal state.
    - Determine which children are allergic to gluten from static facts.
    - Determine the waiting location for each child from static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Initialize total heuristic cost to 0.
    2. Pre-process the current state to quickly find:
       - Locations of all trays.
       - Which sandwiches are on which trays.
       - Which sandwiches are at the kitchen.
       - Which sandwiches do not exist (`notexist`).
       - Which sandwiches are gluten-free.
       - Which bread and content portions are at the kitchen, and which of those are gluten-free.
    3. For each child that needs to be served (identified during initialization):
        a. Check if the child is already served in the current state. If yes, continue to the next child (cost 0 for this child).
        b. If the child is not served, find their waiting location (identified during initialization).
        c. Determine if the child requires a gluten-free sandwich based on initialization data.
        d. Find the minimum estimated cost to get a *suitable* sandwich to this child by checking the pre-processed state information in order of increasing cost:
            i. Check if any suitable sandwich is on a tray at the child's location. If yes, the minimum cost for this child is 1.
            ii. If not, check if any suitable sandwich is on a tray at a *different* location. If yes, the minimum cost for this child is 2.
            iii. If not, check if any suitable sandwich is at the kitchen. If yes, the minimum cost for this child is 3.
            iv. If not, check if a suitable sandwich can be made: requires a `notexist` sandwich object AND appropriate ingredients at the kitchen (checked using pre-processed ingredient info). If yes, the minimum cost for this child is 4.
            v. If none of the above conditions are met (no suitable sandwich exists in state and cannot be made based on checks), assign a large penalty (e.g., 1000) to this child's cost, indicating a potentially stuck state.
        e. Add the minimum estimated cost found for this child to the total heuristic cost.
    4. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Extract children that need to be served from the goals
        self.children_to_serve = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                self.children_to_serve.add(get_parts(goal)[1])

        # Extract allergic children from static facts
        self.allergic_children = set()
        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                self.allergic_children.add(get_parts(fact)[1])

        # Extract child waiting locations from static facts
        self.child_locations = {}
        for fact in self.static:
            if match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.child_locations[child] = place

        # Note: no_gluten_bread/content are properties of objects, not static facts themselves,
        # but they are typically in the initial state and don't change. We will identify
        # which specific bread/content objects are no-gluten by scanning the state in __call__.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        total_cost = 0

        # --- Pre-process state for quick lookups ---
        tray_locations = {} # tray -> place
        sandwich_on_tray = {} # sandwich -> tray
        sandwich_at_kitchen = set() # set of sandwiches
        sandwich_not_exist = set() # set of sandwiches
        no_gluten_sandwiches_in_state = set() # set of no-gluten sandwich objects

        bread_at_kitchen_list = [] # Store objects like 'bread1'
        content_at_kitchen_list = [] # Store objects like 'content1'
        no_gluten_bread_objects_in_state = set() # set of no-gluten bread objects
        no_gluten_content_objects_in_state = set() # set of no-gluten content objects


        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]

            if predicate == "at" and len(parts) == 3: # (at ?t ?p)
                 obj, place = parts[1:]
                 # Assuming only trays can be at places other than kitchen for objects we care about
                 # and kitchen is a special place handled by at_kitchen_* predicates for ingredients/sandwiches
                 if obj.endswith("-tray"): # Check if it's a tray object
                     tray_locations[obj] = place
            elif predicate == "ontray" and len(parts) == 3: # (ontray ?s ?t)
                 sandwich, tray = parts[1:]
                 sandwich_on_tray[sandwich] = tray
            elif predicate == "at_kitchen_sandwich" and len(parts) == 2: # (at_kitchen_sandwich ?s)
                 sandwich_at_kitchen.add(parts[1])
            elif predicate == "notexist" and len(parts) == 2: # (notexist ?s)
                 sandwich_not_exist.add(parts[1])
            elif predicate == "no_gluten_sandwich" and len(parts) == 2: # (no_gluten_sandwich ?s)
                 no_gluten_sandwiches_in_state.add(parts[1])
            elif predicate == "at_kitchen_bread" and len(parts) == 2: # (at_kitchen_bread ?b)
                bread_at_kitchen_list.append(parts[1])
            elif predicate == "at_kitchen_content" and len(parts) == 2: # (at_kitchen_content ?c)
                content_at_kitchen_list.append(parts[1])
            elif predicate == "no_gluten_bread" and len(parts) == 2: # (no_gluten_bread ?b)
                 no_gluten_bread_objects_in_state.add(parts[1])
            elif predicate == "no_gluten_content" and len(parts) == 2: # (no_gluten_content ?c)
                 no_gluten_content_objects_in_state.add(parts[1])

        # Check for existence of ingredients at kitchen
        has_any_bread_at_kitchen = len(bread_at_kitchen_list) > 0
        has_any_content_at_kitchen = len(content_at_kitchen_list) > 0

        # Check for existence of specific gluten-free ingredients at kitchen
        has_ng_bread_at_kitchen = any(b in no_gluten_bread_objects_in_state for b in bread_at_kitchen_list)
        has_ng_content_at_kitchen = any(c in no_gluten_content_objects_in_state for c in content_at_kitchen_list)

        # --- Calculate cost for each unserved child ---
        for child in self.children_to_serve:
            # Check if child is already served
            if f"(served {child})" in state:
                continue # Child is served, cost is 0 for this child

            # Child is not served, calculate cost
            child_place = self.child_locations.get(child) # Get child's waiting place
            if child_place is None:
                 # This child needs serving according to goal, but is not waiting.
                 # This state might be unreachable or indicate an issue.
                 # For a solvable problem, children needing service should be waiting.
                 # Assign a penalty or skip? Skipping might underestimate. Penalty is safer for GBFS.
                 # Let's assign a penalty.
                 total_cost += 1000 # Penalty for child not waiting but needing service
                 continue

            needs_gluten_free = child in self.allergic_children

            min_child_cost = float('inf')

            # Category 1: Suitable sandwich on a tray at the child's location (Cost 1)
            found_ready_sandwich = False
            for s, t in sandwich_on_tray.items():
                if tray_locations.get(t) == child_place:
                    # Check if sandwich s is suitable
                    is_suitable = (not needs_gluten_free) or (s in no_gluten_sandwiches_in_state)
                    if is_suitable:
                        min_child_cost = min(min_child_cost, 1)
                        found_ready_sandwich = True
                        break # Found one ready sandwich, no need to check others for cost 1

            if found_ready_sandwich:
                 total_cost += min_child_cost
                 continue # Move to the next child

            # Category 2: Suitable sandwich on a tray elsewhere (Cost 2)
            found_on_tray_elsewhere = False
            for s, t in sandwich_on_tray.items():
                 if tray_locations.get(t) != child_place:
                    # Check if sandwich s is suitable
                    is_suitable = (not needs_gluten_free) or (s in no_gluten_sandwiches_in_state)
                    if is_suitable:
                        min_child_cost = min(min_child_cost, 2)
                        found_on_tray_elsewhere = True
                        break # Found one on tray elsewhere, no need to check others for cost 2

            if found_on_tray_elsewhere:
                 total_cost += min_child_cost
                 continue # Move to the next child

            # Category 3: Suitable sandwich at the kitchen (Cost 3)
            found_at_kitchen = False
            for s in sandwich_at_kitchen:
                 # Check if sandwich s is suitable
                 is_suitable = (not needs_gluten_free) or (s in no_gluten_sandwiches_in_state)
                 if is_suitable:
                     min_child_cost = min(min_child_cost, 3)
                     found_at_kitchen = True
                     break # Found one at kitchen, no need to check others for cost 3

            if found_at_kitchen:
                 total_cost += min_child_cost
                 continue # Move to the next child

            # Category 4: Needs to be made (Cost 4) or Penalty (Cost 1000)
            if min_child_cost == float('inf'): # No suitable sandwich found in categories 1, 2, or 3
                 # Check if a suitable sandwich can be made
                 has_notexist_sandwich = len(sandwich_not_exist) > 0

                 can_make_suitable = False
                 if has_notexist_sandwich:
                     if needs_gluten_free:
                         # Need specific gluten-free ingredients at kitchen
                         if has_ng_bread_at_kitchen and has_ng_content_at_kitchen:
                             can_make_suitable = True
                     else:
                         # Need any ingredients at kitchen
                         if has_any_bread_at_kitchen and has_any_content_at_kitchen:
                             can_make_suitable = True

                 if can_make_suitable:
                     min_child_cost = 4
                 else:
                     # If we cannot make a suitable sandwich, this child is currently stuck.
                     # Assign a large penalty.
                     min_child_cost = 1000 # Penalty

            total_cost += min_child_cost

        return total_cost
