from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import defaultdict

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
        return []
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

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    who are currently waiting and not yet served. It counts the necessary steps
    (make sandwich, put on tray, move tray, serve) for each unserved child,
    greedily assigning available sandwiches and trays to minimize the estimated cost
    for each child.

    # Assumptions
    - The primary goal is to serve all children specified in the goal state.
    - Children's locations and allergy statuses are static (from the initial state).
    - Bread, content, and 'notexist' sandwich objects are assumed to be available
      if needed to make a sandwich, up to the count of 'notexist' objects.
    - Tray capacity is not explicitly modeled; any tray can hold a needed sandwich
      for the heuristic calculation.
    - The heuristic greedily assigns the "most ready" available sandwich/tray
      combination to each unserved child in an arbitrary order.

    # Heuristic Initialization
    - Extracts the goal conditions (which children need to be served).
    - Extracts static information about children (allergy status and initial waiting location)
      from the task's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all children who are in the goal state as 'served' but are not
       currently 'served' in the current state. These are the unserved children.
    2. For each unserved child, determine their waiting location and allergy status
       using the static information stored during initialization.
    3. Count the available resources/items in the current state that can be used
       to serve children:
       - Sandwiches ready in the kitchen (`at_kitchen_sandwich`), noting if they are gluten-free.
       - Sandwiches already on trays (`ontray`), noting the tray and if the sandwich is gluten-free.
       - Trays at specific locations (`at ?t ?p`).
       - Available 'notexist' sandwich objects (representing potential to make new sandwiches).
    4. Initialize the total heuristic cost to 0.
    5. Iterate through the list of unserved children. For each child:
       - Initialize the cost for this child to 1 (for the final 'serve' action).
       - Check if a suitable sandwich is already on a tray that is *at the child's location*.
         - A sandwich is suitable if it's gluten-free for an allergic child, or any sandwich otherwise.
         - If found among available items: Mark the specific sandwich-on-tray and the tray at that location as used for this child's need. This child's need for delivery is met.
       - If not found at the child's location:
         - Add 1 to the child's cost (for a 'move_tray' action).
         - Check if a suitable sandwich is already on a tray *anywhere*.
           - If found among available items: Mark the specific sandwich-on-tray as used. This child's need for 'sandwich on tray' is met.
       - If not found on any tray:
         - Add 1 to the child's cost (for a 'put_on_tray' action).
         - Check if a suitable sandwich is available *in the kitchen*.
           - If found among available items: Mark the specific kitchen sandwich as used. This child's need for 'sandwich in kitchen' is met.
       - If not found in the kitchen:
         - Add 1 to the child's cost (for a 'make_sandwich' action).
         - Check if a 'notexist' sandwich object is available.
           - If available: Mark one 'notexist' object as used. This child's need for a 'made' sandwich is met (in a relaxed sense).
         - If not available: The heuristic still adds the cost, assuming a relaxed world where making is possible, or simply reflecting the steps needed even if resources are scarce.
       - Add the child's total estimated cost to the total heuristic cost.
    6. Return the total heuristic cost.

    This greedy approach prioritizes using items that require fewer subsequent actions,
    providing a reasonable estimate for a greedy best-first search.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store static child information: allergy status and initial waiting location.
        self.child_info = {} # child_name -> {'allergic': bool, 'location': place_name}

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == "allergic_gluten":
                child_name = parts[1]
                if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                self.child_info[child_name]['allergic'] = True
            elif predicate == "not_allergic_gluten":
                child_name = parts[1]
                if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                self.child_info[child_name]['allergic'] = False
            elif predicate == "waiting":
                child_name, location = parts[1:]
                if child_name not in self.child_info:
                    self.child_info[child_name] = {}
                self.child_info[child_name]['location'] = location

    def is_suitable(self, sandwich_name, is_child_allergic, state):
        """
        Checks if a sandwich is suitable for a child based on allergy status.
        Requires the current state to check the sandwich's gluten status.
        """
        if not is_child_allergic:
            return True # Non-allergic children can eat any sandwich
        else:
            # Allergic children need a gluten-free sandwich
            return f"(no_gluten_sandwich {sandwich_name})" in state

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all unserved children.
        """
        state = node.state  # Current world state.

        # 1. Identify unserved children
        unserved_children = [] # List of (child_name, location, is_allergic)
        for goal_fact in self.goals:
            if match(goal_fact, "served", "*"):
                child_name = get_parts(goal_fact)[1]
                if goal_fact not in state:
                    # This child needs to be served
                    info = self.child_info.get(child_name)
                    if info: # Ensure we have static info for this child
                         unserved_children.append((child_name, info['location'], info['allergic']))
                    # else: Should not happen in valid problems, child in goal but no static info.

        if not unserved_children:
            return 0 # Goal reached for all children in the goal list

        # 3. Count available resources/items in the current state
        available_kitchen_sandwiches = [] # List of (sandwich_name, is_gf)
        available_ontray_sandwiches = [] # List of (sandwich_name, tray_name, is_gf)
        available_trays_at_location = defaultdict(list) # location_name -> list of tray_names
        available_notexist_sandwiches = 0

        # Pre-determine gluten status of all sandwiches mentioned in the state
        sandwich_gluten_status = {}
        for fact in state:
             if match(fact, "no_gluten_sandwich", "*"):
                 s_name = get_parts(fact)[1]
                 sandwich_gluten_status[s_name] = True
             # Assume non-mentioned sandwiches or those explicitly not (no_gluten_sandwich) are regular
             # We populate this more fully when processing sandwich locations

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "at_kitchen_sandwich":
                s_name = parts[1]
                is_gf = sandwich_gluten_status.get(s_name, False) # Default to False if status unknown
                available_kitchen_sandwiches.append((s_name, is_gf))
            elif predicate == "ontray":
                s_name, t_name = parts[1:]
                is_gf = sandwich_gluten_status.get(s_name, False) # Default to False if status unknown
                available_ontray_sandwiches.append((s_name, t_name, is_gf))
            elif predicate == "at" and len(parts) == 3: # Check for (at ?obj ?loc)
                 obj_name, loc_name = parts[1:]
                 if obj_name.startswith("tray"): # Check if the object is a tray
                     available_trays_at_location[loc_name].append(obj_name)
            elif predicate == "notexist":
                available_notexist_sandwiches += 1

        # 4. & 5. Calculate heuristic cost by iterating through unserved children
        total_heuristic_cost = 0

        # Process children in a fixed order (e.g., alphabetical) for determinism,
        # although the greedy assignment makes the order matter for the specific
        # sandwiches/trays consumed in the heuristic calculation.
        unserved_children.sort()

        for child_name, child_location, is_child_allergic in unserved_children:
            child_cost = 0
            found_source = False

            # Cost for the final 'serve' action
            child_cost += 1

            # Check 1: Is a suitable sandwich on a tray already at the child's location?
            trays_at_P = available_trays_at_location.get(child_location, [])[:] # Copy list to modify safely
            ontray_copy = available_ontray_sandwiches[:] # Copy list to modify safely

            for s_info in ontray_copy:
                s_name, t_name, is_gf = s_info
                # Check if sandwich is suitable AND the tray is at the child's location AND the tray is available
                if (not is_child_allergic or is_gf) and t_name in trays_at_P:
                    # Found a suitable sandwich on a tray at the right location
                    available_ontray_sandwiches.remove(s_info) # Consume the sandwich-on-tray
                    available_trays_at_location[child_location].remove(t_name) # Consume the tray at this location
                    found_source = True
                    break # Move to the next child

            # If not found at the child's location, need to move a tray
            if not found_source:
                child_cost += 1 # Cost for 'move_tray'

                # Check 2: Is a suitable sandwich on a tray anywhere?
                ontray_copy = available_ontray_sandwiches[:] # Copy list again
                for s_info in ontray_copy:
                    s_name, t_name, is_gf = s_info
                    # Check if sandwich is suitable
                    if (not is_child_allergic or is_gf):
                        # Found a suitable sandwich on a tray somewhere
                        available_ontray_sandwiches.remove(s_info) # Consume the sandwich-on-tray
                        # Note: We don't consume the tray's location here, as it might be needed for another child
                        # at that *original* location. The move_tray cost covers getting it here.
                        found_source = True
                        break # Move to the next stage for this child

                # If not found on any tray, need to put one on a tray
                if not found_source:
                    child_cost += 1 # Cost for 'put_on_tray'

                    # Check 3: Is a suitable sandwich available in the kitchen?
                    kitchen_copy = available_kitchen_sandwiches[:] # Copy list
                    for s_info in kitchen_copy:
                        s_name, is_gf = s_info
                        # Check if sandwich is suitable
                        if (not is_child_allergic or is_gf):
                            # Found a suitable sandwich in the kitchen
                            available_kitchen_sandwiches.remove(s_info) # Consume the kitchen sandwich
                            found_source = True
                            break # Move to the next stage for this child

                    # If not found in the kitchen, need to make one
                    if not found_source:
                        child_cost += 1 # Cost for 'make_sandwich'

                        # Check 4: Can we make a sandwich? (Relaxed: check 'notexist')
                        # We assume enough bread/content if a 'notexist' object is available.
                        if available_notexist_sandwiches > 0:
                            available_notexist_sandwiches -= 1 # Consume a 'notexist' object
                            found_source = True
                        # Else: Cannot make. Cost is added anyway as an estimate of steps needed.

            # Add the cost for this child to the total
            total_heuristic_cost += child_cost

        # 6. Return the total estimated cost
        return total_heuristic_cost

