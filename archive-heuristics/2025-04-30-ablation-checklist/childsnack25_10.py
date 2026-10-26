from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty facts or malformed strings gracefully, though PDDL facts are structured.
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class ChildsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Childsnack domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It counts the number of unserved children and estimates the steps needed
    per child: make sandwich (if not enough made), put on tray, move tray, serve.
    It accounts for gluten allergies when estimating sandwich making needs.

    # Assumptions (Relaxations for efficiency):
    - Ingredients (bread, content) and sandwich slots (`notexist`) are available
      when needed to make sandwiches, although the heuristic counts how many
      *need* to be made based on current stock.
    - Trays are available when needed to put sandwiches on.
    - Trays can be moved directly to any child's location in one 'move_tray' action.
    - The cost of each relevant action (make, put, move, serve) is 1.

    # Heuristic Initialization
    - Extracts static information about children (allergy status, waiting place)
      and identifies all possible sandwich and tray objects from the initial state and goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are waiting but not yet served.
    2. Separate unserved children into those who are allergic to gluten and those who are not.
    3. Count the number of available made sandwiches (both gluten-free and any type)
       in the kitchen or on trays.
    4. Calculate how many new gluten-free sandwiches need to be made to satisfy
       allergic children, considering available gluten-free sandwiches.
    5. Calculate how many new *any* sandwiches need to be made to satisfy
       non-allergic children, considering all available sandwiches (including
       any leftover gluten-free ones after allergic children are accounted for).
    6. The total number of sandwiches that need to be made is the sum of steps 4 and 5.
    7. Each unserved child requires a sequence of actions:
       - Make sandwich (accounted for in step 6, aggregated).
       - Put sandwich on tray (1 action per child).
       - Move tray to child's location (1 action per child).
       - Serve sandwich (1 action per child).
    8. The heuristic value is the sum of the estimated 'make' actions (step 6)
       plus 3 actions (put, move, serve) for each unserved child.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and object lists.
        """
        self.goals = task.goals

        # Extract static information about children
        self.allergic_children = set()
        self.nongluten_children = set()
        self.child_waiting_place = {} # Map child to their waiting place
        self.all_children = set()
        self.all_sandwiches = set()
        self.all_trays = set()
        self.all_places = set() # Including kitchen

        # Process static facts
        for fact in task.static:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.allergic_children.add(parts[1])
                self.all_children.add(parts[1])
            elif predicate == "not_allergic_gluten":
                self.nongluten_children.add(parts[1])
                self.all_children.add(parts[1])
            # Waiting facts are often in init, but sometimes static? Check both.
            # For robustness, let's extract waiting places from initial state as well.

        # Process initial state and goals to get object lists and waiting places
        # We iterate through initial state and goals to find all objects of relevant types
        # mentioned in key predicates.
        facts_to_scan = set(task.initial_state) | set(task.goals) | set(task.static)
        for fact in facts_to_scan:
             parts = get_parts(fact)
             if not parts: continue
             predicate = parts[0]
             if predicate == "waiting":
                 child, place = parts[1], parts[2]
                 self.child_waiting_place[child] = place
                 self.all_children.add(child)
                 self.all_places.add(place)
             elif predicate == "served":
                 self.all_children.add(parts[1])
             elif predicate in ["notexist", "at_kitchen_sandwich", "no_gluten_sandwich"]:
                 self.all_sandwiches.add(parts[1])
             elif predicate == "ontray":
                 sandwich, tray = parts[1], parts[2]
                 self.all_sandwiches.add(sandwich)
                 self.all_trays.add(tray)
             elif predicate == "at":
                 tray, place = parts[1], parts[2]
                 self.all_trays.add(tray)
                 self.all_places.add(place)

        # Add kitchen constant if not already found (it's a place)
        self.all_places.add('kitchen')

        # Convert sets to frozensets for efficiency if needed elsewhere,
        # but sets are fine for iteration/membership checking here.

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # 1. Identify unserved children
        served_children = {c for c in self.all_children if f'(served {c})' in state}
        unserved_children = self.all_children - served_children

        # If all children are served, the goal is reached, heuristic is 0.
        if not unserved_children:
            return 0

        # 2. Separate unserved children by allergy status
        unserved_allergic = unserved_children.intersection(self.allergic_children)
        unserved_nongluten = unserved_children.intersection(self.nongluten_children)

        num_allergic_unserved = len(unserved_allergic)
        num_nongluten_unserved = len(unserved_nongluten)
        num_unserved = num_allergic_unserved + num_nongluten_unserved # Should be equal to len(unserved_children)

        # 3. Count available made sandwiches
        # A sandwich is "made" if it's in the kitchen or on a tray.
        # We need to iterate through all possible sandwich objects to check their status.
        avail_gf_sandwiches = 0
        avail_any_sandwiches = 0 # Includes GF ones

        for s in self.all_sandwiches:
            is_made = False
            # Check if sandwich is in kitchen
            if f'(at_kitchen_sandwich {s})' in state:
                is_made = True
            # Check if sandwich is on any tray
            if not is_made: # Avoid double counting if a sandwich could somehow be both (shouldn't happen in this domain)
                 for t in self.all_trays:
                     if f'(ontray {s} {t})' in state:
                         is_made = True
                         break # Found it on a tray, no need to check other trays

            if is_made:
                avail_any_sandwiches += 1
                if f'(no_gluten_sandwich {s})' in state:
                    avail_gf_sandwiches += 1

        # 4. Calculate how many new GF sandwiches need to be made
        # Allergic children *must* get GF. Use available GF sandwiches first.
        needed_gf_make = max(0, num_allergic_unserved - avail_gf_sandwiches)

        # 5. Calculate how many new *any* sandwiches need to be made for non-allergic
        # Non-allergic children can use any sandwich.
        # Available sandwiches for non-allergic are all made sandwiches minus those
        # hypothetically reserved for allergic children (which must be GF).
        # Note: This is a simplification. A GF sandwich could be used for a non-allergic
        # child even if an allergic child is unserved, but this heuristic assumes
        # GF sandwiches are prioritized for allergic children.
        available_for_nongluten = max(0, avail_any_sandwiches - num_allergic_unserved)
        needed_nongf_make = max(0, num_nongluten_unserved - available_for_nongluten)

        # 6. Total sandwiches to make
        total_make = needed_gf_make + needed_nongf_make

        # 7. Estimate total actions
        # Each unserved child needs:
        # - 1 'make' action (aggregated across all children needing new sandwiches)
        # - 1 'put_on_tray' action
        # - 1 'move_tray' action (assuming tray is not already at location)
        # - 1 'serve' action

        # The 'make' cost is `total_make`.
        # The 'put_on_tray', 'move_tray', and 'serve' costs are 1 each per unserved child.
        # Total cost = total_make + num_unserved (put) + num_unserved (move) + num_unserved (serve)

        heuristic_cost = total_make + 3 * num_unserved

        return heuristic_cost

