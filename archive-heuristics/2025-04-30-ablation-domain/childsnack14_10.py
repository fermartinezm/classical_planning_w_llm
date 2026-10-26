# from heuristics.heuristic_base import Heuristic # Assuming this base class is provided

from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and not empty
    if not isinstance(fact, str) or len(fact) < 2:
        return []
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It does this by summing up the estimated cost for each unserved
    child independently. The cost for a child is estimated based on how many
    major stages are still needed to get a suitable sandwich to them:
    1. Making a suitable sandwich (if none exists).
    2. Putting a suitable sandwich on a tray (if none is on a tray).
    3. Moving a tray to the child's table (if no tray is there).
    4. Performing the final serve action.

    # Assumptions
    - Each unserved child requires a suitable sandwich.
    - A suitable sandwich must be assembled and, if the child is allergic to gluten,
      it must be gluten-free.
    - A sandwich must be on a tray to be moved and served.
    - A tray must be at the child's table to serve the child.
    - Resources (bread, content, notexist sandwich objects, trays) are assumed
      to be available if needed for the heuristic calculation (i.e., we don't
      check resource counts beyond the existence of assembled/on-tray sandwiches).
      This simplifies computation and makes it suitable for greedy search.
    - The heuristic counts abstract "stages" (make, put, move, serve) rather than
      exact action counts (e.g., make is 3 actions). Each stage is counted as 1.
      This provides a simple progression measure.
    - Predicates like `is_assembled`, `is_gluten_free` (or `no_gluten_sandwich`),
      `ontray`, `at` (for trays), `served`, `waiting`, `allergic_gluten`,
      `not_allergic_gluten`, and potentially `at_kitchen_sandwich` are assumed
      to be the relevant predicates describing the state and goals, based on the
      provided examples. `at_kitchen_sandwich` is treated as indicating an
      assembled sandwich in the kitchen.

    # Heuristic Initialization
    The heuristic extracts static information from the task:
    - Which children are allergic to gluten (`allergic_gluten`, `not_allergic_gluten`).
    - Which table each child is waiting at (`waiting`).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Initialize the total heuristic value `h` to 0.
    2. Identify all assembled sandwiches (including those `at_kitchen_sandwich`),
       their gluten-free status, whether they are on trays, and the locations
       of trays by iterating through the state facts. Store this information
       efficiently (e.g., in sets).
    3. Identify all children who are waiting but not yet served by checking the
       goal conditions against the current state.
    4. For each unserved child:
       a. Determine the child's waiting table and allergy status using the
          pre-calculated static information.
       b. Initialize the child's estimated cost `child_h` to 0.
       c. Check if there exists *any* assembled sandwich in the current state
          that is suitable for this child (i.e., assembled, and gluten-free
          if the child is allergic). If not, add 1 to `child_h` (representing
          the "make sandwich" stage needed).
       d. Check if there exists *any* suitable assembled sandwich (as determined
          in step 4c) that is currently on *any* tray. If not, add 1 to `child_h`
          (representing the "put on tray" stage needed).
       e. Check if there exists *any* tray currently located at the child's
          waiting table. If not, add 1 to `child_h` (representing the "move tray"
          stage needed).
       f. Add 1 to `child_h` for the final "serve" action, which is always needed
          if the child is unserved.
       g. Add `child_h` to the total heuristic value `h`.
    5. Return the total heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting allergy and waiting information
        from the static facts.
        """
        self.goals = task.goals

        # Extract allergy information: child -> True if allergic, False otherwise
        self.allergy_info = {}
        for fact in task.static:
            if match(fact, "allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergy_info[child] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                self.allergy_info[child] = False

        # Extract waiting information: child -> table
        self.waiting_info = {}
        for fact in task.static:
             if match(fact, "waiting", "*", "*"):
                 child, table = get_parts(fact)[1:3]
                 self.waiting_info[child] = table

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all waiting children.
        """
        state = node.state
        h = 0

        # Identify relevant facts from the current state
        assembled_sandwiches = set() # set of sandwich names that are assembled
        gluten_free_sandwiches = set() # set of sandwich names that are gluten-free
        sandwiches_on_tray = set() # set of sandwich names that are on any tray
        trays_at_table = set() # set of table names that have a tray

        for fact in state:
            if match(fact, "is_assembled", "*"):
                assembled_sandwiches.add(get_parts(fact)[1])
            # Handle the predicate seen in the state example, assuming it implies assembled
            elif match(fact, "at_kitchen_sandwich", "*"):
                 assembled_sandwiches.add(get_parts(fact)[1])

            # Using predicate from state example, assuming it means is_gluten_free
            elif match(fact, "no_gluten_sandwich", "*"):
                 gluten_free_sandwiches.add(get_parts(fact)[1])

            elif match(fact, "ontray", "*", "*"):
                sandwiches_on_tray.add(get_parts(fact)[1])

            elif match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:3]
                 # Assuming tray objects start with "tray" based on instance examples
                 if obj.startswith("tray"):
                     trays_at_table.add(loc)

        # Identify unserved children
        unserved_children = set()
        for goal in self.goals:
            if match(goal, "served", "*"):
                if goal not in state:
                    unserved_children.add(get_parts(goal)[1])

        # Calculate heuristic cost for each unserved child
        for child in unserved_children:
            child_h = 0
            child_table = self.waiting_info.get(child) # Get table from static info

            # If a child in the goal is not in the waiting_info static facts,
            # it indicates an invalid problem instance according to typical PDDL structure.
            # However, we should handle this defensively. If the child isn't waiting
            # at a known table, they cannot be served by the standard actions.
            # In a real planner, this might mean the problem is unsolvable or
            # requires actions not considered by this heuristic.
            # For this heuristic, we'll assume valid instances where waiting children
            # are listed in static facts. If not found, the child_table will be None,
            # and the 'Need tray at table?' check will correctly add cost.
            # The allergy info might also be missing, default to not allergic.
            is_allergic = self.allergy_info.get(child, False)


            # --- Estimate stages needed for this child ---

            # Stage 1: Need suitable assembled sandwich?
            # Check if *any* assembled sandwich exists that is suitable for this child.
            suitable_assembled_exists = False
            for s in assembled_sandwiches:
                if is_allergic and s not in gluten_free_sandwiches:
                    continue # This assembled sandwich is not suitable for an allergic child
                suitable_assembled_exists = True
                break # Found at least one suitable assembled sandwich

            if not suitable_assembled_exists:
                child_h += 1 # Add cost for the "make sandwich" stage

            # Stage 2: Need suitable sandwich on tray?
            # Check if *any* suitable assembled sandwich is currently on *any* tray.
            # We iterate through sandwiches known to be on trays and check if they are suitable and assembled.
            suitable_ontray_exists = False
            for s in sandwiches_on_tray:
                 # A sandwich on a tray must be assembled in a valid plan sequence
                 if s in assembled_sandwiches:
                     if is_allergic and s not in gluten_free_sandwiches:
                         continue # This sandwich on a tray is not suitable
                     suitable_ontray_exists = True
                     break # Found at least one suitable sandwich on a tray

            if not suitable_ontray_exists:
                child_h += 1 # Add cost for the "put on tray" stage

            # Stage 3: Need tray at table?
            # Check if *any* tray is currently located at the child's specific table.
            # This check relies on child_table being correctly retrieved from static info.
            if child_table is not None and child_table not in trays_at_table:
                child_h += 1 # Add cost for the "move tray" stage
            # Note: If child_table is None, this stage cost is not added. This is
            # consistent with the idea that if the child isn't waiting at a table,
            # moving a tray to a table isn't a relevant step for serving *this* child
            # via the standard mechanism. However, this case might indicate an unsolvable
            # problem or a problem outside the heuristic's scope. Assuming valid instances.


            # Stage 4: Need to serve?
            # The final serve action is always needed if the child is unserved.
            child_h += 1 # Add cost for the "serve" action

            h += child_h

        return h
