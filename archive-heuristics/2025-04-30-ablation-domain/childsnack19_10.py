from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Assumes fact is like "(predicate arg1 arg2)"
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all
    unserved children. It calculates the minimum estimated cost for each
    unserved child independently and sums these costs. The cost for a child
    depends on the current state of suitable sandwiches and trays.

    # Assumptions
    - Children requiring service are those listed in the task goals with the
      `(served childX)` predicate.
    - Each child waits at a specific table, as defined by the static
      `(waiting childX tableY)` facts. This location is assumed constant
      until the child is served.
    - Children may be allergic to gluten, as defined by static
      `(allergic_gluten childX)` facts. Allergic children require gluten-free
      sandwiches, indicated by the state fact `(no_gluten_sandwich sandwK)`.
      Non-allergic children can eat any sandwich.
    - Sandwiches are made in the kitchen, put on trays in the kitchen, and
      trays are moved from the kitchen to tables. There is no explicit action
      to move trays back from tables to the kitchen in the provided domain
      description. The heuristic assumes that if a tray is needed in the
      kitchen for making/putting actions, one will eventually be available
      (either initially present or implicitly freed/returned).
    - Ingredients (bread and content) are needed in the kitchen to make
      sandwiches. Gluten-free ingredients are needed for gluten-free sandwiches.
      The heuristic checks for the *existence* of suitable ingredients in the
      kitchen, assuming they can be combined.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the task goals.
    - Extracts static facts to map children to their waiting tables and identify
      children with gluten allergies.
    - Infers the names of all objects (children, trays, sandwiches, bread,
      content, tables) by parsing terms in the initial state and static facts,
      filtering out known predicates and constants.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic calculates the total cost as follows:

    1.  Initialize total heuristic cost to 0.
    2.  Identify resource availability in the current state:
        -   Is there any tray currently located in the kitchen?
        -   Are there any bread and content ingredients in the kitchen?
        -   Are there any *gluten-free* bread and content ingredients in the kitchen (needed for allergic children)?
    3.  Identify the set of children that need to be served (from task goals).
    4.  For each child that needs to be served:
        a.  If the child is already served (fact `(served childX)` is in the state), the cost for this child is 0. Continue to the next child.
        b.  If the child is not served:
            i.  Determine the table where the child is waiting (from static facts).
            ii. Determine if the child requires a gluten-free sandwich (from static facts).
            iii. Initialize the minimum estimated cost for this child to infinity.
            iv. Check the state of suitable sandwiches in decreasing order of readiness:
                -   **Path 1 (Cost 1):** Is there a suitable sandwich `sandwK` on a tray `trayZ` that is currently at the child's table (`(ontray sandwK trayZ)` and `(at trayZ tableY)` in state)? If yes, the child can be served directly. Set minimum cost to 1.
                -   **Path 2 (Cost 3):** If Path 1 is not available, is there a suitable sandwich `sandwK` in the kitchen (`(at_kitchen_sandwich sandwK)` in state)? If yes, AND there is a tray available in the kitchen, the sandwich can be put on a tray, the tray moved to the table, and the child served. Set minimum cost to 3.
                -   **Path 3 (Cost 4):** If Paths 1 and 2 are not available, are there suitable ingredients (GF if needed, any otherwise) in the kitchen, AND is there a tray available in the kitchen? If yes, a suitable sandwich can be made, put on a tray, moved to the table, and the child served. Set minimum cost to 4.
                -   **Path 4 (Base Cost 5):** If none of the above paths are immediately available (e.g., due to missing trays in the kitchen or lack of suitable ingredients), assign a base cost of 5. This represents the effort needed to potentially unblock the process or complete it from a less ideal starting point.
            v.  Add the determined minimum cost for this child to the total heuristic cost.
    5.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal information, static facts,
        and object names.
        """
        self.goals = task.goals
        self.static = task.static

        # Known constants and predicates to filter out when extracting objects
        # These are based on the provided domain and instance examples.
        known_constants = {'kitchen'}
        known_predicates = {
            'at', 'at_kitchen_bread', 'at_kitchen_content', 'not_allergic_gluten',
            'waiting', 'notexist', 'served', 'allergic_gluten', 'no_gluten_bread',
            'no_gluten_content', 'no_gluten_sandwich', 'ontray', 'at_kitchen_sandwich'
        }

        # Extract all objects by type from initial state and static facts
        # This is an inference based on naming conventions and appearance in facts.
        all_terms = set()
        all_facts = set(task.initial_state) | set(task.static)
        for fact in all_facts:
            parts = get_parts(fact)
            all_terms.update(parts)

        # Filter terms to get potential objects
        potential_objects = all_terms - known_constants - known_predicates

        # Categorize potential objects by prefix
        self.all_children = {obj for obj in potential_objects if obj.startswith('child')}
        self.all_trays = {obj for obj in potential_objects if obj.startswith('tray')}
        self.all_sandwiches = {obj for obj in potential_objects if obj.startswith('sandw')}
        self.all_breads = {obj for obj in potential_objects if obj.startswith('bread')}
        self.all_contents = {obj for obj in potential_objects if obj.startswith('content')}
        self.all_tables = {obj for obj in potential_objects if obj.startswith('table')}


        # Map children to their waiting tables (from static facts)
        self.child_waiting_table = {}
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'waiting':
                _, child, table = parts
                self.child_waiting_table[child] = table

        # Determine which children are allergic to gluten (from static facts)
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static if get_parts(fact)[0] == 'allergic_gluten'
        }

        # Identify the children that are goals (need to be served)
        self.served_children_goals = {
            get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == 'served'
        }


    def is_suitable(self, sandw, child, state):
        """
        Check if a sandwich is suitable for a child based on gluten allergy.
        Requires the current state to check for the no_gluten_sandwich fact.
        """
        # If child is not allergic, any sandwich is suitable
        if child not in self.allergic_children:
            return True
        # If child is allergic, sandwich must be gluten-free
        return f'(no_gluten_sandwich {sandw})' in state

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all unserved children.
        """
        state = node.state
        total_heuristic_cost = 0

        # Pre-calculate resource availability checks for efficiency
        has_tray_kitchen = any(f'(at {tray} kitchen)' in state for tray in self.all_trays)
        has_any_bread_kitchen = any(f'(at_kitchen_bread {bread})' in state for bread in self.all_breads)
        has_any_content_kitchen = any(f'(at_kitchen_content {content})' in state for content in self.all_contents)
        has_gf_bread_kitchen = any(f'(at_kitchen_bread {bread})' in state and f'(no_gluten_bread {bread})' in self.static for bread in self.all_breads)
        has_gf_content_kitchen = any(f'(at_kitchen_content {content})' in state and f'(no_gluten_content {content})' in self.static for content in self.all_contents)


        # Iterate through children who need to be served according to the goals
        for child in self.served_children_goals:
             # If child is already served, cost is 0 for this child
             if f'(served {child})' in state:
                 continue

             # Child is unserved. Estimate cost for this child.
             # Get the table the child is waiting at from static facts.
             # We assume the child remains waiting at this table until served.
             table = self.child_waiting_table.get(child)
             # This check handles cases where a child in goals might not be in static waiting facts,
             # although in valid problems for this domain, goals should correspond to waiting children.
             if table is None:
                 # This indicates an unexpected problem definition. Assign a high cost.
                 total_heuristic_cost += 1000
                 continue

             needs_gf = child in self.allergic_children
             child_cost = float('inf') # Minimum cost for this specific child

             # --- Check possible paths to serve the child ---

             # Path 1 (Cost 1): Suitable sandwich already on a tray at the child's table?
             # Action: serve_child
             found_on_tray_at_table = False
             for fact in state:
                 parts = get_parts(fact)
                 if parts[0] == 'ontray':
                     sandw, tray = parts[1], parts[2]
                     if f'(at {tray} {table})' in state:
                         if self.is_suitable(sandw, child, state):
                             found_on_tray_at_table = True
                             break # Found one suitable sandwich on a tray at the table

             if found_on_tray_at_table:
                 child_cost = 1

             # Path 2 (Cost 3): Suitable sandwich available in the kitchen?
             # Actions: put_on_tray, move_tray, serve_child
             if child_cost == float('inf'): # Only check if faster path not found
                 found_kitchen_sandwich = False
                 for fact in state:
                     parts = get_parts(fact)
                     if parts[0] == 'at_kitchen_sandwich':
                         sandw = parts[1]
                         if self.is_suitable(sandw, child, state):
                             found_kitchen_sandwich = True
                             break # Found one suitable sandwich in the kitchen

                 if found_kitchen_sandwich and has_tray_kitchen:
                     child_cost = 3

             # Path 3 (Cost 4): Suitable ingredients available in the kitchen to make a sandwich?
             # Actions: make_sandwich, put_on_tray, move_tray, serve_child
             if child_cost == float('inf'): # Only check if faster paths not found
                 can_make_suitable = False
                 if needs_gf:
                     if has_gf_bread_kitchen and has_gf_content_kitchen:
                         can_make_suitable = True
                 else:
                     if has_any_bread_kitchen and has_any_content_kitchen:
                         can_make_suitable = True

                 if can_make_suitable and has_tray_kitchen:
                     child_cost = 4

             # Path 4 (Base Cost 5): Blocked or requires resources/unmodeled actions
             # If none of the above paths are immediately available, it means
             # either resources (like a tray in the kitchen or suitable ingredients)
             # are missing, or no suitable sandwich exists in any stage.
             # Assign a base cost representing the effort needed to potentially unblock or
             # complete the process from a less ideal state.
             if child_cost == float('inf'):
                 child_cost = 5 # Base cost

             total_heuristic_cost += child_cost

        return total_heuristic_cost
