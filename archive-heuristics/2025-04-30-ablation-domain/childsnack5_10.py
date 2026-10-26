from heuristics.heuristic_base import Heuristic

def get_parts(fact_string):
    """Extract the components of a PDDL fact string."""
    # Remove parentheses and split by space
    # Handle potential empty string or malformed fact
    if not isinstance(fact_string, str) or not fact_string or fact_string[0] != '(' or fact_string[-1] != ')':
        return []
    return fact_string[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children who are specified as goals. It sums the estimated cost for each
    such child independently. The cost for a child is estimated based on the
    steps needed to get a suitable sandwich onto a tray and deliver it to the
    child's table, plus the final serving action.

    # Assumptions
    - Each waiting child who is a goal needs one suitable sandwich delivered
      on a tray to their table.
    - A suitable sandwich is complete and meets the child's gluten requirements.
    - If a suitable sandwich doesn't exist, it must be made from available components.
    - The costs for obtaining a tray, moving it, assigning it, obtaining a sandwich,
      making a sandwich, and putting it on a tray are estimated as fixed values
      (typically 1 or 2 actions per step).
    - Resource availability (bread, content, trays) is checked, but if resources
      are insufficient, a fixed cost is still added, assuming resources will
      become available (non-admissible).
    - Goal children are assumed to be waiting at a table until served.

    # Heuristic Initialization
    - Extracts static information about which children are allergic to gluten
      and which bread/content portions are gluten-free.
    - Identifies the set of children that are goals (need to be served).

    # Step-By-Step Thinking for Computing Heuristic
    For each child that is a goal and is currently waiting and not yet served:
    1.  Initialize the cost for this child to 1 (for the final 'serve' action).
    2.  Find the table where the child is waiting.
    3.  Check if any tray is currently located at the child's waiting table. If not, add 1 to the child's cost (for a 'move-tray' action).
    4.  Check if any tray is currently assigned to this child. If not, add 1 to the child's cost (for an 'assign-tray' action).
    5.  Determine the child's gluten allergy status using static information.
    6.  Search for a 'suitable' sandwich in the current state. A sandwich is suitable if it is complete and meets the child's gluten requirement (is gluten-free if the child is allergic).
    7.  Check if any suitable sandwich is currently on any tray.
        - If yes: The sandwich is already on a tray. No additional cost for making or putting on tray.
        - If no: Need to get a suitable sandwich onto a tray.
            - Check if any suitable sandwich is available in the kitchen.
                - If yes: Add 1 to the child's cost (for a 'put-on-tray' action).
                - If no: Need to make a suitable sandwich.
                    - Check if suitable bread and content components are available in the kitchen (considering gluten requirements).
                    - If suitable components exist: Add 3 to the child's cost (for 'make-sandwich-bread', 'make-sandwich-content', and 'put-on-tray' actions).
                    - If suitable components do not exist: Add 3 to the child's cost (assume cost).
    8.  Add the calculated child cost to the total heuristic cost.
    9.  Return the total heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal children.
        """
        self.goals = task.goals

        # Extract static facts
        static_facts = task.static

        self.allergic_children = set()
        self.gluten_free_breads = set()
        self.gluten_free_contents = set()

        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "allergic_gluten":
                self.allergic_children.add(parts[1])
            elif predicate == "no_gluten_bread":
                self.gluten_free_breads.add(parts[1])
            elif predicate == "no_gluten_content":
                self.gluten_free_contents.add(parts[1])

        # Identify goal children
        self.goal_children = set()
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            if parts and parts[0] == "served":
                self.goal_children.add(parts[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to serve all waiting children who are goals.
        """
        state = node.state

        total_heuristic_cost = 0

        # Pre-process state for faster lookups
        state_facts_set = set(state) # Convert to set for faster 'in' checks

        # Collect information about children, sandwiches, trays, and kitchen items
        waiting_children_info = {} # child -> table
        complete_sandwiches = set()
        gluten_free_sandwiches = set()
        sandwiches_on_trays = set() # Store sandwich names
        sandwiches_in_kitchen = set() # Store sandwich names
        breads_in_kitchen = set() # Store bread names
        contents_in_kitchen = set() # Store content names
        trays_at_tables = {} # table -> set of trays
        trays_assigned_to_children = {} # child -> tray

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "waiting":
                # Only track waiting status for children who are goals
                if parts[1] in self.goal_children:
                    child, table = parts[1], parts[2]
                    waiting_children_info[child] = table
            elif predicate == "is_complete":
                complete_sandwiches.add(parts[1])
            elif predicate == "is_gluten_free":
                gluten_free_sandwiches.add(parts[1])
            elif predicate == "ontray":
                sandwiches_on_trays.add(parts[1])
            elif predicate == "at_kitchen_sandwich":
                sandwiches_in_kitchen.add(parts[1])
            elif predicate == "at_kitchen_bread":
                breads_in_kitchen.add(parts[1])
            elif predicate == "at_kitchen_content":
                contents_in_kitchen.add(parts[1])
            elif predicate == "at" and len(parts) == 3 and parts[1].startswith("tray"):
                 # Assuming locations starting with 'table' are tables
                 location = parts[2]
                 if location.startswith("table"):
                     tray = parts[1]
                     trays_at_tables.setdefault(location, set()).add(tray)
            elif predicate == "has_child":
                 tray, child = parts[1], parts[2]
                 trays_assigned_to_children[child] = tray # Assuming one tray per child assignment

        # Calculate cost for each child that is a goal and needs service
        for child in self.goal_children:
            # Check if the child is already served
            if f"(served {child})" in state_facts_set:
                continue # This child is served, cost is 0

            # Check if the child is waiting. If a goal child is not waiting,
            # this heuristic assumes it's not currently on the path it estimates.
            # We only estimate costs for children currently waiting.
            if child not in waiting_children_info:
                 continue # Skip this child

            table = waiting_children_info[child]
            child_cost = 1 # Cost for the final 'serve' action

            # 3. Tray Location Check
            # Check if any tray is at the child's table
            if table not in trays_at_tables or not trays_at_tables[table]:
                 child_cost += 1 # Need to move a tray

            # 4. Tray Assignment Check
            # Check if a tray is assigned to this child
            if child not in trays_assigned_to_children:
                 child_cost += 1 # Need to assign a tray

            # 5. Sandwich Status Check
            is_allergic = child in self.allergic_children

            # Find a suitable sandwich that exists anywhere
            suitable_sandwich_on_tray = False
            suitable_sandwich_in_kitchen = False

            # Check sandwiches on trays
            for s in sandwiches_on_trays:
                if s in complete_sandwiches and (not is_allergic or s in gluten_free_sandwiches):
                    suitable_sandwich_on_tray = True
                    break # Found one on a tray, this is the best case

            if not suitable_sandwich_on_tray:
                # Check sandwiches in kitchen
                for s in sandwiches_in_kitchen:
                     if s in complete_sandwiches and (not is_allergic or s in gluten_free_sandwiches):
                         suitable_sandwich_in_kitchen = True
                         break # Found one in kitchen

            if not suitable_sandwich_on_tray and not suitable_sandwich_in_kitchen:
                 # Need to make a sandwich
                 suitable_components_available = False
                 # Check for suitable bread and content in kitchen
                 for b in breads_in_kitchen:
                     if not is_allergic or b in self.gluten_free_breads:
                         for c in contents_in_kitchen:
                             if not is_allergic or c in self.gluten_free_contents:
                                 suitable_components_available = True
                                 break # Found suitable bread and content
                         if suitable_components_available:
                             break # Found suitable bread and content

                 if suitable_components_available:
                     child_cost += 3 # make-bread, make-content, put-on-tray
                 else:
                     child_cost += 3 # Assume cost even if components not immediately available

            elif suitable_sandwich_in_kitchen:
                 child_cost += 1 # put-on-tray

            # If suitable_sandwich_on_tray is True, no cost added here for sandwich part

            total_heuristic_cost += child_cost

        return total_heuristic_cost
