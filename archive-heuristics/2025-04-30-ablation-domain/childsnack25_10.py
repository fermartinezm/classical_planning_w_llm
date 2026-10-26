from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper functions from Logistics example
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the necessary steps for each child: ensuring a suitable sandwich exists,
    getting a tray to the child's table, getting a sandwich onto a tray at the table,
    and finally serving the child. It aggregates costs for shared resources like trays
    and sandwich making.

    # Assumptions
    - Each child needs exactly one sandwich.
    - Allergic children require gluten-free sandwiches.
    - Non-allergic children can eat any sandwich.
    - Ingredients for making sandwiches (bread, content) are always available in the kitchen.
    - Trays can be moved between the kitchen and tables.
    - Sandwiches can be put on trays only in the kitchen (inferred from typical domain structure, though not explicitly stated in predicates). Let's assume put_on_tray happens in the kitchen.
    - Serving happens at the child's table.

    # Heuristic Initialization
    The heuristic extracts static information:
    - The set of all children, trays, sandwiches, and tables in the problem.
    - Which children are allergic to gluten.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  Identify all children who are currently waiting but not yet served. For each such child, determine their table and whether they are allergic to gluten.
    2.  Initialize the total heuristic cost to 0.
    3.  Add 1 to the cost for each unserved child (representing the final 'serve' action).
    4.  Identify the set of tables where unserved children are waiting.
    5.  For each table identified in step 4, check if there is currently any tray located at that table. If not, add 1 to the cost (representing a 'move_tray' action needed to bring a tray).
    6.  For each unserved child, check if there is a suitable sandwich (gluten status matches child's allergy) currently on a tray located at the child's table. A sandwich S is suitable for child C if C is not allergic OR S is a gluten-free sandwich. If no such suitable sandwich exists for this child on a tray at their table, add 1 to the cost (representing the need to get a suitable sandwich onto a tray at that table, e.g., via 'put_on_tray'). Note: This step counts per child, potentially overcounting if one 'put_on_tray' action serves multiple children at the same table needing sandwiches. This is a simplification for efficiency.
    7.  Calculate the number of new sandwiches that need to be made.
        - Count the total number of unserved children needing GF sandwiches.
        - Count the total number of unserved children who can eat non-GF sandwiches.
        - Count the number of existing GF sandwiches (not marked 'notexist' and marked 'no_gluten_sandwich').
        - Count the number of existing non-GF sandwiches (not marked 'notexist' and not marked 'no_gluten_sandwich').
        - Calculate how many *new* GF sandwiches are strictly required: `max(0, num_allergic_unserved - num_existing_gf)`. Add this count multiplied by 3 (cost for 'add_bread', 'add_content', 'assemble') to the heuristic.
        - Calculate how many *new* non-GF sandwiches are required for the non-allergic children, after potentially using any remaining existing non-GF sandwiches and any excess existing GF sandwiches. Add this count multiplied by 3 to the heuristic.
    8.  The total cost accumulated is the heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information about objects
        and child allergies.
        """
        self.goals = task.goals
        self.static = task.static

        # Extract all objects by type from initial state and goals
        self.all_children = set()
        self.all_trays = set()
        self.all_sandwiches = set()
        self.all_tables = set()
        self.all_places = set() # Includes kitchen

        # Helper to add objects based on predicate patterns
        def add_objects_from_facts(facts):
            for fact in facts:
                parts = get_parts(fact)
                if not parts: continue # Skip empty facts

                predicate = parts[0]
                if predicate == 'waiting' and len(parts) == 3:
                    self.all_children.add(parts[1])
                    self.all_tables.add(parts[2])
                    self.all_places.add(parts[2])
                elif predicate == 'at' and len(parts) == 3:
                    # Could be tray at kitchen or tray at table
                    # We need to distinguish trays from other objects that might be 'at' a place
                    # Assuming only trays are moved between kitchen/tables based on domain structure
                    # A more robust way would be to parse object types from the problem file
                    # For this domain, 'at' seems to apply primarily to trays and the robot (if one existed)
                    # Let's assume anything 'at' a place that isn't a child is a tray or similar movable object.
                    # Given the instance examples, 'at' is used for trays.
                    self.all_trays.add(parts[1])
                    self.all_places.add(parts[2])
                elif predicate == 'at_kitchen_bread' and len(parts) == 2:
                    pass # bread-portion objects, not directly used in heuristic counts
                elif predicate == 'at_kitchen_content' and len(parts) == 2:
                    pass # content-portion objects, not directly used in heuristic counts
                elif predicate == 'at_kitchen_sandwich' and len(parts) == 2:
                     self.all_sandwiches.add(parts[1])
                elif predicate == 'ontray' and len(parts) == 3:
                    self.all_sandwiches.add(parts[1])
                    self.all_trays.add(parts[2])
                elif predicate == 'has_bread' and len(parts) == 3:
                    self.all_sandwiches.add(parts[1])
                    # parts[2] is bread-portion
                elif predicate == 'has_content' and len(parts) == 3:
                    self.all_sandwiches.add(parts[1])
                    # parts[2] is content-portion
                elif predicate == 'is_assembled' and len(parts) == 2:
                    self.all_sandwiches.add(parts[1])
                elif predicate == 'no_gluten_sandwich' and len(parts) == 2:
                    self.all_sandwiches.add(parts[1])
                elif predicate == 'served' and len(parts) == 2:
                    self.all_children.add(parts[1])
                elif predicate == 'notexist' and len(parts) == 2:
                    self.all_sandwiches.add(parts[1])
                elif predicate == 'allergic_gluten' and len(parts) == 2:
                    self.all_children.add(parts[1])
                elif predicate == 'not_allergic_gluten' and len(parts) == 2:
                     self.all_children.add(parts[1])
                elif predicate == 'no_gluten_bread' and len(parts) == 2:
                    pass # bread-portion objects
                elif predicate == 'no_gluten_content' and len(parts) == 2:
                    pass # content-portion objects
                # Add 'kitchen' explicitly as it might not appear in 'at' facts in init
                self.all_places.add('kitchen')


        add_objects_from_facts(task.initial_state)
        add_objects_from_facts(task.goals)
        add_objects_from_facts(task.static) # Allergic status is static

        # Extract allergic children from static facts
        self.allergic_children = {
            get_parts(fact)[1]
            for fact in self.static
            if match(fact, "allergic_gluten", "*")
        }


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        h = 0

        # 1. Identify unserved children and their needs
        unserved_children = set() # Store as (child, table, needs_gf)
        for child in self.all_children:
            # Find the table the child is waiting at
            waiting_fact = None
            child_table = None
            # Efficiently check for the waiting fact in the current state
            for fact in state:
                if fact.startswith(f'(waiting {child} '):
                     parts = get_parts(fact)
                     if len(parts) == 3 and parts[1] == child: # Double check match
                        waiting_fact = fact
                        child_table = parts[2]
                        break # Assuming a child waits at only one table

            # Check if child is waiting and not served
            if waiting_fact and f'(served {child})' not in state:
                 needs_gf = child in self.allergic_children
                 unserved_children.add((child, child_table, needs_gf))

        N_unserved = len(unserved_children)

        # If no unserved children, goal is reached
        if N_unserved == 0:
            return 0

        # 3. Add cost for 'serve' actions
        h += N_unserved

        # 4. Identify tables that need service
        tables_needing_service = {table for _, table, _ in unserved_children}

        # 5. Count tables missing a tray
        tables_with_tray = set()
        for fact in state:
             if fact.startswith('(at '):
                 parts = get_parts(fact)
                 if len(parts) == 3 and parts[1] in self.all_trays:
                     tables_with_tray.add(parts[2])

        tables_missing_tray = tables_needing_service - tables_with_tray
        h += len(tables_missing_tray) # Cost for move_tray

        # 6. Count children who need a suitable sandwich delivered to their table
        children_needing_delivery = set()
        for child, table, needs_gf in unserved_children:
            # Check if a suitable sandwich exists on a tray at this child's table
            suitable_sandwich_found_at_table = False
            # Find trays at the child's table
            trays_at_table = {
                 get_parts(fact)[1]
                 for fact in state
                 if fact.startswith('(at ') and len(get_parts(fact)) == 3 and get_parts(fact)[2] == table and get_parts(fact)[1] in self.all_trays
            }

            if trays_at_table:
                # Find sandwiches on those trays
                sandwiches_on_trays_at_table = {
                    get_parts(fact)[1]
                    for fact in state
                    if fact.startswith('(ontray ') and len(get_parts(fact)) == 3 and get_parts(fact)[2] in trays_at_table
                }

                # Check suitability
                for s_at_table in sandwiches_on_trays_at_table:
                    is_gf_sandwich = f'(no_gluten_sandwich {s_at_table})' in state
                    if (not needs_gf) or (needs_gf and is_gf_sandwich):
                        suitable_sandwich_found_at_table = True
                        break # Found a suitable sandwich for this child at their table

            if not suitable_sandwich_found_at_table:
                children_needing_delivery.add(child)

        h += len(children_needing_delivery) # Cost for put_on_tray (conceptually per child)


        # 7. Count needed new sandwiches (considering gluten)
        N_allergic_unserved = sum(1 for _, _, needs_gf in unserved_children if needs_gf)
        N_non_allergic_unserved = N_unserved - N_allergic_unserved

        # Find existing sandwiches
        existing_sandwiches = {s for s in self.all_sandwiches if f'(notexist {s})' not in state}

        # Classify existing sandwiches by gluten status
        existing_gf_sandwiches = {s for s in existing_sandwiches if f'(no_gluten_sandwich {s})' in state}
        existing_nongf_sandwiches = existing_sandwiches - existing_gf_sandwiches

        N_existing_gf = len(existing_gf_sandwiches)
        N_existing_nongf = len(existing_nongf_sandwiches)

        # Calculate new GF sandwiches needed
        needed_new_gf = max(0, N_allergic_unserved - N_existing_gf)
        h += needed_new_gf * 3 # Cost for add_bread, add_content, assemble

        # Calculate new non-GF sandwiches needed for non-allergic children
        rem_non_allergic = N_non_allergic_unserved
        avail_nongf = N_existing_nongf
        avail_excess_gf = max(0, N_existing_gf - N_allergic_unserved) # GF sandwiches not needed by allergic children

        avail_total_for_non_allergic = avail_nongf + avail_excess_gf

        needed_new_nongf = max(0, rem_non_allergic - avail_total_for_non_allergic)
        h += needed_new_nongf * 3 # Cost for add_bread, add_content, assemble

        return h
