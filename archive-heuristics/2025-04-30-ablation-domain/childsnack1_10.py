from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions outside the class
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children
    specified in the goal. It counts the number of unserved children, the number
    of tables that need a tray, the number of sandwiches that need to be put on
    trays, and the number of sandwiches that need to be made, based on the
    current state and the requirements of the unserved children.

    # Assumptions
    - Each unserved child requires one serve action.
    - Each table with unserved children requires at least one tray to be present.
    - Each sandwich needed for an unserved child must be put on a tray.
    - Sandwiches that do not exist must be made.
    - The heuristic simplifies resource contention (e.g., robot location, tray capacity, ingredient availability beyond simple counts).
    - Gluten-free sandwiches are prioritized for allergic children. Surplus GF sandwiches can serve non-allergic children.

    # Heuristic Initialization
    - Extracts the set of children that need to be served from the task goals.
    - Extracts static information about children's allergies and waiting tables.
    - Stores this information for quick lookup during heuristic computation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of children that are in the goal but not yet served. Let this count be `N_unserved`.
    2. Initialize the heuristic cost with `N_unserved` (representing the final 'serve' action for each child).
    3. Group the unserved children by the table they are waiting at (using static facts). Identify the set of tables that need serving.
    4. Count the number of trays currently located at any of these needed tables.
    5. Calculate the number of tables that require a tray to be moved there: `max(0, number_of_needed_tables - number_of_trays_at_needed_tables)`. Add this to the heuristic cost (representing 'move_tray' actions).
    6. Count the total number of sandwiches currently on any tray in the state.
    7. Calculate the number of sandwiches that still need to be put on trays to serve the unserved children: `max(0, N_unserved - number_of_sandwiches_on_trays)`. Add this to the heuristic cost (representing 'put_on_tray' actions).
    8. Determine the type (gluten-free or not) and quantity of sandwiches required for the unserved children, considering their allergy status.
        - Count unserved allergic children (`N_allergic_unserved`). These need GF sandwiches.
        - Count unserved non-allergic children (`N_non_allergic_unserved`). These need any sandwich.
    9. Count the number of existing sandwiches in the state, categorized by gluten-free status.
    10. Calculate the number of GF sandwiches that must be made: `max(0, N_allergic_unserved - existing_gf_sandwiches)`.
    11. Calculate the number of non-GF sandwiches that must be made: `max(0, N_non_allergic_unserved - existing_any_sandwiches - max(0, existing_gf_sandwiches - N_allergic_unserved))` (using surplus GF for non-allergic).
    12. Sum the required GF and non-GF sandwiches that must be made (`total_must_make`). Add this to the heuristic cost (representing 'make_sandwich' actions).
    13. The total heuristic value is the sum of costs from steps 2, 5, 7, and 12.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Extract children that need to be served from the goals.
        self.goal_children = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "served", "child*")
        }

        # Extract static information about children: allergy status and waiting table.
        self.child_allergy = {}
        self.child_table = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if match(fact, "allergic_gluten", "child*"):
                self.child_allergy[parts[1]] = True
            elif match(fact, "not_allergic_gluten", "child*"):
                 self.child_allergy[parts[1]] = False
            elif match(fact, "waiting", "child*", "table*"):
                self.child_table[parts[1]] = parts[2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify unserved children
        unserved_children = {
            child for child in self.goal_children
            if f"(served {child})" not in state
        }
        N_unserved = len(unserved_children)

        # If all children in the goal are served, the heuristic is 0.
        if N_unserved == 0:
            return 0

        total_cost = N_unserved # Cost for the 'serve' actions

        # 3. Group unserved children by table and identify needed tables
        tables_needed = set()
        unserved_allergic_count = 0
        unserved_non_allergic_count = 0

        for child in unserved_children:
            table = self.child_table.get(child)
            if table: # Ensure table info exists
                tables_needed.add(table)
                # Use .get() with default False in case allergy info is missing for a child
                if self.child_allergy.get(child, False):
                    unserved_allergic_count += 1
                else:
                    unserved_non_allergic_count += 1
            # Note: If a child in the goal has no waiting/allergy info in static, they are ignored for heuristic calculation.
            # This is a limitation based on available static info.

        # 4. Count trays at needed tables
        trays_at_needed_tables = sum(
            1 for fact in state
            if match(fact, "at", "tray*", "*") and get_parts(fact)[2] in tables_needed
        )

        # 5. Calculate tables requiring tray move
        tables_requiring_tray_move = max(0, len(tables_needed) - trays_at_needed_tables)
        total_cost += tables_requiring_tray_move

        # 6. Count sandwiches on trays
        sandwiches_on_trays_count = sum(1 for fact in state if match(fact, "ontray", "sandw*", "tray*"))

        # 7. Calculate sandwiches that need to be put on trays
        need_put_on_tray = max(0, N_unserved - sandwiches_on_trays_count)
        total_cost += need_put_on_tray

        # 8. Determine sandwich requirements and 9. Count existing sandwiches
        # Infer existing sandwich objects from facts they appear in and are not marked 'notexist'
        existing_sandwiches = set()
        potential_sandwiches = set()
        for fact in state:
            parts = get_parts(fact)
            for part in parts:
                if part.startswith("sandw"):
                    potential_sandwiches.add(part)

        for s in potential_sandwiches:
             if f"(notexist {s})" not in state:
                 existing_sandwiches.add(s)

        existing_gf_sandwiches_count = sum(1 for s in existing_sandwiches if f"(no_gluten_sandwich {s})" in state)
        existing_any_sandwiches_count = len(existing_sandwiches) - existing_gf_sandwiches_count

        # 10. Calculate GF sandwiches that must be made
        must_make_gf = max(0, unserved_allergic_count - existing_gf_sandwiches_count)

        # 11. Calculate non-GF sandwiches that must be made (using surplus GF)
        surplus_gf = max(0, existing_gf_sandwiches_count - unserved_allergic_count)
        must_make_any = max(0, unserved_non_allergic_count - existing_any_sandwiches_count - surplus_gf)

        # 12. Sum required makes
        total_must_make = must_make_gf + must_make_any
        total_cost += total_must_make

        # 13. Total heuristic value is calculated iteratively above.

        return total_cost
