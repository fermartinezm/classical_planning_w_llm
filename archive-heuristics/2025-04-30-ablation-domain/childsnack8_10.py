from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts represented as strings
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at tray1 kitchen)" -> ["at", "tray1", "kitchen"]
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't try to match more args than parts in the fact
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It sums up the estimated costs for making necessary sandwiches,
    putting them on trays, moving trays to the tables where children are waiting,
    and finally serving the children. It provides a lower bound on the remaining
    actions by counting necessary steps for each unserved child and required
    resource movements (trays).

    # Assumptions
    - Children are waiting at specific tables, specified by `(waiting ?child ?place)`
      in the static facts.
    - Sandwiches can be made from bread and content in the kitchen.
    - Gluten-free sandwiches require gluten-free bread and content, specified by
      `(no_gluten_bread ?b)` and `(no_gluten_content ?c)` in static facts.
    - Allergic children, specified by `(allergic_gluten ?child)` in static facts,
      must receive gluten-free sandwiches.
    - Sandwiches need to be on a tray (`(ontray ?s ?t)`) to be moved and served.
    - Trays can be moved between the kitchen and tables (`(at ?t ?place)`).
    - Sufficient bread and content (including gluten-free options if needed)
      are available in the kitchen to make any required sandwiches.
    - Trays are available either in the kitchen or at tables.
    - The goal is always to serve a specific set of children (`(served ?child)`).

    # Heuristic Initialization
    The heuristic extracts the following static information from the task:
    - The set of children who are allergic to gluten.
    - The set of bread portions that are gluten-free.
    - The set of content portions that are gluten-free.
    - A mapping from each child to the table where they are waiting.
    - Lists of all objects by type (sandwiches, trays, children, places)
      from the task definition (assuming `task.objects` is available).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic value is calculated as follows:

    1.  Identify all children who are not yet served. This is done by checking
        which `(served ?child)` goals are not present in the current state.
        If all children are served, the heuristic is 0.
    2.  Categorize the unserved children into those who need a gluten-free
        sandwich (allergic) and those who need a regular sandwich (not allergic).
    3.  Count the total number of existing sandwiches in the current state. A
        sandwich `?s` exists if `(notexist ?s)` is not in the state. Categorize
        existing sandwiches as gluten-free or regular based on the presence
        of the `(no_gluten_sandwich ?s)` predicate in the state.
    4.  Calculate the deficit of suitable sandwiches: the number of gluten-free
        and regular sandwiches that still need to be *made* to satisfy the
        needs of the unserved children. Add this deficit count to the total
        heuristic (each represents a 'make' action).
    5.  Count the number of existing sandwiches that are currently on trays
        (`(ontray ?s ?t)` is in the state for some tray `?t`).
    6.  Calculate the number of sandwiches (equal to the total number of
        unserved children) that still need to be *put on a tray*. This is the
        total number of unserved children minus the number of suitable sandwiches
        already on trays. Add this count to the total heuristic (each represents
        a 'put on tray' action).
    7.  Identify the set of tables where unserved children are waiting. This is
        derived from the static `child_table` mapping for the unserved children.
    8.  Identify the set of tables that currently have a tray located at them
        (`(at ?t ?place)` is in the state for some tray `?t`).
    9.  Calculate the number of tables that have unserved children but *do not*
        currently have a tray. Add this count to the total heuristic (each
        represents a 'move tray' action needed to bring a tray to that table).
    10. Add the total number of unserved children to the heuristic. This
        represents the final 'serve' action needed for each child.
    11. The total heuristic value is the sum of the costs from steps 4, 6, 9, and 10.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and object lists.
        """
        self.goals = task.goals  # Goal conditions.
        self.static = task.static  # Facts that are not affected by actions.

        # Get lists of objects by type (assuming task object has an 'objects' attribute)
        # This is based on the structure implied by example heuristics.
        self.all_sandwiches = task.objects.get('sandwich', [])
        self.all_trays = task.objects.get('tray', [])
        self.all_children = task.objects.get('child', [])
        self.all_places = task.objects.get('place', []) # Includes kitchen and tables

        # Extract static information
        self.allergic_children = {
            get_parts(fact)[1] for fact in self.static
            if match(fact, "allergic_gluten", "*")
        }
        self.gf_bread = {
            get_parts(fact)[1] for fact in self.static
            if match(fact, "no_gluten_bread", "*")
        }
        self.gf_content = {
            get_parts(fact)[1] for fact in self.static
            if match(fact, "no_gluten_content", "*")
        }
        self.child_table = {
            get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static
            if match(fact, "waiting", "*", "*")
        }

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.
        total_cost = 0  # Initialize action cost counter.

        # 1. Identify unserved children
        unserved_children = {
            get_parts(goal)[1] for goal in self.goals
            if match(goal, "served", "*") and goal not in state
        }
        num_unserved = len(unserved_children)

        # If all children are served, the goal is reached.
        if num_unserved == 0:
            return 0

        # 2. Categorize unserved children by sandwich need
        unserved_gf_needed = {c for c in unserved_children if c in self.allergic_children}
        unserved_reg_needed = {c for c in unserved_children if c not in self.allergic_children}
        num_gf_needed = len(unserved_gf_needed)
        num_reg_needed = len(unserved_reg_needed)

        # 3. Count existing sandwiches by type
        existing_sandwiches = {
            s for s in self.all_sandwiches if f'(notexist {s})' not in state
        }
        avail_gf_sandwiches = {
            s for s in existing_sandwiches if f'(no_gluten_sandwich {s})' in state
        }
        # Sandwiches that are not GF are considered regular for non-allergic children
        avail_reg_sandwiches = {
             s for s in existing_sandwiches if f'(no_gluten_sandwich {s})' not in state
        }

        # 4. Calculate cost for making sandwiches
        # We need num_gf_needed GF sandwiches. We have len(avail_gf_sandwiches).
        make_gf = max(0, num_gf_needed - len(avail_gf_sandwiches))
        # We need num_reg_needed regular sandwiches. We have len(avail_reg_sandwiches).
        # This heuristic simplifies by assuming GF are primarily for allergic needs.
        make_reg = max(0, num_reg_needed - len(avail_reg_sandwiches))

        total_cost += make_gf + make_reg # Each make action costs 1

        # 5. Count existing sandwiches on trays
        ontray_sandwiches = {
            s for s in existing_sandwiches
            if any(match(fact, "ontray", s, "*") for fact in state)
        }

        # 6. Calculate cost for putting sandwiches on trays
        # We need num_unserved sandwiches on trays in total.
        put_on_tray = max(0, num_unserved - len(ontray_sandwiches))
        total_cost += put_on_tray # Each put on tray action costs 1

        # 7. Identify tables with unserved children
        tables_with_unserved = {self.child_table[c] for c in unserved_children}

        # 8. Identify tables that currently have a tray
        tables_with_tray = {
            get_parts(fact)[2] for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1] in self.all_trays
        }

        # 9. Calculate cost for moving trays
        # Each table with unserved children that doesn't have a tray needs one moved.
        # This counts the number of distinct tables needing a tray delivery.
        tables_need_tray_move = tables_with_unserved - tables_with_tray
        total_cost += len(tables_need_tray_move) # Each move tray action costs 1

        # 10. Calculate cost for serving
        # Each unserved child needs one serve action.
        total_cost += num_unserved # Each serve action costs 1

        return total_cost
