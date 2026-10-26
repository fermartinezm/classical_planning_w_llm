from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Define helper functions outside the class
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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class
class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all children.
    It sums up the estimated costs for making sandwiches, putting them on trays,
    moving trays to children's locations, and serving the children.

    # Assumptions
    - Enough bread, content, and sandwich objects (`notexist`) are available
      to make any required sandwich type if they are initially present.
    - Trays can hold multiple sandwiches.
    - A single tray move action can take a tray from any place to any other place.
    - The primary bottleneck steps are making sandwiches, putting them on trays,
      moving trays to the correct locations, and serving.

    # Heuristic Initialization
    - Extract static information: which children are allergic/not allergic,
      and where each child is waiting.
    - Identify all children, sandwiches, trays, and places from the task objects
      present in the initial state or defined as constants.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic is calculated by summing up the estimated costs for the major
    phases of the task based on the current state:

    1.  **Cost_Serve:** Count the number of children who are not yet `served`. Each requires a final `serve` action.
    2.  **Cost_Make:** Count the number of suitable sandwiches (gluten-free for allergic children, regular for non-allergic) that are *needed* for the unserved children but do *not* yet exist (i.e., are not `at_kitchen_sandwich` or `ontray`). Each such sandwich requires a `make_sandwich` action.
    3.  **Cost_Put:** Count the number of sandwiches that are currently `at_kitchen_sandwich`. These need to be put on trays using the `put_on_tray` action before they can be moved or served.
    4.  **Cost_Move:** Count the number of distinct locations where unserved children are waiting that do *not* currently have any tray (`at ?t ?p`). Each such location will likely require at least one `move_tray` action to bring a tray there.

    The total heuristic value is the sum of these four cost components:
    `h = Cost_Make + Cost_Put + Cost_Move + Cost_Serve`

    This heuristic is non-admissible because actions can contribute to satisfying multiple conditions (e.g., one tray move can help serve multiple children at the same location), but it provides a reasonable estimate of the remaining work by counting items/entities in intermediate states or locations.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract all objects by type from the initial state and static facts/constants
        # Children are identified by the 'waiting' predicate in the initial state
        self.all_children = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "waiting", "*", "*")}

        # Sandwiches are initially marked by 'notexist', but can also exist 'at_kitchen_sandwich' or 'ontray'
        initial_sandwiches = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "notexist", "*")}
        initial_sandwiches.update({get_parts(fact)[1] for fact in task.initial_state if match(fact, "at_kitchen_sandwich", "*")})
        initial_sandwiches.update({get_parts(fact)[1] for fact in task.initial_state if match(fact, "ontray", "*", "*")})
        self.all_sandwiches = list(initial_sandwiches)

        # Trays are identified by the 'at' predicate in the initial state
        self.all_trays = {get_parts(fact)[1] for fact in task.initial_state if match(fact, "at", "*", "*")}

        # Places are where trays are, where children wait, plus the kitchen constant
        initial_places = {get_parts(fact)[2] for fact in task.initial_state if match(fact, "at", "*", "*")}
        initial_places.update({get_parts(fact)[2] for fact in task.initial_state if match(fact, "waiting", "*", "*")})
        initial_places.add('kitchen') # kitchen is a constant place
        self.all_places = list(initial_places)


        # Extract static information about children (allergy status and waiting place)
        self.allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")}
        self.non_allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")}
        # Map child to their waiting place (from static facts)
        self.child_waiting_place = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static_facts if match(fact, "waiting", "*", "*")}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Cost_Serve: Number of unserved children
        served_children_in_state = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = [c for c in self.all_children if c not in served_children_in_state]
        cost_serve = len(unserved_children)

        if cost_serve == 0:
            return 0 # Goal state reached

        # 2. Cost_Make: Number of suitable sandwiches needed but not yet made
        needed_gf = sum(1 for c in unserved_children if c in self.allergic_children)
        needed_reg = sum(1 for c in unserved_children if c in self.non_allergic_children)

        # Count sandwiches that are already made (either at kitchen or ontray)
        available_gf_made = sum(1 for s in self.all_sandwiches if f"(no_gluten_sandwich {s})" in state and (f"(at_kitchen_sandwich {s})" in state or any(match(fact, "ontray", s, "*") for fact in state)))
        available_reg_made = sum(1 for s in self.all_sandwiches if f"(no_gluten_sandwich {s})" not in state and (f"(at_kitchen_sandwich {s})" in state or any(match(fact, "ontray", s, "*") for fact in state)))

        # Number of make actions needed is the deficit of made sandwiches vs needed ones
        cost_make = max(0, needed_gf - available_gf_made) + max(0, needed_reg - available_reg_made)

        # 3. Cost_Put: Number of sandwiches currently at the kitchen that need to be put on a tray
        # This is simply the count of sandwiches currently at the kitchen.
        cost_put = sum(1 for s in self.all_sandwiches if f"(at_kitchen_sandwich {s})" in state)

        # 4. Cost_Move: Number of locations with unserved children that need a tray delivered
        # Identify the distinct locations where unserved children are waiting.
        locations_with_unserved = set(self.child_waiting_place[c] for c in unserved_children)

        # Count how many of these locations currently have *no* tray.
        cost_move = sum(1 for loc in locations_with_unserved if not any(f"(at {t} {loc})" in state for t in self.all_trays))

        # Total heuristic is the sum of estimated actions for each stage.
        total_heuristic = cost_make + cost_put + cost_move + cost_serve

        return total_heuristic
