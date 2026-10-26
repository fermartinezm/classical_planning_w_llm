from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    Estimates the number of actions needed to serve all children.
    The heuristic is the sum of estimated costs for different stages:
    1. Serving: Cost for the final 'serve' action for each unserved child.
    2. Making: Cost for 'make_sandwich' for sandwiches that don't exist but are needed.
    3. Putting on Tray: Cost for 'put_on_tray' for sandwiches currently in the kitchen.
    4. Moving Tray to Child's Location: Cost for 'move_tray' to bring a tray to places with waiting children that lack one.
    5. Moving Tray to Kitchen: Cost to move a tray to the kitchen if needed for 'put_on_tray'.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts."""
        self.goals = task.goals
        self.static_facts = task.static

        # Extract static info: child allergy status, waiting locations
        self.allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "allergic_gluten", "*")}
        self.not_allergic_children = {get_parts(fact)[1] for fact in self.static_facts if match(fact, "not_allergic_gluten", "*")}
        self.all_children = self.allergic_children | self.not_allergic_children

        # Map child to their waiting location (static)
        self.waiting_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in self.static_facts if match(fact, "waiting", "*", "*")}

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        total_cost = 0

        # --- Extract current state information ---
        served_children = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}
        unserved_children = self.all_children - served_children
        num_unserved = len(unserved_children)

        # If goal is reached, heuristic is 0
        if num_unserved == 0:
            return 0

        # Separate unserved children by allergy
        unserved_allergic = {c for c in unserved_children if c in self.allergic_children}
        unserved_reg = {c for c in unserved_children if c in self.not_allergic_children}

        # Find locations of trays
        tray_locations = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("tray")}
        places_with_trays = set(tray_locations.values())

        # Find places with waiting unserved children
        places_with_waiting = {self.waiting_locations[c] for c in unserved_children}

        # Count sandwiches by state (kitchen vs ontray)
        sandwiches_at_kitchen = {get_parts(fact)[1] for fact in state if match(fact, "at_kitchen_sandwich", "*")}
        sandwiches_ontray = {get_parts(fact)[1] for fact in state if match(fact, "ontray", "*", "*")}
        sandwiches_made = sandwiches_at_kitchen.union(sandwiches_ontray)

        # Count gluten-free sandwiches among those made
        gf_sandwiches_made = {s for s in sandwiches_made if f"(no_gluten_sandwich {s})" in state}
        reg_sandwiches_made = sandwiches_made - gf_sandwiches_made

        # --- Calculate heuristic components ---

        # Component 1: Cost of the final 'serve' action for each unserved child
        total_cost += num_unserved

        # Component 2: Cost of 'make_sandwich' for sandwiches that don't exist but are needed
        # Estimate needed makes based on the deficit of available sandwiches vs unserved children
        num_make_gf = max(0, len(unserved_allergic) - len(gf_sandwiches_made))
        num_make_reg = max(0, len(unserved_reg) - len(reg_sandwiches_made))
        total_cost += num_make_gf + num_make_reg

        # Component 3: Cost of 'put_on_tray' for sandwiches currently at kitchen
        total_cost += len(sandwiches_at_kitchen)

        # Component 4: Cost of 'move_tray' to bring a tray to places needing one
        places_needing_tray = places_with_waiting - places_with_trays
        total_cost += len(places_needing_tray)

        # Component 5: Cost of 'move_tray' to bring a tray to the kitchen if needed for put_on_tray
        # This is needed if there are sandwiches in the kitchen that need to be put on a tray,
        # and there is currently no tray at the kitchen.
        if len(sandwiches_at_kitchen) > 0 and 'kitchen' not in places_with_trays:
             total_cost += 1

        return total_cost
