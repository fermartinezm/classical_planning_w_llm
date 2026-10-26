from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the estimated actions needed for each stage of the
    process: making sandwiches, putting them on trays, moving trays to children's
    locations, and finally serving the children.

    # Assumptions
    - Each unserved child requires one sandwich.
    - Allergic children require gluten-free sandwiches.
    - Sandwiches must be made in the kitchen, then put on a tray in the kitchen,
      then the tray must be moved to the child's location before serving.
    - The heuristic counts the minimum number of actions needed for each stage
      independently, summing them up. It considers the number of items currently
      in the state and the total number required based on unserved children.
    - It accounts for the pipeline: sandwiches made appear in the kitchen and
      can then be put on trays.

    # Heuristic Initialization
    - Stores the task object to access static facts (like waiting children and
      allergies) and initial state information if needed (though primarily uses
      dynamic state facts).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are waiting but not yet served. Count the total
       number (`N_unserved`) and the number of allergic ones (`N_allergic_unserved`).
       If `N_unserved` is 0, the goal is reached, heuristic is 0.
    2. Count the number of sandwiches currently existing (either in the kitchen
       or on a tray). Distinguish between gluten-free and regular sandwiches.
       Calculate `Available_GF` and `Available_Any`.
    3. Count the number of sandwiches currently on trays (`len(sandwiches_ontray)`)
       and the number currently in the kitchen (`len(sandwiches_kitchen)`).
    4. Identify the unique locations where unserved children are waiting (`waiting_places`).
    5. Identify the locations where trays are currently present (`current_tray_locations`).
    6. Calculate the heuristic value by summing estimated actions for each stage:
       a.  **Sandwiches to Make:** Estimate the number of `make_sandwich` actions.
           This is the maximum of (GF sandwiches needed - Available GF) and
           (Total sandwiches needed - Total Available). Total needed is `N_unserved`.
           Add this count to the heuristic.
       b.  **Sandwiches to Put on Tray:** Estimate the number of `put_on_tray` actions.
           This is the number of sandwiches that need to end up on trays (`N_unserved`)
           minus those already on trays, limited by the number of sandwiches that
           are or will be in the kitchen (currently in kitchen + those that need
           to be made). Add this count to the heuristic.
       c.  **Tray Movements:** Estimate the number of `move_tray` actions. This is
           the number of unique locations where unserved children are waiting but
           no tray is currently present. Add this count to the heuristic.
       d.  **Serving Actions:** Each unserved child needs one `serve` action. Add
           `N_unserved` to the heuristic.
    7. Return the total sum as the heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by storing the task."""
        self.task = task

    def get_parts(self, fact):
        """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
        return fact[1:-1].split()

    def match(self, fact, *args):
        """
        Check if a PDDL fact matches a given pattern.
        - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
        - `args`: The expected pattern (wildcards `*` allowed).
        - Returns `True` if the fact matches the pattern, else `False`.
        """
        parts = self.get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Unserved Children and their places
        # Waiting children facts are static
        all_waiting_children_and_places = [(self.get_parts(f)[1], self.get_parts(f)[2]) for f in self.task.static if self.match(f, "waiting", "*", "*")]
        # Served children facts are dynamic
        served_children = {self.get_parts(f)[1] for f in state if self.match(f, "served", "*")}

        unserved_children_and_places = [(c, p) for c, p in all_waiting_children_and_places if c not in served_children]
        N_unserved = len(unserved_children_and_places)

        # If no children are unserved, the goal is reached.
        if N_unserved == 0:
            return 0

        # Unique places where unserved children are waiting
        waiting_places = {p for c, p in unserved_children_and_places}

        # 2. Count Allergic Unserved Children (allergy info is static)
        allergic_children = {self.get_parts(f)[1] for f in self.task.static if self.match(f, "allergic_gluten", "*")}
        N_allergic_unserved = len([c for c, p in unserved_children_and_places if c in allergic_children])

        # 3. Count Available Sandwiches (dynamic facts)
        sandwiches_kitchen = {self.get_parts(f)[1] for f in state if self.match(f, "at_kitchen_sandwich", "*")}
        sandwiches_ontray = {self.get_parts(f)[1] for f in state if self.match(f, "ontray", "*", "*")}
        sandwiches_existing = sandwiches_kitchen.union(sandwiches_ontray) # Sandwiches that are either in kitchen or on a tray.
        Available_Any = len(sandwiches_existing)

        # Need to find which *existing* sandwiches are gluten-free.
        # The predicate (no_gluten_sandwich s) is in the state if s is GF.
        gf_sandwiches_existing = {s for s in sandwiches_existing if "(no_gluten_sandwich {})".format(s) in state}
        Available_GF = len(gf_sandwiches_existing)

        # 4. Count Current Tray Locations (dynamic facts)
        current_tray_locations = {self.get_parts(f)[2] for f in state if self.match(f, "at", "*", "*")}

        # --- Heuristic Calculation ---
        h = 0

        # Cost 1: Sandwiches to Make
        # We need N_allergic_unserved GF sandwiches.
        # We need N_unserved total sandwiches.
        # Number of GF sandwiches that must be made:
        make_gf = max(0, N_allergic_unserved - Available_GF)
        # Number of total sandwiches that must be made:
        make_total = max(0, N_unserved - Available_Any)
        # The total number of make actions is the maximum of these two requirements
        # because making a GF sandwich also contributes to the total count.
        h += max(make_gf, make_total)

        # Cost 2: Sandwiches to Put on Tray
        # We need N_unserved sandwiches to be on trays eventually.
        # Number already on trays: len(sandwiches_ontray).
        # Number that still need to be put on trays:
        put_needed = max(0, N_unserved - len(sandwiches_ontray))
        # Number of sandwiches that are or will be in the kitchen to be put on trays:
        # This is the number currently in the kitchen plus the number that need to be made.
        will_be_in_kitchen = len(sandwiches_kitchen) + max(make_gf, make_total)
        # We can only put as many as needed and as many as will be in the kitchen.
        h += min(will_be_in_kitchen, put_needed)

        # Cost 3: Tray Movements
        # Count unique places where unserved children are waiting that don't currently have a tray.
        h += len(waiting_places - current_tray_locations)

        # Cost 4: Serving Actions
        # Each unserved child needs one serve action.
        h += N_unserved

        return h

