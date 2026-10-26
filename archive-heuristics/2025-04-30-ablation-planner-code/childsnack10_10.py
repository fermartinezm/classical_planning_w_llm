from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper function to extract parts of a PDDL fact string
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty string or non-fact string gracefully, though PDDL facts are expected.
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

# Helper function to match a PDDL fact against a pattern
def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
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
    This heuristic estimates the number of actions required to serve all waiting
    children. It counts the minimum necessary actions of each type: making
    sandwiches, putting them on trays, moving trays to children's locations,
    and serving the children.

    # Assumptions
    - Each unserved child requires one sandwich of the appropriate type (gluten-free
      if allergic, any otherwise).
    - Sandwiches must be made in the kitchen, put on a tray in the kitchen,
      the tray must be moved to the child's location, and then the child can be served.
    - A tray at a location can potentially serve all children waiting at that location.
    - Ingredient availability is not strictly checked beyond the initial counts
      when calculating 'make_sandwich' actions; it assumes necessary ingredients
      will be available if the sandwich needs to be made.
    - The heuristic counts the total number of times each action type is needed
      across all unserved children, without considering specific object assignments
      or potential bottlenecks beyond the aggregate counts.

    # Heuristic Initialization
    The heuristic extracts static information from the task definition:
    - Which children are allergic or not allergic.
    - The waiting place for each child.
    - Which bread and content portions are gluten-free.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state, the heuristic is computed as follows:

    1.  **Identify Unserved Children:** Determine the set of children who have not
        yet been served. Count the total number of unserved children (`num_unserved`).
        If `num_unserved` is 0, the state is a goal state, and the heuristic is 0.
        Otherwise, initialize the heuristic value `h` with `num_unserved` (representing
        the `serve` action needed for each child).

    2.  **Count Sandwiches Needing to be on Trays:** Each unserved child needs a
        sandwich that is on a tray. Count the number of sandwiches currently on
        trays (`num_ontray`). The number of sandwiches that still need to be moved
        onto a tray from the kitchen is `max(0, num_unserved - num_ontray)`. Add this
        count to `h` (representing `put_on_tray` actions).

    3.  **Count Sandwiches Needing to be Made:** Each unserved child needs a sandwich
        to exist. Count the number of sandwiches that currently exist (either in the
        kitchen or on a tray) (`num_existing`). The number of sandwiches that still
        need to be created is `max(0, num_unserved - num_existing)`.
        Refine this by considering gluten-free requirements:
        - Count unserved allergic children (`num_allergic_unserved`). These need GF sandwiches.
        - Count existing GF sandwiches (`gf_existing`).
        - Count existing non-GF sandwiches (`non_gf_existing`).
        - The number of GF sandwiches that *must* be made is `make_gf = max(0, num_allergic_unserved - gf_existing)`.
        - The total number of sandwiches to make is `make_total = max(0, num_unserved - (gf_existing + non_gf_existing))`.
        - The number of non-GF sandwiches to make is `make_non_gf = max(0, make_total - make_gf)`.
        - Add `make_gf + make_non_gf` to `h` (representing `make_sandwich` actions).

    4.  **Count Trays Needing to be Moved:** For each unique location where unserved
        children are waiting, a tray is needed at that location. Identify the set
        of waiting places for unserved children (`waiting_places`). Identify the set
        of locations where trays are currently present (`tray_locations_set`). The
        number of waiting places that do not currently have a tray is
        `len(waiting_places - tray_locations_set)`. Add this count to `h`
        (representing `move_tray` actions).

    5.  **Return Total Cost:** The final value of `h` is the estimated number of
        actions to reach the goal state.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts about children,
        waiting places, and gluten-free ingredients.
        """
        # The set of facts that must hold in goal states.
        self.goals = task.goals
        # Static facts are not affected by actions.
        static_facts = task.static

        # Extract static information
        self.child_is_allergic = {
            get_parts(fact)[1]: True
            for fact in static_facts
            if match(fact, "allergic_gluten", "*")
        }
        self.child_is_not_allergic = {
            get_parts(fact)[1]: True
            for fact in static_facts
            if match(fact, "not_allergic_gluten", "*")
        }
        self.child_waiting_place = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, "waiting", "*", "*")
        }
        # Store all children names found in static waiting facts
        self.all_children_names = set(self.child_waiting_place.keys())

        # Note: We don't strictly need GF ingredient info for this heuristic's
        # calculation logic, as we assume ingredients are available if needed
        # for making sandwiches. Keeping the parsing here just in case a more
        # complex heuristic needed it.
        self.bread_is_gf = {
             get_parts(fact)[1]: True
             for fact in static_facts
             if match(fact, "no_gluten_bread", "*")
        }
        self.content_is_gf = {
             get_parts(fact)[1]: True
             for fact in static_facts
             if match(fact, "no_gluten_content", "*")
        }


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all waiting children.
        """
        state = node.state

        # 1. Parse state facts
        served_children = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "served", "*")
        }

        sandwiches_not_exist = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "notexist", "*")
        }

        sandwiches_kitchen = {
            get_parts(fact)[1]
            for fact in state
            if match(fact, "at_kitchen_sandwich", "*")
        }

        sandwiches_ontray_map = {
            get_parts(fact)[1]: get_parts(fact)[2] # sandwich -> tray
            for fact in state
            if match(fact, "ontray", "*", "*")
        }
        num_ontray = len(sandwiches_ontray_map)

        trays_at_map = {
            get_parts(fact)[1]: get_parts(fact)[2] # tray -> place
            for fact in state
            if match(fact, "at", "*", "*")
        }
        tray_locations_set = set(trays_at_map.values())

        # Need GF status of sandwiches from the state
        sandwich_is_gf_state = {
            get_parts(fact)[1]: True
            for fact in state
            if match(fact, "no_gluten_sandwich", "*")
        }

        # 2. Identify Unserved Children
        unserved_children = self.all_children_names - served_children
        num_unserved = len(unserved_children)

        # If all children are served, the heuristic is 0
        if num_unserved == 0:
            return 0

        # Initialize heuristic cost
        h = 0

        # Cost for serving each unserved child
        h += num_unserved # Each needs 1 'serve' action

        # 3. Count Sandwiches Needing to be on Trays
        # Each unserved child needs a sandwich on a tray.
        # Count how many sandwiches are already on trays.
        # The difference needs the 'put_on_tray' action.
        h += max(0, num_unserved - num_ontray)

        # 4. Count Sandwiches Needing to be Made (considering GF)
        # Each unserved child needs a sandwich to exist.
        # Count existing sandwiches (in kitchen or on tray).
        num_existing = len(sandwiches_kitchen) + num_ontray

        # Calculate needed GF and non-GF sandwiches among unserved children
        num_allergic_unserved = sum(1 for c in unserved_children if c in self.child_is_allergic)
        # Assuming all children are either allergic or not_allergic as per domain/instances
        num_non_allergic_unserved = num_unserved - num_allergic_unserved

        # Count existing GF and non-GF sandwiches
        gf_existing = sum(1 for s in sandwiches_kitchen if sandwich_is_gf_state.get(s, False)) + \
                      sum(1 for s in sandwiches_ontray_map if sandwich_is_gf_state.get(s, False))

        non_gf_existing = sum(1 for s in sandwiches_kitchen if not sandwich_is_gf_state.get(s, False)) + \
                          sum(1 for s in sandwiches_ontray_map if not sandwich_is_gf_state.get(s, False))

        # Calculate how many GF sandwiches must be made
        make_gf = max(0, num_allergic_unserved - gf_existing)

        # Calculate total sandwiches that need to be made
        make_total = max(0, num_unserved - num_existing)

        # Calculate how many non-GF sandwiches need to be made
        # This is the total needed minus the GF ones that must be made.
        # Ensure we don't ask to make negative non-GF sandwiches.
        make_non_gf = max(0, make_total - make_gf)

        h += make_gf + make_non_gf # Each needs 1 'make_sandwich' action

        # 5. Count Trays Needing to be Moved
        # Identify unique places where unserved children are waiting
        waiting_places = {
            self.child_waiting_place[c]
            for c in unserved_children
        }

        # Count places where unserved children are waiting but there is no tray
        places_need_tray = waiting_places - tray_locations_set
        num_places_need_tray = len(places_need_tray)

        h += num_places_need_tray # Each needs 1 'move_tray' action

        # Return the total estimated cost
        return h
