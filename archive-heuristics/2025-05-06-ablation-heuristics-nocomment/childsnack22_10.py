from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class childsnackHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    This heuristic estimates the number of actions required to reach a goal state
    by summing up the estimated costs for each unserved child. It considers the
    actions needed to make suitable sandwiches, put them on trays, move trays
    to the children's locations, and finally serve the sandwiches. It accounts
    for existing sandwiches and trays, and groups children by location and
    allergy status to identify shared needs.

    Assumptions:
    - The heuristic is non-admissible; it aims for accuracy and efficiency
      in a greedy best-first search, not optimality guarantees.
    - It assumes that necessary ingredients (bread, content) and sandwich
      objects (`notexist`) are available in the kitchen if a sandwich needs
      to be made.
    - It simplifies tray usage by primarily focusing on whether a location
      has *any* tray present when needed for delivery, rather than tracking
      individual tray capacity or which specific sandwiches are on which tray.
      A single tray move is assumed sufficient to enable serving all children
      at a location who need sandwiches delivered via a tray move.

    Heuristic Initialization:
    The constructor processes the static facts from the task to build data
    structures that are used repeatedly during heuristic computation:
    - `child_waiting_place`: A dictionary mapping each child object to their
      waiting place object, extracted from `(waiting ?c ?p)` facts.
    - `allergic_children`: A set containing child objects who are allergic
      to gluten, extracted from `(allergic_gluten ?c)` facts.
    - `not_allergic_children`: A set containing child objects who are not
      allergic to gluten, extracted from `(not_allergic_gluten ?c)` facts.
    - The goal facts (`(served ?c)`) are also stored.

    Step-By-Step Thinking for Computing Heuristic:
    1.  Identify all children who are in the goal state (`(served ?c)`) but
        are not yet served in the current state. These are the `UnservedChildren`.
    2.  The base heuristic value is the total number of `UnservedChildren`.
        This represents the minimum number of `serve` actions required.
        `h = N_unserved`.
    3.  For each unserved child, determine their waiting place and allergy status
        using the pre-processed static information (`ChildNeeds`).
    4.  Identify which of the `UnservedChildren` can *already* be served without
        requiring a new sandwich delivery action sequence (make, put, move).
        A child is "ready" if a suitable sandwich is already on a tray that
        is located at the child's waiting place. Suitability depends on the
        child's allergy status and the sandwich's gluten status. Count these
        children (`N_already_ready`).
    5.  The number of children who still need a sandwich delivery sequence is
        `N_need_delivery = N_unserved - N_already_ready`.
    6.  Calculate the number of sandwiches of each type (gluten-free and regular)
        that need to be made to satisfy the `N_need_delivery` children, taking
        into account suitable sandwiches already existing in the kitchen or
        on trays elsewhere.
        - Count how many of the `N_need_delivery` children are allergic
          (`N_allergic_need_delivery`) and non-allergic (`N_not_allergic_need_delivery`).
        - Count suitable gluten-free and regular sandwiches currently in the
          kitchen (`N_kitchen_GF`, `N_kitchen_Reg`).
        - Count suitable gluten-free and regular sandwiches currently on trays
          anywhere (`N_ontray_anywhere_GF`, `N_ontray_anywhere_Reg`).
        - Calculate the number of gluten-free sandwiches to make:
          `N_to_make_GF = max(0, N_allergic_need_delivery - (N_kitchen_GF + N_ontray_anywhere_GF))`.
        - Calculate the number of regular sandwiches to make:
          `N_to_make_Reg = max(0, N_not_allergic_need_delivery - (N_kitchen_Reg + N_ontray_anywhere_Reg))`.
        - Total sandwiches to make: `N_to_make = N_to_make_GF + N_to_make_Reg`.
        - Add `N_to_make` to the heuristic `h`. (Cost for `make_sandwich` actions).
    7.  Calculate the number of sandwiches that need to be put on a tray. This includes
        all sandwiches that were just made (`N_to_make`) plus any suitable sandwiches
        that were already in the kitchen (`N_kitchen_GF + N_kitchen_Reg`).
        `N_need_put_on_tray = N_to_make + N_kitchen_GF + N_kitchen_Reg`.
        Add `N_need_put_on_tray` to the heuristic `h`. (Cost for `put_on_tray` actions).
    8.  Identify the unique locations where children needing delivery are waiting.
        `LocationsNeedingDelivery`.
    9.  Identify the locations where trays are currently present (`LocationsWithTray`).
    10. Count the number of locations in `LocationsNeedingDelivery` that do not
        currently have a tray. These locations will require a tray to be moved there.
        `N_locations_needing_tray_move = len(LocationsNeedingDelivery - LocationsWithTray)`.
        Add `N_locations_needing_tray_move` to the heuristic `h`. (Cost for `move_tray` actions).
    11. The final heuristic value is the sum `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        self.child_waiting_place = {}
        self.allergic_children = set()
        self.not_allergic_children = set()

        for fact in static_facts:
            parts = self._get_parts(fact)
            if parts[0] == 'waiting':
                self.child_waiting_place[parts[1]] = parts[2]
            elif parts[0] == 'allergic_gluten':
                self.allergic_children.add(parts[1])
            elif parts[0] == 'not_allergic_gluten':
                self.not_allergic_children.add(parts[1])
            # We don't strictly need no_gluten_bread/content for this heuristic,
            # assuming ingredients are available if needed for making sandwiches.

    def _get_parts(self, fact):
        """Helper to parse fact string into predicate and arguments."""
        # Remove surrounding parentheses and split by space
        return fact[1:-1].split()

    def _match(self, fact, *args):
        """Helper for pattern matching fact parts with arguments."""
        parts = self._get_parts(fact)
        return all(fnmatch(part, arg) for part, arg in zip(parts, args))

    def __call__(self, node):
        state = node.state

        # 1. Identify unserved children
        unserved_children = set()
        for goal in self.goals:
            # Goal facts are expected to be of the form '(served child_name)'
            if self._get_parts(goal)[0] == 'served':
                child_name = self._get_parts(goal)[1]
                if goal not in state:
                    unserved_children.add(child_name)

        # 2. Base heuristic: number of unserved children (min serve actions)
        n_unserved = len(unserved_children)
        if n_unserved == 0:
            return 0 # Goal reached

        h = n_unserved

        # 3. Determine needs (place, allergy) for unserved children
        child_needs = {
            c: (self.child_waiting_place[c], 'allergic_gluten' if c in self.allergic_children else 'not_allergic_gluten')
            for c in unserved_children
        }

        # Extract relevant state information
        ontray_sandwiches = {self._get_parts(fact)[1]: self._get_parts(fact)[2] for fact in state if self._match(fact, 'ontray', '*', '*')} # {sandwich: tray}
        tray_locations = {self._get_parts(fact)[1]: self._get_parts(fact)[2] for fact in state if self._match(fact, 'at', '*', '*')} # {tray: place}
        gluten_free_sandwiches_in_state = {self._get_parts(fact)[1] for fact in state if self._match(fact, 'no_gluten_sandwich', '*')}
        kitchen_sandwiches = {self._get_parts(fact)[1] for fact in state if self._match(fact, 'at_kitchen_sandwich', '*')}
        sandwiches_on_trays_anywhere = set(ontray_sandwiches.keys())


        # 4. Identify children whose needs are already met by a tray at their location
        ready_sandwiches_at_location = set() # Set of children
        for child in unserved_children:
            place, allergy_status = child_needs[child]
            found_ready = False
            # Check if any sandwich on any tray at the child's location is suitable
            for s, t in ontray_sandwiches.items():
                if t in tray_locations and tray_locations[t] == place:
                    is_gluten_free_s = s in gluten_free_sandwiches_in_state
                    is_suitable = False
                    if allergy_status == 'allergic_gluten' and is_gluten_free_s:
                        is_suitable = True
                    elif allergy_status == 'not_allergic_gluten':
                         # Any sandwich is suitable for non-allergic children
                         is_suitable = True

                    if is_suitable:
                        found_ready = True
                        break # Found one ready sandwich for this child

            if found_ready:
                ready_sandwiches_at_location.add(child)

        # 5. Number of children who still need a delivery sequence
        children_need_delivery = unserved_children - ready_sandwiches_at_location
        n_need_delivery = len(children_need_delivery)

        # 6. Calculate sandwiches to make
        n_allergic_need_delivery = len({c for c in children_need_delivery if c in self.allergic_children})
        n_not_allergic_need_delivery = n_need_delivery - n_allergic_need_delivery

        # Count available suitable sandwiches (in kitchen or on trays anywhere)
        n_kitchen_gf = len({s for s in kitchen_sandwiches if s in gluten_free_sandwiches_in_state})
        n_kitchen_reg = len({s for s in kitchen_sandwiches if s not in gluten_free_sandwiches_in_state})

        n_ontray_anywhere_gf = len({s for s in sandwiches_on_trays_anywhere if s in gluten_free_sandwiches_in_state})
        n_ontray_anywhere_reg = len({s for s in sandwiches_on_trays_anywhere if s not in gluten_free_sandwiches_in_state})

        n_available_gf = n_kitchen_gf + n_ontray_anywhere_gf
        n_available_reg = n_kitchen_reg + n_ontray_anywhere_reg

        # Sandwiches to make
        n_to_make_gf = max(0, n_allergic_need_delivery - n_available_gf)
        n_to_make_reg = max(0, n_not_allergic_need_delivery - n_available_reg)
        n_to_make = n_to_make_gf + n_to_make_reg
        h += n_to_make # Cost for make actions

        # 7. Calculate sandwiches needing put_on_tray
        # These are the ones just made, plus suitable ones already in the kitchen
        n_need_put_on_tray = n_to_make + n_kitchen_gf + n_kitchen_reg
        h += n_need_put_on_tray # Cost for put_on_tray actions

        # 8. Identify locations needing delivery
        locations_needing_delivery = {child_needs[c][0] for c in children_need_delivery}

        # 9. Identify locations with trays
        locations_with_tray = set(tray_locations.values())

        # 10. Count locations needing a tray move
        locations_without_tray_needing_delivery = locations_needing_delivery - locations_with_tray
        n_locations_needing_tray_move = len(locations_without_tray_needing_delivery)
        h += n_locations_needing_tray_move # Cost for move_tray actions

        # 11. Total heuristic is the sum
        return h
