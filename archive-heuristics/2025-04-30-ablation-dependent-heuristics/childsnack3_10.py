# No imports needed for this specific implementation using basic Python types and string methods.

class childsnackHeuristic:
    """
    Domain-dependent heuristic for the childsnacks domain.

    Summary:
    The heuristic estimates the number of actions required to reach the goal
    state (all children served). It sums up the estimated number of actions
    needed for each stage of the process: making sandwiches, putting them
    on trays, moving trays to the children's locations, and finally serving
    the children. It considers the types of sandwiches needed (gluten-free
    vs. regular) based on child allergies and the availability of sandwiches
    and trays in the current state.

    Assumptions:
    - The problem is solvable, meaning there are enough ingredients (bread,
      content) and trays available in the initial state to make all necessary
      sandwiches and transport them. The heuristic does not explicitly check
      for resource sufficiency beyond counting available items in the current
      state.
    - The heuristic assumes a simplified flow: make sandwiches (if needed),
      put kitchen sandwiches on trays, move trays to waiting locations, serve.
      It doesn't account for complex tray movements or re-using trays/sandwiches
      optimally across multiple locations or children, but provides a lower-bound-like
      count for each type of necessary action.

    Heuristic Initialization:
    The constructor processes the static facts from the task definition.
    It extracts and stores:
    - A set of all children in the problem.
    - A dictionary mapping each child to their allergy status (True if allergic, False otherwise).
    - A dictionary mapping each child to their waiting place.
    - The constant 'kitchen' place.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1. Identify which children are already served by checking for `(served ?c)` facts in the state.
    2. Determine the set of unserved children by comparing the set of all children (from static info) with the set of served children.
    3. If there are no unserved children, the goal is reached, and the heuristic is 0.
    4. Count the total number of unserved children (`N_unserved`). This is the minimum number of `serve` actions required.
    5. Separate unserved children into those who are allergic (`U_A`) and those who are not (`U_NA`), using the pre-calculated allergy information.
    6. Count the number of suitable sandwiches currently available in the state:
       - Gluten-free sandwiches on trays (`S_GF_on_tray`): count `(ontray ?s ?t)` facts where `(no_gluten_sandwich ?s)` is also in the state.
       - Regular sandwiches on trays (`S_R_on_tray`): count `(ontray ?s ?t)` facts where `(no_gluten_sandwich ?s)` is NOT in the state.
       - Gluten-free sandwiches in the kitchen (`S_GF_kitchen`): count `(at_kitchen_sandwich ?s)` facts where `(no_gluten_sandwich ?s)` is also in the state.
       - Regular sandwiches in the kitchen (`S_R_kitchen`): count `(at_kitchen_sandwich ?s)` facts where `(no_gluten_sandwich ?s)` is NOT in the state.
    7. Calculate the total available gluten-free sandwiches (`S_GF_avail = S_GF_on_tray + S_GF_kitchen`) and regular sandwiches (`S_R_avail = S_R_on_tray + S_R_kitchen`).
    8. Estimate the minimum number of new sandwiches that need to be made (`N_make`). Allergic children *must* have GF sandwiches. Non-allergic children can have regular or GF.
       - GF sandwiches needed for allergic children = `U_A`.
       - Available GF sandwiches = `S_GF_avail`.
       - Minimum GF sandwiches to make = `max(0, gf_needed_for_allergic - S_GF_avail)`.
       - After accounting for allergic children, remaining available GF sandwiches for non-allergic = `max(0, S_GF_avail - gf_needed_for_allergic)`.
       - Regular sandwiches needed for non-allergic children = `U_NA`.
       - Available sandwiches for non-allergic (regular + leftover GF) = `S_R_avail + remaining_gf_avail`.
       - Minimum additional sandwiches (can be regular or GF) to make for non-allergic = `max(0, reg_needed_for_non_allergic - avail_for_non_allergic)`.
       - Total sandwiches to make (`N_make`) = (Min GF to make) + (Min additional regular/GF to make).
    9. Estimate the number of `put_on_tray` actions needed (`N_put`). This is the number of sandwiches currently in the kitchen (those already there plus the newly made ones) that need to be moved onto trays.
       - Sandwiches in kitchen currently = `S_GF_kitchen + S_R_kitchen`.
       - `N_put` = `N_make + sandwiches_in_kitchen_currently`.
    10. Estimate the number of `move_tray` actions needed (`N_move`). Identify the set of places where unserved children are waiting (`WaitingPlaces`). Identify the set of places where trays are currently located (`TrayLocations`). Count the number of places in `WaitingPlaces` (excluding the kitchen) that do not currently have a tray. This is a lower bound on the necessary tray movements to get trays to all required locations.
        - `WaitingPlaces = {P | (waiting C P) is static and (served C) is not in state}`.
        - `TrayLocations = {P | (at T P) is in state}`.
        - `WaitingPlaces_non_kitchen = WaitingPlaces - {self.kitchen}`. # Use self.kitchen
        - `PlacesNeedingTray = WaitingPlaces_non_kitchen - TrayLocations`.
        - `N_move = len(PlacesNeedingTray)`.
    11. The total heuristic value is the sum: `N_unserved + N_make + N_put + N_move`.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static task information.

        Args:
            task: The planning task object.
        """
        self.all_children = set()
        self.child_allergy = {}  # child_name -> True if allergic, False otherwise
        self.child_waiting_place = {} # child_name -> place_name
        self.kitchen = 'kitchen' # Constant defined in domain

        # Process static facts
        for fact in task.static:
            parts = fact.strip('()').split()
            predicate = parts[0]
            objects = parts[1:]

            if predicate == 'allergic_gluten':
                child = objects[0]
                self.all_children.add(child)
                self.child_allergy[child] = True
            elif predicate == 'not_allergic_gluten':
                child = objects[0]
                self.all_children.add(child)
                self.child_allergy[child] = False
            elif predicate == 'waiting':
                child, place = objects
                self.child_waiting_place[child] = place
                # Add child to all_children if not already added by allergy facts
                self.all_children.add(child)


    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        Args:
            state: The current state (frozenset of fact strings).

        Returns:
            An integer heuristic value.
        """
        served_children = set()
        sandwiches_in_state = {} # sandwich_name -> {'location': 'kitchen'/'ontray', 'is_gf': True/False, 'tray': tray_name}
        trays_in_state = {} # tray_name -> place_name

        # Parse state facts
        # First pass to identify sandwich types (GF status) and locations
        for fact in state:
            parts = fact.strip('()').split()
            predicate = parts[0]
            objects = parts[1:]

            if predicate == 'at_kitchen_sandwich':
                sandwich = objects[0]
                if sandwich not in sandwiches_in_state:
                    sandwiches_in_state[sandwich] = {'location': 'kitchen', 'is_gf': False, 'tray': None}
                sandwiches_in_state[sandwich]['location'] = 'kitchen'
            elif predicate == 'ontray':
                sandwich, tray = objects
                if sandwich not in sandwiches_in_state:
                     sandwiches_in_state[sandwich] = {'location': 'ontray', 'is_gf': False, 'tray': tray}
                sandwiches_in_state[sandwich]['location'] = 'ontray'
                sandwiches_in_state[sandwich]['tray'] = tray
            elif predicate == 'no_gluten_sandwich':
                 sandwich = objects[0]
                 if sandwich not in sandwiches_in_state:
                      # Should not happen based on domain actions, but handle defensively
                      sandwiches_in_state[sandwich] = {'location': None, 'is_gf': True, 'tray': None}
                 sandwiches_in_state[sandwich]['is_gf'] = True
            elif predicate == 'at':
                tray, place = objects
                trays_in_state[tray] = place
            elif predicate == 'served':
                served_children.add(objects[0])


        # 1. Identify unserved children
        unserved_children = self.all_children - served_children
        N_unserved = len(unserved_children)

        # Goal check
        if N_unserved == 0:
            return 0

        # 2. Separate unserved children by allergy and find their waiting places
        U_A = 0 # Unserved allergic
        U_NA = 0 # Unserved non-allergic
        waiting_places_with_unserved = set()

        for child in unserved_children:
            if self.child_allergy.get(child, False): # Default to False if allergy not specified
                U_A += 1
            else:
                U_NA += 1
            # 10. Identify places with unserved children
            place = self.child_waiting_place.get(child)
            if place: # Child should have a waiting place based on domain
                 waiting_places_with_unserved.add(place)


        # 6. Count available sandwiches based on parsed info
        S_GF_on_tray = 0
        S_R_on_tray = 0
        S_GF_kitchen = 0
        S_R_kitchen = 0

        for sandwich_data in sandwiches_in_state.values():
            is_gf = sandwich_data['is_gf']
            location = sandwich_data['location']

            if location == 'ontray':
                if is_gf:
                    S_GF_on_tray += 1
                else:
                    S_R_on_tray += 1
            elif location == 'kitchen':
                if is_gf:
                    S_GF_kitchen += 1
                else:
                    S_R_kitchen += 1

        S_GF_avail = S_GF_on_tray + S_GF_kitchen
        S_R_avail = S_R_on_tray + S_R_kitchen

        # 8. Estimate sandwiches to make (N_make)
        # Prioritize GF for allergic
        gf_needed_for_allergic = U_A
        gf_to_make = max(0, gf_needed_for_allergic - S_GF_avail)

        # Remaining GF available for non-allergic
        remaining_gf_avail = max(0, S_GF_avail - gf_needed_for_allergic)

        # Sandwiches needed for non-allergic
        reg_needed_for_non_allergic = U_NA

        # Available sandwiches for non-allergic (regular + remaining GF)
        avail_for_non_allergic = S_R_avail + remaining_gf_avail

        # Additional sandwiches (can be regular or GF) to make for non-allergic
        additional_to_make_for_non_allergic = max(0, reg_needed_for_non_allergic - avail_for_non_allergic)

        N_make = gf_to_make + additional_to_make_for_non_allergic

        # 9. Estimate put_on_tray actions (N_put)
        # Sandwiches in kitchen that need to go on a tray
        sandwiches_in_kitchen_currently = S_GF_kitchen + S_R_kitchen
        N_put = N_make + sandwiches_in_kitchen_currently

        # 10. Estimate move_tray actions (N_move)
        tray_locations = set(trays_in_state.values())
        waiting_places_non_kitchen = {p for p in waiting_places_with_unserved if p != self.kitchen}

        places_needing_tray_moved = waiting_places_non_kitchen - tray_locations
        N_move = len(places_needing_tray_moved)

        # Total heuristic value
        heuristic_value = N_unserved + N_make + N_put + N_move

        return heuristic_value
