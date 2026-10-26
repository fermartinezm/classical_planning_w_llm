from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the number of actions required to serve all waiting children.
    It counts the number of unserved children and adds the estimated cost to get a suitable
    sandwich on a tray to their location for those who don't have one ready.

    # Assumptions
    - Children wait at a fixed location specified in the initial state.
    - All children listed in the goal must be served.
    - A child needs a gluten-free sandwich if and only if they are allergic to gluten.
    - Any non-gluten-allergic child can receive any sandwich.
    - A sandwich on a tray at a child's location is considered "ready to serve" for that child.
    - Getting a sandwich ready for a child who doesn't have one involves a pipeline of actions:
        - Getting a suitable sandwich (make or use one from the kitchen).
        - Putting the sandwich on a tray (requires a tray at the kitchen).
        - Moving the tray from the kitchen to the child's location.
        - Serving the sandwich (this action is counted separately for all unserved children).
    - The estimated cost for the 'get sandwich onto tray and move' pipeline part is:
        - 3 actions (make + put + move) if a new sandwich must be made.
        - 2 actions (put + move) if a suitable sandwich is available in the kitchen.
    - Resource constraints (bread, content, sandwich objects, trays) are partially
      accounted for by checking available kitchen sandwiches, but not strictly enforced
      beyond that count.

    # Heuristic Initialization
    - Extracts which children are allergic from static facts.
    - Extracts the waiting location for each child (who needs to be served) from the initial state.
    - Identifies the set of all children who need to be served from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all children who are required to be served (from goal facts).
    2. From the current state, identify which of these children are already served.
    3. Count the total number of unserved children (`N_unserved`).
    4. For each unserved child, determine their waiting location and gluten requirement (allergic or not).
    5. For each unserved child, check if there is already a suitable sandwich (gluten-free if allergic, any if not) on a tray that is currently located at the child's waiting place.
    6. Count the number of unserved children who *do* have a suitable sandwich on a tray at their location (`N_ready_to_serve`).
    7. The number of unserved children who *do not* have a suitable sandwich on a tray at their location is `N_need_delivery = N_unserved - N_ready_to_serve`. These children require the sandwich/tray delivery pipeline.
    8. Count the number of sandwiches currently available in the kitchen (`N_avail_Kitchen`). These can potentially be used for the delivery pipeline, saving the 'make_sandwich' step.
    9. Estimate the total cost:
       - Each unserved child needs a 'serve_sandwich' action (cost +1 per child). Total: `N_unserved`.
       - For the `N_need_delivery` children, the 'get sandwich onto tray and move' pipeline part is needed.
       - Estimate the cost of this pipeline part:
         - If a kitchen sandwich is used (up to `N_avail_Kitchen` times): put_on_tray (1) + move_tray (1) = 2 actions.
         - If a new sandwich must be made: make_sandwich (1) + put_on_tray (1) + move_tray (1) = 3 actions.
       - The number of pipelines using kitchen sandwiches is `min(N_need_delivery, N_avail_Kitchen)`.
       - The number of pipelines needing new sandwiches is `max(0, N_need_delivery - N_avail_Kitchen)`.
       - Total pipeline part cost = `min(N_need_delivery, N_avail_Kitchen) * 2 + max(0, N_need_delivery - N_avail_Kitchen) * 3`.
       - This simplifies to `3 * N_need_delivery - min(N_need_delivery, N_avail_Kitchen)`.
    10. The total heuristic value is the sum of the serving cost and the pipeline part cost:
        `h = N_unserved + (3 * N_need_delivery - min(N_need_delivery, N_avail_Kitchen))`.

    11. Return the calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal children (who need to be served).
        - Allergic children (from static facts).
        - Waiting locations for children (from initial state).
        """
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # Identify all children who need to be served from goals
        self.all_children_to_serve = {get_parts(goal)[1] for goal in self.goals if get_parts(goal)[0] == "served"}

        # Identify allergic children from static facts
        self.allergic_children = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] == "allergic_gluten":
                self.allergic_children.add(parts[1])

        # Identify waiting locations for children from initial state
        self.child_waiting_place = {}
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == "waiting":
                 child, place = parts[1], parts[2]
                 self.child_waiting_place[child] = place


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse state to extract relevant facts ---
        served_children = set()
        kitchen_sandwiches = set()
        current_gf_sandwiches = set()
        ontray_sandwiches_map = {} # tray -> set of sandwiches
        tray_locations = {} # tray -> place

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue

            predicate = parts[0]
            if predicate == "served":
                served_children.add(parts[1])
            elif predicate == "at_kitchen_sandwich":
                kitchen_sandwiches.add(parts[1])
            elif predicate == "no_gluten_sandwich":
                current_gf_sandwiches.add(parts[1])
            elif predicate == "ontray":
                s, t = parts[1], parts[2]
                if t not in ontray_sandwiches_map:
                    ontray_sandwiches_map[t] = set()
                ontray_sandwiches_map[t].add(s)
            elif predicate == "at" and len(parts) == 3 and parts[1].startswith("tray"):
                 t, p = parts[1], parts[2]
                 tray_locations[t] = p

        # --- Calculate N_unserved and N_ready_to_serve ---
        N_unserved = 0
        N_ready_to_serve = 0

        for child in self.all_children_to_serve:
            if child not in served_children:
                N_unserved += 1
                place = self.child_waiting_place.get(child)
                if place is None:
                     # Child in goal but not in initial waiting facts? Assume they are waiting.
                     # This case indicates a potentially malformed problem instance or a misunderstanding
                     # of state representation. Assuming valid problems where goal children are initially waiting.
                     continue

                needs_gf = child in self.allergic_children

                # Check if a suitable sandwich is already on a tray at the child's location
                suitable_sandwich_found_at_location = False
                trays_at_p = {t for t, loc in tray_locations.items() if loc == place}

                for tray in trays_at_p:
                    for sandwich in ontray_sandwiches_map.get(tray, set()):
                        is_gf_sandwich = sandwich in current_gf_sandwiches
                        if needs_gf and is_gf_sandwich:
                            suitable_sandwich_found_at_location = True
                            break
                        if not needs_gf: # Regular child accepts any sandwich
                            suitable_sandwich_found_at_location = True
                            break
                    if suitable_sandwich_found_at_location:
                        break

                if suitable_sandwich_found_at_location:
                    N_ready_to_serve += 1

        # --- Calculate N_avail_Kitchen ---
        N_avail_Kitchen = len(kitchen_sandwiches)

        # --- Calculate N_need_delivery ---
        # These are the unserved children who do NOT have a suitable sandwich ready at their location
        N_need_delivery = N_unserved - N_ready_to_serve

        # --- Calculate heuristic ---
        # Heuristic = (Cost for serving each unserved child)
        #           + (Cost for getting sandwich/tray ready for those who need delivery)
        # Cost for serving = N_unserved * 1
        # Cost for getting ready pipeline part = (Cost to make if needed) + (Cost to put on tray) + (Cost to move tray)
        # Cost to make = max(0, N_need_delivery - N_avail_Kitchen) * 1
        # Cost to put on tray = N_need_delivery * 1 (each delivery needs a sandwich put on a tray)
        # Cost to move tray = N_need_delivery * 1 (each delivery needs a tray moved to location)
        # Total h = N_unserved + max(0, N_need_delivery - N_avail_Kitchen) + N_need_delivery + N_need_delivery
        # Simplified: h = N_unserved + 2 * N_need_delivery + max(0, N_need_delivery - N_avail_Kitchen)

        h = N_unserved + 2 * N_need_delivery + max(0, N_need_delivery - N_avail_Kitchen)

        # Ensure heuristic is non-negative (guaranteed by the formula if N_unserved >= N_ready_to_serve)
        # N_unserved >= 0, N_need_delivery >= 0, max(...) >= 0. So h >= 0.

        return h
