from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Helper function to split a PDDL fact string into its parts."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Helper function to match fact parts with pattern arguments."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    This heuristic estimates the cost to reach the goal state by simulating a greedy plan.
    The goal is to tighten a specific set of nuts. To tighten a nut, the man must be
    at the nut's location, carrying a usable spanner. The heuristic simulates the
    process of tightening each required nut one by one. In each step, it selects
    the closest remaining loose goal nut. It calculates the cost to move the man
    to this nut's location. Then, it calculates the cost to acquire a usable spanner
    at that location (either by using a spanner the man is already carrying, or by
    moving to the closest available usable spanner at a location, picking it up,
    and returning to the nut's location). Finally, it adds the cost of the tighten
    action. Spanners are treated as consumable resources; once used to tighten a nut
    in the simulation, they are no longer available. The total accumulated cost
    from this greedy simulation is the heuristic value.

    Assumptions:
    - There is exactly one man object in the domain.
    - The location graph defined by 'link' predicates is static and does not change.
    - The location graph is connected for all locations relevant to the problem
      (man's initial location, locations of goal nuts, locations of usable spanners).
      If locations are disconnected, distances will be infinite, leading to an
      infinite heuristic value, correctly indicating unsolvability from that state.
    - The only way to make a nut tightened is by using the 'tighten_nut' action.
    - The only way to get a usable spanner is to pick up one that is currently
      usable and at a location, or already be carrying one that is usable.
      Spanners do not become usable again after being used in a 'tighten_nut' action.
    - Nuts needing tightening are initially loose and at a location.
    - The greedy strategy of processing the closest nut first and picking up the
      closest available spanner provides a reasonable estimate of the remaining cost.

    Heuristic Initialization:
    - The constructor identifies all potential locations mentioned in static 'link'
      facts and 'at' facts from the initial state and goals.
    - It builds an adjacency list representation of the location graph based on
      the 'link' predicates in the static facts. Links are assumed to be bidirectional.
    - It computes all-pairs shortest paths between all identified locations using
      Breadth-First Search (BFS). These distances are stored in a dictionary `self.dist`.
    - It identifies the names of all nut and spanner objects by parsing initial
      state and goal facts.
    - It identifies the name of the man object, assuming it is the unique object
      involved in 'carrying' predicates or 'at' predicates that is not a nut or spanner.
    - It identifies the set of nuts that are specified as goals (i.e., need to be tightened).

    Step-By-Step Thinking for Computing Heuristic:
    1. The `__call__` method receives a search node containing the current state.
    2. It parses the current state to extract key information: the man's current
       location, the locations of all locatable objects (spanners, nuts), which
       nuts are currently loose or tightened, which spanners are currently usable,
       and if the man is currently carrying a spanner.
    3. It determines the set of loose nuts that are also present in the goal
       conditions. These are the nuts that still need to be tightened.
    4. If there are no loose goal nuts, the current state is a goal state, and the
       heuristic returns 0.
    5. It identifies the set of all usable spanners currently available in the state
        (either carried or at a location).
    6. It checks if the number of loose goal nuts is greater than the number of
       available usable spanners. If so, the problem is unsolvable from this state
       (assuming no new spanners appear or become usable), and the heuristic returns
       `float('inf')`.
    7. A greedy simulation of the plan execution begins to estimate the cost.
       Variables are initialized to track the man's current location in the simulation,
       the set of usable spanners not yet consumed in the simulation, and whether
       the man is currently carrying a usable spanner in the simulation. The total
       estimated cost is initialized to 0.
    8. The simulation proceeds iteratively while there are still loose goal nuts
       remaining to be tightened:
        a. From the remaining loose goal nuts, select the one whose location is
           closest to the man's current location in the simulation, using the
           precomputed distances.
        b. Add the distance from the man's current location to the selected nut's
           location to the total cost. Update the man's current location in the
           simulation to the nut's location.
        c. Determine the cost to acquire a usable spanner at the current location
           (the nut's location):
            i. If the man is currently carrying a usable spanner in the simulation,
               conceptually use this spanner. The cost to acquire it is 0. The man
               is no longer carrying a usable spanner after this step in the simulation.
            ii. If the man is not carrying a usable spanner, find the closest
                available usable spanner that is currently located at a location
                (not already used in the simulation). Add the distance from the
                man's current location (the nut location) to this spanner's location,
                plus 1 for the 'pickup_spanner' action, to the total cost. Update
                the man's current location in the simulation to the spanner's location
                (since pickup happens at the spanner's location). Conceptually, the
                man is now carrying this spanner in the simulation. Remove this
                spanner from the pool of available usable spanners for future steps.
        d. After acquiring the spanner, the man is at `current_L_M` (either the nut
           location if he was already carrying a spanner, or the spanner location
           if he picked one up). He needs to be at the nut location to tighten.
           Add the distance from the man's current location to the nut's location
           to the total cost. Update the man's current location in the simulation
           to the nut's location.
        e. Add 1 to the total cost for the 'tighten_nut' action.
        f. Remove the selected nut from the list of remaining loose goal nuts.
    9. Once all loose goal nuts have been processed in the simulation, the final
       accumulated total cost is returned as the heuristic value.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by precomputing location distances and
        identifying key objects and goal conditions.

        Args:
            task: The planning task object.
        """
        self.goals = task.goals
        static_facts = task.static

        # --- Heuristic Initialization ---
        # 1. Identify all locations and build the location graph.
        self.all_locations = set()
        self.adj = {} # Adjacency list for the location graph

        # Collect locations from static links
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link' and len(parts) == 3:
                loc1, loc2 = parts[1], parts[2]
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        # Collect locations from initial state and goals 'at' facts
        for fact in task.initial_state | task.goals:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 # The second argument of 'at' is a location
                 self.all_locations.add(parts[2])

        # Initialize adjacency list entries for locations that might not have links
        for loc in self.all_locations:
             self.adj.setdefault(loc, set())

        # 2. Compute all-pairs shortest paths between locations using BFS.
        self.dist = {}
        for start_loc in self.all_locations:
            self.dist[start_loc] = {loc: float('inf') for loc in self.all_locations}
            self.dist[start_loc][start_loc] = 0
            queue = deque([start_loc])

            while queue:
                curr = queue.popleft()
                current_d = self.dist[start_loc][curr]

                # Check if curr is in adj before iterating (handles isolated locations)
                if curr in self.adj:
                    for neighbor in self.adj[curr]:
                        if self.dist[start_loc][neighbor] == float('inf'): # Not visited
                            self.dist[start_loc][neighbor] = current_d + 1
                            queue.append(neighbor)

        # 3. Identify goal nuts.
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'tightened' and len(get_parts(g)) == 2}

        # 4. Identify the man object (assuming there's only one man).
        self.man_name = None
        # Infer man name by looking for objects involved in 'carrying' or 'at' facts
        # that are not nuts or spanners.
        potential_men = set()
        nuts_and_spanners = set()

        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 obj = parts[1]
                 potential_men.add(obj) # Assume anything 'at' a location could be the man initially
            elif parts[0] == 'carrying' and len(parts) == 3:
                 m, s = parts[1], parts[2]
                 self.man_name = m # Man is the first arg of carrying
                 nuts_and_spanners.add(s) # Second arg is spanner
            elif parts[0] == 'loose' and len(parts) == 2:
                 nuts_and_spanners.add(parts[1]) # Arg is nut
            elif parts[0] == 'usable' and len(parts) == 2:
                 nuts_and_spanners.add(parts[1]) # Arg is spanner

        # If man not found via 'carrying', assume it's the object at a location that isn't a nut/spanner
        if self.man_name is None:
             remaining_potential_men = potential_men - nuts_and_spanners
             if len(remaining_potential_men) == 1:
                 self.man_name = list(remaining_potential_men)[0]
             # else: handle error or assume problem structure guarantees finding man

        # 5. Identify all nuts and spanners mentioned in initial state/goals/static
        self.all_nuts = set()
        self.all_spanners = set()
        for fact in task.initial_state | task.goals | static_facts:
             parts = get_parts(fact)
             if parts[0] in ['loose', 'tightened'] and len(parts) == 2:
                  self.all_nuts.add(parts[1])
             elif parts[0] in ['usable'] and len(parts) == 2:
                  self.all_spanners.add(parts[1])
             elif parts[0] == 'carrying' and len(parts) == 3:
                  self.all_spanners.add(parts[2]) # Second arg is spanner


    def __call__(self, node):
        """
        Computes the heuristic value for the given state.

        Args:
            node: The search node containing the state.

        Returns:
            An integer estimate of the remaining cost to the goal, or float('inf').
        """
        state = node.state

        # --- Step-By-Step Thinking for Computing Heuristic ---
        # 1. Parse the current state
        locations = {} # obj -> loc
        loose_nuts_in_state = set()
        tightened_nuts_in_state = set()
        usable_spanners_in_state = set()
        man_location = None
        carried_spanner_name = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                locations[obj] = loc
                if obj == self.man_name:
                    man_location = loc
            elif parts[0] == 'loose' and len(parts) == 2:
                loose_nuts_in_state.add(parts[1])
            elif parts[0] == 'tightened' and len(parts) == 2:
                tightened_nuts_in_state.add(parts[1])
            elif parts[0] == 'usable' and len(parts) == 2:
                usable_spanners_in_state.add(parts[1])
            elif parts[0] == 'carrying' and len(parts) == 3:
                 m, s = parts[1], parts[2]
                 if m == self.man_name:
                     carried_spanner_name = s

        # Ensure man location is known
        if man_location is None:
             # Man is not at any location in the state. This shouldn't happen in a valid state.
             return float('inf')

        # 2. Identify loose nuts that are goal nuts.
        loose_nuts_needing_tightening = {n for n in self.goal_nuts if n in loose_nuts_in_state}

        # 3. Check for goal state.
        if not loose_nuts_needing_tightening:
            return 0 # All goal nuts are tightened

        # 4. Check for unsolvability based on available usable spanners.
        # We need one usable spanner per loose nut to tighten.
        # Total usable spanners available in the current state (carried or at locations)
        total_usable_spanners_in_state = set(usable_spanners_in_state)
        if carried_spanner_name is not None and carried_spanner_name in usable_spanners_in_state:
             # The carried spanner is usable, it's counted in usable_spanners_in_state
             pass
        else:
             # If man is carrying a spanner that is NOT usable, it doesn't help.
             pass # usable_spanners_in_state already filters for usable ones

        if len(loose_nuts_needing_tightening) > len(total_usable_spanners_in_state):
            # Not enough usable spanners in the state to tighten all required nuts.
            return float('inf')

        # 5. Simulate a greedy plan to estimate cost.
        current_L_M = man_location
        # Pool of usable spanners not yet consumed in the simulation
        sim_usable_spanners_pool = set(usable_spanners_in_state)
        # Track if man is carrying a usable spanner in the simulation
        sim_man_carrying = carried_spanner_name if (carried_spanner_name is not None and carried_spanner_name in sim_usable_spanners_pool) else None

        # Remove the initially carried usable spanner from the pool if it exists,
        # as it's handled separately by sim_man_carrying.
        if sim_man_carrying and sim_man_carrying in sim_usable_spanners_pool:
             sim_usable_spanners_pool.remove(sim_man_carrying)


        cost = 0
        nuts_remaining = list(loose_nuts_needing_tightening) # Use a list to remove elements

        while nuts_remaining:
            # a. Select the nut closest to the man's current location
            closest_nut = None
            min_dist_to_nut = float('inf')
            nut_location = None

            for nut in nuts_remaining:
                loc = locations.get(nut) # Get nut location from current state
                if loc is None:
                    # Nut not at any location? Problematic state.
                    return float('inf')

                # Ensure locations are in our distance map
                if current_L_M not in self.dist or loc not in self.dist[current_L_M]:
                     # Cannot calculate distance, likely disconnected graph or missing location
                     return float('inf')

                d = self.dist[current_L_M][loc]
                if d < min_dist_to_nut:
                    min_dist_to_nut = d
                    closest_nut = nut
                    nut_location = loc

            if closest_nut is None:
                 # Should not happen if nuts_remaining is not empty
                 return float('inf')

            # b. Move to the closest nut location
            cost += self.dist[current_L_M][nut_location]
            current_L_M = nut_location

            # c. Get a spanner at the current location (nut_location)
            spanner_used = None
            spanner_travel_cost = 0 # Travel cost to get spanner
            spanner_pickup_cost = 0 # Pickup action cost

            if sim_man_carrying is not None:
                # Use the spanner already carried
                spanner_used = sim_man_carrying
                sim_man_carrying = None # Spanner consumed conceptually
                # No travel or pickup needed, spanner is already with the man at the nut location.
            else:
                # Need to pick up a spanner from a location.
                closest_spanner = None
                min_dist_to_spanner = float('inf')
                spanner_loc_to_go = None

                # Find the closest usable spanner *at a location* from the remaining pool.
                usable_spanners_at_locs_in_pool = {s for s in sim_usable_spanners_pool if s in locations and locations[s] in self.all_locations}

                for spanner in usable_spanners_at_locs_in_pool:
                    spanner_location = locations[spanner]
                    # Ensure locations are in our distance map
                    if current_L_M not in self.dist or spanner_location not in self.dist[current_L_M]:
                         return float('inf') # Defensive

                    d = self.dist[current_L_M][spanner_location]
                    if d < min_dist_to_spanner:
                        min_dist_to_spanner = d
                        closest_spanner = spanner
                        spanner_loc_to_go = spanner_location

                if closest_spanner is None:
                     # Should not happen if initial check passed and logic is correct
                     return float('inf') # Defensive

                spanner_used = closest_spanner
                # Travel from current_L_M (nut location) to spanner_loc_to_go
                spanner_travel_cost = self.dist[current_L_M][spanner_loc_to_go]
                spanner_pickup_cost = 1 # pickup action
                current_L_M = spanner_loc_to_go # Man is now at spanner location after pickup
                sim_man_carrying = spanner_used # Man is now carrying this spanner conceptually

            # Remove the used spanner from the simulation's available pool
            if spanner_used in sim_usable_spanners_pool:
                 sim_usable_spanners_pool.remove(spanner_used)

            cost += spanner_travel_cost + spanner_pickup_cost

            # d. Go back to the nut location (if spanner was picked up elsewhere)
            # Man is currently at current_L_M (either nut_location or spanner_loc_to_go)
            # He needs to be at nut_location to tighten.
            cost += self.dist[current_L_M][nut_location]
            current_L_M = nut_location # Man is now at the nut location

            # e. Tighten the nut
            cost += 1

            # Remove the nut from the list
            nuts_remaining.remove(closest_nut)

        # 6. Return the total estimated cost.
        return cost
