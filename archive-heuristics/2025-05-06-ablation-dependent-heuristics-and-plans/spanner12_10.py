from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It sums the estimated cost for each
    individual loose goal nut, assuming resources (man and spanners) can be
    reused across nuts in a relaxed way (h_add style).

    For each loose goal nut, the heuristic estimates the cost as:
    1 (tighten action)
    + cost to get the man to the nut's location
    + cost to get a usable spanner to the man at the nut's location.

    The cost to get a usable spanner to the man at the nut's location is
    estimated as the minimum of:
    - 0, if the man is already carrying a usable spanner.
    - The cost to walk from the man's current location to a usable spanner
      on the ground, pick it up, and walk with it to the nut's location.

    # Assumptions:
    - There is exactly one man object.
    - Nuts and spanners are distinct types of objects from the man.
    - The location graph defined by 'link' predicates is static and does not change.
    - All locations are reachable from each other if the problem is solvable.
    - Each usable spanner can tighten exactly one nut.
    - The number of usable spanners available in the initial state is sufficient
      to tighten all goal nuts in solvable problems.

    # Heuristic Initialization
    - Identifies the man object, nut objects, and spanner objects from the task's
      initial state and goals.
    - Builds the location graph based on 'link' static facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Stores the set of goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Determine if the man is currently carrying a usable spanner.
    3. Identify all usable spanners currently on the ground and their locations.
    4. Identify all nuts that are currently loose and are part of the goal set.
    5. If there are no loose goal nuts, the heuristic value is 0.
    6. Initialize the total heuristic cost to 0.
    7. For each loose goal nut N at location L_N:
       a. Add 1 to the cost (for the 'tighten_nut' action).
       b. Add the shortest path distance from the man's current location to L_N
          to the cost (cost for the man to walk to the nut).
       c. Calculate the minimum cost to get a usable spanner to the man *at location L_N*.
          - If the man is currently carrying a usable spanner, this cost is 0.
          - If the man is not carrying a usable spanner, iterate through all
            usable spanners S on the ground at location L_S. The cost to get
            spanner S to the man at L_N is the distance from the man's current
            location to L_S, plus 1 (for pickup), plus the distance from L_S
            to L_N (for the man to carry it there). Find the minimum such cost
            over all available usable spanners.
          - If no usable spanners are available (neither carried nor on ground),
            the problem is likely unsolvable from this state towards this nut.
            Return infinity.
       d. Add this minimum spanner acquisition cost (at L_N) to the total cost.
    8. Return the total accumulated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal details.
        """
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Identify object types (man, nuts, spanners)
        self.man = None
        self.nuts = set()
        self.spanners = set()
        locations = set()

        # Infer types based on predicates in initial state and goals
        potential_nuts = {get_parts(f)[1] for f in self.initial_state if get_parts(f)[0] == 'loose'} | \
                         {get_parts(f)[1] for f in self.goals if get_parts(f)[0] == 'tightened'}
        self.nuts = potential_nuts

        potential_spanners = {get_parts(f)[1] for f in self.initial_state if get_parts(f)[0] == 'usable'} | \
                            {get_parts(f)[2] for f in self.initial_state if get_parts(f)[0] == 'carrying'} | \
                            {get_parts(f)[2] for f in self.goals if get_parts(f)[0] == 'carrying'} # Carrying might be in goals? (Unlikely in spanner, but general)

        # Add spanners that might only appear in 'at' facts in initial state and are not nuts
        spanners_from_at = {get_parts(f)[1] for f in self.initial_state if get_parts(f)[0] == 'at' and get_parts(f)[1] not in self.nuts}
        self.spanners = potential_spanners.union(spanners_from_at)

        # Man is the locatable object in initial state 'at' fact that is not a nut or spanner
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] not in self.nuts and parts[1] not in self.spanners:
                 self.man = parts[1]
                 locations.add(parts[2]) # Add man's initial location
            elif parts[0] == 'at':
                 locations.add(parts[2]) # Add other objects' initial locations

        # Identify all locations from link facts
        self.location_graph = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, set()).add(l2)
                self.location_graph.setdefault(l2, set()).add(l1)

        self.locations = list(locations) # Store list of all locations

        # Compute all-pairs shortest paths
        self.distances = {}
        for start_loc in self.locations:
            self.distances[(start_loc, start_loc)] = 0
            queue = deque([(start_loc, 0)])
            visited = {start_loc}

            while queue:
                current_loc, dist = queue.popleft()

                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            queue.append((neighbor, dist + 1))

        # Identify goal nuts
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'tightened'}


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = set()
        nut_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'loose' and parts[1] in self.goal_nuts:
                loose_goal_nuts_in_state.add(parts[1])
            if parts[0] == 'at' and parts[1] in self.nuts:
                 nut_locations[parts[1]] = parts[2]

        if not loose_goal_nuts_in_state:
            return 0 # Goal reached for all nuts

        # 2. Find man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man:
                man_location = parts[2]
                break
        if man_location is None:
             # Man's location not found - problem state? Return large value.
             return float('inf')


        # 3. Determine if man is carrying a usable spanner
        man_carrying_usable_spanner = False
        carried_spanner = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and parts[1] == self.man:
                carried_spanner = parts[2]
                if f'(usable {carried_spanner})' in state:
                    man_carrying_usable_spanner = True
                break # Assuming man carries at most one spanner

        # 4. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground = [] # List of (spanner_obj, location)
        spanner_locations = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] in self.spanners and parts[1] != carried_spanner:
                 spanner_locations[parts[1]] = parts[2]

        for spanner in self.spanners:
             if spanner != carried_spanner and f'(usable {spanner})' in state and spanner in spanner_locations:
                  usable_spanners_on_ground.append((spanner, spanner_locations[spanner]))


        # 5. Calculate heuristic based on loose goal nuts
        total_cost = 0

        for nut in loose_goal_nuts_in_state:
            nut_loc = nut_locations.get(nut)
            if nut_loc is None:
                 # Nut location not found - problem state? Return large value.
                 return float('inf')

            # Cost for this specific nut
            cost_this_nut = 1 # tighten action

            # Cost to get man to the nut's location
            dist_man_to_nut = self.distances.get((man_location, nut_loc), float('inf'))
            if dist_man_to_nut == float('inf'):
                 # Nut location unreachable from man's current location
                 return float('inf')
            cost_this_nut += dist_man_to_nut

            # Cost to get a usable spanner to the man *at location nut_loc*
            min_spanner_acquisition_cost_at_nut_loc = float('inf')

            # Option 1: Use the spanner man is currently carrying (if usable)
            if man_carrying_usable_spanner:
                 # The cost to get the spanner to nut_loc is just the man's walk cost,
                 # which is already included in dist_man_to_nut. So, no *additional*
                 # cost for spanner acquisition if man arrives with it.
                 min_spanner_acquisition_cost_at_nut_loc = 0

            # Option 2: Pick up a usable spanner from the ground and bring it to nut_loc
            for s, s_loc in usable_spanners_on_ground:
                 # Cost to go from man_location to s_loc, pickup, then go from s_loc to nut_loc
                 dist_man_to_spanner = self.distances.get((man_location, s_loc), float('inf'))
                 dist_spanner_to_nut = self.distances.get((s_loc, nut_loc), float('inf'))

                 if dist_man_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                      cost_get_s_to_nut = dist_man_to_spanner + 1 + dist_spanner_to_nut # walk M->S + pickup + walk S->N
                      min_spanner_acquisition_cost_at_nut_loc = min(min_spanner_acquisition_cost_at_nut_loc, cost_get_s_to_nut)

            # If no way to get a usable spanner to the nut location
            if min_spanner_acquisition_cost_at_nut_loc == float('inf'):
                 return float('inf') # Problem unsolvable from this state

            total_cost += cost_this_nut + min_spanner_acquisition_cost_at_nut_loc

        return total_cost
