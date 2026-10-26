import math
from collections import deque
from heuristics.heuristic_base import Heuristic
from task import Task # Used for type hinting in __init__

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
    Estimates the number of actions required to reach a goal state by summing
    three components: the number of loose goal nuts (representing tighten actions),
    the estimated walk cost to reach the locations of all loose goal nuts, and
    the estimated cost to acquire enough usable spanners for all tightening
    operations.

    Assumptions:
    - The domain follows the structure defined in the provided PDDL (man, spanners,
      nuts, locations, links).
    - Nut locations are static (defined in the initial state and do not change).
    - A spanner becomes unusable after a single tighten_nut action.
    - The location graph defined by 'link' predicates is connected (or unreachable
      locations result in infinite distance).
    - There is exactly one man object in the domain.
    - The heuristic is designed for greedy best-first search and does not need
      to be admissible, but aims to be informative and efficiently computable.

    Heuristic Initialization:
    In the constructor, the heuristic precomputes static information:
    1. Parses 'link' facts from task.static to build the location graph.
    2. Identifies all locations involved in the problem by parsing 'link' and 'at'
       facts from static and initial state.
    3. Identifies the man, all spanners, all nuts, and specifically the goal nuts
       by parsing initial state and goal facts, inferring types from predicates.
    4. Stores the static location for each nut by parsing 'at' facts in the
       initial state for objects identified as nuts.
    5. Computes all-pairs shortest paths between all locations using Breadth-First
       Search (BFS) starting from each location. These distances are stored
       in a dictionary `self.dist` for quick lookup during heuristic computation.

    Step-By-Step Thinking for Computing Heuristic:
    For a given state:
    1.  Identify the man's current location by searching for the '(at man_name location)'
        fact in the state.
    2.  Find all goal nuts that are currently in a 'loose' state by checking
        '(loose nut_name)' facts for each nut in the precomputed set of goal nuts.
        If there are no loose goal nuts, the goal is effectively reached for this
        part, and the heuristic component is 0.
    3.  Identify all spanners that are currently 'usable' by checking '(usable spanner_name)'
        facts for each spanner in the precomputed set of spanners.
    4.  Calculate the base cost: This is the number of loose goal nuts. Each requires
        at least one 'tighten_nut' action. This forms a lower bound on the number
        of actions directly achieving the goal facts.
    5.  Calculate the walk cost to nuts: For each loose goal nut, find its static
        location using the precomputed `self.nut_locations` map. Sum the shortest
        path distances from the man's current location to the location of each
        loose goal nut. This estimates the minimum travel needed to reach the
        places where tightening actions must occur. This is an additive relaxation
        of the travel cost.
    6.  Calculate the spanner acquisition cost:
        -   Determine the total number of usable spanners required, which is equal
            to the number of loose goal nuts (since each tightening consumes one
            spanner's usability).
        -   Check if the man is currently carrying a usable spanner.
        -   Identify all usable spanners that are currently at specific locations
            (not carried by the man) by checking '(at spanner_name location)' facts.
        -   Calculate the total number of usable spanners available in the world
            (the one potentially carried + those at locations).
        -   If the number of required spanners exceeds the total available usable
            spanners, the state is likely a dead end (as not enough tools exist),
            and the heuristic returns infinity.
        -   Determine how many additional spanners need to be picked up from
            locations (required spanners minus the one potentially already carried).
        -   If pickups are needed, find the usable spanners at locations, sort
            them by their shortest distance from the man's current location using
            the precomputed distances, and sum the costs (distance + 1 for the
            pickup action) for the required number of nearest spanners. This
            estimates the minimum cost to gather the necessary tools.
    7.  The total heuristic value is the sum of the base cost (tighten actions),
        the walk cost to nuts, and the spanner acquisition cost.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.locations = set()
        self.man = None
        self.spanners = set()
        self.nuts = set()
        self.goal_nuts = set()
        self.nut_locations = {}  # Map nut name to location
        self.adj = {}  # Adjacency list for location graph

        # --- Parsing to collect all locations ---
        for f in task.static:
            pred, args = self.parse_fact(f)
            if pred == 'link':
                l1, l2 = args
                self.locations.add(l1)
                self.locations.add(l2)
        for f in task.initial_state:
            pred, args = self.parse_fact(f)
            if pred == 'at':
                obj, loc = args
                self.locations.add(loc)
        # Goal facts usually don't add new locations for objects in this domain

        # --- Initialize adjacency list and populate from static links ---
        self.adj = {l: [] for l in self.locations}
        for f in task.static:
            pred, args = self.parse_fact(f)
            if pred == 'link':
                l1, l2 = args
                # Ensure locations were collected before adding edges
                if l1 in self.adj and l2 in self.adj:
                    self.adj[l1].append(l2)
                    self.adj[l2].append(l1)
                # else: Warning can be added here if needed, but assuming valid PDDL

        # --- Parsing to collect objects and goal nuts ---
        # Use sets to collect potential objects based on predicates
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()

        # Collect objects from initial state
        for f in task.initial_state:
            pred, args = self.parse_fact(f)
            if pred == 'carrying':
                m, s = args
                potential_men.add(m)
                potential_spanners.add(s)
            elif pred == 'usable':
                s = args[0]
                potential_spanners.add(s)
            elif pred == 'loose' or pred == 'tightened':
                n = args[0]
                potential_nuts.add(n)
            elif pred == 'at':
                 obj, loc = args
                 # Objects at locations could be man, spanner, or nut
                 # Add to all potential sets for now, refine later
                 potential_men.add(obj)
                 potential_spanners.add(obj)
                 potential_nuts.add(obj)

        # Collect objects from goals
        for f in task.goals:
            pred, args = self.parse_fact(f)
            if pred == 'tightened':
                n = args[0]
                potential_nuts.add(n)
                self.goal_nuts.add(n) # Directly add to goal nuts

        # Refine object sets based on typical predicates and domain structure
        # Assuming there's only one man
        self.man = next(iter(potential_men), None)

        # Spanners are objects that appear in 'usable' or 'carrying' predicates
        self.spanners = {obj for obj in potential_spanners if any(
            self.parse_fact(f)[0] in ['usable', 'carrying'] and obj in self.parse_fact(f)[1]
            for f in task.initial_state
        )}

        # Nuts are objects that appear in 'loose' or 'tightened' predicates
        self.nuts = {obj for obj in potential_nuts if any(
            self.parse_fact(f)[0] in ['loose', 'tightened'] and obj in self.parse_fact(f)[1]
            for f in task.initial_state | task.goals # Check both initial state and goals
        )}

        # Populate nut locations (nuts are static)
        for f in task.initial_state:
            pred, args = self.parse_fact(f)
            if pred == 'at':
                obj, loc = args
                if obj in self.nuts:
                    self.nut_locations[obj] = loc

        # --- Compute shortest paths ---
        self.dist = {}
        self.compute_distances()


    def parse_fact(self, fact_string):
        """Helper to parse a PDDL fact string into predicate and arguments."""
        # Remove parentheses and split by space
        # Handles cases like '(at obj loc)' or '(predicate)'
        parts = fact_string[1:-1].split()
        if not parts: # Handle empty fact string inside parentheses, though unlikely
            return None, []
        predicate = parts[0]
        args = parts[1:]
        return predicate, args

    def compute_distances(self):
        """Computes all-pairs shortest paths using BFS from each location."""
        self.dist = {l: {} for l in self.locations}
        for start_loc in self.locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.dist[start_loc][start_loc] = 0

            while q:
                current_loc, d = q.popleft()

                # Check if current_loc exists in adj list (should if collected properly)
                if current_loc in self.adj:
                    for neighbor in self.adj.get(current_loc, []): # Use .get for safety
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.dist[start_loc][neighbor] = d + 1
                            q.append((neighbor, d + 1))

            # Ensure all locations have a distance (infinity if unreachable)
            for loc in self.locations:
                 if loc not in self.dist[start_loc]:
                     self.dist[start_loc][loc] = float('inf')


    def __call__(self, node):
        """Computes the domain-dependent heuristic value for a given state."""
        state = node.state

        # 1. Find man's current location
        man_location = None
        # Assuming there is only one man object
        if self.man:
            for fact in state:
                pred, args = self.parse_fact(fact)
                if pred == 'at' and args and args[0] == self.man:
                    man_location = args[1]
                    break

        if man_location is None:
             # Man's location should always be known in a valid state
             # If not found, something is wrong or state is unreachable
             return float('inf')

        # 2. Find loose goal nuts in state
        loose_goal_nuts_in_state = [n for n in self.goal_nuts if f'(loose {n})' in state]

        # If no loose goal nuts, the goal is reached for these nuts
        if not loose_goal_nuts_in_state:
            return 0

        # 3. Find usable spanners in state
        usable_spanners_in_state = [s for s in self.spanners if f'(usable {s})' in state]

        # 4. Calculate base cost (tighten actions)
        h_tighten = len(loose_goal_nuts_in_state)

        # 5. Calculate walk cost to nuts
        h_walk_nuts = 0
        loose_goal_nuts_with_locs = []
        for nut in loose_goal_nuts_in_state:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None:
                 # Nut location not found (should not happen if parsed correctly)
                 return float('inf') # Indicates a problem with the state or parsing
            loose_goal_nuts_with_locs.append((nut, nut_loc))

            # Add distance from man's current location to nut location
            if man_location in self.dist and nut_loc in self.dist[man_location]:
                 dist_to_nut = self.dist[man_location][nut_loc]
                 if dist_to_nut == float('inf'):
                      return float('inf') # Cannot reach a needed nut location
                 h_walk_nuts += dist_to_nut
            else:
                 # Location not in precomputed distances (should not happen if locations are collected correctly)
                 return float('inf')


        # 6. Calculate spanner acquisition cost
        h_spanner = 0
        num_spanners_needed = len(loose_goal_nuts_in_state)

        # Check if man is carrying a usable spanner
        man_carrying_usable = any(f'(carrying {self.man} {s})' in state for s in usable_spanners_in_state)
        num_carried = 1 if man_carrying_usable else 0

        # Find usable spanners at locations (not carried)
        usable_at_locs = []
        for spanner in usable_spanners_in_state:
             # Check if the spanner is usable AND not carried by the man
             if f'(usable {spanner})' in state and not f'(carrying {self.man} {spanner})' in state:
                 # Find location of this spanner in the current state
                 spanner_loc = None
                 for fact in state:
                      pred, args = self.parse_fact(fact)
                      if pred == 'at' and args and args[0] == spanner:
                           spanner_loc = args[1]
                           break
                 if spanner_loc:
                      usable_at_locs.append((spanner, spanner_loc))

        num_at_locs = len(usable_at_locs)
        num_total_usable = num_carried + num_at_locs

        # Check if enough spanners exist in total
        if num_spanners_needed > num_total_usable:
            return float('inf') # Problem likely unsolvable with available spanners

        # Calculate how many additional spanners need to be picked up
        spanners_to_pickup = max(0, num_spanners_needed - num_carried)

        if spanners_to_pickup > 0:
            # Sort usable spanners at locations by distance from man
            # Use .get with infinity default for safety, though compute_distances should cover all locations
            sorted_usable_at_locs = sorted(usable_at_locs, key=lambda item: self.dist[man_location].get(item[1], float('inf')))

            # Sum costs (distance + 1 for pickup) for the nearest spanners to pick up
            # Take minimum in case there are fewer usable spanners at locations than needed pickups
            # (This shouldn't happen if num_spanners_needed <= num_total_usable, but min is safe)
            for i in range(min(spanners_to_pickup, len(sorted_usable_at_locs))):
                 spanner, spanner_loc = sorted_usable_at_locs[i]
                 dist_to_spanner = self.dist[man_location].get(spanner_loc, float('inf'))
                 if dist_to_spanner == float('inf'):
                      return float('inf') # Cannot reach a needed spanner location
                 h_spanner += dist_to_spanner + 1 # Distance + 1 for pickup action

        # 7. Total heuristic value
        h = h_tighten + h_walk_nuts + h_spanner

        return h

