import collections
import math

# Assume the Task class definition is available from the provided code file.
# from task import Task

# Helper function to parse PDDL facts
def parse_fact(fact_string):
    """Parses a fact string like '(at bob shed)' into ('at', 'bob', 'shed')."""
    # Remove surrounding brackets and split by space
    # Handle potential empty strings from multiple spaces
    parts = fact_string[1:-1].split()
    return tuple(parts)

class spannerHeuristic:
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
        Estimates the cost to reach the goal (tighten all required nuts)
        by simulating a greedy plan: repeatedly walk to the closest remaining
        loose goal nut, pick up a usable spanner if needed (from the closest
        available location), and perform the tighten action. The heuristic
        value is the sum of estimated actions (walk, pickup, tighten) in this
        greedy simulation.

    Assumptions:
        - The problem is solvable (there are enough usable spanners in the initial state).
        - The location graph defined by 'link' facts is connected.
        - The man, spanners, and nuts are located at valid locations.
        - Spanners become unusable after one tighten action.
        - Nuts do not move.
        - Links are bidirectional.
        - There is exactly one man object.
        - The man object can be identified by looking for the object involved in a 'carrying' fact, or failing that, the single object 'at' a location that is not a known goal nut or usable spanner in the initial state. (This is a heuristic guess due to lack of type info).

    Heuristic Initialization:
        - Parses static 'link' facts to build a graph of locations.
        - Computes all-pairs shortest paths between locations using BFS, storing
          distances in a dictionary.
        - Identifies the set of goal nuts from the task's goal state.
        - Identifies the man object name (heuristic guess).
        - Precomputes initial locations of goal nuts.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Parse the current state to extract:
            -   The man's current location.
            -   The set of loose nuts.
            -   The set of usable spanners the man is carrying.
            -   A mapping of locations to usable spanners present at those locations.
        2.  Identify the set of loose nuts that are part of the goal. If this set is empty, the goal is reached, and the heuristic is 0.
        3.  Initialize the estimated cost to 0.
        4.  Initialize conceptual state variables: current man location, current set of usable spanners carried, current usable spanners available at locations, and the set of remaining loose goal nuts.
        5.  While there are remaining loose goal nuts:
            a.  Find the loose goal nut closest to the current man location using the precomputed distance map and the precomputed nut locations.
            b.  Add the distance to this nut's location to the estimated cost. Update the current man location to the nut's location.
            c.  Check if the man is carrying a usable spanner (conceptually).
            d.  If not carrying a usable spanner:
                i.  Find the closest location to the current man location that has a usable spanner available (conceptually).
                ii. If no usable spanners are available anywhere (conceptually), return infinity as the problem is likely unsolvable from this state.
                iii. Add the distance to this spanner location to the estimated cost. Update the current man location to the spanner location.
                iv. Add 1 to the estimated cost for the 'pickup_spanner' action.
                v.  Conceptually move one usable spanner from the location to the man's carried spanners.
            e.  Add 1 to the estimated cost for the 'tighten_nut' action.
            f.  Conceptually use one spanner from the man's carried spanners.
            g.  Remove the tightened nut from the set of remaining loose goal nuts.
        6.  Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initializes the heuristic.

        @param task: The planning task object (instance of Task class).
        """
        self.task = task
        # Goal nuts are the objects that need to be tightened
        self.goal_nuts = {parse_fact(g)[1] for g in task.goals if parse_fact(g)[0] == 'tightened'}

        # Extract locations and links from static facts and initial state
        locations = set()
        links = set()

        for fact_str in task.static:
            fact = parse_fact(fact_str)
            if fact[0] == 'link':
                l1, l2 = fact[1], fact[2]
                links.add((l1, l2))
                locations.add(l1)
                locations.add(l2)

        # Add locations mentioned in initial state 'at' facts
        for fact_str in task.initial_state:
             fact = parse_fact(fact_str)
             if fact[0] == 'at':
                 # The second argument of 'at' is always a location
                 locations.add(fact[2])

        self.locations = locations

        # Build distance map
        self.dist = self._build_distance_map(self.locations, links)

        # Precompute initial locations of goal nuts (nuts don't move)
        self.nut_locations = {}
        for fact_str in task.initial_state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at' and fact[1] in self.goal_nuts:
                 self.nut_locations[fact[1]] = fact[2]

        # Identify the man object name (heuristic guess)
        self.man_name = None
        # Look for an object involved in a 'carrying' fact in the initial state
        for fact_str in task.initial_state:
             fact = parse_fact(fact_str)
             if fact[0] == 'carrying':
                  self.man_name = fact[1]
                  break
        # If no 'carrying' fact in initial state, look for the single object 'at' a location
        # that is not a goal nut. This is fragile.
        if self.man_name is None:
             man_candidates = []
             initial_usable_spanners = {parse_fact(f)[1] for f in task.initial_state if parse_fact(f)[0] == 'usable'}
             for fact_str in task.initial_state:
                  fact = parse_fact(fact_str)
                  if fact[0] == 'at':
                       obj_name = fact[1]
                       if obj_name not in self.goal_nuts and obj_name not in initial_usable_spanners:
                            man_candidates.append(obj_name)
             if len(man_candidates) == 1:
                  self.man_name = man_candidates[0]
             # else: Man name could not be uniquely identified heuristically.
             # The heuristic might fail or be inaccurate in such cases.
             # We proceed assuming man_name is found or the problem structure allows inference.


    def _build_distance_map(self, locations, links):
        """Builds an all-pairs shortest path distance map using BFS."""
        dist = {loc: {other_loc: math.inf for other_loc in locations} for loc in locations}
        for loc in locations:
            dist[loc][loc] = 0

        adj = collections.defaultdict(set)
        for l1, l2 in links:
            adj[l1].add(l2)
            adj[l2].add(l1) # Assuming bidirectional links

        for start_node in locations:
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_loc, current_dist = q.popleft()
                dist[start_node][current_loc] = current_dist

                for neighbor in adj[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, current_dist + 1))
        return dist

    def __call__(self, state):
        """
        Computes the heuristic value for a given state.

        @param state: The current state (frozenset of fact strings).
        @return: The estimated cost (number of actions) to reach a goal state,
                 or infinity if likely unsolvable.
        """
        # 1. Parse current state
        man_loc = None
        loose_nuts_in_state = set()
        usable_spanners_carried = set()
        usable_spanners_at_loc = collections.defaultdict(list)
        usable_spanners_in_state = set()
        carried_spanner_names = set()

        # First pass to find usable spanners and carried spanners
        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'usable':
                usable_spanners_in_state.add(fact[1])
            elif fact[0] == 'carrying' and (self.man_name is None or fact[1] == self.man_name):
                 # Assuming the second argument is the spanner name
                 carried_spanner_names.add(fact[2])
                 # If man_name wasn't identified in init, assume the carrier is the man
                 if self.man_name is None:
                      self.man_name = fact[1]


        # Second pass to get locations and identify usable carried/at_loc spanners
        for fact_str in state:
            fact = parse_fact(fact_str)
            if fact[0] == 'at':
                obj_name = fact[1]
                loc_name = fact[2]

                # Identify man's location
                if self.man_name is not None and obj_name == self.man_name:
                     man_loc = loc_name

                # Identify usable spanners at locations
                if obj_name in usable_spanners_in_state and obj_name not in carried_spanner_names:
                     usable_spanners_at_loc[loc_name].append(obj_name)

            elif fact[0] == 'loose':
                nut_name = fact[1]
                if nut_name in self.goal_nuts:
                    loose_nuts_in_state.add(nut_name)

        # Identify usable spanners carried by the man
        for spanner_name in carried_spanner_names:
             if spanner_name in usable_spanners_in_state:
                  usable_spanners_carried.add(spanner_name)

        # If man_name was not found via 'carrying' fact, try the fallback from init
        if self.man_name is None:
             # This fallback logic is already attempted in __init__
             # If it failed there, man_name remains None.
             # We might still find the man's location if he's just 'at' somewhere
             # and we can guess his identity from the state facts.
             # Let's re-apply the fallback guess based on current state if man_name is still None
             man_candidates = []
             current_usable_spanners = usable_spanners_in_state # Usable spanners in current state
             for fact_str in state:
                  fact = parse_fact(fact_str)
                  if fact[0] == 'at':
                       obj_name = fact[1]
                       # Guess: object 'at' a location that is not a goal nut and not a usable spanner
                       if obj_name not in self.goal_nuts and obj_name not in current_usable_spanners:
                            man_candidates.append((obj_name, loc_name))
             if len(man_candidates) == 1:
                  self.man_name = man_candidates[0][0]
                  man_loc = man_candidates[0][1]
             # else: Still couldn't identify man_name or man_loc reliably.

        if man_loc is None:
             # Could not find man's location. This state is likely invalid or unsolvable.
             return math.inf


        # 2. Identify loose goal nuts
        # loose_goal_nuts are the nuts that are loose in the current state AND are in the goal
        loose_goal_nuts = {nut for nut in loose_nuts_in_state if nut in self.goal_nuts}

        if not loose_goal_nuts:
            return 0 # Goal reached

        # 3. Initialize conceptual state
        estimated_cost = 0
        current_man_loc = man_loc
        current_usable_spanners_carried = set(usable_spanners_carried)
        # Need a mutable copy of usable spanners at locations
        current_usable_spanners_at_loc = collections.defaultdict(list)
        for loc, spanners in usable_spanners_at_loc.items():
             current_usable_spanners_at_loc[loc] = list(spanners) # Use list for pop()

        remaining_loose_goal_nuts = set(loose_goal_nuts)

        # Use precomputed nut locations (nuts don't move)
        nut_current_locations = self.nut_locations # Assuming all goal nuts are in initial state

        # 4. Simulate greedy plan
        while remaining_loose_goal_nuts:
            # a. Find the closest loose goal nut
            min_dist_to_nut = math.inf
            next_nut = None
            next_nut_loc = None

            for nut in remaining_loose_goal_nuts:
                nut_loc = nut_current_locations.get(nut) # Get location from precomputed map
                if nut_loc is None:
                     # This loose goal nut was not in the initial state 'at' facts. Problematic.
                     return math.inf # Indicate unsolvability

                # Check if locations exist in distance map
                if current_man_loc not in self.dist or nut_loc not in self.dist[current_man_loc]:
                     return math.inf # Indicate unsolvability or error

                d = self.dist[current_man_loc][nut_loc]
                if d < min_dist_to_nut:
                    min_dist_to_nut = d
                    next_nut = nut
                    next_nut_loc = nut_loc

            if next_nut is None:
                 # Should not happen if remaining_loose_goal_nuts is not empty and graph is connected
                 return math.inf # Indicate unsolvability

            # Add walk cost to the nut
            estimated_cost += min_dist_to_nut
            current_man_loc = next_nut_loc

            # c. Check if spanner is needed
            if not current_usable_spanners_carried:
                # d. Need to pick up a spanner
                min_dist_to_spanner = math.inf
                best_spanner_loc = None

                for loc, spanners in current_usable_spanners_at_loc.items():
                    if spanners: # Location has usable spanners
                        # Check if location exists in distance map
                        if current_man_loc not in self.dist or loc not in self.dist[current_man_loc]:
                             return math.inf # Indicate unsolvability or error

                        d = self.dist[current_man_loc][loc]
                        if d < min_dist_to_spanner:
                            min_dist_to_spanner = d
                            best_spanner_loc = loc

                if best_spanner_loc is None:
                    # No usable spanners left anywhere in the conceptual state
                    # Problem is unsolvable from here
                    return math.inf

                # Add walk cost to the spanner
                estimated_cost += min_dist_to_spanner
                current_man_loc = best_spanner_loc

                # Add pickup cost
                estimated_cost += 1
                # Conceptually pick up one spanner
                current_usable_spanners_at_loc[best_spanner_loc].pop()
                current_usable_spanners_carried.add("dummy_spanner") # Add placeholder

            # e. Perform tighten action
            estimated_cost += 1
            # f. Conceptually use one spanner
            # Ensure there is a spanner to use (should be if logic is correct)
            if not current_usable_spanners_carried:
                 # This indicates an error in the heuristic's logic or state parsing
                 return math.inf # Should not happen if pickup logic was followed

            current_usable_spanners_carried.pop()
            # g. Remove the tightened nut
            remaining_loose_goal_nuts.remove(next_nut)

        return estimated_cost
