from collections import deque

# Assume Heuristic and Task classes are available from the planner's framework
# from heuristics.heuristic_base import Heuristic
# from task import Operator, Task

# The code will be placed within the planner's structure, so Heuristic and Task
# are expected to be in scope.

class spannerHeuristic: # Inherit from Heuristic if required by framework
    """
    Domain-dependent heuristic for the spanner domain.

    Summary:
        Estimates the number of actions required to reach a goal state
        by summing up the costs associated with tightening each loose
        goal nut. The cost for a single nut includes the tighten action itself,
        the estimated walking cost for the man to reach the nut's location,
        and the estimated cost to acquire a usable spanner if the man
        doesn't currently carry one and needs one for the first time.

    Assumptions:
        - There is exactly one man object.
        - Spanners become unusable after one tighten action.
        - The locations form a connected graph (or reachable locations are considered).
        - Nut locations are static (defined in initial state or static facts).

    Heuristic Initialization:
        The constructor precomputes static information from the task:
        - Identifies the names of the man, spanners, nuts, and locations
          by parsing all possible facts and initial state/static facts.
        - Stores the fixed location for each nut found in the initial state or static facts.
        - Builds the location graph based on 'link' facts.
        - Computes all-pairs shortest paths (distances) between locations
          using Breadth-First Search (BFS).

    Step-By-Step Thinking for Computing Heuristic:
        1. Identify all nuts that are goals (i.e., '(tightened nut_name)' is a goal fact)
           and are currently not tightened (i.e., '(tightened nut_name)' is not in the state).
           Let this set be `LooseGoalNuts`.
        2. If `LooseGoalNuts` is empty, the state is a goal state, return 0.
        3. Find the man's current location (`L_man`) by searching the state facts.
        4. Find the set of usable spanners the man is currently carrying (`CarriedUsableSpanners`)
           by searching the state facts for '(carrying man_name spanner_name)' and checking
           if '(usable spanner_name)' is also in the state.
        5. Find the set of locations where usable spanners are currently located
           (`LocatedUsableSpannerLocs`) by searching the state facts for '(at spanner_name location_name)'
           where the spanner is usable and not carried by the man.
        6. Collect all usable spanners available in the state (carried or at locations).
        7. Check if the total number of available usable spanners is less than the number
           of loose goal nuts. If so, the problem is unsolvable from this state (as each
           tighten action consumes a usable spanner), return infinity.
        8. Initialize the heuristic value `h` to 0.
        9. Add the cost for the tighten actions: `h += len(LooseGoalNuts)`. Each tighten
           action costs 1.
        10. Add the estimated cost for the man to reach the nuts: Calculate the minimum
            distance from `L_man` to the location of any nut in `LooseGoalNuts` using
            the precomputed distances. Add this minimum distance to `h`. If any required
            nut location is unreachable from the man's current location, return infinity.
        11. Add the estimated cost for acquiring a spanner if needed: If the man is not
            carrying any usable spanners (`len(CarriedUsableSpanners) == 0`) and there
            are loose goal nuts (`len(LooseGoalNuts) > 0`), the man needs to acquire
            a spanner. Calculate the minimum distance from `L_man` to any location
            in `LocatedUsableSpannerLocs` using the precomputed distances. If no usable
            spanners are reachable from the man's current location, return infinity.
            Add this minimum distance plus 1 (for the pickup action) to `h`.
        12. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        # super().__init__() # Uncomment if inheriting from Heuristic base class
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static
        self.all_facts = task.facts # Contains all possible fact strings

        self._precompute_static_info(task) # Pass task to access types if needed

    def _precompute_static_info(self, task):
        """
        Precomputes location graph, distances, nut locations, and object names.
        Infers object types from predicate structure in all facts.
        """
        self.man_name = None
        self.spanner_names = set()
        self.nut_names = set()
        self.location_names = set()
        self.nut_locations = {}  # nut_name -> location_name
        self.location_graph = {} # loc -> set of connected locs

        # Infer object types and names from all possible facts
        # This is more robust than relying on initial state structure
        for fact_str in self.all_facts:
            parts = fact_str.strip('()').split()
            if not parts: continue
            pred = parts[0]
            args = parts[1:]

            if pred == 'at' and len(args) == 2:
                 # (at ?obj ?loc) - args[1] is location
                 self.location_names.add(args[1])
                 # args[0] is a locatable, type determined by other predicates
            elif pred == 'carrying' and len(args) == 2:
                 # (carrying ?m - man ?s - spanner)
                 self.man_name = args[0] # Assuming only one man
                 self.spanner_names.add(args[1])
            elif pred == 'usable' and len(args) == 1:
                 # (usable ?s - spanner)
                 self.spanner_names.add(args[0])
            elif pred == 'link' and len(args) == 2:
                 # (link ?l1 - location ?l2 - location)
                 self.location_names.add(args[0])
                 self.location_names.add(args[1])
                 self.location_graph.setdefault(args[0], set()).add(args[1])
                 self.location_graph.setdefault(args[1], set()).add(args[0]) # Links are bidirectional
            elif pred == 'tightened' and len(args) == 1:
                 # (tightened ?n - nut)
                 self.nut_names.add(args[0])
            elif pred == 'loose' and len(args) == 1:
                 # (loose ?n - nut)
                 self.nut_names.add(args[0])

        # Populate nut locations from initial state and static facts
        # Nuts have fixed locations
        for fact_str in task.initial_state | task.static:
             if fact_str.startswith('(at '):
                 parts = fact_str.strip('()').split()
                 if len(parts) == 3:
                     obj_name = parts[1]
                     loc_name = parts[2]
                     if obj_name in self.nut_names:
                         self.nut_locations[obj_name] = loc_name
                     # Also add locations found here, just in case they weren't in links
                     self.location_names.add(loc_name)


        # Ensure all locations mentioned are in the graph keys
        for loc in self.location_names:
             self.location_graph.setdefault(loc, set())

        # Compute all-pairs shortest paths (distances)
        self.distances = {}
        for start_loc in self.location_names:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all reachable nodes.
        """
        distances = {node: float('inf') for node in self.location_names}
        if start_node not in self.location_names:
             # Start node is not a known location, cannot compute distances
             return distances # All distances remain inf

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in self.location_graph: # Check if node exists in graph keys
                for neighbor in self.location_graph.get(current_node, []): # Use .get for safety
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """
        Returns the precomputed distance between two locations.
        Returns infinity if locations are not known or unreachable.
        """
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             # loc1 is not a known start node for BFS, or loc2 is not in its distance map
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.
        """
        state = node.state
        h = 0

        # 1. Identify loose goal nuts
        loose_goal_nuts = set() # Stores nut names
        for goal_fact in self.goals:
            # Goal facts are like '(tightened nut1)'
            if goal_fact.startswith('(tightened '):
                # A nut needs tightening if the goal fact is not yet in the state
                if goal_fact not in state:
                     nut_name = goal_fact.strip('()').split()[1]
                     loose_goal_nuts.add(nut_name)

        num_loose_goal_nuts = len(loose_goal_nuts)

        # 2. If LooseGoalNuts is empty, the state is a goal state
        if num_loose_goal_nuts == 0:
            return 0

        # 3. Find the man's current location
        man_location = None
        for fact in state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                if len(parts) == 3 and parts[1] == self.man_name:
                    man_location = parts[2]
                    break

        if man_location is None or man_location not in self.location_names:
             # Man must be at a known location. If not, something is wrong.
             # Treat as unreachable goal.
             return float('inf')

        # 4. Find carried usable spanners
        carried_spanners = set()
        for fact in state:
            if fact.startswith('(carrying '):
                parts = fact.strip('()').split()
                if len(parts) == 3 and parts[1] == self.man_name:
                    carried_spanners.add(parts[2])

        carried_usable_spanners = {s for s in carried_spanners if f'(usable {s})' in state}
        num_carried_usable = len(carried_usable_spanners)

        # 5. Find locations of usable spanners not carried
        located_usable_spanner_locs = set() # Stores locations
        all_usable_spanners = set(carried_usable_spanners) # Stores spanner names

        for fact in state:
            if fact.startswith('(at '):
                parts = fact.strip('()').split()
                if len(parts) == 3:
                    obj_name = parts[1]
                    loc_name = parts[2]
                    # Check if obj_name is a spanner, is usable, and is not carried
                    if obj_name in self.spanner_names and f'(usable {obj_name})' in state and obj_name not in carried_spanners:
                         if loc_name in self.location_names: # Ensure location is known
                             located_usable_spanner_locs.add(loc_name)
                             all_usable_spanners.add(obj_name) # Collect all usable spanners

        # 7. Check if enough usable spanners exist in total
        if len(all_usable_spanners) < num_loose_goal_nuts:
            return float('inf') # Unsolvable

        # 8. Initialize heuristic value (already h=0)

        # 9. Add cost for tighten actions
        h += num_loose_goal_nuts

        # 10. Add estimated cost for man to reach nuts
        min_dist_to_nut = float('inf')
        all_nuts_reachable = True
        for nut_name in loose_goal_nuts:
            if nut_name not in self.nut_locations:
                 # Location of a goal nut is unknown - should not happen in valid problems
                 return float('inf')

            nut_location = self.nut_locations[nut_name]
            dist = self.get_distance(man_location, nut_location)

            if dist == float('inf'):
                all_nuts_reachable = False
                break # Man cannot reach this nut location

            min_dist_to_nut = min(min_dist_to_nut, dist)

        if not all_nuts_reachable:
             return float('inf')

        h += min_dist_to_nut

        # 11. Add estimated cost for acquiring a spanner if needed
        # Need a spanner if we have loose goal nuts but no usable spanner carried
        if num_carried_usable == 0 and num_loose_goal_nuts > 0:
            min_dist_to_spanner_loc = float('inf')
            spanner_locs_reachable = False
            for spanner_loc in located_usable_spanner_locs:
                dist = self.get_distance(man_location, spanner_loc)
                if dist != float('inf'):
                    spanner_locs_reachable = True
                    min_dist_to_spanner_loc = min(min_dist_to_spanner_loc, dist)

            if not spanner_locs_reachable:
                # Need a spanner, but no usable spanners are reachable from man's location
                return float('inf')

            # Cost to get the first spanner: walk to it + pickup (cost 1)
            h += min_dist_to_spanner_loc + 1

        # 12. Return total cost
        return h
