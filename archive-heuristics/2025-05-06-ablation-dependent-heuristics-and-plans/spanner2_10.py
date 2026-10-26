from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It uses a greedy approach, estimating the cost to tighten nuts sequentially,
    always picking the closest available usable spanner and moving to the closest
    remaining loose goal nut.

    # Assumptions
    - There is exactly one man object. The heuristic attempts to identify it.
    - The man can carry at most one spanner at a time.
    - A spanner becomes unusable after tightening one nut.
    - Enough usable spanners exist in the initial state to tighten all goal nuts.
    - The location graph defined by 'link' predicates is static.
    - All locations relevant to the problem (initial locations, goal locations, spanner locations)
      are part of the graph or reachable within connected components.

    # Heuristic Initialization
    - Identify all locations mentioned in initial state and static facts.
    - Build the graph based on 'link' predicates.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the man object, all nut objects, and all spanner objects from the initial state.
    - Identify the set of goal nuts from the goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify State Elements:** Determine the man's current location, which goal nuts are still loose, which spanners are currently usable and their locations (on the ground), and if the man is currently carrying a usable spanner.
    2.  **Goal Check:** If there are no loose goal nuts, the goal is reached, and the heuristic value is 0.
    3.  **Initialize Cost and State Variables:** Set the initial heuristic cost to 0. Keep track of the man's current location (starts as the actual man's location in the state). Keep track of the set of usable spanners available on the ground.
    4.  **Handle Initially Carried Spanner:** Check if the man starts in the current state carrying a usable spanner. If yes, assume he will use this spanner for the first nut he tightens.
    5.  **Process Loose Nuts Sequentially:** Create a list of loose goal nuts with their current locations. Sort this list greedily (e.g., by distance from the man's *current* location). Iterate through the sorted list:
        a.  **Acquire Spanner (if needed):** If the man is not currently carrying a usable spanner (either didn't start with one, or used the one he had on a previous nut), he needs to get one. Find the closest usable spanner available on the ground from his current location. Add the cost to walk to the spanner's location plus the 'pickup_spanner' action cost (1). Update the man's location to the spanner's location. Mark the spanner as unavailable for future pickups. If no usable spanners are available on the ground but are needed, the state is likely unsolvable, return infinity.
        b.  **Move to Nut:** Add the cost to walk from the man's current location to the nut's location. Update the man's location to the nut's location.
        c.  **Tighten Nut:** Add the cost of the 'tighten_nut' action (1). The spanner used becomes unusable (implicitly handled by the spanner tracking).
    6.  **Return Total Cost:** The accumulated cost is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and computing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        self.locations = set()
        self.man_obj = None
        self.spanner_objs = set()
        self.nut_objs = set()
        self.initial_at = {} # obj: loc

        # --- Parse Initial State to identify objects and initial facts ---
        # First pass to collect all objects and locations from 'at' facts
        all_initial_objects_at_loc = set()
        for fact in self.initial_state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                self.initial_at[obj] = loc
                self.locations.add(loc)
                all_initial_objects_at_loc.add(obj)
            elif parts[0] == "loose":
                 self.nut_objs.add(parts[1])
            elif parts[0] == "tightened":
                 self.nut_objs.add(parts[1])

        # Second pass to identify man and spanners based on predicates
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "carrying":
                  # Man is the first arg in (carrying ?m - man ?s - spanner)
                  self.man_obj = parts[1]
                  self.spanner_objs.add(parts[2])

        # Infer spanner objects that are not carried initially but are at a location
        if self.man_obj:
             all_initial_objects_at_loc.discard(self.man_obj)
        self.spanner_objs.update(all_initial_objects_at_loc - self.nut_objs)

        # If man_obj wasn't found via 'carrying', try finding a locatable that isn't a spanner or nut
        if self.man_obj is None:
             for obj in self.initial_at:
                  if obj not in self.spanner_objs and obj not in self.nut_objs:
                       self.man_obj = obj
                       break # Assuming only one man

        # --- Parse Static Facts to build graph ---
        self.graph = {} # adjacency list
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                if l1 not in self.graph: self.graph[l1] = []
                if l2 not in self.graph: self.graph[l2] = []
                self.graph[l1].append(l2)
                self.graph[l2].append(l1)

        # Ensure all locations from initial state are in graph nodes even if no links
        for loc in self.locations:
            if loc not in self.graph:
                self.graph[loc] = []

        # --- Compute All-Pairs Shortest Paths using BFS ---
        self.distances = {} # dist[loc1][loc2]
        for start_loc in self.locations:
            self.distances[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[start_loc][start_loc] = 0
            while q:
                curr_loc, d = q.popleft()
                for neighbor in self.graph.get(curr_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_loc][neighbor] = d + 1
                        q.append((neighbor, d + 1))

        # --- Parse Goal Conditions ---
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened":
                self.goal_nuts.add(parts[1])

        # Ensure goal nuts are actually identified nut objects
        self.goal_nuts = self.goal_nuts.intersection(self.nut_objs)


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions using a
        greedy sequential approach.
        """
        state = node.state

        # 1. Identify State Elements
        man_loc = None
        nut_loc_current = {} # nut_obj: loc_obj
        available_usable_spanners = {} # s_obj: s_loc (spanners on the ground)
        man_carrying_usable = None # spanner_obj if carrying usable, else None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj == self.man_obj:
                    man_loc = loc
                elif obj in self.nut_objs:
                    nut_loc_current[obj] = loc
                elif obj in self.spanner_objs:
                    # Check if this spanner is usable
                    if f"(usable {obj})" in state:
                         available_usable_spanners[obj] = loc
            elif parts[0] == "carrying":
                 carrier, carried_obj = parts[1], parts[2]
                 if carrier == self.man_obj and carried_obj in self.spanner_objs:
                      # Check if this carried spanner is usable
                      if f"(usable {carried_obj})" in state:
                           man_carrying_usable = carried_obj

        # Identify loose goal nuts and their current locations
        loose_goal_nuts_info = [] # List of (nut_obj, nut_loc)
        for nut in self.goal_nuts:
            if f"(loose {nut})" in state:
                if nut in nut_loc_current:
                    loose_goal_nuts_info.append((nut, nut_loc_current[nut]))
                else:
                    # This nut is loose and a goal, but its location is unknown?
                    # Should not happen in valid states if initial state is consistent.
                    # Treat as unreachable?
                    return float('inf')


        # 2. Goal Check
        if not loose_goal_nuts_info:
            return 0 # All goal nuts are tightened

        # 3. Initialize Cost and State Variables
        heuristic_cost = 0
        current_loc = man_loc

        # Sort loose nuts by distance from the man's current location
        # This is the greedy choice for the *next* nut to tighten
        # Use .get(item[1], float('inf')) to handle cases where a nut location might not be in the computed distances (e.g., disconnected graph)
        loose_goal_nuts_info.sort(key=lambda item: self.distances.get(current_loc, {}).get(item[1], float('inf')))


        # 4. Handle Initially Carried Spanner
        if man_carrying_usable:
            # Use the carried spanner for the first nut in the sorted list
            if loose_goal_nuts_info: # Should be true if we are here and h > 0
                first_nut, first_nut_loc = loose_goal_nuts_info.pop(0)

                # Cost to reach the first nut
                dist_to_nut = self.distances.get(current_loc, {}).get(first_nut_loc, float('inf'))
                if dist_to_nut == float('inf'):
                    return float('inf') # Cannot reach this nut

                heuristic_cost += dist_to_nut
                current_loc = first_nut_loc
                heuristic_cost += 1 # Tighten action
                # Spanner used, man is no longer carrying a usable spanner
                man_carrying_usable = None # Explicitly set to None

        # 5. Process Remaining Loose Nuts Sequentially
        # For each remaining nut, the man needs to acquire a spanner and tighten the nut.
        for nut, nut_loc in loose_goal_nuts_info:
            # Need to get a spanner (man is not carrying a usable one at this point)
            # Find closest usable spanner object S_obj at L_S_obj from CurrentLoc
            closest_spanner_obj = None
            closest_spanner_loc = None
            min_dist_spanner = float('inf')

            # Consider spanners available on the ground
            # Need to use a copy of items() as we might delete from available_usable_spanners
            for s_obj, s_loc in list(available_usable_spanners.items()):
                 dist_s = self.distances.get(current_loc, {}).get(s_loc, float('inf'))
                 if dist_s < min_dist_spanner:
                     min_dist_spanner = dist_s
                     closest_spanner_obj = s_obj
                     closest_spanner_loc = s_loc

            if closest_spanner_obj is None:
                # No usable spanners available on the ground for remaining nuts.
                # If there are still loose nuts, this state is likely unsolvable.
                return float('inf') # Cannot proceed

            # Cost to get spanner
            if min_dist_spanner == float('inf'):
                 return float('inf') # Spanner location unreachable

            heuristic_cost += min_dist_spanner # Walk to spanner
            current_loc = closest_spanner_loc
            heuristic_cost += 1 # Pickup spanner
            # Remove this spanner from available list (it's now carried and will be used)
            del available_usable_spanners[closest_spanner_obj]
            # Man is now carrying this spanner, but it will be used on *this* nut immediately.
            # We don't need to update man_carrying_usable here as it's consumed within this nut's steps.

            # Need to get to the nut location
            dist_to_nut = self.distances.get(current_loc, {}).get(nut_loc, float('inf'))
            if dist_to_nut == float('inf'):
                 return float('inf') # Nut location unreachable

            heuristic_cost += dist_to_nut # Walk to nut
            current_loc = nut_loc

            # Need to tighten the nut
            heuristic_cost += 1 # Tighten action
            # Spanner used becomes unusable (implicitly handled)

        # 6. Return Total Cost
        return heuristic_cost
