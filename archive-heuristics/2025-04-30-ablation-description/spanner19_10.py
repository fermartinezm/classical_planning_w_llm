from fnmatch import fnmatch
from collections import deque
import math # For float('inf')


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if pattern is longer than fact parts
    if len(args) > len(parts):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic: # Assuming Heuristic base class is implicitly handled by the environment
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose nuts specified in the goal.
    It considers the number of loose nuts remaining, the minimum distance for the
    man to reach any of these loose nuts, and the cost to acquire a usable spanner
    if the man is not already carrying one.

    # Assumptions
    - There is exactly one man object in the domain.
    - Nuts that are not in the goal do not need to be tightened.
    - Nuts stay at their initial locations throughout the plan.
    - Spanners stay at their locations unless carried by the man.
    - The location graph formed by 'link' predicates is connected, or relevant
      locations (man's start, nut locations, spanner locations) are in the same
      connected component.
    - Enough usable spanners exist in the problem instance (across initial state
      facts) to tighten all goal nuts. If not enough are *currently* usable/available
      on the ground or carried, the heuristic might return infinity.

    # Heuristic Initialization
    - Extract all location objects and link relationships from static facts and
      object definitions to build a location graph (adjacency list).
    - Compute all-pairs shortest path distances between locations using BFS on the graph.
    - Identify all goal nuts from the task's goal conditions.
    - Identify the man object and all spanner objects based on their types.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. **Identify Loose Goal Nuts:** Determine which nuts are currently in a `(loose ?n)` state and are also part of the task's goal (`(tightened ?n)`). Let this set be `LooseGoalNuts`.
    2. **Goal Check:** If `LooseGoalNuts` is empty, the goal has been reached, and the heuristic value is 0.
    3. **Find Man's Location:** Determine the current location of the man object from the state facts.
    4. **Base Cost (Tighten Actions):** Initialize the heuristic value `h` with the count of loose goal nuts (`len(LooseGoalNuts)`). Each loose nut requires at least one `tighten_nut` action.
    5. **Movement Cost (to Nearest Nut):**
       a. Find the current location for each nut in `LooseGoalNuts`.
       b. Calculate the minimum shortest path distance from the man's current location to the location of any nut in `LooseGoalNuts` using the precomputed distances.
       c. Add this minimum distance to `h`. This estimates the cost for the man to reach the vicinity of the first nut that needs tightening.
    6. **Spanner Acquisition Cost (if needed):**
       a. Check if the man is currently carrying a spanner that is also in a `(usable ?s)` state.
       b. If the man is NOT carrying a usable spanner:
          i. Identify all spanners that are currently located on the ground (`(at ?s ?l)`) and are in a `(usable ?s)` state, and are not currently being carried by the man.
          ii. If no such usable spanners are available anywhere (neither carried nor on the ground), the problem is likely unsolvable from this state. Return `float('inf')`.
          iii. Find the minimum shortest path distance from the man's current location to the location of any of these available usable spanners.
          iv. Add this minimum distance plus 1 (for the `pickup_spanner` action) to `h`. This estimates the cost to acquire the necessary tool for the first tightening action.
    7. **Return Heuristic Value:** The final value of `h` is the estimated cost.
    """

    def __init__(self, task):
        # The task object is expected to have attributes: goals, static, objects
        self.goals = task.goals
        self.static = task.static
        self.objects = task.objects # Dictionary: {obj_name: obj_type}

        # Extract locations from task objects
        self.locations = {obj_name for obj_name, obj_type in task.objects.items() if obj_type == 'location'}

        # Build location graph from static link facts
        self.location_graph = {loc: set() for loc in self.locations} # Initialize with all locations
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations from link facts are known locations
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1)
                # else: print(f"Warning: Link fact involves non-location object: {fact}") # Optional warning

        # Compute all-pairs shortest paths
        self.dist = self._compute_all_pairs_shortest_paths()

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Identify man and spanner objects
        self.man_name = None
        self.all_spanners = set()
        for obj_name, obj_type in task.objects.items():
            if obj_type == 'man':
                self.man_name = obj_name
            elif obj_type == 'spanner':
                self.all_spanners.add(obj_name)

        if self.man_name is None:
             # This heuristic assumes a single man. If none exists, it's an invalid problem for this heuristic.
             # print("Error: No object of type 'man' found in the task.")
             pass # Handle potential error later if man_name is needed and None

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS. Returns a dictionary dist[start_loc][end_loc].
        """
        dist = {loc: {l: math.inf for l in self.locations} for loc in self.locations}

        for start_node in self.locations:
            dist[start_node][start_node] = 0
            queue = deque([start_node])
            visited = {start_node}

            while queue:
                u = queue.popleft()

                if u in self.location_graph: # Ensure u is a valid node in the graph
                    for v in self.location_graph.get(u, []): # Use .get for safety
                        if v not in visited:
                            visited.add(v)
                            dist[start_node][v] = dist[start_node][u] + 1
                            queue.append(v)
        return dist

    def __call__(self, node):
        """
        Estimate the minimum cost to tighten all remaining goal nuts.
        """
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {
            get_parts(fact)[1] for fact in state
            if match(fact, "loose", "*") and get_parts(fact)[1] in self.goal_nuts
        }

        # 2. Goal Check
        if not loose_goal_nuts:
            return 0

        # Ensure man object exists
        if self.man_name is None:
             # Cannot compute heuristic without a man
             return math.inf # Or raise error

        # 3. Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None or man_location not in self.locations:
             # Man must be at a known location
             return math.inf # Should not happen in valid states

        # 4. Base cost: number of tighten actions
        h = len(loose_goal_nuts)

        # 5a. Find locations of loose goal nuts
        nut_locations = {}
        for nut in loose_goal_nuts:
             for fact in state:
                 if match(fact, "at", nut, "*"):
                     nut_location = get_parts(fact)[2]
                     if nut_location in self.locations:
                         nut_locations[nut] = nut_location
                     else:
                         # Nut must be at a known location
                         return math.inf # Should not happen
                     break
             if nut not in nut_locations:
                  # Nut must always be somewhere
                  return math.inf # Should not happen in valid states

        # 5b. Add minimum distance to nearest loose nut location
        min_dist_to_nut_loc = math.inf
        for loc in nut_locations.values():
             if man_location in self.dist and loc in self.dist[man_location]:
                min_dist_to_nut_loc = min(min_dist_to_nut_loc, self.dist[man_location][loc])
             else:
                 # Man cannot reach this nut location
                 return math.inf

        if min_dist_to_nut_loc == math.inf:
             # Man cannot reach any loose nut location
             return math.inf

        h += min_dist_to_nut_loc

        # 5c. Check if man is carrying a usable spanner
        is_carrying_usable_spanner = False
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                # Check if the carried spanner is usable
                if any(match(f, "usable", carried_spanner) for f in state):
                    is_carrying_usable_spanner = True
                break # Assuming man carries at most one spanner

        # 5d. Add cost to get a usable spanner if needed
        if not is_carrying_usable_spanner:
            available_usable_spanners = [] # List of (spanner_name, location)
            for spanner_name in self.all_spanners:
                 # Check if spanner is usable and not carried
                 is_usable = any(match(f, "usable", spanner_name) for f in state)
                 is_carried = any(match(f, "carrying", self.man_name, spanner_name) for f in state)

                 if is_usable and not is_carried:
                      # Find its location
                      spanner_location = None
                      for fact in state:
                           if match(fact, "at", spanner_name, "*"):
                                spanner_location = get_parts(fact)[2]
                                break
                      if spanner_location and spanner_location in self.locations:
                           available_usable_spanners.append((spanner_name, spanner_location))
                      # else: print(f"Warning: Usable spanner {spanner_name} not at a known location.") # Optional warning


            # If no usable spanners are available anywhere (on ground or carried)
            # Note: We already checked if carried spanner is usable.
            # So if not is_carrying_usable_spanner and available_usable_spanners is empty,
            # it means no usable spanner exists in the state.
            if not available_usable_spanners:
                 # Cannot get a usable spanner
                 return math.inf

            min_dist_to_spanner_loc = math.inf
            for s_name, s_loc in available_usable_spanners:
                 if man_location in self.dist and s_loc in self.dist[man_location]:
                      min_dist_to_spanner_loc = min(min_dist_to_spanner_loc, self.dist[man_location][s_loc])
                 else:
                      # Man cannot reach this spanner location
                      return math.inf

            if min_dist_to_spanner_loc == math.inf:
                 # Man cannot reach any available usable spanner location
                 return math.inf

            spanner_cost = min_dist_to_spanner_loc + 1 # +1 for pickup action
            h += spanner_cost

        # 6. Return heuristic value
        return h
