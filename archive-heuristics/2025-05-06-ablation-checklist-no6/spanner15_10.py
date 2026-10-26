from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
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
    loose goal nuts. It sums the estimated cost for each loose goal nut
    independently. The cost for a single nut includes the tighten action
    and the estimated cost to get the man to the nut's location with a
    usable spanner.

    # Assumptions:
    - A nut is either loose or tightened. If a goal nut is not tightened,
      it is assumed to be loose.
    - Nut locations are static and given in the initial state.
    - Spanners become unusable after one use.
    - The man can carry multiple spanners.
    - The man object is named 'bob' (based on examples).
    - The location graph is static and defined by 'link' predicates.

    # Heuristic Initialization
    - Extracts goal nuts from the task goals.
    - Builds the location graph from static 'link' facts.
    - Identifies all spanners and nuts and their initial locations from
      the initial state.
    - Precomputes all-pairs shortest paths between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Check if the goal is reached (all goal nuts tightened). If yes, return 0.
    2. Identify the man's current location and which usable spanners he is carrying.
    3. Identify all goal nuts that are not yet tightened (assumed loose) and their static locations.
    4. Identify all usable spanners currently on the ground and their locations.
    5. Count the total number of usable spanners available (carried or on ground). If this is less
       than the number of loose goal nuts, the state is likely unsolvable; return infinity.
    6. Initialize total heuristic cost to 0.
    7. For each loose goal nut:
        a. Add 1 to the cost (for the `tighten_nut` action).
        b. Calculate the minimum cost to get the man to this nut's location while carrying a usable spanner.
           - If the man is already carrying a usable spanner: The cost is the shortest distance from the man's current location to the nut's location.
           - If the man is not carrying a usable spanner: The cost is the minimum over all usable spanners on the ground of (distance from man to spanner + 1 (pickup) + distance from spanner to nut).
           - If it's impossible to get a usable spanner to the nut's location (e.g., locations are disconnected), return infinity.
        c. Add this minimum cost to the nut's cost.
    8. Sum the costs calculated for each loose goal nut to get the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        initial state information, and precomputing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Extract goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

        # Build location graph and get all locations
        self.location_graph = {}
        self.all_locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)[1:]
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Identify all spanners, nuts, and the man from initial state and goals
        self.all_spanners = set()
        self.all_nuts = set()
        self.nut_initial_locations = {} # Store initial/static nut locations
        self.man_name = None # Store man's name

        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj_name, loc = parts[1:]
                 if obj_name.startswith("spanner"):
                     self.all_spanners.add(obj_name)
                 elif obj_name.startswith("nut"):
                     self.all_nuts.add(obj_name)
                     self.nut_initial_locations[obj_name] = loc # Store nut location
                 # Assume the object named 'bob' is the man based on examples
                 if obj_name == 'bob':
                     self.man_name = obj_name

             elif parts[0] == "carrying":
                  carrier, obj = parts[1:]
                  # Assume the carrier 'bob' is the man
                  if carrier == 'bob' and obj.startswith("spanner"):
                      self.all_spanners.add(obj)
                      self.man_name = carrier # Confirm man's name

        # Fallback if man's name wasn't found (e.g., not in initial state facts)
        if self.man_name is None:
             self.man_name = 'bob' # Assume 'bob' based on examples

        self.all_nuts.update(self.goal_nuts) # Ensure all goal nuts are known

        # Precompute all-pairs shortest paths
        self.all_pairs_distances = self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS from each location.
        Returns a dictionary {loc1: {loc2: distance}}.
        """
        distances = {}
        for start_loc in self.all_locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_location):
        """
        Performs BFS from start_location to find distances to all reachable locations.
        Returns a dictionary {location: distance}.
        """
        dists = {loc: float('inf') for loc in self.all_locations}
        if start_location not in self.all_locations:
             return dists # Start location not in graph

        dists[start_location] = 0
        queue = deque([start_location])

        while queue:
            current_loc = queue.popleft()
            current_dist = dists[current_loc]

            if current_loc in self.location_graph: # Check if node has neighbors
                for neighbor in self.location_graph[current_loc]:
                    if dists[neighbor] == float('inf'):
                        dists[neighbor] = current_dist + 1
                        queue.append(neighbor)
        return dists

    def get_distance(self, loc1, loc2):
        """Retrieves precomputed distance between loc1 and loc2."""
        if loc1 in self.all_pairs_distances and loc2 in self.all_pairs_distances[loc1]:
            return self.all_pairs_distances[loc1][loc2]
        return float('inf') # Locations are unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Extract state information
        man_loc = None
        man_carried_usable_spanners = set()
        usable_spanner_ground_locs = set() # {location}

        spanner_locations = {} # {spanner_name: location}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1:]
                if obj == self.man_name:
                    man_loc = loc
                elif obj in self.all_spanners:
                    spanner_locations[obj] = loc

            elif parts[0] == "carrying":
                 carrier, obj = parts[1:]
                 if carrier == self.man_name and obj in self.all_spanners:
                     # Check if the carried spanner is usable
                     if f"(usable {obj})" in state:
                         man_carried_usable_spanners.add(obj)

        man_carrying_usable_spanner = len(man_carried_usable_spanners) > 0

        # Find loose goal nuts and their static locations
        loose_goal_nuts_with_loc = {} # {nut_name: location}
        for nut in self.goal_nuts:
            # Assume a goal nut is loose if it's not tightened
            if f"(tightened {nut})" not in state:
                 # Get its static location from initial state
                 if nut in self.nut_initial_locations:
                     loose_goal_nuts_with_loc[nut] = self.nut_initial_locations[nut]
                 # else: This loose goal nut has no known location, likely an issue with problem definition

        # Find usable spanners on the ground
        for spanner, loc in spanner_locations.items():
             if f"(usable {spanner})" in state:
                 usable_spanner_ground_locs.add(loc)

        # If no loose goal nuts, goal is reached (checked at start)
        if not loose_goal_nuts_with_loc:
             return 0

        # Check if enough usable spanners exist in total (carried + on ground)
        usable_spanners_total_set = set(man_carried_usable_spanners)
        for spanner, loc in spanner_locations.items():
             if f"(usable {spanner})" in state:
                  usable_spanners_total_set.add(spanner)

        if len(loose_goal_nuts_with_loc) > len(usable_spanners_total_set):
             # Cannot tighten all nuts with available usable spanners
             return float('inf') # Unsolvable

        total_heuristic_cost = 0

        # Calculate cost for each loose goal nut independently
        for nut, nut_loc in loose_goal_nuts_with_loc.items():
            cost_for_this_nut = 0

            # Cost for tighten_nut action
            cost_for_this_nut += 1

            # Cost to get man to nut_loc with a usable spanner
            cost_to_get_man_with_spanner = float('inf')

            # Option 1: Use a spanner already carried by the man (if any usable)
            if man_carried_usable_spanners:
                 dist_to_nut = self.get_distance(man_loc, nut_loc)
                 if dist_to_nut != float('inf'):
                     cost_to_get_man_with_spanner = min(cost_to_get_man_with_spanner, dist_to_nut)

            # Option 2: Pick up a usable spanner from the ground
            if usable_spanner_ground_locs:
                # Find the minimum cost path: man_loc -> spanner_loc -> nut_loc + pickup cost
                min_spanner_pickup_walk_cost = float('inf')
                for s_loc in usable_spanner_ground_locs:
                    dist_man_to_spanner = self.get_distance(man_loc, s_loc)
                    dist_spanner_to_nut = self.get_distance(s_loc, nut_loc)
                    if dist_man_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                         # Cost is walk to spanner + pickup + walk to nut
                         current_path_cost = dist_man_to_spanner + 1 + dist_spanner_to_nut
                         min_spanner_pickup_walk_cost = min(min_spanner_pickup_walk_cost, current_path_cost)

                cost_to_get_man_with_spanner = min(cost_to_get_man_with_spanner, min_spanner_pickup_walk_cost)


            # Add the cost to get man with spanner to the nut location
            if cost_to_get_man_with_spanner == float('inf'):
                 # If man cannot reach the nut location with a spanner, this nut is currently untightenable
                 return float('inf') # Unsolvable

            cost_for_this_nut += cost_to_get_man_with_spanner

            total_heuristic_cost += cost_for_this_nut

        return total_heuristic_cost
