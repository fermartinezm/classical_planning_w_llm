from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper functions for parsing PDDL facts
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
    if len(parts) < len(args):
        return False # Fact is shorter than the pattern

    for i, arg in enumerate(args):
        if not fnmatch(parts[i], arg):
            return False

    return True # All pattern parts matched corresponding fact parts

# BFS function to compute distances from a start node in a graph
def bfs(start_node, graph):
    """
    Performs a Breadth-First Search starting from start_node to compute distances
    to all reachable nodes in the graph.
    """
    distances = {node: float('inf') for node in graph}
    if start_node in graph: # Ensure start_node is a valid key in the graph
        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph: # Ensure node exists in graph keys
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It counts the number of loose goal nuts, adds a cost if a spanner needs to be
    acquired, and estimates the movement cost to reach the necessary locations
    (spanner location if needed, and then a nut location).

    # Assumptions
    - The man is the only agent capable of performing actions.
    - Spanners and nuts are static objects at fixed locations unless carried (spanner).
    - The 'link' predicate defines a bidirectional graph of locations.
    - All goal nuts are initially loose and need tightening.
    - There is at least one usable spanner available somewhere if the problem is solvable.
    - The man starts at a location present in the location graph or initial state.
    - All nut and spanner locations are present in the location graph or initial state.

    # Heuristic Initialization
    - Extracts goal conditions to identify nuts that need tightening.
    - Builds a graph of locations based on 'link' facts, including all locations mentioned in links.
    - Adds any locations mentioned in the initial state or goals to the graph nodes to ensure they are included in distance calculations, even if isolated.
    - Precomputes shortest path distances between all pairs of relevant locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all nuts that are goals and are currently loose. Let this set of nuts be `loose_goal_nuts`.
    2. If `loose_goal_nuts` is empty, the heuristic is 0 (goal state).
    3. Initialize the heuristic value `h` with the number of loose goal nuts. This accounts for the 'tighten' action for each nut.
    4. Determine the man's current location (`man_location`).
    5. Check if the man is currently carrying a usable spanner (`carrying_usable_spanner`).
    6. Find the locations of all usable spanners that are currently on the ground (`usable_spanner_locs`).
    7. Find the locations of all loose goal nuts (`loose_goal_nut_locs`).
    8. Calculate the movement cost:
       - If `carrying_usable_spanner` is True: The man needs to move from `man_location` to the location of at least one loose goal nut. The estimated movement cost is the shortest distance from `man_location` to the *closest* location in `loose_goal_nut_locs`.
       - If `carrying_usable_spanner` is False: The man first needs to acquire a usable spanner, then move to the location of at least one loose goal nut. The estimated cost includes:
         - 1 action for picking up the spanner.
         - Movement from `man_location` to a spanner location (`loc_s`).
         - Movement from `loc_s` (after pickup) to a nut location (`loc_n`).
         - Add 1 to `h` for the pickup action.
         - The movement cost is estimated as the minimum of `dist(man_location, loc_s) + dist(loc_s, loc_n)` over all pairs of `loc_s` in `usable_spanner_locs` and `loc_n` in `loose_goal_nut_locs`. This finds the best spanner to go to and the best nut to go to *after* getting that spanner.
    9. Add the calculated movement cost to `h`.
    10. Return `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and precomputing distances."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Store goal nuts (objects that need to be tightened)
        self.goal_nuts = set()
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "tightened":
                self.goal_nuts.add(args[0])

        # Build the location graph from 'link' facts
        self.location_graph = {}
        self.all_linked_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1) # Links are bidirectional
                self.all_linked_locations.add(loc1)
                self.all_linked_locations.add(loc2)

        # Add any locations mentioned in goals or initial state that might not be linked
        # This ensures BFS covers all potentially relevant locations
        all_relevant_locations = set(self.all_linked_locations)
        for fact in task.initial_state | task.goals:
             if match(fact, "at", "*", "*"):
                  all_relevant_locations.add(get_parts(fact)[2])

        # Ensure all relevant locations are in the graph keys for BFS, even if they have no links
        for loc in all_relevant_locations:
             self.location_graph.setdefault(loc, []) # Add nodes with no links if necessary

        # Compute all-pairs shortest paths using BFS from each relevant location
        self.dist = {}
        for start_loc in self.location_graph.keys():
            self.dist[start_loc] = bfs(start_loc, self.location_graph)

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts = set()
        nut_locations = {}
        # First pass to find locations of all goal nuts
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:3]
                 if obj in self.goal_nuts:
                      nut_locations[obj] = loc

        # Second pass to find which goal nuts are loose
        for nut in self.goal_nuts:
             if f"(loose {nut})" in state:
                  loose_goal_nuts.add(nut)

        # Get locations of loose goal nuts
        loose_goal_nut_locs = {nut_locations[nut] for nut in loose_goal_nuts if nut in nut_locations}

        # 2. If no loose goal nuts, we are in a goal state (or equivalent for this heuristic)
        if not loose_goal_nuts:
            return 0

        # 3. Initialize heuristic with tighten actions
        h = len(loose_goal_nuts)

        # 4. Determine man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                man_location = get_parts(fact)[2]
                break
        # If man_location is not in our precomputed distances, something is wrong or state is malformed
        if man_location is None or man_location not in self.dist:
             # This should not happen in a valid spanner instance, but return inf
             return float('inf')

        # 5. Check if man is carrying a usable spanner
        carrying_usable_spanner = False
        carried_spanner_obj = None
        for fact in state:
            if match(fact, "carrying", "bob", "*"):
                carried_spanner_obj = get_parts(fact)[2]
                break # Assuming bob carries at most one spanner

        if carried_spanner_obj:
             # Check if the carried spanner is usable
             if f"(usable {carried_spanner_obj})" in state:
                  carrying_usable_spanner = True

        # 6. Find locations of usable spanners on the ground
        usable_spanners_on_ground = set()
        # Find all usable spanners
        for fact in state:
             if match(fact, "usable", "*"):
                  spanner = get_parts(fact)[1]
                  # Check if this usable spanner is on the ground (not carried)
                  if spanner != carried_spanner_obj:
                       usable_spanners_on_ground.add(spanner)

        # Find locations of usable spanners on the ground
        usable_spanner_locs = set()
        for spanner in usable_spanners_on_ground:
             for fact in state:
                  if match(fact, "at", spanner, "*"):
                       usable_spanner_locs.add(get_parts(fact)[2])
                       break # Assume spanner is only at one location

        # 7. Calculate movement cost
        movement_cost = 0

        if carrying_usable_spanner:
            # Man has spanner, just needs to go to the closest nut location
            min_dist_to_nut = float('inf')
            for loc_n in loose_goal_nut_locs:
                 if loc_n in self.dist[man_location]: # Check if destination is reachable from man's location
                    min_dist_to_nut = min(min_dist_to_nut, self.dist[man_location][loc_n])

            if min_dist_to_nut == float('inf'):
                 # Cannot reach any loose goal nut location from current position
                 return float('inf')
            movement_cost = min_dist_to_nut

        else: # Not carrying usable spanner
            # Man needs to pick up a spanner first, then go to a nut location
            h += 1 # Cost for pickup action

            min_total_dist = float('inf')

            if not usable_spanner_locs:
                 # No usable spanners available on the ground.
                 # If there are loose goal nuts, this state is likely unsolvable.
                 # Return inf.
                 return float('inf') # Cannot pick up a spanner

            # Find the best spanner location (loc_s) and best nut location (loc_n) combination
            # that minimizes dist(man_location, loc_s) + dist(loc_s, loc_n)
            for loc_s in usable_spanner_locs:
                 if loc_s not in self.dist[man_location]:
                      # Cannot reach this spanner location
                      continue

                 dist_to_spanner = self.dist[man_location][loc_s]

                 min_dist_s_to_nut = float('inf')
                 for loc_n in loose_goal_nut_locs:
                      if loc_n in self.dist[loc_s]: # Check if nut location is reachable from spanner location
                           min_dist_s_to_nut = min(min_dist_s_to_nut, self.dist[loc_s][loc_n])

                 if min_dist_s_to_nut == float('inf'):
                      # Cannot reach any loose goal nut location from this spanner location
                      continue

                 total_dist_via_spanner = dist_to_spanner + min_dist_s_to_nut
                 min_total_dist = min(min_total_dist, total_dist_via_spanner)

            if min_total_dist == float('inf'):
                 # Cannot reach any spanner and then any nut
                 return float('inf')

            movement_cost = min_total_dist

        h += movement_cost

        return h
