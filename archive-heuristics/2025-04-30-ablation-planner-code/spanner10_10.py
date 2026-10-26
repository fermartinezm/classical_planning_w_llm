# Need to import necessary modules
from fnmatch import fnmatch
from collections import deque
import math

# Helper functions used by the heuristic
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
    # Ensure fact is a string before attempting to split
    if not isinstance(fact, str):
        return False
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, unless args has wildcards
    # A simple check is to just zip and compare, which handles differing lengths implicitly
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Assume Heuristic base class exists and has __init__(self, task) and __call__(self, node)
# The problem asks for *only* the domain-dependent heuristic code, so we won't include the base class definition.
# The class should inherit from Heuristic in a real planning system context.
# class spannerHeuristic(Heuristic): # Assuming inheritance
class spannerHeuristic: # Standalone for code output

    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose
    nuts specified in the goal. It uses a greedy approach: repeatedly calculate
    the minimum cost to tighten any *one* remaining loose nut from the man's
    current location (including acquiring a spanner if needed), add this cost
    to the total, and update the man's location and available spanners.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Each nut requires a separate, usable spanner.
    - The man can only carry one spanner at a time.
    - The problem instance is solvable (i.e., there are enough usable spanners
      available throughout the problem to tighten all goal nuts, and the graph
      of locations is connected such that all necessary locations are reachable).
    - Action costs are 1.

    # Heuristic Initialization
    - Identify all locations from the problem objects.
    - Build a graph of locations based on the `link` static facts.
    - Precompute all-pairs shortest path distances between locations using BFS.
    - Identify the set of goal nuts from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify the set of loose nuts that are also goal nuts. If this set is empty, the heuristic is 0.
    3. Identify the set of usable spanners currently available (either on the ground or carried by the man) and their locations.
    4. Check if the man is currently carrying a usable spanner.
    5. Initialize the total heuristic cost to 0.
    6. Initialize the man's current location for the heuristic calculation to his actual current location.
    7. Create mutable lists/dictionaries of the remaining loose goal nuts and available usable spanners on the ground.
    8. While there are still loose goal nuts remaining:
       a. Calculate the estimated cost to tighten *each* remaining loose nut `n` at `loc_n` from the man's `current_man_loc`:
          - If the man is currently carrying a usable spanner: The cost is the distance from `current_man_loc` to `loc_n` plus 1 (for the `tighten_nut` action).
          - If the man is *not* currently carrying a usable spanner: Find the nearest available usable spanner `s` at `loc_s` from `current_man_loc`. The cost is the distance from `current_man_loc` to `loc_s` plus 1 (for `pickup_spanner`), plus the distance from `loc_s` to `loc_n` plus 1 (for `tighten_nut`).
       b. Select the loose nut `n_best` that has the minimum calculated cost `cost_best`.
       c. Add `cost_best` to the total heuristic cost.
       d. Update the man's `current_man_loc` to `loc(n_best)`.
       e. Update spanner availability: If the man was carrying a spanner, he is no longer carrying one. If he wasn't carrying one, remove the spanner he picked up from the available ground spanners. In either case, the man ends up *not* carrying a usable spanner after tightening a nut.
       f. Remove `n_best` from the list of remaining loose nuts.
    9. Return the total heuristic cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        # Access goals and static facts from the task object
        self.goals = task.goals
        static_facts = task.static
        # Assuming task.objects is available, mapping object name to type
        # Example: task.objects = {'bob': 'man', 'spanner1': 'spanner', ...}
        self.objects = task.objects # Access objects from task

        # 1. Identify all locations
        self.locations = {obj for obj, obj_type in self.objects.items() if obj_type == 'location'}

        # 2. Build a graph of locations based on the `link` static facts.
        self.graph = {loc: set() for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations from link facts are actually defined objects of type location
                if loc1 in self.locations and loc2 in self.locations:
                    self.graph[loc1].add(loc2)
                    self.graph[loc2].add(loc1) # Links are bidirectional

        # 3. Precompute all-pairs shortest path distances using BFS.
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # 4. Identify the set of goal nuts from the task's goal conditions.
        self.goal_nuts = set()
        # The goal can be a single predicate or a conjunction (and ...)
        # We need to find all (tightened ?n) predicates within the goal structure.
        goal_list = []
        if isinstance(self.goals, str) and match(self.goals, "and", "*"):
             # If it's a conjunction, get the list of sub-goals
             goal_list = get_parts(self.goals)[1:]
        elif isinstance(self.goals, str):
             # If it's a single goal predicate
             goal_list = [self.goals]
        # Handle other potential goal representations if necessary, but string is common

        for goal in goal_list:
            if match(goal, "tightened", "*"):
                 _, nut = get_parts(goal)
                 self.goal_nuts.add(nut)


    def _bfs(self, start_node):
        """Perform BFS from a start node to find distances to all reachable nodes."""
        distances = {node: math.inf for node in self.graph}
        if start_node in self.graph: # Ensure start node is a valid location in the graph
            distances[start_node] = 0
            queue = deque([start_node])

            while queue:
                current_node = queue.popleft()

                # current_node should always be in self.graph if start_node was valid
                for neighbor in self.graph.get(current_node, []):
                    if distances[neighbor] == math.inf:
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def _get_distance(self, loc1, loc2):
        """Get the precomputed distance between two locations."""
        # Return infinity if either location is not in our precomputed distances (e.g., not a valid location object)
        if loc1 not in self.distances or loc2 not in self.distances.get(loc1, {}):
             return math.inf
        return self.distances[loc1][loc2]

    def _find_nearest_object_location(self, from_loc, object_locations_map):
        """
        Find the object with the minimum distance from from_loc among those in object_locations_map.
        Assumes object_locations_map is a dict {object_name: location_name}.
        Returns (nearest_object_name, nearest_object_location, min_distance).
        Returns (None, None, math.inf) if object_locations_map is empty or no reachable object exists.
        """
        min_dist = math.inf
        nearest_obj = None
        nearest_loc = None

        for obj, loc in object_locations_map.items():
            dist = self._get_distance(from_loc, loc)
            if dist < min_dist:
                min_dist = dist
                nearest_obj = obj
                nearest_loc = loc

        return nearest_obj, nearest_loc, min_dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify the man's current location.
        man_loc = None
        man_name = None
        # Assuming there is exactly one man object
        for obj, obj_type in self.objects.items():
            if obj_type == 'man':
                man_name = obj
                break

        if man_name:
             for fact in state:
                 if match(fact, "at", man_name, "*"):
                     _, _, man_loc = get_parts(fact)
                     break

        # If man's location is unknown or not a valid location in the graph, heuristic is infinity
        if man_loc is None or man_loc not in self.locations:
             return math.inf

        # 2. Identify the set of loose nuts that are also goal nuts and their locations.
        loose_goal_nuts_in_state = set()
        nut_locations = {} # {nut_name: location_name}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 # Store location for all nuts (we only care about goal nuts later, but good to have all)
                 if obj in self.objects and self.objects[obj] == 'nut':
                     nut_locations[obj] = loc

        # Filter for loose goal nuts that are currently loose in the state
        for nut in self.goal_nuts:
             if f"(loose {nut})" in state:
                  # Ensure we know the location of this loose goal nut and it's a valid location
                  if nut in nut_locations and nut_locations[nut] in self.locations:
                      loose_goal_nuts_in_state.add(nut)
                  else:
                      # Loose goal nut location unknown or not a valid location - problem state issue?
                      # Assume solvable means all goal nuts have a location in the graph.
                      # If not, this state is likely unreachable or invalid.
                      return math.inf


        # If all goal nuts are tightened (i.e., none are loose goal nuts in state), heuristic is 0.
        if not loose_goal_nuts_in_state:
            return 0

        # 3. Identify the set of usable spanners currently available and their locations.
        usable_spanners_on_ground = {} # {spanner_name: location_name}
        man_carrying_usable_spanner = False

        for fact in state:
            if match(fact, "usable", "*"):
                _, spanner_name = get_parts(fact)
                # Check if this usable spanner is carried by the man
                is_carried = False
                if man_name: # Ensure we found the man
                    if f"(carrying {man_name} {spanner_name})" in state:
                        man_carrying_usable_spanner = True
                        # We don't need the name of the carried spanner for the heuristic logic,
                        # just the boolean flag.
                        is_carried = True

                # If not carried, find its location on the ground
                if not is_carried:
                    for at_fact in state:
                        if match(at_fact, "at", spanner_name, "*"):
                            _, _, spanner_loc = get_parts(at_fact)
                            # Ensure spanner location is a valid location object
                            if spanner_loc in self.locations:
                                usable_spanners_on_ground[spanner_name] = spanner_loc
                            # Found location, move to next usable spanner
                            break


        # 5. Initialize the total heuristic cost to 0.
        h = 0

        # 6. Initialize the man's current location for the heuristic calculation.
        current_man_loc = man_loc

        # 7. Create mutable lists/dictionaries for the simulation.
        remaining_nuts = list(loose_goal_nuts_in_state)
        # We need a copy of ground spanners that we can modify (remove from)
        current_available_ground_spanners = dict(usable_spanners_on_ground)

        # 8. While there are still loose goal nuts remaining:
        while remaining_nuts:
            min_cost_to_tighten_next_nut = math.inf
            best_nut_for_next_step = None
            # We don't strictly need to track the spanner name used in the simulation,
            # just whether one was used from the ground pool.
            spanner_picked_up_for_best_nut = None # Will store the name if a ground spanner is picked up

            # Calculate cost to tighten each remaining nut
            for nut in remaining_nuts:
                # Ensure the nut's location is known and valid (checked already when building loose_goal_nuts_in_state)
                loc_n = nut_locations[nut]
                cost_to_tighten_this_nut = math.inf # Initialize cost for this nut

                if man_carrying_usable_spanner:
                    # Man has a spanner, just need to walk to the nut and tighten
                    dist_to_nut = self._get_distance(current_man_loc, loc_n)
                    if dist_to_nut != math.inf:
                         cost_to_tighten_this_nut = dist_to_nut + 1 # +1 for tighten action
                         # No spanner picked up from ground in this branch
                         spanner_picked_up_for_this_nut = None
                    # else: cost remains math.inf (nut unreachable)

                else: # Man does not have a spanner, needs to get one first
                    # Find the nearest available spanner (on ground)
                    nearest_spanner_name, loc_s, dist_to_spanner = self._find_nearest_object_location(
                        current_man_loc, current_available_ground_spanners
                    )

                    if nearest_spanner_name is not None and dist_to_spanner != math.inf:
                        # Found a spanner, calculate cost to get it and go to nut
                        dist_spanner_to_nut = self._get_distance(loc_s, loc_n)
                        if dist_spanner_to_nut != math.inf:
                             # Cost = walk_to_spanner + pickup + walk_spanner_to_nut + tighten
                             cost_to_tighten_this_nut = dist_to_spanner + 1 + dist_spanner_to_nut + 1
                             spanner_picked_up_for_this_nut = nearest_spanner_name
                        # else: cost remains math.inf (nut unreachable from spanner)
                    # else: cost remains math.inf (no ground spanners or unreachable)


                # Select the nut with the minimum cost among reachable ones
                if cost_to_tighten_this_nut < min_cost_to_tighten_next_nut:
                    min_cost_to_tighten_next_nut = cost_to_tighten_this_nut
                    best_nut_for_next_step = nut
                    spanner_picked_up_for_best_nut = spanner_picked_up_for_this_nut


            # If we found a reachable nut to tighten in this iteration
            if best_nut_for_next_step is not None and min_cost_to_tighten_next_nut != math.inf:
                # c. Add cost_best to the total heuristic cost.
                h += min_cost_to_tighten_next_nut

                # d. Update the man's current_man_loc to loc(n_best).
                current_man_loc = nut_locations[best_nut_for_next_step]

                # e. Update spanner availability
                if man_carrying_usable_spanner:
                    # The spanner he was carrying is now used.
                    man_carrying_usable_spanner = False
                    # The carried spanner is not in current_available_ground_spanners, so no removal needed there.

                else: # Man was not carrying, picked up a ground spanner
                    # The spanner picked up is now used. Remove it from available ground spanners.
                    if spanner_picked_up_for_best_nut in current_available_ground_spanners:
                         del current_available_ground_spanners[spanner_picked_up_for_best_nut]
                    # Man is NOT carrying a spanner after tightening the nut.
                    man_carrying_usable_spanner = False # Already False, but explicit for clarity


                # f. Remove n_best from the list of remaining loose nuts.
                remaining_nuts.remove(best_nut_for_next_step)

            else:
                 # No reachable loose goal nuts found in this iteration.
                 # This state is likely unsolvable or a dead end.
                 # This check should ideally not be hit in solvable problems
                 # before remaining_nuts is empty.
                 return math.inf


        # 9. Return the total heuristic cost.
        return h
