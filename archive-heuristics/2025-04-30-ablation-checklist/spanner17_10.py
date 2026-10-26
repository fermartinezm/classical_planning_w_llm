from collections import deque
from fnmatch import fnmatch
import math

# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided externally for testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

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

# Heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all required nuts by greedily selecting
    the next nut that is cheapest to tighten from the current location,
    considering the need to pick up a spanner if not already carrying one.
    Assumes the man can carry only one spanner at a time.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        computing shortest paths.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find initial nut locations

        # Build the location graph from link facts
        self.graph = {}
        self.locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.graph.setdefault(loc1, []).append(loc2)
                self.graph.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Collect all potential locations from initial state, goals, and static facts.
        potential_locations = set()
        for fact in self.initial_state | self.goals | self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 potential_locations.add(parts[2])
            elif parts[0] == 'link' and len(parts) == 3:
                 potential_locations.add(parts[1])
                 potential_locations.add(parts[2])

        self.locations.update(potential_locations)

        # Add isolated locations to the graph structure
        for loc in self.locations:
             self.graph.setdefault(loc, [])

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = self._bfs(start_loc)

        # Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                _, nut_name = get_parts(goal)
                self.goal_nuts.add(nut_name)

        # Find initial locations of all goal nuts
        self.nut_initial_locations = {}
        for nut_name in self.goal_nuts:
             # Find the (at nut_name loc) fact in the initial state
             found = False
             for fact in self.initial_state:
                  if match(fact, "at", nut_name, "*"):
                       _, obj_name, loc_name = get_parts(fact)
                       self.nut_initial_locations[nut_name] = loc_name
                       found = True
                       break
             # If a goal nut is not in the initial state with an (at) predicate, something is wrong.
             # This heuristic assumes goal nuts are at fixed initial locations.
             if not found:
                  # If not found, the heuristic might return inf later if this nut needs tightening.
                  pass


    def _bfs(self, start_node):
        """Performs BFS to find shortest paths from start_node to all reachable nodes."""
        distances = {loc: math.inf for loc in self.locations}
        if start_node in distances: # Ensure start_node is a known location
             distances[start_node] = 0
             queue = deque([start_node])

             while queue:
                 current = queue.popleft()
                 if current in self.graph: # Handle isolated nodes
                     for neighbor in self.graph[current]:
                         if distances[neighbor] == math.inf:
                             distances[neighbor] = distances[current] + 1
                             queue.append(neighbor)
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the shortest path distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This happens if loc1 or loc2 is not a recognized location, or if loc2 is unreachable from loc1.
             # Unreachable locations will have distance math.inf.
             # If loc1 or loc2 is not in the precomputed distances dicts, it means it wasn't identified as a location.
             # This shouldn't happen if location parsing is correct.
             return math.inf
        return self.distances[loc1][loc2]

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify man's current location and name
        man_location = None
        man_name = None

        # Identify all spanners and nuts currently in the state
        all_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}
        all_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*") or match(fact, "tightened", "*")}

        # Find the man by looking for the object at a location that is not a spanner or nut.
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc_name = get_parts(fact)[1:]
                # If this object is not a known spanner and not a known nut, it's likely the man.
                if obj_name not in all_spanners_in_state and obj_name not in all_nuts_in_state:
                    man_name = obj_name
                    man_location = loc_name
                    break # Assuming one man

        # If man not found via (at), try finding the object that is 'carrying' something.
        # This object must be the man.
        if man_name is None:
             for fact in state:
                  if match(fact, "carrying", "*", "*"):
                       man_name = get_parts(fact)[1]
                       # If man is carrying, his location must still be given by (at man loc)
                       # Re-search for his location now that we know his name.
                       for at_fact in state:
                            if match(at_fact, "at", man_name, "*"):
                                 man_location = get_parts(at_fact)[2]
                                 break
                       break # Found the man

        if man_location is None:
             # Cannot find man's location. This state is likely unreachable or malformed.
             return math.inf


        # 2. Identify loose nuts in the goal and their locations
        untightened_goal_nuts = {} # {nut_name: location}
        for nut_name in self.goal_nuts:
            # Check if (tightened nut_name) is NOT in state
            if f"(tightened {nut_name})" not in state:
                 # Find the location of this nut (assuming nuts don't move from initial location)
                 # We stored initial locations in __init__
                 if nut_name in self.nut_initial_locations:
                      untightened_goal_nuts[nut_name] = self.nut_initial_locations[nut_name]
                 else:
                      # Location for a goal nut not found in initial state. Problematic.
                      # Return infinity as we cannot estimate cost to reach it.
                      return math.inf


        # 3. Identify usable spanners and their status (carried/at location)
        usable_spanners_carried = [] # List of spanner names
        usable_spanners_at_locs = [] # List of (spanner_name, location) tuples

        for fact in state:
            if match(fact, "usable", "*"):
                _, spanner_name = get_parts(fact)
                # Check if the identified man is carrying it
                if man_name and f"(carrying {man_name} {spanner_name})" in state:
                    usable_spanners_carried.append(spanner_name)
                else:
                    # Check its location
                    found_at_loc = False
                    for at_fact in state:
                        if match(at_fact, "at", spanner_name, "*"):
                            _, obj, loc = get_parts(at_fact)
                            usable_spanners_at_locs.append((spanner_name, loc))
                            found_at_loc = True
                            break
                    # If usable but neither carried nor at location, it's an issue. Ignore it for heuristic.

        # 4. Check solvability based on spanner count
        num_nuts_to_tighten = len(untightened_goal_nuts)
        num_usable_spanners = len(usable_spanners_carried) + len(usable_spanners_at_locs)

        if num_nuts_to_tighten > num_usable_spanners:
            return math.inf # Problem is unsolvable with available usable spanners

        # If no nuts need tightening, goal is reached, heuristic is 0.
        if num_nuts_to_tighten == 0:
             return 0

        # 5. Compute heuristic cost using greedy approach
        current_man_location = man_location
        remaining_nuts = list(untightened_goal_nuts.items()) # [(nut_name, loc), ...]
        available_spanners_at_locs = list(usable_spanners_at_locs) # [(spanner_name, loc), ...]
        available_spanners_carried_count = len(usable_spanners_carried)
        total_cost = 0

        while remaining_nuts:
            min_cost_for_next_nut = math.inf
            best_nut_info = None # (nut_name, L_N)
            best_spanner_info = None # (spanner_name, L_S) or None if carried

            # Find the best nut to process next
            for nut_name, L_N in remaining_nuts:
                cost_to_tighten_this_nut = math.inf
                spanner_used_info = None # To track which spanner option was cheaper for this nut candidate

                # Option 1: Use a carried spanner (if available)
                cost_option_carried = math.inf
                if available_spanners_carried_count > 0:
                    cost_walk = self.get_distance(current_man_location, L_N)
                    if cost_walk != math.inf:
                         cost_option_carried = cost_walk + 1 # +1 for tighten action

                # Option 2: Pick up a spanner (if available at locations)
                cost_option_pickup = math.inf
                spanner_info_for_pickup = None # (spanner_name, L_S)

                if available_spanners_at_locs:
                    min_spanner_pickup_path_cost = math.inf
                    temp_spanner_info = None
                    for spanner_name, L_S in available_spanners_at_locs:
                        cost_walk_to_spanner = self.get_distance(current_man_location, L_S)
                        cost_walk_spanner_to_nut = self.get_distance(L_S, L_N)

                        if cost_walk_to_spanner != math.inf and cost_walk_spanner_to_nut != math.inf:
                            # Cost = walk to spanner + pickup + walk to nut
                            current_spanner_path_cost = cost_walk_to_spanner + 1 + cost_walk_spanner_to_nut

                            if current_spanner_path_cost < min_spanner_pickup_path_cost:
                                min_spanner_pickup_path_cost = current_spanner_path_cost
                                temp_spanner_info = (spanner_name, L_S)

                    if min_spanner_pickup_path_cost != math.inf:
                         # Cost = min spanner path cost + tighten action
                         cost_option_pickup = min_spanner_pickup_path_cost + 1
                         spanner_info_for_pickup = temp_spanner_info

                # Choose the minimum cost option for this nut
                if cost_option_carried <= cost_option_pickup:
                    cost_to_tighten_this_nut = cost_option_carried
                    spanner_used_info = None # Indicates carried spanner used
                else:
                    cost_to_tighten_this_nut = cost_option_pickup
                    spanner_used_info = spanner_info_for_pickup # Indicates spanner picked up

                # Compare with the minimum cost found so far for any nut
                if cost_to_tighten_this_nut < min_cost_for_next_nut:
                     min_cost_for_next_nut = cost_to_tighten_this_nut
                     best_nut_info = (nut_name, L_N)
                     best_spanner_info = spanner_used_info # Store which spanner was chosen for the best nut

            # If no nut is reachable (either directly or via a spanner), the remaining goals are unreachable
            if min_cost_for_next_nut == math.inf:
                 return math.inf

            # Process the best nut found
            total_cost += min_cost_for_next_nut
            best_nut_name, best_L_N = best_nut_info
            current_man_location = best_L_N

            # Remove the processed nut
            remaining_nuts = [(n, l) for n, l in remaining_nuts if n != best_nut_name]

            # Consume the spanner
            if best_spanner_info is None: # Used a carried spanner
                available_spanners_carried_count -= 1
            else: # Picked up a spanner
                used_spanner_name, used_spanner_loc = best_spanner_info
                available_spanners_at_locs = [(s, l) for s, l in available_spanners_at_locs if s != used_spanner_name]
                # The spanner is consumed by the tighten action.
                # If a spanner was picked up, available_spanners_carried_count must have been 0.
                # After tightening, available_spanners_carried_count remains 0.

        return total_cost
