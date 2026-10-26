from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque
import sys # Import sys for float('inf')

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
    This heuristic estimates the number of actions required to tighten all loose
    nuts specified in the goal. It uses a greedy approach, iteratively selecting
    the next loose goal nut that can be tightened with the minimum cost from the
    current estimated state (man's location, spanners available).

    # Assumptions
    - There is exactly one man object.
    - Nuts are static and do not change location.
    - Spanners become unusable after one use for tightening a nut.
    - Spanners cannot be dropped once picked up.
    - The graph of locations connected by 'link' predicates is undirected.
    - All necessary spanners exist and are initially usable and on the ground
      or carried by the man.
    - All locations mentioned in 'at' or 'link' facts are part of a single
      connected graph, or relevant locations are reachable from each other.

    # Heuristic Initialization
    - Identify all nuts that are required to be tightened in the goal state.
    - Build a graph of locations based on 'link' predicates and locations
      mentioned in initial 'at' facts and static 'at' facts for nuts.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Store the static locations of all nuts.
    - Identify the man object.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by simulating a greedy process of tightening
    the remaining loose goal nuts one by one.

    1. Identify the set of loose nuts that are also goal conditions in the current state. If this set is empty, the heuristic is 0.
    2. Determine the man's current location.
    3. Identify usable spanners currently carried by the man and usable spanners on the ground. Also identify unusable spanners carried by the man.
    4. Initialize the total estimated cost to 0.
    5. Maintain the current estimated state: man's location, set of carried usable spanners, dictionary of usable spanners on the ground (spanner -> location), and set of carried unusable spanners. These are copies of the current state's facts.
    6. While there are still loose goal nuts remaining:
       a. For each remaining loose goal nut `n`:
          i. Calculate the estimated cost to tighten this nut *from the current estimated state*. This cost is the sum of actions in the sequence:
             - If the man is not currently carrying a usable spanner:
               - Walk from the estimated man's location to the location of the closest available usable spanner on the ground.
               - Pick up the spanner (cost 1).
               - Walk from the spanner's location to the nut's location.
             - If the man is already carrying a usable spanner:
               - Walk from the estimated man's location directly to the nut's location.
             - The tighten action itself (cost 1).
          ii. If no usable spanners are available anywhere (carried or on ground) but there are still nuts to tighten, the state is likely a dead end; return infinity.
       b. Select the nut `n_best` that has the minimum calculated cost for its sequence.
       c. Add this minimum cost to the total estimated cost.
       d. Update the estimated state based on performing the sequence for `n_best`:
          - The man's location becomes the location of `n_best`.
          - The spanner used for `n_best` is marked as unusable. If it was picked up from the ground, it's removed from the ground set and added to the carried unusable set. If it was already carried usable, it's moved from the carried usable set to the carried unusable set.
          - Remove `n_best` from the set of remaining loose goal nuts.
    7. Return the total estimated cost.

    Shortest path distances between locations are precomputed using BFS on the
    graph formed by 'link' predicates and locations from 'at' facts.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, static facts,
        and precomputing location distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find man and initial locations

        # Identify all nuts that are goals (need to be tightened)
        self.goal_nuts = {
            args[0] for goal in self.goals for predicate, *args in [get_parts(goal)] if predicate == "tightened"
        }

        # Store static nut locations and collect all locations
        self.nut_locations = {}
        locations = set()
        links = []

        # Locations from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == "at" and parts[1].startswith("nut"):
                 self.nut_locations[parts[1]] = parts[2]
                 locations.add(parts[2])
            elif parts[0] == "link":
                 l1, l2 = parts[1], parts[2]
                 links.append((l1, l2))
                 locations.add(l1)
                 locations.add(l2)
            elif parts[0] == "at": # Also include locations of other static objects if any
                 locations.add(parts[2])

        # Locations from initial state 'at' facts (man, spanners, etc.)
        for fact in initial_state:
             parts = get_parts(fact)
             if parts[0] == "at":
                  locations.add(parts[2])
                  if parts[1].startswith("nut"): # Ensure nut locations from initial state are also captured
                      self.nut_locations[parts[1]] = parts[2]


        # Build location graph
        self.location_graph = {loc: set() for loc in locations}
        for l1, l2 in links:
            if l1 in self.location_graph and l2 in self.location_graph: # Only add links between known locations
                self.location_graph[l1].add(l2)
                self.location_graph[l2].add(l1) # Links are bidirectional
            # else: Warning? Or assume problem is well-formed? Assume well-formed.

        # Precompute all-pairs shortest paths
        # Ensure all locations found are included in the graph keys before computing distances
        all_locations_in_graph = list(self.location_graph.keys())
        self.distances = self._compute_all_pairs_shortest_paths(all_locations_in_graph)

        # Identify the man object (assuming only one)
        self.man_object = None
        for fact in initial_state:
             parts = get_parts(fact)
             # Assuming the man is the only object of type 'man' and is located somewhere initially
             # A more robust way would parse object types if available in Task
             if parts[0] == "at" and not parts[1].startswith(("nut", "spanner")):
                 self.man_object = parts[1]
                 break
        if not self.man_object:
             # Fallback or error if man not found in initial 'at' facts
             raise ValueError("Could not identify the man object in the initial state.")


    def _compute_all_pairs_shortest_paths(self, locations):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        for start_loc in locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all reachable locations."""
        distances = {loc: float('inf') for loc in self.location_graph}
        if start_loc in self.location_graph: # Ensure start_loc is in the graph nodes
            distances[start_loc] = 0
            queue = deque([start_loc])

            while queue:
                current_loc = queue.popleft()
                current_dist = distances[current_loc]

                if current_loc in self.location_graph: # Should always be true here
                    for neighbor in self.location_graph[current_loc]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = current_dist + 1
                            queue.append(neighbor)
        # Locations not in the graph or unreachable from start_loc will remain with distance infinity
        return distances

    def get_distance(self, loc1, loc2):
        """Returns the precomputed shortest distance between two locations."""
        if loc1 == loc2:
            return 0
        # Check if locations exist in the precomputed distances map
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This indicates one or both locations were not included in the graph
             # or are unreachable from each other. Return infinity.
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify loose goal nuts
        nuts_remaining = {
            n for n in self.goal_nuts if f"(loose {n})" in state
        }

        if not nuts_remaining:
            return 0 # Goal reached for all nuts

        # 2. Determine man's current location
        current_man_loc = None
        for fact in state:
            if match(fact, "at", self.man_object, "*"):
                current_man_loc = get_parts(fact)[2]
                break
        if not current_man_loc:
             # Man's location not found? Should not happen in valid states.
             return float('inf') # Indicate error or unsolvable state

        # 3. Identify usable spanners and their status
        current_carrying_usable = set()
        current_spanners_on_ground = {} # {spanner_name: location}
        estimated_spanners_carried_unusable = set() # For tracking in estimated state

        usable_spanners_in_state = set()
        # First pass: find all usable spanners in the current state
        for fact in state:
            if match(fact, "usable", "*"):
                usable_spanners_in_state.add(get_parts(fact)[1])

        # Second pass: find locations/carrying status of all spanners
        all_spanners_in_state = set()
        for fact in state:
             if match(fact, "carrying", self.man_object, "*"):
                 spanner = get_parts(fact)[2]
                 all_spanners_in_state.add(spanner)
                 if spanner in usable_spanners_in_state:
                      current_carrying_usable.add(spanner)
                 else:
                      estimated_spanners_carried_unusable.add(spanner) # Initially carried but unusable
             elif match(fact, "at", "*", "*") and get_parts(fact)[1].startswith("spanner"):
                  spanner, loc = get_parts(fact)[1], get_parts(fact)[2]
                  all_spanners_in_state.add(spanner)
                  if spanner in usable_spanners_in_state:
                       current_spanners_on_ground[spanner] = loc
                  # Unusable spanners on ground are irrelevant as they cannot be used


        # 4. Initialize total cost
        total_cost = 0

        # 5. Maintain estimated state (using copies)
        estimated_man_loc = current_man_loc
        estimated_carrying_usable = current_carrying_usable.copy()
        estimated_spanners_on_ground = current_spanners_on_ground.copy()
        estimated_spanners_carried_unusable = estimated_spanners_carried_unusable.copy()


        # 6. Greedy loop
        while nuts_remaining:
            best_nut = None
            min_cost_for_nut = float('inf')
            spanner_to_use = None # Track which spanner would be used for the best nut

            # Find the best nut to tighten next (the one with the minimum sequence cost)
            for nut in nuts_remaining:
                nut_loc = self.nut_locations[nut]

                current_cost_for_nut = 0
                temp_man_loc = estimated_man_loc
                temp_spanner_used = None

                if estimated_carrying_usable:
                    # Use carried usable spanner
                    temp_spanner_used = next(iter(estimated_carrying_usable)) # Pick any carried usable spanner
                    # Walk to nut
                    dist_to_nut = self.get_distance(temp_man_loc, nut_loc)
                    if dist_to_nut == float('inf'): return float('inf') # Nut location unreachable
                    current_cost_for_nut += dist_to_nut
                    temp_man_loc = nut_loc
                    # Tighten
                    current_cost_for_nut += 1
                else:
                    # Need to pick up spanner from the ground
                    closest_spanner = None
                    min_spanner_path_cost = float('inf') # Cost to reach spanner from current man loc

                    # Find the closest usable spanner on the ground based on travel cost
                    for s, s_loc in estimated_spanners_on_ground.items():
                         path_cost = self.get_distance(temp_man_loc, s_loc)
                         if path_cost < min_spanner_path_cost:
                             min_spanner_path_cost = path_cost
                             closest_spanner = s

                    if closest_spanner is None:
                        # No usable spanners available anywhere (carried or on ground)
                        return float('inf') # Indicate unsolvable

                    temp_spanner_used = closest_spanner
                    spanner_loc = estimated_spanners_on_ground[temp_spanner_used]

                    # Walk to spanner
                    dist_to_spanner = self.get_distance(temp_man_loc, spanner_loc)
                    if dist_to_spanner == float('inf'): return float('inf') # Spanner location unreachable
                    current_cost_for_nut += dist_to_spanner
                    temp_man_loc = spanner_loc
                    # Pickup spanner
                    current_cost_for_nut += 1
                    # Walk to nut
                    dist_spanner_to_nut = self.get_distance(temp_man_loc, nut_loc)
                    if dist_spanner_to_nut == float('inf'): return float('inf') # Nut location unreachable from spanner loc
                    current_cost_for_nut += dist_spanner_to_nut
                    temp_man_loc = nut_loc
                    # Tighten
                    current_cost_for_nut += 1

                # Compare and select the best nut
                if current_cost_for_nut < min_cost_for_nut:
                    min_cost_for_nut = current_cost_for_nut
                    best_nut = nut
                    spanner_to_use = temp_spanner_used

            # If min_cost_for_nut is still infinity, it means no nuts are reachable
            if min_cost_for_nut == float('inf'):
                 return float('inf')

            # Add cost of the best nut sequence to total cost
            total_cost += min_cost_for_nut

            # Update estimated state based on the chosen nut and spanner
            nut_loc = self.nut_locations[best_nut]
            estimated_man_loc = nut_loc # Man ends up at the nut location

            if spanner_to_use in estimated_carrying_usable:
                 # Spanner was carried, now unusable
                 estimated_carrying_usable.remove(spanner_to_use)
                 estimated_spanners_carried_unusable.add(spanner_to_use)
            elif spanner_to_use in estimated_spanners_on_ground:
                 # Spanner was on ground, now carried unusable
                 del estimated_spanners_on_ground[spanner_to_use]
                 estimated_spanners_carried_unusable.add(spanner_to_use)
            # else: Should not happen if spanner_to_use was correctly identified

            nuts_remaining.remove(best_nut)

        return total_cost

