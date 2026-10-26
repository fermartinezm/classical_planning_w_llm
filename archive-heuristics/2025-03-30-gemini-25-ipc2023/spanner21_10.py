import heapq
from collections import deque
from fnmatch import fnmatch
# Ensure this import path matches the environment where the heuristic will be used.
# For example, if the heuristic file is in a 'heuristics' directory:
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at bob shed)" -> ["at", "bob", "shed"]
    Returns an empty list if the fact is not a valid parenthesized string
    or is empty.
    """
    if isinstance(fact, str) and fact.startswith("(") and fact.endswith(")"):
        # Remove parentheses and split, handling potential extra spaces
        content = fact[1:-1].strip()
        if content:
            return content.split()
    return []

class SpannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the PDDL Spanner domain.

    # Summary
    Estimates the cost to tighten all remaining loose nuts required by the goal.
    The cost for each nut is estimated independently based on the current state,
    and the total heuristic value is the sum of these individual costs. Special
    handling is included for the case where the man is already carrying a usable
    spanner, assigning its benefit (action savings) to the nut where it is most
    advantageous compared to fetching a spanner from the ground.

    # Assumptions
    - There is exactly one object of type 'man' in the problem.
    - Nuts do not move from their initial locations; their positions are static.
    - 'link' predicates define static, undirected connections between locations.
    - The heuristic is designed for informativeness in Greedy Best-First Search
      and is not necessarily admissible (it might overestimate the true cost).

    # Heuristic Initialization
    - Identifies the 'man' object based on its type provided in the task description.
    - Identifies all 'nut' objects and stores their fixed locations derived from
      the initial state predicates.
    - Determines the set of nuts that must be in the 'tightened' state to satisfy
      the goal conditions.
    - Parses the static 'link' predicates to construct an undirected graph
      representing the connectivity of locations.
    - Computes all-pairs shortest path distances between all reachable locations
      using Breadth-First Search (BFS) on the location graph. Stores distances
      in `self.dist`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the set of goal nuts that are currently in the 'loose' state (`LooseNuts`)
        based on the current state and the task goals. If this set is empty, the goal
        is satisfied regarding nuts, and the heuristic value is 0.
    2.  Determine the man's current location (`man_loc`) from '(at man loc)' facts.
        If the man's location cannot be determined, return infinity (invalid state).
    3.  Check if the man is carrying a spanner (`carried_spanner`) and if that spanner
        is 'usable' (`has_usable_carried`) based on '(carrying ...)' and '(usable ...)' facts.
    4.  Find all 'usable' spanners currently located on the ground (`usable_spanners_ground`)
        by checking '(at spanner loc)' and '(usable spanner)' facts.
    5.  Check for immediate unsolvability: If `LooseNuts` is not empty, but there are no
        usable spanners available (neither carried nor on the ground), return infinity.
    6.  Check if the number of loose goal nuts exceeds the total number of available usable
        spanners (carried + ground). If so, return infinity, as not all nuts can be tightened.
    7.  Initialize the total heuristic cost `h = 0`.
    8.  If the man is carrying a usable spanner (`has_usable_carried`), determine which
        loose nut (`best_nut_for_carried`) would benefit most from using this spanner.
        This is the nut for which the cost difference [Cost(fetch spanner for n) - Cost(use carried spanner for n)]
        is maximized. Store the index of this nut in `nuts_list`.
    9.  Iterate through each nut `n` in `LooseNuts` (using `nuts_list` and index `i`):
        a. Get the static location `nut_loc` of nut `n`. If unknown, return infinity.
        b. Calculate the estimated cost `cost_n` to tighten this specific nut `n`:
           i.  If `has_usable_carried` is True AND the current nut's index `i` matches
               `best_nut_index_for_carried`:
               - The cost involves traveling from `man_loc` to `nut_loc` and performing the tighten action (cost 1).
               - `cost_n = distance(man_loc, nut_loc) + 1`. If `nut_loc` is unreachable from `man_loc`, return infinity.
           ii. Otherwise (either the man is not carrying a usable spanner, or the carried
               spanner is optimally used for a different nut):
               - The man must fetch a usable spanner from the ground. This involves traveling
                 from `man_loc` to a ground spanner's location (`spanner_loc`), picking it
                 up (cost 1), traveling from `spanner_loc` to `nut_loc`, and tightening (cost 1).
               - Find the ground spanner `s` at `spanner_loc` that minimizes the total travel:
                 `dist(man_loc, spanner_loc) + dist(spanner_loc, nut_loc)`.
               - If no usable ground spanner exists, or if no path exists via any ground spanner
                 to the nut location (i.e., `min_combined_distance` remains infinity), return infinity.
               - `cost_n = min_combined_distance + 1 (pickup) + 1 (tighten)`.
        c. Add `cost_n` to the total heuristic cost `h`.
    10. Return the total estimated cost `h`. If any step determined the state is unsolvable
        or a required location is unreachable, infinity will have been returned earlier.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information from the task.

        Args:
            task: A Task object containing the problem definition (goals, initial state,
                  static facts, objects, types, etc.).
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state

        # Assume task.objects is a dict {'name': 'type'} provided by the planner framework.
        # If the framework provides object info differently, this needs adaptation.
        if not hasattr(task, 'objects') or not isinstance(task.objects, dict):
             raise ValueError("Task object does not provide object types as expected (task.objects dict).")
        self.objects = task.objects

        # 1. Identify goal nuts
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'tightened':
                # Ensure the object is actually a nut if type info is reliable
                nut_name = parts[1]
                if nut_name in self.objects and self.objects.get(nut_name) == 'nut':
                    self.goal_nuts.add(nut_name)
                # else: Optional: Warn if goal refers to non-nut object

        # 2. Identify the man
        self.man = None
        for obj_name, obj_type in self.objects.items():
            # Simple check for 'man' type. Add subtype checks if framework supports/requires it.
            if obj_type == 'man':
                if self.man is not None:
                    # Optional: Add warning or error if multiple men found, as domain implies one.
                    print(f"Warning: Multiple objects of type 'man' found ({self.man}, {obj_name}). Using the first one found.")
                else:
                    self.man = obj_name
        if self.man is None:
            raise ValueError("Could not find an object of type 'man' in the task.")

        # 3. Identify nut locations (static) from initial state
        self.nut_locations = {}
        for fact in initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at':
                obj_name, loc = parts[1], parts[2]
                # Check if the object is a nut using type info
                if obj_name in self.objects and self.objects.get(obj_name) == 'nut':
                    self.nut_locations[obj_name] = loc

        # 4. Build location graph
        self.locations = set()
        adj = {} # Adjacency list for graph

        # Add locations and links from static facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                adj.setdefault(loc1, set()).add(loc2)
                adj.setdefault(loc2, set()).add(loc1) # Assumes bidirectional links

        # Add locations from initial object placements (if not already added via links)
        # Also ensure all locations mentioned in nut_locations are included.
        all_initial_locations = set(self.nut_locations.values())
        for fact in initial_state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at':
                 loc = parts[2]
                 all_initial_locations.add(loc)

        for loc in all_initial_locations:
             if loc not in self.locations:
                 self.locations.add(loc)
                 # Ensure location exists as a key in adj, even if isolated
                 adj.setdefault(loc, set())

        self.adj = adj

        # 5. Compute all-pairs shortest paths using BFS
        self.dist = {loc: {other: float('inf') for other in self.locations} for loc in self.locations}
        for start_node in self.locations:
            # Check if start_node is actually part of the graph connectivity defined by links
            # A location might exist but have no links.
            if start_node not in self.adj:
                 if start_node in self.locations: # If it's a known location
                      self.dist[start_node][start_node] = 0 # Distance to self is 0
                 continue # Skip BFS if no outgoing links

            self.dist[start_node][start_node] = 0
            queue = deque([start_node])
            # Keep track of visited nodes and their distances from start_node
            distances = {start_node: 0}

            while queue:
                current_node = queue.popleft()
                current_dist = distances[current_node]

                # Iterate through neighbors safely using .get() for nodes that might not be in adj keys
                for neighbor in self.adj.get(current_node, set()):
                    if neighbor not in distances: # If neighbor not visited from start_node yet
                        distances[neighbor] = current_dist + 1
                        self.dist[start_node][neighbor] = current_dist + 1
                        queue.append(neighbor)


    def get_distance(self, loc1, loc2):
        """
        Safely retrieves the precomputed shortest path distance between two locations.
        Returns float('inf') if locations are invalid or not connected.
        """
        if loc1 not in self.dist or loc2 not in self.dist[loc1]:
            # This can happen if loc1 or loc2 were not found during init,
            # or if they are in different connected components of the graph.
            return float('inf')
        return self.dist[loc1][loc2]

    def __call__(self, node):
        """ Calculates the heuristic value for the given state node. """
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = set()
        for fact in state:
            parts = get_parts(fact)
            # Ensure parts[1] exists before checking if it's in goal_nuts
            if parts and len(parts) > 1 and parts[0] == 'loose' and parts[1] in self.goal_nuts:
                loose_goal_nuts.add(parts[1])

        if not loose_goal_nuts:
            return 0 # Goal state w.r.t. nuts

        # 2. Find man's location
        man_loc = None
        carried_spanner = None
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue
            # Check for man's location
            if parts[0] == 'at' and parts[1] == self.man:
                man_loc = parts[2]
            # Check if man is carrying something
            elif parts[0] == 'carrying' and parts[1] == self.man:
                # Ensure it's a spanner if type info is reliable
                carried_obj = parts[2]
                if carried_obj in self.objects and self.objects.get(carried_obj) == 'spanner':
                     carried_spanner = carried_obj
                # else: Optional: Warn if carrying non-spanner? Domain seems specific.

        if man_loc is None:
            # Man must have a location defined by 'at' in a valid state.
            # print(f"Warning: Man {self.man} location not found in state.")
            return float('inf') # Indicate invalid or unreachable state

        # 3. Check if carried spanner is usable
        has_usable_carried = False
        if carried_spanner:
            # Check for '(usable spanner_name)' fact in the state set
            if f'(usable {carried_spanner})' in state:
                has_usable_carried = True

        # 4. Find usable spanners on the ground
        usable_spanners_ground = {} # {spanner_name: location}
        spanner_locations = {} # Track all spanner locations on ground
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at':
                 obj_name, loc = parts[1], parts[2]
                 # Check if it's a spanner using type info
                 if obj_name in self.objects and self.objects.get(obj_name) == 'spanner':
                     spanner_locations[obj_name] = loc

        for spanner, loc in spanner_locations.items():
             # Check for '(usable spanner_name)' fact
             if f'(usable {spanner})' in state:
                 usable_spanners_ground[spanner] = loc

        # 5. Check for immediate unsolvability (no usable spanners at all)
        if not has_usable_carried and not usable_spanners_ground:
            # print("Heuristic: Unsolvable - no usable spanners available.")
            return float('inf')

        # 6. Check if enough total usable spanners exist for remaining nuts
        num_nuts_to_tighten = len(loose_goal_nuts)
        total_usable_spanners = len(usable_spanners_ground) + (1 if has_usable_carried else 0)
        if num_nuts_to_tighten > total_usable_spanners:
             # print(f"Heuristic: Unsolvable - need {num_nuts_to_tighten} spanners, only {total_usable_spanners} available.")
             return float('inf')

        # 7. Initialize cost and identify best nut for carried spanner
        h = 0
        nuts_list = list(loose_goal_nuts) # Use a list for indexing
        best_nut_index_for_carried = -1 # Index in nuts_list
        max_saving = -float('inf') # Use -inf to handle cases where saving might be negative

        if has_usable_carried:
            for i, nut in enumerate(nuts_list):
                if nut not in self.nut_locations: continue # Should not happen
                nut_loc = self.nut_locations[nut]

                # Cost using carried spanner: travel + tighten
                dist_mn = self.get_distance(man_loc, nut_loc)
                if dist_mn == float('inf'): continue # Cannot reach this nut, so cannot use carried spanner for it
                cost_carry = dist_mn + 1

                # Cost fetching spanner from ground (only possible if ground spanners exist)
                cost_fetch = float('inf')
                if usable_spanners_ground:
                    min_combined_dist = float('inf')
                    found_fetch_path = False
                    for spanner, spanner_loc in usable_spanners_ground.items():
                        dist_ms = self.get_distance(man_loc, spanner_loc)
                        dist_sn = self.get_distance(spanner_loc, nut_loc)
                        # Ensure both path segments are possible
                        if dist_ms != float('inf') and dist_sn != float('inf'):
                            min_combined_dist = min(min_combined_dist, dist_ms + dist_sn)
                            found_fetch_path = True

                    if found_fetch_path:
                        cost_fetch = min_combined_dist + 2 # travel + pickup + tighten

                # Calculate saving = Cost(fetch) - Cost(carry) if fetching is possible
                if cost_fetch != float('inf'):
                    saving = cost_fetch - cost_carry
                    if saving > max_saving:
                        max_saving = saving
                        best_nut_index_for_carried = i
                # If fetching is impossible, but using carried is possible, this nut is a candidate
                # Assign it if no other positive saving has been found yet.
                elif best_nut_index_for_carried == -1:
                     best_nut_index_for_carried = i

            # Final check: if a best_nut was chosen, ensure it's actually reachable
            if best_nut_index_for_carried != -1:
                 nut_for_check = nuts_list[best_nut_index_for_carried]
                 if nut_for_check not in self.nut_locations or \
                    self.get_distance(man_loc, self.nut_locations[nut_for_check]) == float('inf'):
                      best_nut_index_for_carried = -1 # Reset if chosen nut is unreachable


        # 8. Calculate total cost summing per-nut costs
        for i, nut in enumerate(nuts_list):
            if nut not in self.nut_locations:
                # print(f"Error: Location for nut {nut} not found during calculation.")
                return float('inf') # Should not happen if goal/init state is valid
            nut_loc = self.nut_locations[nut]
            cost_n = 0

            # Determine if the carried spanner (if available) is used for THIS nut
            use_carried = (has_usable_carried and i == best_nut_index_for_carried)

            if use_carried:
                dist_mn = self.get_distance(man_loc, nut_loc)
                # Reachability should have been confirmed when selecting best_nut_index_for_carried
                if dist_mn == float('inf'):
                     # This indicates an issue in the saving calculation logic or state inconsistency
                     # print(f"Error: Assigned carried spanner to unreachable nut {nut}.")
                     return float('inf')
                cost_n = dist_mn + 1 # travel + tighten
            else:
                # Need to fetch a spanner from the ground
                if not usable_spanners_ground:
                    # No ground spanners, and not using the carried one for this nut
                    # This case should be caught by the earlier check num_nuts > total_spanners
                    # If has_usable_carried is true, but best_nut_index is different, we hit this.
                    # If has_usable_carried is false, we also hit this.
                    # This implies impossibility if no ground spanners remain.
                    # print(f"Error: No ground spanners available to tighten nut {nut}.")
                    return float('inf')

                min_combined_dist = float('inf')
                found_path = False
                for spanner, spanner_loc in usable_spanners_ground.items():
                    dist_ms = self.get_distance(man_loc, spanner_loc)
                    dist_sn = self.get_distance(spanner_loc, nut_loc)
                    # Ensure both path segments are possible
                    if dist_ms != float('inf') and dist_sn != float('inf'):
                        min_combined_dist = min(min_combined_dist, dist_ms + dist_sn)
                        found_path = True

                if not found_path:
                    # Cannot reach this nut via any usable ground spanner path
                    # print(f"Error: Nut {nut} at {nut_loc} unreachable via any ground spanner from {man_loc}.")
                    return float('inf')
                # Cost = travel_man_spanner + travel_spanner_nut + pickup + tighten
                cost_n = min_combined_dist + 1 + 1

            h += cost_n

        # 9. Return final heuristic value
        # Ensure heuristic is non-negative
        return max(0, h)

