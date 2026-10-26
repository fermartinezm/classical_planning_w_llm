from fnmatch import fnmatch
from collections import deque
import math

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Note: The Heuristic base class is assumed to be provided by the planning framework.
# The code below defines the spannerHeuristic class inheriting from it.
# If running this code standalone, you would need a definition for Heuristic.

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and starts/ends with parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected input, maybe raise an error or return None/empty list
        # For PDDL facts represented as strings, this should be fine.
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions (tighten, pickup, walk)
    required to tighten all loose nuts specified in the goal. It considers
    the number of nuts to tighten, the travel needed for the man to visit
    the nut locations, and the cost to acquire a usable spanner if the man
    doesn't currently possess one.

    # Assumptions
    - There is exactly one man object.
    - Nut locations are static.
    - Spanner usability decreases after one use (tightening a nut).
    - The graph of locations connected by 'link' predicates is static.
    - All locations relevant to the problem (man's location, nut locations, usable spanner locations) are reachable from each other in solvable instances.

    # Heuristic Initialization
    - Identify all locations and build a graph based on 'link' predicates.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify the man object and all spanner objects.
    - Identify the set of nuts that must be tightened to reach the goal.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location. If not found, return infinity.
    2. Identify the set of nuts that are currently loose but need to be tightened according to the goal.
    3. If this set is empty, the heuristic value is 0 (goal reached for these nuts).
    4. The base cost includes one 'tighten_nut' action for each loose nut that needs tightening.
    5. Get the locations of these loose nuts. If any nut location is not found in the state, return infinity.
    6. Calculate the travel cost for the man to visit all locations where these loose nuts are situated.
       - This includes the travel from the man's current location to the *closest* nut location (min_dist_m_cn). If unreachable, return infinity.
       - It also includes estimated travel between the remaining nut locations (estimated as one 'walk' action per additional nut location after the first), stored in `inter_nut_travel`.
    7. Determine if the man is currently carrying a usable spanner.
    8. If the man does *not* have a usable spanner:
       - Find the locations of all usable spanners that are currently on the ground (not carried).
       - If no usable spanners are available anywhere (neither carried nor on the ground), the problem is likely unsolvable from this state, return infinity.
       - Find the closest usable spanner location from the man's current location (min_dist_m_cs). If unreachable, return infinity.
       - Add 1 to the cost for the 'pickup_spanner' action.
       - Add the estimated travel cost required to first reach the closest usable spanner location AND then reach the closest nut location. This is estimated as the minimum of (travel man->spanner->nut) and (travel man->nut->spanner->nut->nut). Specifically, `min(dist(M, CS) + dist(CS, CN), dist(M, CN) + dist(CN, CS) + dist(CS, CN))`.
    9. If the man *does* have a usable spanner:
       - The initial travel is just from the man's current location to the closest nut location (min_dist_m_cn).
    10. Sum up the base cost (tightens), spanner pickup cost (if applicable), the initial travel cost (to get spanner and/or reach first nut), and the inter-nut travel cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Location graph and distances.
        - Man object.
        - All spanner objects.
        - Goal nuts.
        """
        self.task = task
        self.locations = []
        self.man_obj = None
        self.all_spanners = []
        self.goal_nuts = set()

        # 1. Identify objects by type
        # task.objects is a list of tuples like (obj1, obj2, ..., '-', type)
        for obj_list_info in task.objects:
            # Ensure obj_list_info is a tuple/list and has at least 3 elements ('obj', '-', 'type')
            if not isinstance(obj_list_info, (tuple, list)) or len(obj_list_info) < 3:
                continue # Skip malformed object lists

            obj_type = obj_list_info[-1]
            # Get object names, excluding the '-' and type
            objs = obj_list_info[:-2]

            if obj_type == 'location':
                self.locations.extend(objs)
            elif obj_type == 'man':
                # Assuming one man
                if self.man_obj is not None:
                     # print(f"Warning: Found multiple man objects. Using the first one: {self.man_obj}")
                     pass # Suppress warning for cleaner output
                self.man_obj = objs[0]
            elif obj_type == 'spanner':
                self.all_spanners.extend(objs)
            # We don't strictly need nut objects here, but could store them
            # elif obj_type == 'nut':
            #     pass

        # Ensure we found the man
        if self.man_obj is None:
             # This should ideally be caught by the parser or be an invalid problem
             # print("Error: Could not identify the man object in the task.")
             pass # Suppress error for cleaner output


        # 2. Build location graph from link facts
        self.graph = {loc: [] for loc in self.locations}
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                if len(parts) == 3: # Ensure it's (link loc1 loc2)
                    _, loc1, loc2 = parts
                    if loc1 in self.graph and loc2 in self.graph:
                        self.graph[loc1].append(loc2)
                        self.graph[loc2].append(loc1) # Links are bidirectional
                    else:
                         # Warning for links involving unknown locations
                         # print(f"Warning: Link fact involves unknown location: {fact}")
                         pass # Suppress warning for cleaner output


        # 3. Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[start_node][start_node] = 0

            while q:
                current_node, dist = q.popleft()

                # Ensure current_node is a valid key in graph
                if current_node not in self.graph:
                    continue # Should not happen if locations list is correct

                for neighbor in self.graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[start_node][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

            # Mark unreachable locations with infinity
            for loc in self.locations:
                if loc not in self.distances[start_node]:
                    self.distances[start_node][loc] = float('inf')


        # 4. Identify goal nuts
        # task.goals is a frozenset of facts like '(tightened nut1)'
        for goal_fact in task.goals:
            if match(goal_fact, "tightened", "*"):
                parts = get_parts(goal_fact)
                if len(parts) == 2: # Ensure it's (tightened nut)
                    _, nut_name = parts
                    self.goal_nuts.add(nut_name)
                else:
                     # print(f"Warning: Unexpected goal fact format: {goal_fact}")
                     pass # Suppress warning for cleaner output


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # 1. Find man's current location
        man_loc = None
        # Ensure self.man_obj was found during init
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    parts = get_parts(fact)
                    if len(parts) == 3: # Ensure it's (at man loc)
                        man_loc = parts[2]
                        break
        if man_loc is None:
             # Man location not found, problem state is likely invalid or unsolvable
             # This also covers the case where self.man_obj was not found in init
             return float('inf')


        # 2. Identify loose nuts needing tightening
        loose_nuts_needed = set()
        nut_locations = {} # Map nut -> location
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 2 and parts[0] == 'loose' and parts[1] in self.goal_nuts:
                 nut_name = parts[1]
                 loose_nuts_needed.add(nut_name)
                 # Find its location from state facts
                 found_loc = False
                 for loc_fact in state:
                     if match(loc_fact, "at", nut_name, "*"):
                         loc_parts = get_parts(loc_fact)
                         if len(loc_parts) == 3: # Ensure it's (at nut loc)
                             nut_loc = loc_parts[2]
                             nut_locations[nut_name] = nut_loc
                             found_loc = True
                             break # Found location for this nut
                 if not found_loc:
                      # Nut location not found in state, problem state is likely invalid
                      return float('inf')


        # 3. If no loose nuts needing tightening, goal is reached for these nuts
        if not loose_nuts_needed:
            return 0

        # 4. Base cost: one tighten action per nut
        num_nuts_to_tighten = len(loose_nuts_needed)
        h = num_nuts_to_tighten

        # Get the set of locations where these nuts are
        required_nut_locations = {nut_locations[n] for n in loose_nuts_needed}

        # 5. Calculate travel cost for the man
        # Find closest nut location from man's current location
        closest_nut_loc = None
        min_dist_m_cn = float('inf')
        for loc in required_nut_locations:
            # Use .get with default inf in case loc is not in self.distances[man_loc]
            # (shouldn't happen if BFS was correct and locations are valid)
            dist = self.distances[man_loc].get(loc, float('inf'))
            if dist < min_dist_m_cn:
                min_dist_m_cn = dist
                closest_nut_loc = loc

        # If closest nut is unreachable, state is unsolvable
        if closest_nut_loc is None or min_dist_m_cn == float('inf'):
             return float('inf')

        # Estimated travel between remaining nuts: num_nuts_to_tighten - 1 walks
        # This is the travel *after* reaching the first nut location
        inter_nut_travel = max(0, num_nuts_to_tighten - 1)


        # 6. Check if man has a usable spanner
        man_has_usable_spanner = False
        # Ensure self.all_spanners was populated
        if self.all_spanners:
            for spanner in self.all_spanners:
                if (f'carrying {self.man_obj} {spanner}') in state and (f'usable {spanner}') in state:
                    man_has_usable_spanner = True
                    break

        # 7. Handle spanner acquisition if needed
        if not man_has_usable_spanner:
            # Find locations of usable spanners on the ground
            usable_spanner_locs = set()
            # Ensure self.all_spanners was populated
            if self.all_spanners:
                for spanner in self.all_spanners:
                     # Check if spanner is usable and not carried by the man
                     if (f'usable {spanner}') in state and (f'carrying {self.man_obj} {spanner}') not in state:
                         # Find its location on the ground
                         for fact in state:
                             if match(fact, "at", spanner, "*"):
                                 loc_parts = get_parts(fact)
                                 if len(loc_parts) == 3: # Ensure it's (at spanner loc)
                                     spanner_loc = loc_parts[2]
                                     usable_spanner_locs.add(spanner_loc)
                                     break # Found location for this spanner

            # If no usable spanners available anywhere to the man (neither carried nor on the ground)
            if not usable_spanner_locs:
                 # If man needs a spanner and none are available on ground/carried usable, it's unsolvable.
                 if num_nuts_to_tighten > 0: # Only need spanner if there are nuts to tighten
                     return float('inf')
                 else:
                     # Should have returned 0 earlier if num_nuts_to_tighten == 0
                     pass # This case should not be reached


            # Find closest usable spanner location from man's current location
            closest_spanner_loc = None
            min_dist_m_cs = float('inf')
            for loc in usable_spanner_locs:
                dist = self.distances[man_loc].get(loc, float('inf'))
                if dist < min_dist_m_cs:
                    min_dist_m_cs = dist
                    closest_spanner_loc = loc

            # If closest spanner is unreachable, state is unsolvable
            if closest_spanner_loc is None or min_dist_m_cs == float('inf'):
                 return float('inf')

            # Add cost for pickup action
            h += 1 # pickup_spanner cost

            # Add travel cost to get spanner AND reach the first nut location
            dist_m_cn = self.distances[man_loc].get(closest_nut_loc, float('inf'))
            dist_cn_cs = self.distances[closest_nut_loc].get(closest_spanner_loc, float('inf')) # Distance from closest nut to closest spanner

            # Check for unreachable paths
            if dist_m_cn == float('inf') or dist_cn_cs == float('inf'):
                 return float('inf')

            # Minimum travel to visit closest spanner (CS) and closest nut (CN) starting from man (M)
            # Option 1: M -> CS -> CN
            travel_option1 = min_dist_m_cs + dist_cn_cs
            # Option 2: M -> CN -> CS -> CN
            travel_option2 = dist_m_cn + dist_cn_cs + dist_cn_cs # Assumes dist(CS, CN) == dist(CN, CS)

            initial_travel = min(travel_option1, travel_option2)

            total_travel = initial_travel + inter_nut_travel
            h += total_travel

        else: # Man already has a usable spanner
            # Travel is just from man's current location to the first nut location + between nuts
            travel_to_first_nut = self.distances[man_loc].get(closest_nut_loc, float('inf'))
            if travel_to_first_nut == float('inf'):
                 return float('inf')

            total_travel = travel_to_first_nut + inter_nut_travel
            h += total_travel

        return h
