from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
import math # For float('inf')
import collections # For defaultdict

# Define a dummy Heuristic base class if not provided externally for standalone testing
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # print("Warning: heuristics.heuristic_base not found. Using a dummy base class.")
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

# Helper functions to parse PDDL facts
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

# Helper function for All-Pairs Shortest Paths (Floyd-Warshall)
def floyd_warshall(locations, links):
    """
    Computes all-pairs shortest paths on the location graph.

    Args:
        locations: A list or set of location names.
        links: A set of (loc1, loc2) tuples representing bidirectional links.

    Returns:
        A dictionary mapping (loc1, loc2) pairs to their shortest distance.
        Returns float('inf') if no path exists.
    """
    loc_list = list(locations)
    n = len(loc_list)
    loc_to_idx = {loc: i for i, loc in enumerate(loc_list)}

    # Initialize distance matrix
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0

    # Add direct links
    for l1, l2 in links:
        if l1 in loc_to_idx and l2 in loc_to_idx: # Ensure locations are in our list
            i, j = loc_to_idx[l1], loc_to_idx[l2]
            dist[i][j] = 1
            dist[j][i] = 1 # Links are bidirectional

    # Apply Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Convert matrix back to dictionary using location names
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            dist_dict[(loc_list[i], loc_list[j])] = dist[i][j]

    return dist_dict

# Helper function to find closest location from a set
def find_closest_location(start_loc, target_locs, dist_matrix):
    """
    Finds the location in target_locs closest to start_loc using precomputed distances.

    Args:
        start_loc: The starting location.
        target_locs: A set of potential target locations.
        dist_matrix: The precomputed distance dictionary (from floyd_warshall).

    Returns:
        The closest location name, or None if target_locs is empty or all targets are unreachable.
    """
    min_dist = float('inf')
    closest_loc = None

    for target_loc in target_locs:
        distance = dist_matrix.get((start_loc, target_loc), float('inf'))
        if distance < min_dist:
            min_dist = distance
            closest_loc = target_loc

    # Return None if no reachable location was found
    if closest_loc is None or dist_matrix.get((start_loc, closest_loc), float('inf')) == float('inf'):
         return None

    return closest_loc


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It simulates a greedy plan: repeatedly acquire a usable spanner (if not carrying one)
    by going to the closest available spanner location and picking it up, then go to the
    closest remaining loose goal nut location and tighten it. The cost is the sum of
    walk, pickup, and tighten actions in this simulated plan.

    # Assumptions
    - There is only one man.
    - Links between locations are bidirectional.
    - Nuts are static (do not move).
    - Spanners become unusable after one use for tightening a nut.
    - Solvable instances have enough usable spanners available throughout the plan.

    # Heuristic Initialization
    - Identify all locations and links from static facts to build the location graph.
    - Compute all-pairs shortest paths (APSP) on the location graph using Floyd-Warshall.
    - Identify all goal nuts from the task goals.
    - Identify the static location of each nut from initial state or static facts.
    - Identify the man object.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic simulates a greedy process to tighten all loose goal nuts:

    1.  **Identify Current State:** Determine the man's current location, whether he is carrying a usable spanner, the set of usable spanners on the ground and their locations, and the set of goal nuts that are currently loose.
    2.  **Initialize Simulation:** Set the current man's location, spanner-in-hand status, remaining loose goal nuts, and available usable spanner objects on the ground based on the current state. Initialize total cost to 0.
    3.  **Iterate Until Goals Met:** While there are still loose goal nuts remaining:
        a.  **Acquire Spanner (if needed):** If the man is not currently carrying a usable spanner:
            i.  Find the closest usable spanner location among the available spanners on the ground (in the simulation), relative to the man's current location, using the precomputed APSP distances.
            ii. If no usable spanners are available on the ground, the problem is unsolvable from this state; return infinity.
            iii. Add the walk cost (distance) from the man's current location to the closest spanner location to the total cost. Update the man's current location.
            iv. Add the pickup action cost (1) to the total cost.
            v. Update the man's state to carrying a usable spanner. Remove *one* spanner object at the picked-up location from the set of available usable spanners on the ground in the simulation.
        b.  **Tighten Nut:** Find the closest loose goal nut location among the remaining loose nuts, relative to the man's current location, using the precomputed APSP distances.
        c.  Add the walk cost (distance) from the man's current location to the closest nut location to the total cost. Update the man's current location.
        d.  Add the tighten action cost (1) to the total cost.
        e.  Update the man's state to not carrying a usable spanner (as it becomes unusable). Remove the tightened nut from the set of remaining loose goal nuts in the simulation.
    4.  **Return Total Cost:** Once all loose goal nuts have been tightened in the simulation, the accumulated total cost is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and computing APSP.
        """
        # Identify goal nuts from the task goals
        self.goal_nuts = {get_parts(goal)[1] for goal in task.goals if match(goal, "tightened", "*")}

        locations = set()
        links = set()
        self.nut_locations = {} # Map nut object to its static location
        self.man_object = None # Assume one man

        # Collect all facts from initial state and static
        all_facts = set(task.initial_state) | set(task.static)

        # First pass: Identify locations and links
        for fact in all_facts:
            parts = get_parts(fact)
            if parts[0] == "link":
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                links.add((l1, l2))
                links.add((l2, l1)) # Links are bidirectional
            elif parts[0] == "at":
                 # Collect locations mentioned in 'at' facts
                 locations.add(parts[2])

        # Compute APSP on the identified locations
        self.locations = list(locations) # Store as list for consistent indexing if needed, though dict is used
        self.dist_matrix = floyd_warshall(self.locations, links)

        # Second pass: Identify object types and static nut locations
        potential_men = set()
        potential_spanners = set()
        potential_nuts = set()
        potential_locatables = set()

        for fact in all_facts:
             parts = get_parts(fact)
             if parts[0] == "at":
                 obj, loc = parts[1], parts[2]
                 potential_locatables.add(obj)
                 # Store initial nut locations (nuts are static)
                 if obj in self.goal_nuts:
                     self.nut_locations[obj] = loc
             elif parts[0] == "carrying":
                 m, s = parts[1], parts[2]
                 potential_men.add(m)
                 potential_spanners.add(s)
             elif parts[0] == "usable":
                 s = parts[1]
                 potential_spanners.add(s)
             elif parts[0] == "loose" or parts[0] == "tightened":
                 n = parts[1]
                 potential_nuts.add(n)

        # Identify the man object (assuming a single man)
        # Prioritize finding the object involved in 'carrying'
        if len(potential_men) == 1:
            self.man_object = list(potential_men)[0]
        else:
             # Fallback: Find the unique locatable that isn't a spanner or nut
             # This assumes all spanners and nuts are locatable, and the man is the only other locatable
             locatables_not_spanner_nut = potential_locatables - potential_spanners - potential_nuts
             if len(locatables_not_spanner_nut) == 1:
                 self.man_object = list(locatables_not_spanner_nut)[0]
             else:
                 # If still ambiguous, try finding the object that is 'at' a location and potentially 'carrying'
                 men_at_and_carrying = {p[1] for p in all_facts if match(p, "at", "*", "*")} & {p[1] for p in all_facts if match(p, "carrying", "*", "*")}
                 if len(men_at_and_carrying) == 1:
                      self.man_object = list(men_at_and_carrying)[0]
                 else:
                      # If we still can't find a unique man, the heuristic might fail.
                      # For typical spanner problems, one of the above should work.
                      self.man_object = None # Indicate failure to find man
                      # print("Warning: Could not uniquely identify the man object.")


        # Ensure nut locations are mapped for all goal nuts (they should be in initial state)
        # This is a safety check; PDDL requires objects to be located in initial state.
        for nut in self.goal_nuts:
             if nut not in self.nut_locations:
                  # Find the nut's location in the initial state
                  found_loc = False
                  for fact in task.initial_state:
                       if match(fact, "at", nut, "*"):
                            self.nut_locations[nut] = get_parts(fact)[2]
                            found_loc = True
                            break
                  if not found_loc:
                       # This would make the problem unsolvable, but the heuristic might not detect it cleanly.
                       # For now, we proceed, but find_closest_location might return None later.
                       pass # print(f"Error: Location for goal nut {nut} not found in initial state.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Parse current state
        at_map = {} # object -> location
        carrying_map = {} # man -> spanner
        usable_spanners = set() # spanners that are currently usable
        loose_nuts = set() # nuts that are currently loose

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                at_map[obj] = loc
            elif parts[0] == "carrying":
                m, s = parts[1], parts[2]
                carrying_map[m] = s
            elif parts[0] == "usable":
                s = parts[1]
                usable_spanners.add(s)
            elif parts[0] == "loose":
                n = parts[1]
                loose_nuts.add(n)

        # Identify current man location
        man_loc = at_map.get(self.man_object)
        if man_loc is None:
             # Man must be somewhere if the state is valid
             return float('inf') # Should not happen in valid states

        # Identify loose goal nuts in the current state
        loose_goal_nuts = {n for n in self.goal_nuts if n in loose_nuts}

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # Identify usable spanners currently carried by the man
        man_carried_spanner = carrying_map.get(self.man_object)
        man_is_carrying_usable_spanner = (man_carried_spanner is not None) and (man_carried_spanner in usable_spanners)

        # Identify usable spanners currently on the ground
        # These are usable spanners that are not being carried by the man, and are at a location
        usable_spanners_on_ground = {s for s in usable_spanners if s in at_map and s not in carrying_map.values()}


        # --- Greedy Simulation ---
        current_man_loc = man_loc
        spanner_in_hand = man_is_carrying_usable_spanner
        remaining_nuts_to_tighten = set(loose_goal_nuts)
        available_usable_spanners_on_ground_sim = set(usable_spanners_on_ground) # Track actual spanner objects available in simulation

        total_cost = 0

        while remaining_nuts_to_tighten:
            # 1. Acquire Spanner (if needed)
            if not spanner_in_hand:
                # Need a spanner. Find closest available one on the ground in the simulation.
                sim_available_spanner_locs = {at_map.get(s) for s in available_usable_spanners_on_ground_sim if at_map.get(s)}

                closest_s_loc = find_closest_location(current_man_loc, sim_available_spanner_locs, self.dist_matrix)

                if closest_s_loc is None:
                    # No usable spanners left on the ground and not carrying one. Unsolvable.
                    return float('inf')

                # Walk to spanner
                walk_cost_to_spanner = self.dist_matrix.get((current_man_loc, closest_s_loc), float('inf'))
                # Check if reachable (find_closest_location should handle this, but double check)
                if walk_cost_to_spanner == float('inf'):
                     return float('inf')

                total_cost += walk_cost_to_spanner
                current_man_loc = closest_s_loc

                # Pickup spanner
                total_cost += 1 # pickup_spanner action
                spanner_in_hand = True

                # Remove *one* spanner object at closest_s_loc from the available set in simulation
                spanner_picked_up = None
                for s in available_usable_spanners_on_ground_sim:
                    if at_map.get(s) == closest_s_loc:
                        spanner_picked_up = s
                        break
                if spanner_picked_up:
                     available_usable_spanners_on_ground_sim.remove(spanner_picked_up)
                # else: This shouldn't happen if closest_s_loc was found from this set


            # 2. Tighten Nut
            # Now carrying a usable spanner. Find closest remaining loose goal nut.
            nut_locations_to_tighten = {self.nut_locations[n] for n in remaining_nuts_to_tighten}
            closest_n_loc = find_closest_location(current_man_loc, nut_locations_to_tighten, self.dist_matrix)

            if closest_n_loc is None:
                 # Should not happen if remaining_nuts_to_tighten is not empty and nut_locations are mapped
                 return float('inf')

            # Walk to nut
            walk_cost_to_nut = self.dist_matrix.get((current_man_loc, closest_n_loc), float('inf'))
            # Check if reachable
            if walk_cost_to_nut == float('inf'):
                 return float('inf')

            total_cost += walk_cost_to_nut
            current_man_loc = closest_n_loc

            # Tighten nut
            total_cost += 1 # tighten_nut action
            spanner_in_hand = False # Spanner becomes unusable

            # Remove the tightened nut from the remaining set in simulation
            # Find *one* nut object at this location that is still loose and a goal nut
            nut_tightened = None
            for nut in remaining_nuts_to_tighten:
                 if self.nut_locations.get(nut) == closest_n_loc:
                      nut_tightened = nut
                      break
            if nut_tightened:
                 remaining_nuts_to_tighten.remove(nut_tightened)
            # else: Should not happen if closest_n_loc was found from nut_locations_to_tighten

        return total_cost
