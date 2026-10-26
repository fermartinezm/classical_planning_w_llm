# Need to import Heuristic base class. Assuming it's in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Need fnmatch and deque
from fnmatch import fnmatch
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if parts and args have different lengths
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It uses a greedy approach: repeatedly find the closest loose goal nut,
    determine the cost to get the man (with a usable spanner if needed) to that nut's location,
    and add the cost of the tighten action. The spanner used is considered consumed.

    # Assumptions
    - There is only one man.
    - Spanners become unusable after one tighten action.
    - The man can only carry one spanner at a time.
    - There are enough usable spanners available somewhere to tighten all goal nuts in solvable problems.
    - The man can ignore an unusable spanner he might be carrying (no explicit drop action needed for unusable spanners).
    - Object names follow conventions (e.g., 'nut' for nuts, 'spanner' for spanners).
    - Nuts always have an 'at' location.
    - Usable spanners are either carried by the man or are at an 'at' location.

    # Heuristic Initialization
    - Identify all locations mentioned in the problem (from 'link' and initial 'at' facts).
    - Build an adjacency list representing the connectivity between locations based on 'link' facts.
    - Compute all-pairs shortest path distances between locations using Breadth-First Search (BFS).
    - Identify the names of all nuts that are part of the goal condition.
    - Identify the name of the man (inferred from initial state facts).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Parse State:** Extract the man's current location, the locations of all nuts and spanners, which spanners are usable, which nuts are loose, and if the man is carrying a spanner.
    2.  **Identify Needed Nuts:** Determine the set of nuts that are goal conditions and are currently loose in the state. If this set is empty, the heuristic is 0 (goal reached).
    3.  **Identify Available Spanners:** Determine the set of usable spanners that are currently on the ground (not carried by the man).
    4.  **Check Carried Spanner:** Determine if the man is currently carrying a *usable* spanner.
    5.  **Initialize Heuristic Calculation State:** Set the current heuristic cost to 0. Set the man's current location for the calculation to his actual location in the state. Create working copies of the set of available usable spanners on the ground and the boolean indicating if the man is carrying a usable spanner. Create a working copy of the set of loose goal nuts.
    6.  **Iterate Through Needed Nuts (Greedy):** While there are still loose goal nuts remaining in the working set:
        a.  **Select Next Nut:** Find the nut in the working set of loose goal nuts that is closest to the man's current location (in the heuristic calculation state) based on the precomputed shortest path distances.
        b.  **Calculate Cost for Selected Nut:**
            - If the man is currently carrying a usable spanner (in the heuristic calculation state):
                - The cost to tighten this nut is the shortest distance from the man's current location to the selected nut's location (walk action) plus 1 (tighten action).
                - Update the heuristic calculation state: the man is no longer carrying a usable spanner (it was used).
            - If the man is not currently carrying a usable spanner:
                - Find the usable spanner on the ground (from the working set of available spanners) that is closest to the man's current location.
                - If no usable spanners are available on the ground, the problem is unsolvable from this state; return infinity.
                - The cost to get the spanner is the shortest distance from the man's current location to the closest spanner's location (walk) + 1 (pickup).
                - The cost to get from the spanner location to the nut location is the shortest distance from the spanner's location to the selected nut's location (walk).
                - The cost to tighten is 1.
                - Total cost for this nut = Cost to get spanner + Cost to get to nut + Cost to tighten.
                - Update the heuristic calculation state: remove the chosen spanner from the working set of available spanners on the ground. The man is still not carrying a usable spanner (the one he picked up was immediately used).
        c.  **Update Heuristic Cost:** Add the calculated cost for this nut to the total heuristic cost.
        d.  **Update Man's Location:** Set the man's current location (in the heuristic calculation state) to the location of the nut that was just processed.
        e.  **Remove Nut:** Remove the selected nut from the working set of loose goal nuts.
    7.  **Return Total Cost:** After processing all loose goal nuts, return the accumulated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal details.
        """
        super().__init__(task)

        # Identify goal nuts from goal conditions like (tightened nut1)
        self.goal_nuts = {
            n for goal in self.goals if get_parts(goal)[0] == 'tightened' for n in get_parts(goal)[1:]
        }

        locations = set()
        self.adj = {} # Adjacency list for locations

        # Collect locations and build adjacency list from links
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                locations.add(l1)
                locations.add(l2)
                self.adj.setdefault(l1, set()).add(l2)
                self.adj.setdefault(l2, set()).add(l1) # Links are bidirectional

        # Collect locations from initial 'at' facts as well to ensure all relevant locations are included
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at':
                 # The third argument is the location
                 locations.add(parts[2])

        self.locations = list(locations) # Store locations

        # Compute all-pairs shortest paths using BFS
        self.distance = {}
        for start_loc in self.locations:
            self.distance[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distance[start_loc][start_loc] = 0 # Distance to self is 0

            while q:
                current_loc, d = q.popleft()

                # Add neighbors to queue
                for neighbor in self.adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distance[start_loc][neighbor] = d + 1
                        q.append((neighbor, d + 1))

            # Ensure all locations have a distance entry (infinity if unreachable)
            for loc in self.locations:
                 if loc not in self.distance[start_loc]:
                     self.distance[start_loc][loc] = float('inf')


        # Identify the man's name
        # Infer by finding the object in an initial 'at' or 'carrying' fact
        # that is not a spanner, nut, or known location.
        man_candidates = set()
        # Collect potential man names from initial state facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj = parts[1]
                # Heuristic guess: if the object is not a spanner or nut (based on naming)
                # and is not a location, it might be the man.
                if not (obj.startswith('spanner') or obj.startswith('nut') or obj in self.locations):
                     man_candidates.add(obj)
            elif parts[0] == 'carrying':
                 # The first argument of carrying is the man
                 man_candidates.add(parts[1])

        # Assuming there's exactly one man and he appears in an initial 'at' or 'carrying' fact
        # If multiple candidates, pick one. If none, problem. Assume one exists and is identifiable.
        self.man_name = man_candidates.pop() if man_candidates else None
        if self.man_name is None:
             # This case indicates a problem with parsing or instance structure
             # In a real system, proper PDDL parsing including types is needed.
             # For this problem, we proceed assuming the man was identified.
             print("Warning: Could not identify man's name from initial state.")
             # Depending on the problem constraints, raising an error might be appropriate.
             # raise ValueError("Could not identify the man object.")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # --- Parse current state ---
        man_location = None
        nut_locations = {} # {nut_name: location}
        spanner_locations = {} # {spanner_name: location}
        carried_spanner = None
        usable_spanners_in_state = set() # {spanner_name}
        loose_nuts_in_state = set() # {nut_name}
        # tightened_nuts_in_state = set() # Not strictly needed for heuristic calculation

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                    man_location = loc
                elif obj.startswith('nut'):
                    nut_locations[obj] = loc
                elif obj.startswith('spanner'):
                    spanner_locations[obj] = loc
            elif parts[0] == 'carrying':
                m, s = parts[1], parts[2]
                if m == self.man_name:
                    carried_spanner = s
            elif parts[0] == 'usable':
                s = parts[1]
                usable_spanners_in_state.add(s)
            elif parts[0] == 'loose':
                n = parts[1]
                loose_nuts_in_state.add(n)
            # elif parts[0] == 'tightened':
            #     tightened_nuts_in_state.add(parts[1])

        # --- Heuristic Calculation ---

        # Identify loose goal nuts
        needed_nuts = {n for n in self.goal_nuts if n in loose_nuts_in_state}

        if not needed_nuts:
            return 0 # Goal reached for all nuts

        h = 0
        current_location = man_location
        # Usable spanners that are currently on the ground
        current_usable_spanners_on_ground = {s for s in usable_spanners_in_state if s in spanner_locations}
        # Is the man currently carrying a spanner that is usable?
        current_man_carrying_usable = carried_spanner in usable_spanners_in_state if carried_spanner else False

        remaining_nuts_to_tighten = set(needed_nuts)

        while remaining_nuts_to_tighten:
            # Find the closest remaining loose goal nut from the current location
            closest_nut = None
            min_dist_to_nut = float('inf')

            for nut in remaining_nuts_to_tighten:
                nut_loc = nut_locations.get(nut) # Get nut location
                if nut_loc is None:
                    # This nut is a goal but doesn't have an 'at' location? Problematic state.
                    # Assume nuts always have an 'at' location in valid states.
                    return float('inf') # Should not happen in valid states

                dist = self.distance[current_location].get(nut_loc, float('inf')) # Get distance, handle unreachable
                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    closest_nut = nut

            if closest_nut is None:
                 # Cannot reach any remaining nut? Should not happen if needed_nuts was not empty
                 return float('inf')

            nut_loc = nut_locations[closest_nut]

            # Calculate cost to get to the nut with a spanner and tighten it
            if current_man_carrying_usable:
                # Already have a usable spanner, just need to walk to the nut and tighten
                dist_to_nut = self.distance[current_location].get(nut_loc, float('inf'))
                if dist_to_nut == float('inf'):
                    return float('inf') # Cannot reach the nut

                cost_to_tighten_this_nut = dist_to_nut + 1 # walk + tighten
                current_man_carrying_usable = False # Spanner is used

            else:
                # Need to find and pick up a usable spanner first
                closest_spanner = None
                min_dist_to_spanner = float('inf')
                closest_spanner_loc = None

                # Find the closest usable spanner on the ground
                for spanner in current_usable_spanners_on_ground:
                    spanner_loc = spanner_locations.get(spanner) # Get spanner location
                    if spanner_loc is None:
                         # Usable spanner doesn't have an 'at' location? Skip it.
                         continue

                    dist = self.distance[current_location].get(spanner_loc, float('inf')) # Get distance, handle unreachable
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        closest_spanner = spanner
                        closest_spanner_loc = spanner_loc

                if closest_spanner is None:
                    # No usable spanners available on the ground and man isn't carrying one
                    # Problem is unsolvable from this state
                    return float('inf')

                # Cost: walk to spanner + pickup + walk to nut + tighten
                dist_spanner_to_nut = self.distance[closest_spanner_loc].get(nut_loc, float('inf'))
                if min_dist_to_spanner == float('inf') or dist_spanner_to_nut == float('inf'):
                     # Cannot reach the spanner or cannot reach the nut from the spanner location
                     return float('inf')

                cost_to_get_spanner = min_dist_to_spanner + 1 # walk to spanner + pickup
                cost_to_get_to_nut = dist_spanner_to_nut # walk from spanner location to nut location
                cost_to_tighten = 1 # tighten action

                cost_to_tighten_this_nut = cost_to_get_spanner + cost_to_get_to_nut + cost_to_tighten

                # Remove the used spanner from available ones
                current_usable_spanners_on_ground.remove(closest_spanner)
                current_man_carrying_usable = False # Spanner is used

            # If the cost involves infinity, the nut is unreachable
            if cost_to_tighten_this_nut == float('inf'):
                 return float('inf')

            h += cost_to_tighten_this_nut
            current_location = nut_loc # Man is now at the nut's location
            remaining_nuts_to_tighten.remove(closest_nut) # This nut is now tightened

        return h
