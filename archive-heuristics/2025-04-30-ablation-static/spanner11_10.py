from fnmatch import fnmatch
from collections import defaultdict, deque
import math

# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided externally, you would need a definition like:
# class Heuristic:
#     def __init__(self, task):
#         pass
#     def __call__(self, node):
#         pass


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
    # Ensure the fact has at least as many parts as the pattern args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions required to tighten all
    loose nuts that are part of the goal. It calculates the cost for each such nut
    by finding the cheapest way for the man to acquire a usable spanner and travel
    to the nut's location to tighten it. It greedily processes the nuts, assuming
    the man moves from the location of the last tightened nut to the next spanner/nut pair.

    # Assumptions:
    - Each usable spanner can tighten exactly one nut.
    - The man can carry only one spanner at a time.
    - The graph of locations connected by 'link' predicates is static and traversable.
    - All nuts that are part of the goal and are not 'tightened' are considered 'loose'
      and need tightening.
    - All objects (man, spanners, nuts) are initially at a defined location or carried.
    - There are enough usable spanners initially for all nuts that need tightening in the goal.

    # Heuristic Initialization
    - Extracts the set of nuts that must be tightened to reach a goal state.
    - Builds a graph of locations based on 'link' predicates.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the man object (assuming it's the object involved in 'carrying' or starting with 'bob').
    - Stores initial locations of all objects and initial usable spanners for reference.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Nuts to Tighten:** Determine which goal nuts are not currently in the 'tightened' state. These are the nuts that still need to be processed. If this set is empty, the heuristic is 0.
    2.  **Get Object Locations:** Find the current location of the man and each nut that needs tightening. Also, find the location of all spanners that are currently 'usable'. A spanner's location is either where it is 'at' or the man's location if he is 'carrying' it.
    3.  **Identify Available Spanners:** Collect all spanners that are currently marked as 'usable'.
    4.  **Check Solvability:** If the number of usable spanners currently available is less than the number of nuts that still need tightening, the problem is unsolvable from this state. Return infinity.
    5.  **Initialize Calculation:** Set the total estimated cost to 0. Keep track of the man's current location for the calculation (starting with his actual location in the state). Maintain mutable lists of the nuts still needing tightening and the usable spanners still available.
    6.  **Handle Carried Spanner:** If the man is currently carrying a usable spanner, prioritize using it for the first tightening. Find the nut needing tightening that is closest to the man's current location. Calculate the cost to walk to this nut and tighten it (distance + 1 action). Add this cost to the total. Update the man's location for the next step to be the location of the nut just tightened. Remove the processed nut and the carried spanner from the remaining lists.
    7.  **Process Remaining Nuts:** While there are still nuts needing tightening:
        a.  Iterate through all remaining usable spanners and all remaining nuts needing tightening.
        b.  For each pair of (spanner S, nut N), calculate the estimated cost to process this nut using this spanner, starting from the man's current location. This cost involves: walking from the man's current location to the spanner's location, picking up the spanner (1 action), walking from the spanner's location to the nut's location, and tightening the nut (1 action). The cost is `dist(ManLoc, SpannerLoc(S)) + 1 + dist(SpannerLoc(S), NutLoc(N)) + 1`.
        c.  Find the pair (S, N) that results in the minimum cost.
        d.  If no reachable spanner/nut pair is found (cost is infinity), the state is unsolvable. Return infinity.
        e.  Add the minimum cost found to the total cost.
        f.  Update the man's current location for the next step to be the location of the nut just processed.
        g.  Remove the processed nut and spanner from their respective remaining lists.
    8.  **Return Total Cost:** Once all nuts needing tightening have been processed in the calculation, the total accumulated cost is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts, location graph, and computing distances.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Identify goal nuts
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # Build location graph and compute distances
        self.locations = set()
        self.links = defaultdict(set)
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.locations.add(l1)
                self.locations.add(l2)
                self.links[l1].add(l2)
                self.links[l2].add(l1) # Links are bidirectional

        self.dist = self._compute_all_pairs_shortest_paths()

        # Identify the man object (assuming it's the one involved in 'carrying' or starts with 'bob')
        self.man_obj = None
        for fact in self.initial_state:
             if match(fact, "carrying", "*", "*"):
                  self.man_obj = get_parts(fact)[1]
                  break
        if self.man_obj is None:
             # Fallback: Assume object starting with 'bob' is the man
             for fact in self.initial_state:
                  if match(fact, "at", "*", "*"):
                       obj = get_parts(fact)[1]
                       if obj.startswith('bob'):
                            self.man_obj = obj
                            break


    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations using BFS.
        Returns a dictionary dist[l1][l2] = shortest_distance.
        """
        dist = {}
        for start_loc in self.locations:
            dist[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, d = q.popleft()
                dist[start_loc][current_loc] = d
                for neighbor in self.links.get(current_loc, []): # Use .get for safety
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))
            # Ensure all locations have an entry, even if unreachable from start_loc
            for loc in self.locations:
                 if loc not in dist[start_loc]:
                     dist[start_loc][loc] = float('inf') # No path

        return dist

    def get_man_location(self, state):
         """Finds the current location of the man object."""
         if not self.man_obj: return None # Should have been found in init

         for fact in state:
              if match(fact, "at", self.man_obj, "*"):
                   return get_parts(fact)[2]
         return None # Man must be at a location in a valid state


    def get_object_location(self, state, obj):
        """
        Finds the current location of an object in the given state.
        Returns the location string or None if not found at a location or carried.
        Assumes object is either at a location or carried by the man.
        """
        # Check if the object is at a location
        for fact in state:
            if match(fact, "at", obj, "*"):
                return get_parts(fact)[2]

        # Check if the object is carried by the man
        if self.man_obj:
            for fact in state:
                if match(fact, "carrying", self.man_obj, obj):
                    # If carried, its location is the man's location
                    return self.get_man_location(state) # Use the dedicated helper

        # Object location not found
        return None


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify goal nuts that are not yet tightened
        tightened_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "tightened", "*")}
        nuts_to_tighten = list(self.goal_nuts - tightened_nuts_in_state) # Use list for mutability

        # If all goal nuts are tightened, heuristic is 0.
        if not nuts_to_tighten:
             return 0

        # Get locations for nuts that need tightening
        nut_locations = {}
        for nut in nuts_to_tighten:
             loc = self.get_object_location(state, nut)
             if loc is None:
                  # A goal nut that needs tightening must be at a location.
                  # If not found, state is likely invalid or unsolvable.
                  return float('inf')
             nut_locations[nut] = loc


        # 2. Identify man's current location
        man_loc = self.get_man_location(state)
        if man_loc is None:
             return float('inf') # Man must be at a location


        # 3. Identify current usable spanners and their locations
        current_usable_spanners = [] # List of spanner names
        spanner_locations = {} # Map spanner name to location string
        carried_spanner = None

        for fact in state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                current_usable_spanners.append(spanner)

        # Find locations for these usable spanners
        for spanner in current_usable_spanners:
             loc = self.get_object_location(state, spanner)
             if loc: # Spanner must be at a location or carried
                  spanner_locations[spanner] = loc
                  # Check if this usable spanner is the one carried
                  if self.man_obj and loc == man_loc:
                       for fact_c in state:
                            if match(fact_c, "carrying", self.man_obj, spanner):
                                 carried_spanner = spanner
                                 break # Found carried spanner
             # else: usable spanner location not found? Treat as unavailable.

        # Filter out usable spanners whose location couldn't be determined
        current_usable_spanners = [s for s in current_usable_spanners if s in spanner_locations]


        # Check if there are enough usable spanners for the nuts that need tightening
        if len(current_usable_spanners) < len(nuts_to_tighten):
             # Not enough spanners to tighten all required nuts. Unsolvable from here.
             return float('inf')


        # 5. Initialize heuristic calculation variables
        total_cost = 0
        current_man_loc_calc = man_loc
        remaining_nuts_to_tighten = list(nuts_to_tighten) # Make a mutable copy
        remaining_usable_spanners = list(current_usable_spanners) # Make a mutable copy


        # 6. Handle carried spanner first if available
        # Check if the carried spanner is actually one of the *current* usable spanners
        if carried_spanner is not None and carried_spanner in remaining_usable_spanners and remaining_nuts_to_tighten:
             # Find the nut to tighten closest to the man's current location
             closest_nut = None
             min_dist_to_nut = float('inf')
             for nut in remaining_nuts_to_tighten:
                 nut_loc = nut_locations[nut]
                 # Ensure location is valid in the distance map
                 if current_man_loc_calc not in self.dist or nut_loc not in self.dist[current_man_loc_calc]:
                      dist = float('inf')
                 else:
                      dist = self.dist[current_man_loc_calc][nut_loc]

                 if dist < min_dist_to_nut:
                     min_dist_to_nut = dist
                     closest_nut = nut

             # If a reachable closest nut was found
             if closest_nut is not None and min_dist_to_nut != float('inf'):
                 # Cost to use carried spanner on closest nut: walk to nut + tighten
                 cost_this_nut = min_dist_to_nut + 1
                 total_cost += cost_this_nut
                 current_man_loc_calc = nut_locations[closest_nut]
                 remaining_nuts_to_tighten.remove(closest_nut)
                 remaining_usable_spanners.remove(carried_spanner) # This spanner is now used up
             # else: If no reachable nut, the loop below will handle remaining nuts and return inf if needed.


        # 7. Process remaining nuts to tighten using available ground spanners
        while remaining_nuts_to_tighten:
            # Find the pair (S, N) from remaining_usable_spanners and remaining_nuts_to_tighten
            # that minimizes the cost of the sequence:
            # Man at current_man_loc_calc -> SpannerLoc(S) -> NutLoc(N) -> Tighten
            # Cost = dist(current_man_loc_calc, SpannerLoc(S)) + 1 (pickup) + dist(SpannerLoc(S), NutLoc(N)) + 1 (tighten)
            min_cost_pair = float('inf')
            best_spanner = None
            best_nut = None

            if not remaining_usable_spanners:
                 # No more usable spanners available. Unsolvable if nuts remain.
                 return float('inf')

            for spanner in remaining_usable_spanners:
                spanner_loc = spanner_locations[spanner]
                for nut in remaining_nuts_to_tighten:
                    nut_loc = nut_locations[nut]
                    # Ensure locations are valid in the distance map
                    if spanner_loc not in self.dist or nut_loc not in self.dist[spanner_loc] or current_man_loc_calc not in self.dist:
                         cost = float('inf')
                    else:
                         cost = self.dist[current_man_loc_calc][spanner_loc] + 1 + self.dist[spanner_loc][nut_loc] + 1

                    if cost < min_cost_pair:
                        min_cost_pair = cost
                        best_spanner = spanner
                        best_nut = nut

            if best_nut is None or min_cost_pair == float('inf'):
                 # No reachable spanner/nut pair found for remaining nuts. Unsolvable.
                 return float('inf')


            total_cost += min_cost_pair
            current_man_loc_calc = nut_locations[best_nut]
            remaining_nuts_to_tighten.remove(best_nut)
            remaining_usable_spanners.remove(best_spanner) # This spanner is now used up

        # 8. Return the total estimated cost
        return total_cost
