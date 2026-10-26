# Add necessary imports
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available
import heapq # Not strictly needed for BFS, but useful for priority queues
import collections # For BFS deque
import math # For infinity

# Helper function to parse PDDL fact strings
def parse_fact(fact_str):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove parentheses and split by spaces
    parts = fact_str[1:-1].split()
    # The first part is the predicate, the rest are arguments
    return tuple(parts)

# Helper function to get the location of a locatable object from the state
def get_location(obj, state):
    """Finds the location of an object in the current state."""
    for fact_str in state:
        if fact_str.startswith('(at '):
            parts = parse_fact(fact_str)
            if len(parts) == 3:
                 predicate, item, loc = parts
                 if item == obj:
                     return loc
    return None # Object is not at a location (e.g., carried)

class spannerHeuristic(Heuristic):
    """
    Summary:
    Domain-dependent heuristic for the spanner domain. Estimates the cost
    to reach the goal by summing the number of untightened goal nuts,
    the minimum walking distance to reach any required location (either a
    nut location or a spanner pickup location), the number of spanner
    pickup actions needed, and the walking cost to reach the locations
    of the spanners that need to be picked up.

    Assumptions:
    - There is exactly one man object.
    - Nut locations are static (do not change during planning).
    - Spanners can be at a location or carried by the man.
    - The graph of locations connected by 'link' predicates is connected
      for all relevant locations (initial man location, nut locations,
      initial spanner locations).
    - The heuristic assumes that to tighten N nuts, N usable spanners are
      consumed.
    - Object types (man, spanner, nut, location) are inferred from predicate
      usage and common naming conventions in the initial state and operators.
      This inference might be fragile for arbitrary PDDL instances.

    Heuristic Initialization:
    - Parses initial state, static facts, and operators to identify objects
      (man, spanners, nuts, locations) and their initial/static properties
      (like nut locations). Object types are inferred.
    - Builds an undirected graph of locations based on 'link' facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the set of goal nuts from the task goals.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state and find the man's current location. If the man's
       location cannot be determined (e.g., not in an '(at ...)' fact), return
       infinity (invalid state for this heuristic).
    2. Identify the set of goal nuts that are currently loose in the state.
    3. If this set of loose goal nuts is empty, the goal is reached, return 0.
    4. Initialize the heuristic value `h` with the number of loose goal nuts.
       This represents the minimum number of 'tighten_nut' actions required.
    5. Identify the set of usable spanners currently carried by the man.
    6. Identify the set of usable spanners available at locations (not carried)
       and their respective locations.
    7. Calculate the total number of usable spanners available in the state
       (carried + available at locations).
    8. Calculate the number of spanners needed to tighten all loose goal nuts.
       This is equal to the number of loose goal nuts.
    9. If the total number of usable spanners available is less than the number
       of spanners needed, the problem is unsolvable, return infinity.
    10. Calculate the number of additional spanners that need to be picked up.
        This is `max(0, num_spanners_needed - num_carried_usable_spanners)`.
    11. Initialize a set of required locations the man must visit. Add the
        locations of all loose goal nuts to this set. Check if all these nut
        locations are known; if not, return infinity (problem setup issue).
    12. If additional spanners need to be picked up:
        a. Add the number of needed pickups to `h`. This represents the minimum
           number of 'pickup_spanner' actions required.
        b. Find the `num_pickup_needed` available usable spanners at locations
           that are closest to the man's current location.
        c. If there are not enough *reachable* usable spanners available at
           locations to satisfy `num_pickup_needed`, the problem is unsolvable,
           return infinity.
        d. Add the locations of these closest spanners to the set of required
           locations the man must visit.
    13. Calculate the minimum shortest path distance from the man's current
        location to any location in the set of required locations to visit.
        If no required location is reachable from the man's current location,
        return infinity.
    14. Add this minimum distance to `h`. This represents the estimated initial
        walking cost to get to the vicinity of the required tasks (tightening
        nuts or picking up spanners).
    15. Return the final heuristic value `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.goal_nuts = set()
        self.nut_locations = {} # Nut -> Location (static)
        self.spanners = set()
        self.man = None # Assume one man
        self.locations = set()
        self.location_graph = collections.defaultdict(set)
        self.dist = {} # Stores shortest paths: self.dist[l1][l2]

        # --- Precomputation ---

        # Identify objects and locations from initial state, static facts, and operators
        # This inference is based on common naming conventions and predicate usage
        # and might be fragile for arbitrary PDDL instances.
        all_relevant_items = set()
        for fact_str in task.initial_state:
             all_relevant_items.update(parse_fact(fact_str))
        for fact_str in task.static:
             all_relevant_items.update(parse_fact(fact_str))
        for op in task.operators:
             all_relevant_items.update(op.name.split()) # Include operator name parts
             for fact_str in op.preconditions | op.add_effects | op.del_effects:
                  all_relevant_items.update(parse_fact(fact_str))

        # Simple inference based on common names
        for item in all_relevant_items:
             if isinstance(item, str):
                  item_lower = item.lower()
                  if 'bob' in item_lower or 'man' in item_lower:
                       if self.man is None: # Assume the first one found is the man
                            self.man = item
                  elif 'spanner' in item_lower:
                       self.spanners.add(item)
                  elif 'shed' in item_lower or 'gate' in item_lower or 'location' in item_lower:
                       self.locations.add(item)
                  # Nuts are primarily identified via initial 'at' facts

        # Get static nut locations from initial state
        for fact_str in task.initial_state:
             parts = parse_fact(fact_str)
             if parts[0] == 'at' and len(parts) == 3:
                  obj, loc = parts[1], parts[2]
                  # If obj is not the man and not a spanner we identified, assume it's a nut
                  # This relies on spanners and man being identified first
                  if obj != self.man and obj not in self.spanners and obj not in self.locations:
                       self.nut_locations[obj] = loc
                       self.locations.add(loc) # Nut location is a location
                  elif obj in self.spanners:
                       self.locations.add(loc) # Spanner location is a location
                  elif obj == self.man:
                       self.locations.add(loc) # Man location is a location


        # Add any locations mentioned in links that weren't in 'at' facts
        for fact_str in task.static:
             parts = parse_fact(fact_str)
             if parts[0] == 'link' and len(parts) == 3:
                  loc1, loc2 = parts[1], parts[2]
                  self.locations.add(loc1)
                  self.locations.add(loc2)
                  self.location_graph[loc1].add(loc2)
                  self.location_graph[loc2].add(loc1) # Links are bidirectional

        # Identify goal nuts
        for goal_fact_str in task.goals:
            parts = parse_fact(goal_fact_str)
            if parts[0] == 'tightened' and len(parts) == 2:
                self.goal_nuts.add(parts[1])

        # Compute all-pairs shortest paths
        self._compute_shortest_paths()

        # Basic validation (optional, but helpful for debugging)
        # if self.man is None: print("Warning: Man object not identified.")
        # if not self.locations: print("Warning: No locations identified.")
        # if not self.goal_nuts: print("Warning: No goal nuts identified.")
        # if not self.nut_locations and self.goal_nuts: print("Warning: Goal nuts identified, but no nut locations found in initial state.")


    def _compute_shortest_paths(self):
        """Computes shortest paths between all pairs of locations using BFS."""
        for start_node in self.locations:
            self.dist[start_node] = {}
            q = collections.deque([(start_node, 0)])
            visited = {start_node}

            while q:
                current_node, distance = q.popleft()
                self.dist[start_node][current_node] = distance

                # Ensure neighbor is a known location before adding
                for neighbor in self.location_graph.get(current_node, []):
                    if neighbor in self.locations and neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, distance + 1))


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state.
        """
        state = node.state

        # 1. Get man's current location
        man_loc = get_location(self.man, state)
        # If man_loc is None (man object not found or state is weird)
        # or if man_loc is a location name not seen during init (shouldn't happen with valid states)
        if man_loc is None or man_loc not in self.locations:
             return float('inf')

        # 2. Identify loose goal nuts
        # Only consider nuts that are in the goal set and are currently loose
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}

        # 3. If all goal nuts are tightened, goal is reached
        if len(loose_goal_nuts) == 0:
            return 0

        # 4. Initialize heuristic with minimum tighten actions
        h = len(loose_goal_nuts)

        # 5. Identify usable spanners carried
        carried_usable_spanners = {s for s in self.spanners if f'(carrying {self.man} {s})' in state and f'(usable {s})' in state}

        # 6. Identify usable spanners not carried and their locations
        available_usable_spanners_locs = []
        for s in self.spanners:
             if s not in carried_usable_spanners:
                  spanner_loc = get_location(s, state)
                  # Spanner must be at a location to be available for pickup
                  if spanner_loc is not None and f'(usable {s})' in state:
                       available_usable_spanners_locs.append((s, spanner_loc))

        # 7. Calculate total usable spanners
        total_usable_spanners = len(carried_usable_spanners) + len(available_usable_spanners_locs)

        # 8. Calculate the number of spanners needed
        num_spanners_needed = len(loose_goal_nuts)

        # 9. Check solvability based on spanners
        if total_usable_spanners < num_spanners_needed:
            return float('inf') # Not enough usable spanners in the world

        # 10. Calculate additional spanners to pick up
        num_pickup_needed = max(0, num_spanners_needed - len(carried_usable_spanners))

        # 11. Initialize locations to visit
        locations_to_visit = set()
        unreachable_required_locations = False

        # Add locations of loose goal nuts
        for n in loose_goal_nuts:
             nut_loc = self.nut_locations.get(n)
             if nut_loc is None or nut_loc not in self.locations:
                  # Goal nut location not found or not a recognized location - problem likely malformed
                  unreachable_required_locations = True
                  break
             # Check if nut location is reachable from man's current location
             if nut_loc not in self.dist.get(man_loc, {}):
                  unreachable_required_locations = True
                  break
             locations_to_visit.add(nut_loc)

        if unreachable_required_locations:
             return float('inf')

        # 12. If additional spanners needed, add pickup cost and spanner locations to visit
        if num_pickup_needed > 0:
            h += num_pickup_needed # Add cost for pickup actions

            # Find the num_pickup_needed closest available usable spanners
            spanner_distances = []
            for s, l in available_usable_spanners_locs:
                 # Only consider spanners at reachable locations
                 if l in self.dist.get(man_loc, {}):
                      spanner_distances.append((self.dist[man_loc][l], s, l))

            # If there aren't enough *reachable* usable spanners at locations
            # to satisfy num_pickup_needed, it's unsolvable.
            if len(spanner_distances) < num_pickup_needed:
                 return float('inf')

            # Sort available spanners by distance from man's current location
            spanner_distances.sort()

            # Add the locations of the num_pickup_needed closest spanners to locations_to_visit
            for i in range(num_pickup_needed):
                 _dist, s, l = spanner_distances[i]
                 locations_to_visit.add(l)

        # 13. Calculate walking cost to visit required locations
        # Find the minimum distance from man_loc to any location in locations_to_visit
        min_dist_to_visit = float('inf')
        if locations_to_visit: # Ensure there's at least one location to visit
            for loc in locations_to_visit:
                 # All locations in locations_to_visit should be reachable from man_loc
                 # due to earlier checks.
                 min_dist_to_visit = min(min_dist_to_visit, self.dist[man_loc][loc])
        else:
             # This case should only happen if loose_goal_nuts was empty,
             # which is handled at the beginning. If we reach here and
             # locations_to_visit is empty, something is wrong, or maybe
             # all needed spanners are carried and all nuts are at man's loc?
             # If loose_goal_nuts > 0, locations_to_visit will contain nut locations.
             # If loose_goal_nuts == 0, we return 0 earlier.
             # So locations_to_visit should not be empty if we are here.
             # Add a safeguard, though it indicates a potential logic issue or malformed problem.
             if len(loose_goal_nuts) > 0:
                  # This shouldn't happen based on logic
                  return float('inf')
             else:
                  # This branch should ideally not be reached if loose_goal_nuts == 0
                  # as it's handled at the start. Return 0 just in case.
                  return 0


        # If min_dist_to_visit is still infinity, it means no required location is reachable.
        # This should have been caught earlier, but check again.
        if min_dist_to_visit == float('inf'):
             return float('inf')

        h += min_dist_to_visit

        # 14. Return the final heuristic value
        return h
