# Helper functions
from fnmatch import fnmatch
import collections # Required for BFS queue

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact.strip()[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define a placeholder Heuristic class if the import is not available
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError("Heuristic base class not implemented")


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose nuts by simulating a greedy process
    where the man sequentially addresses each loose nut. For each nut, it estimates the cost
    to acquire a usable spanner (if not already carrying one) and then the cost to travel
    to the nut's location and tighten it.

    # Assumptions
    - There is exactly one man.
    - The man can carry at most one spanner at a time.
    - A usable spanner becomes unusable after tightening one nut.
    - Links between locations are bidirectional.
    - The problem is solvable from the initial state (i.e., there are enough usable spanners
      in total in the initial state to tighten all nuts in the goal state, and the graph is connected).
      The heuristic checks solvability from the *current* state based on remaining usable spanners.

    # Heuristic Initialization
    - Parses objects (man, spanners, nuts, locations) from the task definition.
    - Parses location links from the initial state facts.
    - Computes all-pairs shortest path distances between all locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify all nuts that are currently loose. If none are loose, the state is a goal state, return 0.
    2. Initialize the heuristic value `h` to 0.
    3. Find the man's current location.
    4. Determine if the man is currently carrying a usable spanner.
    5. Identify the locations of all usable spanners that are not currently being carried by the man.
    6. Check if the total number of usable spanners available (carried + at locations) is less than the number of loose nuts remaining. If so, the state is a dead end, return infinity.
    7. Iterate through the loose nuts one by one (using a deterministic order, e.g., alphabetical). For each loose nut:
       a. Find the location of the current loose nut.
       b. If the man is *not* currently carrying a usable spanner:
          i. Find the usable spanner at a location that is nearest to the man's current location among the available ones.
          ii. Add the distance from the man's current location to this spanner's location to `h`.
          iii. Add 1 to `h` for the `pickup_spanner` action.
          iv. Update the man's current location to the spanner's location.
          v. Update the man's status to indicate he is now carrying a usable spanner.
          vi. Remove the picked-up spanner from the list of available usable spanners at locations.
          vii. If no usable spanners were available at locations (should be caught by step 6), return infinity.
       c. Add the distance from the man's current location to the nut's location to `h`.
       d. Update the man's current location to the nut's location.
       e. Add 1 to `h` for the `tighten_nut` action.
       f. Update the man's status to indicate he is no longer carrying a usable spanner (it was used).
    8. Return the final heuristic value `h`.
    """
    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and computing distances."""
        self.goals = task.goals # Keep goals for potential future use, though not strictly needed for this heuristic logic

        # Extract objects by type
        # Assuming task.objects is a dict like {'man': ['bob'], 'spanner': ['spanner1', ...], ...}
        self.man = list(task.objects.get('man', []))[0] # Assuming exactly one man
        self.spanners = set(task.objects.get('spanner', []))
        self.nuts = set(task.objects.get('nut', []))
        self.locations = set(task.objects.get('location', []))

        # Build graph from links found in the initial state
        self.adj = {loc: set() for loc in self.locations}
        self.links = set()
        for fact in task.init:
             if match(fact, 'link', '*', '*'):
                 l1, l2 = get_parts(fact)[1], get_parts(fact)[2]
                 self.links.add(tuple(sorted((l1, l2))))
                 # Ensure locations from links are added, although task.objects should cover this
                 self.locations.add(l1)
                 self.locations.add(l2)

        # Rebuild adj list just in case locations were added from links
        self.adj = {loc: set() for loc in self.locations}
        for l1, l2 in self.links:
            self.adj[l1].add(l2)
            self.adj[l2].add(l1)


        # Compute all-pairs shortest paths using BFS from each location
        self.distances = {}
        for start_node in self.locations:
            self.distances[start_node] = {}
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                (curr, d) = queue.popleft()
                self.distances[start_node][curr] = d
                for neighbor in self.adj.get(curr, []): # Use .get for safety
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

    def get_distance(self, loc1, loc2):
        """Get shortest path distance between two locations."""
        if loc1 == loc2:
            return 0
        # Return infinity if locations are not in the graph or not connected
        # The BFS computes distances only within connected components reachable from start_node.
        # If loc2 is not in distances[loc1], it's unreachable.
        return self.distances.get(loc1, {}).get(loc2, float('inf'))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find loose nuts
        loose_nuts = sorted([n for n in self.nuts if f'(loose {n})' in state]) # Sort for deterministic order
        num_loose_nuts = len(loose_nuts)

        # Goal state check
        if num_loose_nuts == 0:
            return 0

        h = 0

        # Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, 'at', self.man, '*'):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man's location must be known in a valid state
             return float('inf') # Should not happen in valid states

        # Check if man is carrying a usable spanner
        man_carrying_usable = False
        carried_spanner = None
        for fact in state:
            if match(fact, 'carrying', self.man, '*'):
                carried_spanner = get_parts(fact)[2]
                # Check if the carried spanner is usable
                if f'(usable {carried_spanner})' in state:
                    man_carrying_usable = True
                # Assuming man carries at most one spanner, break after finding the first one
                break

        # Find locations of usable spanners not carried by the man
        usable_spanner_locations = {} # {spanner: location}
        for s in self.spanners:
            # Check if spanner is usable
            if f'(usable {s})' in state:
                # Check if it's the one the man is carrying (and it's usable)
                if carried_spanner == s and man_carrying_usable:
                    continue # This spanner is already accounted for as 'carried'

                # Find location of this usable spanner
                spanner_loc = None
                for fact in state:
                    if match(fact, 'at', s, '*'):
                        spanner_loc = get_parts(fact)[2]
                        break
                if spanner_loc:
                    usable_spanner_locations[s] = spanner_loc

        # Check if enough usable spanners exist in total (carried + at locations)
        total_usable_spanners = (1 if man_carrying_usable else 0) + len(usable_spanner_locations)
        if total_usable_spanners < num_loose_nuts:
            # Not enough spanners remaining to tighten all loose nuts
            return float('inf')

        current_man_loc = man_loc
        current_carrying_usable = man_carrying_usable
        # Create a mutable copy of available spanners at locations
        current_usable_spanner_locations = dict(usable_spanner_locations)

        # Get locations of loose nuts
        loose_nut_locations = {}
        for nut in loose_nuts:
            nut_loc = None
            for fact in state:
                if match(fact, 'at', nut, '*'):
                    nut_loc = get_parts(fact)[2]
                    break
            if nut_loc is None:
                # Nut location must be known in a valid state
                return float('inf') # Should not happen
            loose_nut_locations[nut] = nut_loc

        # Simulate sequential tightening process for each loose nut
        for nut in loose_nuts:
            nut_loc = loose_nut_locations[nut]

            # 1. Get a usable spanner if needed
            if not current_carrying_usable:
                # Find nearest usable spanner from current_man_loc among available ones
                min_dist_spanner = float('inf')
                best_spanner_loc = None
                spanner_to_remove = None

                # Iterate through available usable spanners at locations
                for s, s_loc in current_usable_spanner_locations.items():
                     dist_s = self.get_distance(current_man_loc, s_loc)
                     # If spanner location is unreachable, skip it
                     if dist_s == float('inf'):
                         continue

                     if dist_s < min_dist_spanner:
                         min_dist_spanner = dist_s
                         best_spanner_loc = s_loc
                         spanner_to_remove = s

                # If no reachable usable spanners found among available ones
                if best_spanner_loc is None:
                     # This state is a dead end as we need a spanner but cannot get one
                     return float('inf')

                # Add cost to get spanner
                h += min_dist_spanner # Walk to spanner
                h += 1 # Pickup spanner
                current_man_loc = best_spanner_loc
                current_carrying_usable = True
                del current_usable_spanner_locations[spanner_to_remove]

            # 2. Get to the nut location
            dist_to_nut = self.get_distance(current_man_loc, nut_loc)
            if dist_to_nut == float('inf'):
                 # Cannot reach the nut location from the current position
                 return float('inf')
            h += dist_to_nut # Walk to nut
            current_man_loc = nut_loc

            # 3. Tighten the nut
            h += 1 # Tighten action
            current_carrying_usable = False # Spanner is used up

        return h
