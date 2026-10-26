from collections import deque
from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle empty fact string if necessary, though unlikely in valid states
    if not fact or fact[0] != '(' or fact[-1] != ')':
        return [] # Or raise error, depending on expected input robustness
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at bob shed)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    num_args = len(args)
    num_parts = len(parts)

    # Check length compatibility based on trailing wildcard
    # If args has a trailing wildcard, parts can be equal or longer.
    # If args does not have a trailing wildcard, parts must be equal length.
    if num_args > 0 and args[-1] == '*':
        if num_parts < num_args - 1:
            return False
    else:
        if num_parts != num_args:
            return False

    # Compare parts and args element by element using fnmatch up to the length of args
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It sums the estimated costs for three main components: performing the tighten
    actions, getting Bob to the locations of the loose nuts, and acquiring a
    usable spanner if Bob doesn't currently have one.

    # Assumptions
    - The goal is to achieve the `(tightened ?nut)` predicate for all nuts
      specified in the task goals that are initially loose.
    - Bob can carry multiple spanners simultaneously.
    - Spanner usability is static and determined by the initial state.
    - Locations are connected by bidirectional `link` predicates, forming a graph.
    - The cost of any primitive action (move, pick, drop, tighten) is 1.
    - The problem is solvable (i.e., usable spanners exist and locations are connected).

    # Heuristic Initialization
    - Stores the task's goal conditions.
    - Builds a graph of locations based on `link` predicates found in the static facts.
    - Computes all-pairs shortest path distances between locations using Breadth-First Search (BFS) on the location graph.
    - Identifies the set of all spanners that are initially marked as usable.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1.  Identify Bob's current location (`bob_loc`).
    2.  Identify all nuts that are currently `loose` and collect their locations into a set (`loose_nut_locs`).
    3.  Determine if Bob is currently carrying at least one usable spanner (`bob_carrying_usable`). This requires checking if any spanner Bob is carrying is in the set of initially usable spanners.
    4.  If Bob is not carrying a usable spanner, identify the locations of all usable spanners that are currently on the ground (`available_spanner_locs`).
    5.  Count the total number of loose nuts (`num_loose_nuts`).
    6.  If `num_loose_nuts` is 0, the state is a goal state, so the heuristic value is 0.
    7.  Initialize the heuristic value `h` with `num_loose_nuts`. This accounts for the minimum of one `tighten` action required for each loose nut.
    8.  If Bob is not carrying a usable spanner (`not bob_carrying_usable`):
        -   Find the minimum distance from Bob's current location (`bob_loc`) to any location in `available_spanner_locs`.
        -   If available spanners exist and are reachable, add this minimum distance plus 1 (for the `pick` action) to `h`. If no reachable usable spanners are available, the state is likely unsolvable, return infinity.
    9.  If there are loose nuts (`num_loose_nuts > 0`):
        -   Find the minimum distance from Bob's current location (`bob_loc`) to any location in `loose_nut_locs`.
        -   If loose nut locations exist and are reachable, add this minimum distance to `h`. This estimates the cost to reach the first nut location. If no loose nut locations are reachable, the state is likely unsolvable, return infinity.
        -   Add `num_loose_nuts - 1` to `h`. This is a simple, non-admissible estimate for the additional movement cost required to visit the remaining loose nut locations after reaching the first one.
    10. Return the calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information:
        - Location graph and distances.
        - Initially usable spanners.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find usable spanners

        # Build location graph and compute distances
        self.location_graph = {}
        locations = set()

        # Collect all locations mentioned in links and initial/goal states
        # This helps ensure all relevant locations are in the graph even if isolated initially
        all_facts = list(static_facts) + list(initial_state) + list(self.goals)
        for fact in all_facts:
             parts = get_parts(fact)
             if parts:
                  # Look for arguments that are likely locations based on predicate context
                  if parts[0] in ['link', 'at']:
                       # 'link' has two location arguments
                       if parts[0] == 'link' and len(parts) == 3:
                            locations.add(parts[1])
                            locations.add(parts[2])
                       # 'at' has object and location arguments
                       elif parts[0] == 'at' and len(parts) == 3:
                            # The third part is the location
                            locations.add(parts[2])
                  # Add other potential location predicates if known, e.g., 'in-city' in Logistics
                  # For spanner, 'link' and 'at' are the main ones defining locations.


        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                # Ensure locations from links are added, even if not in initial/goal 'at'
                locations.add(loc1)
                locations.add(loc2)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        self.locations = list(locations) # Store locations for iteration
        self.distances = self._compute_all_pairs_shortest_paths()

        # Identify initially usable spanners
        self.initially_usable_spanners = {
            s for fact in initial_state if match(fact, "usable", s)
        }

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of locations
        using BFS.
        """
        distances = {}
        for start_node in self.locations:
            distances[start_node] = self._bfs(start_node)
        return distances

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all
        reachable nodes.
        """
        dist = {node: float('inf') for node in self.locations}
        # Handle case where start_node might not be in self.locations (e.g., malformed state)
        if start_node not in dist:
             return {} # Cannot compute distances from an unknown location

        dist[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            # Ensure current_node is in the graph before accessing neighbors
            if current_node in self.location_graph:
                for neighbor in self.location_graph[current_node]:
                    # Ensure neighbor is a known location before updating distance
                    if neighbor in dist and dist[neighbor] == float('inf'):
                        dist[neighbor] = dist[current_node] + 1
                        queue.append(neighbor)
        return dist


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Bob's current location.
        bob_loc = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_loc = get_parts(fact)[2]
                break
        # If Bob's location is not found, the state is invalid/unsolvable.
        # Also check if bob_loc is a known location in our graph
        if bob_loc is None or bob_loc not in self.locations:
             return float('inf')

        # 2. Identify all nuts that are currently loose and their locations.
        loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}
        loose_nut_locs = {
            get_parts(fact)[2]
            for fact in state
            if match(fact, "at", "*", "*") and get_parts(fact)[1] in loose_nuts
        }
        # Filter out loose nut locations that are not in our known locations graph
        loose_nut_locs = {loc for loc in loose_nut_locs if loc in self.locations}


        # 3. Determine if Bob is currently carrying any usable spanner.
        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", "bob", "*")}
        bob_carrying_usable = any(s in carried_spanners for s in self.initially_usable_spanners)

        # 4. If Bob is not carrying a usable spanner, identify available spanners on the ground.
        available_spanner_locs = set()
        if not bob_carrying_usable:
             for spanner in self.initially_usable_spanners:
                  if spanner not in carried_spanners:
                      for fact in state:
                          if match(fact, "at", spanner, "*"):
                              loc = get_parts(fact)[2]
                              # Only consider spanners at known locations
                              if loc in self.locations:
                                  available_spanner_locs.add(loc)
                              break # Assuming a spanner is only at one location if not carried


        # 5. Count the total number of loose nuts.
        num_loose_nuts = len(loose_nuts)

        # 6. If num_loose_nuts is 0, the state is a goal state, return h = 0.
        if num_loose_nuts == 0:
            return 0

        # 7. Initialize the heuristic value h with num_loose_nuts (tighten actions).
        h = num_loose_nuts

        # 8. If Bob is not carrying a usable spanner:
        if not bob_carrying_usable:
            # Find the minimum distance from Bob's current location to any location
            # containing an available usable spanner.
            min_dist_to_spanner = float('inf')
            if available_spanner_locs:
                # Distances from bob_loc should be precomputed
                bob_distances = self.distances.get(bob_loc, {})
                min_dist_to_spanner = min(
                    bob_distances.get(spanner_loc, float('inf'))
                    for spanner_loc in available_spanner_locs
                )

            # If such a spanner exists and is reachable, add cost to get it.
            if min_dist_to_spanner != float('inf'):
                 h += min_dist_to_spanner + 1 # move to spanner + pick up
            else:
                 # No reachable usable spanners available outside Bob's hands, and he has none.
                 # This state is likely unsolvable if tightening is required.
                 return float('inf')


        # 9. If there are loose nuts (num_loose_nuts > 0):
        # Find the minimum distance from Bob's current location to any location
        # containing a loose nut.
        min_dist_bob_to_nut = float('inf')
        if loose_nut_locs:
             # Distances from bob_loc should be precomputed
             bob_distances = self.distances.get(bob_loc, {})
             min_dist_bob_to_nut = min(
                 bob_distances.get(nut_loc, float('inf'))
                 for nut_loc in loose_nut_locs
             )

        # Add cost to reach the first nut location and proxy for moves between others.
        if min_dist_bob_to_nut != float('inf'):
            h += min_dist_bob_to_nut # move to the first nut
            # Add proxy for moves between remaining nuts *if* there's more than one nut
            if num_loose_nuts > 1:
                 h += num_loose_nuts - 1
        else:
             # Loose nuts exist, but their locations are not reachable from Bob's location.
             # This state is likely unsolvable.
             return float('inf')

        # 10. Return the calculated h.
        return h
