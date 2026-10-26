from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this is available in the environment

# Helper functions
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

# Heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose nuts.
    It sums the number of loose nuts (representing tighten actions),
    the cost to acquire a usable spanner if Bob is not carrying one,
    and the cost for Bob to reach the location of the closest loose nut.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - Nut locations are static (do not change during planning).
    - Spanner usability is static.
    - Links between locations are bidirectional and static.
    - Bob can carry multiple spanners.
    - Any usable spanner can tighten any loose nut.
    - The graph of locations connected by links is connected (or solvable instances only involve connected parts).

    # Heuristic Initialization
    - Extracts location links to build a graph.
    - Computes all-pairs shortest paths between locations using BFS.
    - Stores the set of nuts that need to be tightened (goal nuts) and their initial locations.
    - Stores the set of usable spanners.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Goal Check:** If the current state satisfies the task's goal conditions, the heuristic value is 0.
    2.  **Identify Bob's Location:** Find the current location of 'bob'. If Bob's location is not found or is not a known location, the state is likely invalid or unsolvable, return infinity.
    3.  **Identify Loose Nuts:** Determine which of the goal nuts are currently in a 'loose' state by checking the state facts.
    4.  **Count Loose Nuts:** The number of loose nuts is a lower bound on the number of 'tighten' actions required. Add this count to the total cost. If there are no loose nuts, the goal is met (handled by step 1 or return 0 here).
    5.  **Check for Carried Spanner:** Determine if Bob is currently carrying *any* spanner that was marked as 'usable' in the initial state.
    6.  **Estimate Spanner Acquisition Cost:** If Bob is *not* carrying a usable spanner, he will need to acquire one.
        - Find all usable spanners that are currently on the ground (not being carried) by checking the state facts.
        - Calculate the shortest distance from Bob's current location to the location of the closest usable spanner on the ground using the precomputed distances.
        - Add this minimum distance plus 1 (for the 'pickup' action) to the total cost. If no reachable usable spanners are on the ground and Bob isn't carrying one, the state is likely unsolvable, return infinity.
    7.  **Estimate Movement Cost:** Bob needs to reach the location of each loose nut. A lower bound on the movement cost is the shortest distance from Bob's current location to the location of the *closest* loose nut.
        - Find the locations of all loose nuts using the stored initial nut locations.
        - Calculate the shortest distance from Bob's current location to each of these loose nut locations using the precomputed distances.
        - Find the minimum of these distances.
        - Add this minimum distance to the total cost. If loose nuts exist but none are reachable, return infinity.
    8.  **Return Total Cost:** The sum calculated in steps 4, 6 (if applicable), and 7 is the heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find initial nut locations

        # Extract location links to build the graph
        self.links = {}
        all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                loc1, loc2 = get_parts(fact)[1:]
                self.links.setdefault(loc1, set()).add(loc2)
                self.links.setdefault(loc2, set()).add(loc1) # Assuming links are bidirectional
                all_locations.add(loc1)
                all_locations.add(loc2)

        # Add locations mentioned in initial state 'at' predicates
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                all_locations.add(get_parts(fact)[2])

        self.locations = list(all_locations) # Use a list for consistent ordering if needed, or just iterate

        # Precompute shortest paths between all locations
        self.dist = self._compute_all_pairs_shortest_paths()

        # Store goal nuts and their initial locations (assuming nut locations are static)
        self.goal_nuts = set()
        self.nut_locations = {}
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut = get_parts(goal)[1]
                self.goal_nuts.add(nut)
                # Find the initial location of this nut
                for fact in initial_state:
                    if match(fact, "at", nut, "*"):
                        self.nut_locations[nut] = get_parts(fact)[2]
                        break # Found location for this nut
                # If a goal nut doesn't have an initial 'at' location, it's problematic.
                # Assuming valid problems have initial locations for all goal nuts.


        # Identify usable spanners (assuming usability is static)
        self.usable_spanners = set()
        for fact in initial_state:
            if match(fact, "usable", "*"):
                spanner = get_parts(fact)[1]
                self.usable_spanners.add(spanner)


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        dist = {}
        for start_loc in self.locations:
            dist[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, d = q.popleft()
                dist[start_loc][current_loc] = d
                if current_loc in self.links:
                    for neighbor in self.links[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))
        return dist

    def get_distance(self, loc1, loc2):
        """Returns the shortest path distance between two locations."""
        # Ensure both locations are known
        if loc1 not in self.locations or loc2 not in self.locations:
            # This indicates an issue with location extraction or a malformed state/goal
            # For heuristic purposes, treat as unreachable
            return float('inf')

        # Distance to self is 0
        if loc1 == loc2:
            return 0

        # Look up precomputed distance
        # Handle cases where a location might be in self.locations but not in self.dist keys
        # if it has no links (isolated node). BFS from such a node would only find itself.
        if loc1 not in self.dist or loc2 not in self.dist[loc1]:
             # No path found during BFS
             return float('inf')

        return self.dist[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Find Bob's current location
        bob_location = None
        for fact in state:
            if match(fact, "at", "bob", "*"):
                bob_location = get_parts(fact)[2]
                break
        # If bob_location is None, the state is malformed or terminal in an unexpected way.
        # Returning infinity signals this state is bad or unsolvable from here.
        if bob_location is None or bob_location not in self.locations:
             return float('inf')


        # Identify loose nuts among the goal nuts
        current_loose_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}
        num_loose_nuts = len(current_loose_nuts)

        # If no loose nuts, but goal not reached, the goal must contain other conditions
        # not handled by this heuristic. Assuming goal is *only* tightening nuts.
        if num_loose_nuts == 0:
             # This case should be covered by the initial goal check if the goal is just tightening.
             # If we reach here, it implies the goal has non-tightening conditions which are met,
             # but the tightening conditions were already met (0 loose nuts).
             # In a pure spanner domain, this means the goal is reached.
             return 0


        # Heuristic component 1: Tightening actions
        # Each loose nut requires one tighten action.
        total_cost = num_loose_nuts

        # Heuristic component 2: Getting a spanner
        # Check if Bob is carrying *any* usable spanner
        bob_carrying_usable_spanner = False
        for spanner in self.usable_spanners:
             if f"(carrying bob {spanner})" in state:
                 bob_carrying_usable_spanner = True
                 break # Bob just needs to carry *a* usable spanner

        if not bob_carrying_usable_spanner:
             # Find locations of usable spanners on the ground
             usable_spanners_on_ground_locations = set()
             for spanner in self.usable_spanners:
                  # Check if the spanner is on the ground
                  for fact in state:
                      if match(fact, "at", spanner, "*"):
                          spanner_loc = get_parts(fact)[2]
                          if spanner_loc in self.locations: # Ensure location is known
                             usable_spanners_on_ground_locations.add(spanner_loc)
                          break # Found location for this spanner

             # Calculate minimum distance from Bob's current location to any usable spanner on the ground
             min_dist_to_spanner = float('inf')
             if usable_spanners_on_ground_locations:
                  for spanner_loc in usable_spanners_on_ground_locations:
                      dist = self.get_distance(bob_location, spanner_loc)
                      if dist != float('inf'): # Only consider reachable spanners
                        min_dist_to_spanner = min(min_dist_to_spanner, dist)

             if min_dist_to_spanner == float('inf'):
                 # No reachable usable spanners on the ground and Bob isn't carrying one.
                 # This state is likely unsolvable.
                 return float('inf')

             # Add cost to get the closest spanner (moves + pickup)
             total_cost += min_dist_to_spanner + 1


        # Heuristic component 3: Movement to nut locations
        # Bob needs to reach the location of each loose nut.
        # Find the locations of all loose nuts.
        loose_nut_locations = []
        for nut in current_loose_nuts:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc and nut_loc in self.locations: # Ensure location is known
                loose_nut_locations.append(nut_loc)
            # else: # Loose nut location not found or not a known location - indicates problem setup issue

        # Find the closest loose nut location from Bob's current location
        min_dist_to_nut = float('inf')
        if loose_nut_locations:
             for nut_loc in loose_nut_locations:
                  dist = self.get_distance(bob_location, nut_loc)
                  if dist != float('inf'): # Only consider reachable nuts
                     min_dist_to_nut = min(min_dist_to_nut, dist)

        if min_dist_to_nut == float('inf'):
            # Loose nuts exist, but Bob cannot reach any of their locations.
            # This state is likely unsolvable.
            return float('inf')

        # Add cost to reach the closest loose nut
        total_cost += min_dist_to_nut

        return total_cost
