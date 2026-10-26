from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact."""
    # Remove parentheses and split by spaces
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a PDDL fact matches a given pattern."""
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args, and if each part matches the corresponding arg pattern
    return len(parts) == len(args) and all(fnmatch(part, arg) for part, arg in zip(parts, args))

# BFS for shortest path
def bfs(graph, start):
    """Compute shortest distances from start node in a graph."""
    distances = {node: float('inf') for node in graph}
    if start not in graph:
        # Start node might not be in the graph keys if it has no outgoing links defined by 'link' facts,
        # but it should still be a valid location node if objects are placed there.
        # We added all_locations from initial state/goals to graph keys in __init__.
        # So start should be in graph keys if it's a location from the problem.
        # If somehow not, distances remain inf.
        return distances

    distances[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        # Ensure current is in graph keys before accessing neighbors
        if current in graph: # This check is technically redundant if all_locations are graph keys
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It considers the number of nuts to tighten, the number of spanners to pick up,
    and the travel cost to reach the first required location (nearest loose nut or nearest usable spanner on ground if needed first).

    # Assumptions
    - There is exactly one man, and his name is 'bob'.
    - Spanners become unusable after one use.
    - The problem is solvable (enough usable spanners exist in total across initial state).
    - Travel cost is approximated by the shortest path distance in the link graph.
    - All relevant locations (where man, nuts, spanners are) are part of the linked graph defined by 'link' facts or object placements.

    # Heuristic Initialization
    - Build the location graph from static 'link' facts and locations mentioned in the initial state.
    - Precompute shortest path distances between all pairs of locations using BFS.
    - Identify all nut objects from the initial state and goal state.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify all loose nuts in the current state. If none, heuristic is 0.
    2. Count the number of loose nuts (N_loose). This is a lower bound on tighten actions.
    3. Find the man's current location.
    4. Check if the man is currently carrying a usable spanner.
    5. Calculate the number of pickup actions needed: max(0, N_loose - (1 if man has usable spanner else 0)).
    6. Find the locations of all loose nuts.
    7. Find the locations of all usable spanners on the ground.
    8. Determine the set of potential first target locations: all loose nut locations.
    9. If pickups are needed and usable spanners are on the ground, add the nearest usable spanner location on the ground to the set of potential first target locations.
    10. Calculate the minimum travel cost to reach any of these potential first target locations from the man's current location using precomputed distances.
    11. The heuristic value is the sum of:
        - N_loose (for tighten actions)
        - Number of pickup actions needed
        - Minimum travel cost to the first required location.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static facts and building the graph."""
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state # Need initial state to find all nuts

        # Build location graph from link facts
        self.location_graph = {}
        self.all_locations = set()
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, []).append(loc2)
                self.location_graph.setdefault(loc2, []).append(loc1)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Add any locations mentioned in initial state that weren't in links
        # Ensure these locations are nodes in the graph, even if they only have incoming/outgoing links
        # implied by object placement, not explicit 'link' predicates.
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.all_locations.add(loc)
                 self.location_graph.setdefault(loc, []) # Ensure location is a node even if no links

        # Precompute shortest distances between all locations
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = bfs(self.location_graph, start_loc)

        # Identify all nut objects (assume they are mentioned in initial state or goals)
        self.all_nuts = set()
        for fact in self.initial_state:
             if match(fact, "loose", "*"):
                 _, nut = get_parts(fact)
                 self.all_nuts.add(nut)
             # Also check 'at' facts for objects starting with 'nut' - less reliable but might catch some
             if match(fact, "at", "nut*", "*"):
                 _, nut, _ = get_parts(fact)
                 self.all_nuts.add(nut)

        # Also check goals for nuts
        for goal in self.goals:
             if match(goal, "tightened", "*"):
                 _, nut = get_parts(goal)
                 self.all_nuts.add(nut)


    def get_distance(self, loc1, loc2):
        """Get precomputed shortest distance between two locations."""
        # If either location is not in our precomputed distances (e.g., not a node in the graph),
        # it implies it's unreachable from relevant parts of the map.
        # Return infinity in this case.
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             return float('inf')
        return self.distances[loc1][loc2]

    def find_man_location(self, state):
        """Find the location of the man (assuming there's one man named 'bob')."""
        for fact in state:
             if match(fact, "at", "bob", "*"):
                 _, _, loc = get_parts(fact)
                 return loc
        # Man should always be at a location in a valid state
        return None # Indicate location not found

    def check_man_carrying_usable_spanner(self, state):
        """Check if the man is carrying a usable spanner."""
        # Assuming man is 'bob'
        carried_spanner = None
        for fact in state:
             if match(fact, "carrying", "bob", "*"):
                 _, _, spanner = get_parts(fact)
                 carried_spanner = spanner
                 break

        if carried_spanner:
             # Check if the carried spanner is usable
             if f"(usable {carried_spanner})" in state:
                 return True
        return False

    def find_nut_location(self, nut_name, state):
        """Find the location of a specific nut."""
        for fact in state:
             if match(fact, "at", nut_name, "*"):
                 _, _, loc = get_parts(fact)
                 return loc
        # A loose nut must be at a location. If not found, something is wrong.
        return None # Indicate location not found

    def find_usable_spanner_locations_on_ground(self, state):
        """Find locations of usable spanners that are on the ground."""
        usable_spanner_locs = set()
        # Find all usable spanners
        usable_spanners = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # Check if each usable spanner is on the ground (has an 'at' predicate)
        for spanner in usable_spanners:
             for fact in state:
                 if match(fact, "at", spanner, "*"):
                     # This spanner is on the ground at this location
                     _, _, loc = get_parts(fact)
                     usable_spanner_locs.add(loc)
                     break # Found location for this spanner, move to next usable spanner
        return usable_spanner_locs

    def find_nearest_location(self, start_loc, target_locs):
        """Find the location in target_locs nearest to start_loc based on precomputed distances."""
        if not target_locs or start_loc is None or start_loc not in self.distances:
            return None # Cannot find nearest if no targets or start is invalid/unreachable

        min_dist = float('inf')
        nearest_loc = None
        for target_loc in target_locs:
            dist = self.get_distance(start_loc, target_loc)
            if dist < min_dist:
                min_dist = dist
                nearest_loc = target_loc
        return nearest_loc # Returns None if target_locs is empty or all are unreachable

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose nuts
        loose_nuts = {nut for nut in self.all_nuts if f"(loose {nut})" in state}
        N_loose = len(loose_nuts)

        # 2. If no loose nuts, goal is reached
        if N_loose == 0:
            return 0

        # Initialize heuristic cost
        h = 0

        # Add cost for tighten actions (one per loose nut)
        h += N_loose

        # 3. Find man's location
        man_loc = self.find_man_location(state)
        if man_loc is None:
             # Man location not found, indicates an issue or unsolvable state
             return float('inf')

        # 4. Check if man is carrying a usable spanner
        man_has_usable_spanner = self.check_man_carrying_usable_spanner(state)

        # 5. Calculate pickup actions needed
        # Man needs a spanner for each nut. If he starts with one usable, he needs N_loose - 1 more pickups.
        num_pickups_needed = max(0, N_loose - (1 if man_has_usable_spanner else 0))
        h += num_pickups_needed

        # 6. Find locations of loose nuts
        loose_nut_locs = {self.find_nut_location(nut, state) for nut in loose_nuts}
        loose_nut_locs.discard(None) # Remove any potential None if nut location wasn't found

        # 7. Find locations of usable spanners on ground
        usable_spanner_locs_on_ground = self.find_usable_spanner_locations_on_ground(state)

        # 8. Determine potential first target locations
        required_first_locs = set(loose_nut_locs)

        # 9. If pickups are needed, consider nearest spanner location on ground as a potential first stop
        if num_pickups_needed > 0:
             if usable_spanner_locs_on_ground:
                  nearest_spanner_loc = self.find_nearest_location(man_loc, usable_spanner_locs_on_ground)
                  if nearest_spanner_loc is not None:
                       required_first_locs.add(nearest_spanner_loc)
             # else: If pickups are needed but no usable spanners on ground, problem is unsolvable.
             # This case is implicitly handled below if required_first_locs becomes empty or unreachable.

        # 10. Calculate minimum travel cost to the first required location
        if required_first_locs:
             nearest_required_loc = self.find_nearest_location(man_loc, required_first_locs)
             if nearest_required_loc is not None:
                 h += self.get_distance(man_loc, nearest_required_loc)
             else:
                 # If required locations exist but are unreachable from man_loc
                 return float('inf')
        else:
             # This case should only happen if N_loose > 0 but loose_nut_locs is empty,
             # and num_pickups_needed > 0 but usable_spanner_locs_on_ground is empty/unreachable.
             # This implies an unsolvable state.
             if N_loose > 0:
                  return float('inf')
             # If N_loose was 0, we would have returned 0 earlier.

        # 11. The heuristic is the sum calculated above.

        # Ensure heuristic is 0 only for goal states (handled by N_loose check).
        # Ensure heuristic is finite for solvable states (handled by returning inf for unreachable).

        return h
