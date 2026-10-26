from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available as heuristics.heuristic_base.Heuristic
# from heuristics.heuristic_base import Heuristic

# Utility functions to parse PDDL facts represented as strings
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
    # Basic check for number of parts vs args, allowing trailing wildcards
    if len(parts) < len(args) or (len(parts) > len(args) and args[-1] != '*'):
         return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Define the heuristic class inheriting from Heuristic
# class spannerHeuristic(Heuristic): # Use this line in the actual environment
class spannerHeuristic: # Use this for standalone testing/development

    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all loose
    nuts specified in the goal. It considers the number of nuts remaining, the
    need to acquire usable spanners, and the travel cost for the man to reach
    relevant locations (nuts and spanners).

    # Assumptions:
    - Nut locations are static (do not change during planning).
    - Spanners become unusable after one use for tightening a nut.
    - There is only one man object.
    - Object types (man, nut, spanner, location) are inferred based on naming
      conventions (e.g., 'nut' prefix for nuts, 'spanner' prefix for spanners,
      the single non-nut/spanner locatable object is the man) and predicate
      structure in the initial state. This is fragile and relies on standard
      PDDL problem file structure.
    - Links between locations are bidirectional.
    - The graph of locations is connected, or relevant locations are reachable.

    # Heuristic Initialization
    - Precomputes all-pairs shortest path distances between locations based on
      `link` facts using BFS.
    - Stores initial nut locations.
    - Identifies the man object.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all loose nuts that are required to be tightened in the goal.
       Let `k` be the count of these nuts. If `k` is 0, the heuristic is 0.
    3. Count the number of usable spanners the man is currently carrying (`c`).
    4. Count the number of usable spanners available on the ground and their locations.
    5. Calculate the number of additional spanners the man needs to pick up from
       the ground: `pickups_needed = max(0, k - c)`. If `pickups_needed > 0` but
       no usable spanners are on the ground, the state is likely unsolvable,
       return infinity.
    6. The base heuristic cost is the sum of the required `tighten_nut` actions
       (`k`) and the required `pickup_spanner` actions (`pickups_needed`).
       `h = k + pickups_needed`.
    7. Estimate the travel cost (walk actions):
       - The man needs to travel from his current location to interact with
         relevant objects (nuts to tighten, spanners to pick up).
       - Identify the set of locations where loose nuts needing tightening are
         located (`L_nuts`).
       - Identify the set of locations where usable spanners are on the ground
         (`L_spanners_ground`).
       - The set of potential first stops includes `L_nuts` (if `k > 0`) and
         `L_spanners_ground` (if `pickups_needed > 0`).
       - Calculate the shortest distance from the man's current location to the
         closest reachable location in this set of potential first stops. Add
         this distance to `h`.
       - After the first interaction, the man needs to perform `k + pickups_needed - 1`
         additional interactions (tighten or pickup). Assume each subsequent
         interaction requires at least one walk action to move to the next
         location. Add `max(0, k + pickups_needed - 1)` to `h`.
    8. Return the total estimated cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by precomputing distances and storing static info.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # 1. Build the location graph from link facts and identify all locations
        self.locations = set()
        self.adj = {} # Adjacency list: location -> set of connected locations
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adj.setdefault(loc1, set()).add(loc2)
                self.adj.setdefault(loc2, set()).add(loc1) # Links are bidirectional

        # Add all locations mentioned in the initial state or goals to ensure they are in self.locations
        # even if they have no links defined.
        for fact in self.initial_state:
             if match(fact, "at", "*", "*"):
                 _, obj, loc = get_parts(fact)
                 self.locations.add(loc)
                 self.adj.setdefault(loc, set()) # Add location even if no links

        # Goal locations are implicitly where the nuts are, which we get from initial state.

        # 2. Precompute all-pairs shortest paths using BFS
        self.distance = {}
        for start_loc in self.locations:
            self.distance[start_loc] = {loc: float('inf') for loc in self.locations}
            self.distance[start_loc][start_loc] = 0
            queue = deque([start_loc])

            while queue:
                curr = queue.popleft()
                # If a location has no links, it won't be in adj, skip its neighbors
                if curr not in self.adj:
                    continue

                for neighbor in self.adj[curr]:
                    if self.distance[start_loc][neighbor] == float('inf'):
                        self.distance[start_loc][neighbor] = self.distance[start_loc][curr] + 1
                        queue.append(neighbor)

        # 3. Store initial nut locations (assuming they are static)
        self.nut_locations = {}
        for fact in self.initial_state:
            if match(fact, "at", "?n", "?l"):
                parts = get_parts(fact)
                # Fragile: Assumes objects starting with 'nut' are nuts
                if parts[1].startswith('nut'):
                     self.nut_locations[parts[1]] = parts[2]

        # 4. Identify the man object (assuming only one man and it's the only locatable not a nut/spanner)
        self.man = None
        for fact in self.initial_state:
             if match(fact, "at", "?m", "*"):
                 obj_name = get_parts(fact)[1]
                 # Fragile: Check if object name doesn't start with 'nut' or 'spanner'
                 if not obj_name.startswith('nut') and not obj_name.startswith('spanner'):
                      self.man = obj_name
                      break # Assuming only one man


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
             # Man's location should always be known in a valid state
             return float('inf') # Indicates an invalid or unsolvable state

        # 2. Identify loose nuts that need tightening (goal is tightened, state is loose)
        loose_nuts_to_go = set()
        goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "?n"):
                goal_nuts.add(get_parts(goal)[1])

        for nut in goal_nuts:
            if f"(loose {nut})" in state:
                 loose_nuts_to_go.add(nut)

        k = len(loose_nuts_to_go)

        # If all required nuts are tightened, goal reached
        if k == 0:
            return 0

        # 3. Identify usable spanners carried by the man
        usable_spanners_carried = set()
        for fact in state:
            if match(fact, "carrying", self.man, "?s"):
                 spanner = get_parts(fact)[2]
                 # Fragile: Check if object name starts with 'spanner'
                 if spanner.startswith('spanner') and f"(usable {spanner})" in state:
                      usable_spanners_carried.add(spanner)
        c = len(usable_spanners_carried)

        # 4. Identify usable spanners on the ground and their locations
        usable_spanners_on_ground = {} # spanner -> location
        for fact in state:
            if match(fact, "at", "?s", "?l"):
                 parts = get_parts(fact)
                 spanner = parts[1]
                 location = parts[2]
                 # Fragile: Check if object name starts with 'spanner'
                 if spanner.startswith('spanner') and f"(usable {spanner})" in state:
                      usable_spanners_on_ground[spanner] = location

        L_spanners_ground = set(usable_spanners_on_ground.values())

        # 5. Calculate number of pickups needed
        pickups_needed = max(0, k - c)

        # If pickups are needed but no usable spanners are on the ground, it's unsolvable
        if pickups_needed > 0 and not L_spanners_ground:
             return float('inf')

        # 6. Calculate base heuristic: tighten actions + pickup actions
        h = k + pickups_needed

        # 7. Calculate walk cost
        L_nuts = {self.nut_locations[nut] for nut in loose_nuts_to_go if nut in self.nut_locations} # Ensure nut location is known

        relevant_locations_for_first_stop = set()
        if k > 0:
            relevant_locations_for_first_stop.update(L_nuts)
        if pickups_needed > 0:
            relevant_locations_for_first_stop.update(L_spanners_ground)

        walk_cost = 0
        if relevant_locations_for_first_stop:
            # Find the closest reachable relevant location
            closest_relevant_dist = float('inf')
            closest_relevant_loc = None

            # Ensure man_location is in the distance map (should be if in self.locations)
            if man_location not in self.distance:
                 return float('inf') # Should not happen if __init__ is correct

            for loc in relevant_locations_for_first_stop:
                # Ensure the relevant location is in the distance map and is reachable
                if loc in self.distance[man_location]:
                    dist = self.distance[man_location][loc]
                    if dist != float('inf') and dist < closest_relevant_dist:
                        closest_relevant_dist = dist
                        closest_relevant_loc = loc

            if closest_relevant_loc is None:
                 # No relevant location is reachable from man's current location. Unsolvable.
                 return float('inf')

            walk_cost = closest_relevant_dist # Walk to the first relevant location

            # Add cost for subsequent walks between interaction events
            # Total interaction events = k (tightens) + pickups_needed (pickups)
            # After the first stop, there are (k + pickups_needed - 1) events left.
            # Assume 1 walk action is needed between each subsequent event location.
            walk_cost += max(0, k + pickups_needed - 1)

        h += walk_cost

        return h
