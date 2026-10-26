import fnmatch
import collections
import math

# Assume Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static
        # Assuming task object also has initial_state
        self.initial_state = task.initial_state

    def __call__(self, node):
        raise NotImplementedError

# Helper functions from examples
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
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It counts the number of 'tighten_nut' actions required, the number of
    'pickup_spanner' actions required, and adds the minimum travel cost for the
    man to reach the first location where a necessary action (pickup or tighten)
    can occur.

    # Assumptions
    - The goal is always to tighten a specific set of nuts.
    - The man can carry multiple spanners (based on PDDL definition, although
      typical interpretations might restrict this; we follow the provided PDDL).
    - A usable spanner is consumed (becomes unusable) after one 'tighten_nut' action.
    - The graph of locations connected by 'link' predicates is static.

    # Heuristic Initialization
    - Parse static facts to build the location graph and compute all-pairs shortest paths.
    - Identify all locations, nuts, spanners, and the man from the initial state.
    - Identify the set of nuts that are required to be tightened in the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are currently loose and are part of the goal. Let this set be `N_loose_goal`.
    3. If `N_loose_goal` is empty, the heuristic is 0 (goal state reached).
    4. Identify all usable spanners and their current locations (carried by man or on the ground).
    5. Count the number of loose goal nuts (`k = |N_loose_goal|`).
    6. Count the number of usable spanners currently carried by the man (`c`).
    7. Count the number of usable spanners currently on the ground (`g`).
    8. Check if there are enough usable spanners in total (`c + g`) to tighten all loose goal nuts (`k`). If `c + g < k`, the state is likely unsolvable, return infinity.
    9. The base heuristic value is the sum of:
       - The number of 'tighten_nut' actions needed: `k`.
       - The number of 'pickup_spanner' actions needed: We need `k` usable spanners in total. `c` are already carried. We need `max(0, k - c)` more from the ground. Cost: `max(0, k - c)`.
       Base heuristic = `k + max(0, k - c)`.
    10. Add the minimum travel cost for the man to reach a location where the *next* necessary action can occur:
        - If the man is currently carrying at least one usable spanner (`c > 0`), the next useful action is likely 'tighten_nut'. He needs to go to the location of a loose goal nut. Find the minimum distance from the man's current location to any loose goal nut location.
        - If the man is not carrying any usable spanner (`c == 0`), the next useful action must be 'pickup_spanner'. He needs to go to the location of a usable spanner on the ground. Find the minimum distance from the man's current location to any usable spanner location on the ground.
        - Add this minimum distance to the base heuristic.
    11. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and goal details.
        """
        super().__init__(task)

        self.locations = set()
        self.all_nuts = set()
        self.all_spanners = set()
        self.man = None
        self.goal_nuts = set()

        # 1. Parse initial state and static facts to identify objects and locations
        all_facts = set(self.initial_state).union(self.static)
        for fact in all_facts:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                self.locations.add(loc)
                # Attempt to identify object types from initial state facts
                # This is a simplification; ideally, types come from the domain file
                if obj.startswith('nut'):
                    self.all_nuts.add(obj)
                elif obj.startswith('spanner'):
                    self.all_spanners.add(obj)
                elif 'bob' in obj.lower() or 'man' in obj.lower(): # Assuming man name contains 'bob' or 'man'
                     # Find the actual man object name from initial state if not 'bob'
                     # A more robust parser would get types from domain.
                     # For this problem, let's assume the first object of type man is the man.
                     # We can iterate through initial state to find the man object.
                     pass # Will find man below

            elif predicate == 'link':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)

        # Find the man object name explicitly from initial state if possible
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # This is a weak way to identify the man, relies on object naming convention
                 # A proper PDDL parser would use types. Let's assume the instance files
                 # follow a convention where the man object is easily identifiable,
                 # e.g., it's the only object of type 'man'. We can iterate through
                 # initial state 'at' facts and assume the first 'locatable' that isn't
                 # a known nut or spanner is the man.
                 if obj not in self.all_nuts and obj not in self.all_spanners:
                      self.man = obj # Assuming there's only one man object

        # If man wasn't found this way, maybe it's the first object of type man in the problem file?
        # Given the examples, the man object name is explicitly listed and used in 'at' facts.
        # Let's assume the man object name is 'bob' based on example 1 and 2.
        # A robust solution would parse the :objects section and types.
        # For this specific problem, let's rely on finding the object in 'at' facts
        # that isn't a nut or spanner. If that fails, default to 'bob'.
        if self.man is None:
             # Fallback: Assume 'bob' if not found by location check
             # This is fragile but might work for the given problem structure.
             # A better way: parse :objects and :types from the domain/instance file.
             # Since we only have facts, we infer.
             # Let's assume the man object is the one involved in 'carrying' or the one whose location changes.
             # From example 1 and 2, the man is named 'bob'. Let's hardcode or infer 'bob'.
             # Inferring from 'at' facts where the object is not a known nut/spanner is the best we can do with just facts.
             pass # self.man should be set by the loop above if 'bob' is in an 'at' fact.


        # 2. Build location graph
        self.graph = {loc: set() for loc in self.locations}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.graph and l2 in self.graph: # Ensure locations are known
                    self.graph[l1].add(l2)
                    self.graph[l2].add(l1) # Links are bidirectional

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {loc: float('inf') for loc in self.locations}
            self.dist[start_node][start_node] = 0
            queue = collections.deque([start_node])

            while queue:
                u = queue.popleft()
                if u in self.graph: # Ensure node exists in graph
                    for v in self.graph[u]:
                        if self.dist[start_node][v] == float('inf'):
                            self.dist[start_node][v] = self.dist[start_node][u] + 1
                            queue.append(v)

        # 4. Identify goal nuts
        # Assuming goals are conjunctions of (tightened ?n)
        for goal in self.goals:
             # Goal can be a single fact or a conjunction
             if isinstance(goal, str): # Single fact goal
                 if match(goal, "tightened", "*"):
                     self.goal_nuts.add(get_parts(goal)[1])
             elif isinstance(goal, list) and goal[0] == 'and': # Conjunction goal
                 for fact in goal[1:]:
                     if match(fact, "tightened", "*"):
                         self.goal_nuts.add(get_parts(fact)[1])


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # Extract current state information
        loc_m = None
        nut_locations = {}
        N_loose = set()
        SpannersCarried = set()
        spanner_locations = {} # Location if on ground
        spanner_usable = {} # Usability status

        # Need to ensure self.man is correctly identified.
        # Let's re-find man object name from state if needed, more robustly.
        current_man = None
        current_nuts = set()
        current_spanners = set()

        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at':
                obj, loc = parts[1], parts[2]
                # Infer object types from names if not already known
                if obj.startswith('nut'):
                    current_nuts.add(obj)
                    nut_locations[obj] = loc
                elif obj.startswith('spanner'):
                    current_spanners.add(obj)
                    spanner_locations[obj] = loc
                else:
                    # Assume the remaining 'at' object is the man
                    current_man = obj
                    loc_m = loc
            elif predicate == 'loose':
                N_loose.add(parts[1])
            elif predicate == 'carrying':
                 # Assuming the first argument of carrying is the man
                 current_man = parts[1]
                 SpannersCarried.add(parts[2])
            elif predicate == 'usable':
                spanner_usable[parts[1]] = True

        # If man wasn't found via 'at' or 'carrying', something is wrong with state or parsing
        if current_man is None:
             # This shouldn't happen in valid states, but handle defensively
             # Maybe assume the man object name found in __init__?
             # Let's assume the state is valid and contains man's location or carrying status.
             pass # current_man and loc_m should be set

        # Update self.man if found in state (more reliable than __init__ inference)
        if current_man:
             self.man = current_man
        elif self.man is None:
             # If still not found, this is a problem. Assume 'bob' as a last resort.
             self.man = 'bob' # Fragile fallback
             # Try to find bob's location if state contains it
             for fact in state:
                 if match(fact, 'at', self.man, '*'):
                     loc_m = get_parts(fact)[2]
                     break


        # Identify loose goal nuts
        N_loose_goal = N_loose.intersection(self.goal_nuts)
        k = len(N_loose_goal)

        # If all goal nuts are tightened, heuristic is 0
        if k == 0:
            return 0

        # Identify usable spanners
        UsableSpanners = {s for s, usable in spanner_usable.items() if usable}
        SpannersCarriedUsable = SpannersCarried.intersection(UsableSpanners)
        c = len(SpannersCarriedUsable)

        # Identify usable spanners on the ground and their locations
        SpannersGroundUsable_locs = [(s, spanner_locations[s]) for s in UsableSpanners if s not in SpannersCarried and s in spanner_locations]
        g = len(SpannersGroundUsable_locs)

        # Check if enough usable spanners exist in total
        if c + g < k:
            return float('inf') # Unsolvable state

        # Base heuristic: number of tighten actions + number of pickup actions needed
        h = k + max(0, k - c)

        # Add minimum travel cost to reach the first relevant location
        min_dist = float('inf')

        if loc_m is None or loc_m not in self.dist:
             # Man's location is unknown or not in the graph, likely an invalid state
             return float('inf')

        if c > 0:
            # Man has a usable spanner, next step is likely moving to a nut
            # Find the minimum distance to any loose goal nut location
            min_dist_to_nut = float('inf')
            for n in N_loose_goal:
                if n in nut_locations and nut_locations[n] in self.dist[loc_m]:
                     dist = self.dist[loc_m][nut_locations[n]]
                     if dist != float('inf'):
                         min_dist_to_nut = min(min_dist_to_nut, dist)
            min_dist = min_dist_to_nut

        else: # c == 0
            # Man needs to pick up a spanner first
            # Find the minimum distance to any usable spanner location on the ground
            min_dist_to_spanner = float('inf')
            for s, loc_s in SpannersGroundUsable_locs:
                 if loc_s in self.dist[loc_m]:
                     dist = self.dist[loc_m][loc_s]
                     if dist != float('inf'):
                         min_dist_to_spanner = min(min_dist_to_spanner, dist)
            min_dist = min_dist_to_spanner

        # If min_dist is still infinity, it means required locations are unreachable
        if min_dist == float('inf'):
             return float('inf')

        h += min_dist

        return h

