# Required imports
from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

# Helper functions (as seen in example heuristics)
def get_parts(fact):
    """Helper to parse a PDDL fact string into a list of parts."""
    # Remove parentheses and split by space
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class inheriting from Heuristic
class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
    Estimates the cost to reach the goal by summing the estimated costs
    to tighten each remaining loose nut that is part of the goal. The
    heuristic models the process as a sequence of tasks for the man:
    for each nut to be tightened, the man must either go directly to the
    nut (if carrying a usable spanner) or first go to a usable spanner,
    pick it up, and then go to the nut. The heuristic greedily selects
    the seemingly cheapest next nut/spanner combination at each step of
    this sequence estimation. It includes costs for walking, picking up
    spanners, and tightening nuts.

    Assumptions:
    - There is exactly one man, assumed to be named 'bob' based on examples.
    - Nuts do not move from their initial locations. Their initial locations
      are determined from the initial state facts. Objects starting with 'nut'
      are assumed to be nuts.
    - Spanners are single-use (become unusable after one tightening). Objects
      starting with 'spanner' are assumed to be spanners.
    - Links between locations defined by 'link' predicates are bidirectional.
    - The goal is a conjunction of (tightened nut) facts.

    Heuristic Initialization:
    - Parses static facts to build a graph of locations based on 'link' predicates.
      All locations mentioned in 'link' facts and initial 'at' facts are included.
      Links are treated as bidirectional.
    - Computes all-pairs shortest paths between all locations using BFS.
      Distances are stored in a dictionary `self.dist[l1][l2]`. Unreachable
      locations will have infinite distance.
    - Identifies the man's name (assumed 'bob').
    - Identifies the static location of each nut from the initial state facts
      where the object name starts with 'nut'. Stores in `self.nut_locations[nut_name] = loc_name`.
    - Stores the set of goal nuts from the goal facts.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the set of nuts that are goals but are currently loose. If this set is empty, the heuristic is 0.
    2. Find the man's current location from the state.
    3. Check if the man is currently carrying a usable spanner. This requires checking for a fact `(carrying man_name spanner_name)` and a corresponding fact `(usable spanner_name)` in the state.
    4. Identify the locations of all usable spanners that are currently at a location (not carried by the man).
    5. Identify the locations of all loose nuts that are goals.
    6. Initialize the total heuristic cost `h = 0`.
    7. Initialize the man's current estimated location, estimated carrying status, set of available usable spanner locations, and set of remaining loose nut locations for the sequential estimation. These are mutable copies of the values found in steps 2-5.
    8. While there are still loose nuts that are goals (represented by their locations in `current_loose_nuts_locs`):
        a. If the man is currently estimated to be carrying a usable spanner (`current_carrying_usable` is True):
            i. Find the closest remaining loose nut location (`l_n`) from the man's current estimated location (`current_man_loc`) using the precomputed distances.
            ii. If no reachable nut location is found, the state is a dead end, return infinity.
            iii. Add the walk cost (`dist(current_man_loc, l_n)`) plus the tighten cost (1) to `h`.
            iv. Update the man's estimated location to `l_n`.
            v. Update the man's estimated carrying status to false (the spanner is used).
            vi. Remove `l_n` from the set of `current_loose_nuts_locs`.
        b. If the man is currently estimated to NOT be carrying a usable spanner (`current_carrying_usable` is False):
            i. If there are no usable spanners available at locations (`current_usable_spanners_locs` is empty), the state is a dead end, return infinity.
            ii. Find the best pair of (available usable spanner location `l_s`, remaining loose nut location `l_n`) that minimizes the total cost of walking from the man's current estimated location to `l_s`, picking up the spanner (cost 1), walking from `l_s` to `l_n`, and tightening the nut (cost 1). The total step cost for a pair (l_s, l_n) is `dist(current_man_loc, l_s) + 1 + dist(l_s, l_n) + 1`.
            iii. If no reachable spanner-nut pair is found, the state is a dead end, return infinity.
            iv. Add the minimum total step cost found in the previous step to `h`.
            v. Update the man's estimated location to the chosen `best_nut_loc`.
            vi. Update the man's estimated carrying status to false (the spanner is used).
            vii. Remove the chosen `best_spanner_loc` from the set of `current_usable_spanners_locs`.
            viii. Remove the chosen `best_nut_loc` from the set of `current_loose_nuts_locs`.
    9. Return the total accumulated cost `h`.
    """
    def __init__(self, task):
        # 1. Build location graph and compute distances
        self.locations = set()
        self.graph = {} # Adjacency list

        # Extract locations and links from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1) # Assume links are bidirectional

        # Also add locations mentioned in initial state 'at' facts if not already included
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 self.graph.setdefault(loc, set()) # Ensure all locations are keys in graph

        # Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.locations:
            self.dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_node, distance = q.popleft()
                self.dist[start_node][current_node] = distance
                for neighbor in self.graph.get(current_node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, distance + 1))

        # 2. Identify man's name and static nut locations
        # Assuming 'bob' is the man's name based on examples
        self.man_name = 'bob' # This is an assumption based on example files

        self.nut_locations = {} # nut_name -> location_name
        # Identify nut locations from initial state facts
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Assuming objects starting with 'nut' are nuts
                if obj.lower().startswith('nut'):
                    self.nut_locations[obj] = loc

        # 3. Identify goal nuts
        # task.goals is a frozenset of goal fact strings, e.g., frozenset({'(tightened nut1)', '(tightened nut2)'})
        self.goal_nuts = set()
        for goal_fact_str in task.goals:
             parts = get_parts(goal_fact_str)
             if parts[0] == 'tightened' and len(parts) == 2:
                 self.goal_nuts.add(parts[1])


    def __call__(self, node):
        state = node.state

        # 1. Identify nuts to tighten (goal nuts that are currently loose)
        # These are the nuts in the goal whose '(tightened nut)' fact is not in the state.
        nuts_to_tighten = {n for n in self.goal_nuts if '(tightened ' + n + ')' not in state}

        if not nuts_to_tighten:
            return 0 # Goal reached

        # 2. Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man location not found, indicates an invalid state or problem setup
             return float('inf')

        # 3. Check if man is carrying a usable spanner
        man_carrying_usable = False
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                # Check if the carried spanner is usable in the current state
                if '(usable ' + carried_spanner + ')' in state:
                    man_carrying_usable = True
                break # Assuming man can carry only one spanner

        # 4. Identify locations of usable spanners (not carried)
        usable_spanners_locs = set()
        # Find all usable spanners by name
        usable_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        for spanner in usable_spanners_in_state:
             # If the spanner is not the one the man is currently carrying
             if not (man_carrying_usable and carried_spanner == spanner):
                 # Find where this usable spanner is located
                 spanner_loc = None
                 for fact in state:
                     if match(fact, "at", spanner, "*"):
                         spanner_loc = get_parts(fact)[2]
                         break
                 # If the spanner is at a location (not carried by someone else, which is not possible with one man)
                 if spanner_loc:
                     usable_spanners_locs.add(spanner_loc)

        # 5. Identify locations of loose nuts to tighten
        # These are the locations of nuts in nuts_to_tighten
        loose_nuts_target_locs = set()
        for nut_name in nuts_to_tighten:
             loc = self.nut_locations.get(nut_name)
             if loc:
                 loose_nuts_target_locs.add(loc)
             else:
                 # Goal nut location not found in initial state - problem setup error?
                 return float('inf')


        # 6. Sequential greedy cost estimation
        h = 0
        current_man_loc = man_loc
        current_carrying_usable = man_carrying_usable
        current_usable_spanners_locs = set(usable_spanners_locs) # Use a mutable copy
        current_loose_nuts_locs = set(loose_nuts_target_locs) # Use a mutable copy

        # Pre-check reachability for all relevant locations from current man_loc
        # This avoids repeated checks inside the loop and handles initial unreachable states
        all_target_locs = current_usable_spanners_locs | current_loose_nuts_locs
        for target_loc in all_target_locs:
             if current_man_loc not in self.dist or target_loc not in self.dist[current_man_loc]:
                  # Man cannot reach a required spanner or nut location initially
                  return float('inf')

        # Also check reachability between spanner and nut locations if needed later
        # This is implicitly handled by checking self.dist[l_s].get(l_n, float('inf')) inside the loop

        while current_loose_nuts_locs:
            if current_carrying_usable:
                # Man has spanner, go to closest remaining nut.
                min_nut_dist = float('inf')
                next_nut_loc = None
                for l_n in current_loose_nuts_locs:
                    d = self.dist[current_man_loc].get(l_n, float('inf'))
                    if d < min_nut_dist:
                        min_nut_dist = d
                        next_nut_loc = l_n

                if next_nut_loc is None or min_nut_dist == float('inf'):
                     # Cannot reach any remaining nut. Dead end.
                     return float('inf')

                step_cost = min_nut_dist + 1 # walk + tighten
                h += step_cost
                current_man_loc = next_nut_loc
                current_carrying_usable = False # Spanner used

                # Remove the location from the set of locations needing a nut tightened
                current_loose_nuts_locs.remove(next_nut_loc)


            else:
                # Man needs to get a spanner first, then go to a nut.
                # Find the best (spanner_loc, nut_loc) pair to minimize travel + actions.
                # Cost = dist(current_man_loc, l_s) + 1 (pickup) + dist(l_s, l_n) + 1 (tighten)
                min_total_cost_step = float('inf')
                best_spanner_loc = None
                best_nut_loc = None

                if not current_usable_spanners_locs:
                     # No usable spanners left, but nuts remain. Dead end.
                     return float('inf')

                for l_s in current_usable_spanners_locs:
                    # Check if spanner location is reachable from current man location
                    dist_m_s = self.dist[current_man_loc].get(l_s, float('inf'))
                    if dist_m_s == float('inf'):
                         continue # Cannot reach this spanner

                    for l_n in current_loose_nuts_locs:
                        # Check if nut location is reachable from spanner location
                        dist_s_n = self.dist[l_s].get(l_n, float('inf'))
                        if dist_s_n == float('inf'):
                            continue # Cannot reach this nut from this spanner

                        # Cost to get spanner and reach nut location
                        total_travel = dist_m_s + dist_s_n
                        action_cost = 1 + 1 # pickup + tighten
                        step_cost = total_travel + action_cost

                        if step_cost < min_total_cost_step:
                            min_total_cost_step = step_cost
                            best_spanner_loc = l_s
                            best_nut_loc = l_n

                if best_spanner_loc is None or best_nut_loc is None or min_total_cost_step == float('inf'):
                     # Cannot reach any spanner-nut pair. Dead end.
                     return float('inf')

                h += min_total_cost_step
                current_man_loc = best_nut_loc
                current_carrying_usable = False # Spanner used
                current_usable_spanners_locs.remove(best_spanner_loc)
                current_loose_nuts_locs.remove(best_nut_loc)

        return h
