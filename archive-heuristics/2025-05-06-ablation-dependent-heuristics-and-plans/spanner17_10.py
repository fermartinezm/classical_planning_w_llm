# Need to import Heuristic base class if it's in a specific path
# Assuming it's available as heuristics.heuristic_base.Heuristic
from fnmatch import fnmatch
from collections import deque # For BFS

# Dummy Heuristic base class for standalone testing if needed
class Heuristic:
    def __init__(self, task):
        self.task = task
        pass
    def __call__(self, node):
        pass

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not fact.strip() or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts that are specified in the goal. It sums the estimated cost
    for each individual loose goal nut, considering the need to move the
    man to the nut's location and ensure he has a usable spanner there.

    # Assumptions
    - Each nut needs exactly one 'tighten_nut' action.
    - Each 'tighten_nut' action consumes one 'usable' spanner.
    - The man can carry spanners. The heuristic assumes the man needs to
      acquire a usable spanner if he isn't already carrying one when he
      needs to tighten a nut.
    - The location graph defined by 'link' predicates is static.
    - The heuristic calculates the minimum cost to get the man *with a usable
      spanner* to each loose goal nut location independently and sums these costs,
      plus the cost of the 'tighten_nut' action for each nut. This might
      overestimate travel costs but provides a reasonable estimate of the
      required effort involving movement and resource acquisition.
    - Object types (man, spanner, nut) are inferred from initial state facts
      based on the predicates they appear in. This relies on standard domain structure.

    # Heuristic Initialization
    - Parses static 'link' facts to build the location graph and compute
      all-pairs shortest paths between locations using BFS.
    - Identifies the names of goal nuts from the task's goal conditions.
    - Infers the names of the man, spanner, and nut objects by examining
      which objects appear in specific predicates ('carrying', 'usable',
      'loose', 'tightened') within the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify all nuts that are currently 'loose' and are specified in the goal.
    3. If there are no such nuts, the heuristic is 0.
    4. Calculate the shortest path distance from the man's current location to all other locations using BFS.
    5. Identify all usable spanners (either on the ground or carried by the man) and their effective locations.
    6. For each loose goal nut 'N' at location 'L_N':
       a. The cost includes 1 action for 'tighten_nut'.
       b. The cost includes the minimum effort to get the man *with a usable spanner* to location 'L_N'.
          - This minimum effort is calculated as the minimum of two possibilities:
            i. If the man is currently carrying *any* usable spanner: The cost is just the distance from his current location to 'L_N'.
            ii. If the man is not carrying *any* usable spanner: He must go to a location 'L_S' with a usable spanner, pick it up (cost 1), and then travel from 'L_S' to 'L_N'. The minimum cost for this is `min(dist(man_loc, L_S) + 1 + dist(L_S, L_N))` over all usable spanners 'S' at 'L_S'.
          - The minimum of these possibilities (i and ii) is the cost to have a spanner ready at 'L_N'.
       c. Sum `1 + cost_to_have_spanner_at_nut_loc` for each loose goal nut.
    7. Return the total sum. If any required location is unreachable, the heuristic is infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph, computing
        distances, and identifying goal nuts and object types.
        """
        self.task = task
        self.goals = task.goals
        self.static_facts = task.static

        # --- Build Location Graph and Compute Distances ---
        self.adj = {}
        self.locations = set()
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.locations.add(loc1)
                self.locations.add(loc2)
                self.adj.setdefault(loc1, []).append(loc2)
                self.adj.setdefault(loc2, []).append(loc1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc, self.adj, self.locations)

        # --- Identify Goal Nuts ---
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, "tightened", "*")}

        # --- Infer Object Types (Man, Spanners, Nuts) from initial state ---
        self.man_obj = None
        self.spanner_objs = set()
        self.nut_objs = set()
        locatable_objs = set()

        # Collect objects mentioned in relevant initial state predicates
        objects_in_carrying = set()
        objects_in_usable = set()
        objects_in_loose = set()
        objects_in_tightened = set()
        objects_at_location = set() # Objects that are 'at' a location

        for fact in task.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            predicate = parts[0]
            if predicate == 'at' and len(parts) == 3:
                objects_at_location.add(parts[1])
                locatable_objs.add(parts[1])
            elif predicate == 'carrying' and len(parts) == 3:
                objects_in_carrying.add(parts[1]) # Man
                objects_in_carrying.add(parts[2]) # Spanner
            elif predicate == 'usable' and len(parts) == 2:
                objects_in_usable.add(parts[1]) # Spanner
            elif predicate == 'loose' and len(parts) == 2:
                objects_in_loose.add(parts[1]) # Nut
            elif predicate == 'tightened' and len(parts) == 2:
                objects_in_tightened.add(parts[1]) # Nut

        # Infer types based on predicate usage and locatable property
        # Man: Appears in 'carrying' (as first arg) and is locatable (most reliable)
        # Fallback: Appears in 'at' but not in spanner/nut predicates and is locatable
        potential_men_carrying = {obj for obj in objects_in_carrying if obj in locatable_objs}
        if potential_men_carrying:
             # Assuming there is exactly one man object
             self.man_obj = next(iter(potential_men_carrying))
        else:
             # Fallback: Assume the single locatable object not clearly a spanner or nut is the man
             # Identify objects clearly spanners or nuts first
             clearly_spanners = (objects_in_usable | {obj for obj in objects_in_carrying if obj not in potential_men_carrying}) & locatable_objs
             clearly_nuts = (objects_in_loose | objects_in_tightened) & locatable_objs
             other_locatables = {obj for obj in locatable_objs if obj not in clearly_spanners and obj not in clearly_nuts}

             if len(other_locatables) == 1:
                  self.man_obj = next(iter(other_locatables))
             elif len(other_locatables) > 1:
                  # This case is problematic, heuristic might be unreliable
                  print(f"Warning: Multiple potential man objects inferred: {other_locatables}. Picking one arbitrarily.")
                  self.man_obj = next(iter(other_locatables))
             else:
                  # This case is also problematic
                  print("Warning: Could not infer man object from initial state facts.")


        # Spanners: Appear in 'usable' or 'carrying' (as second arg) and are locatable
        self.spanner_objs = (objects_in_usable | {obj for obj in objects_in_carrying if obj != self.man_obj}) & locatable_objs

        # Nuts: Appear in 'loose' or 'tightened' and are locatable
        self.nut_objs = (objects_in_loose | objects_in_tightened) & locatable_objs


    def _bfs(self, start_node, adj, nodes):
        """
        Performs BFS from a start node on the given adjacency list.
        Returns a dictionary of distances from the start node.
        Returns float('inf') for unreachable nodes.
        """
        distances = {node: float('inf') for node in nodes}
        if start_node not in nodes:
             # Start node is not a known location, cannot reach anything
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in adj:
                for neighbor in adj[current_node]:
                    if distances[neighbor] == float('inf'): # Not visited
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state

        # 1. Identify man's current location
        man_location = None
        if self.man_obj:
            for fact in state:
                if match(fact, "at", self.man_obj, "*"):
                    man_location = get_parts(fact)[2]
                    break

        if man_location is None or man_location not in self.locations:
             # Man's location is unknown or not a valid location node
             return float('inf')

        # 2. Identify loose nuts that are goals
        loose_goal_nuts = {nut for nut in self.goal_nuts if f"(loose {nut})" in state}

        # 3. If no loose goal nuts, goal is reached
        if not loose_goal_nuts:
            return 0

        # 4. Calculate distances from man's current location
        dist_man = self._bfs(man_location, self.adj, self.locations)

        # 5. Identify usable spanners and their effective locations
        usable_spanner_locs = {} # {spanner_name: effective_location}
        carried_spanners = {get_parts(fact)[2] for fact in state if self.man_obj and match(fact, "carrying", self.man_obj, "*")}
        man_carrying_usable_spanner_exists = False

        for s_name in self.spanner_objs:
            if f"(usable {s_name})" in state:
                if s_name in carried_spanners:
                    usable_spanner_locs[s_name] = man_location # Effective location is man's location
                    man_carrying_usable_spanner_exists = True
                else: # On the ground
                    for fact in state:
                        if match(fact, "at", s_name, "*"):
                            usable_spanner_locs[s_name] = get_parts(fact)[2]
                            break
                    # If a usable spanner is not carried and not 'at' a location, it's in an invalid state
                    # but the BFS/distance check below will handle reachability implicitly.


        # 6. Find locations of loose goal nuts
        nut_locs = {}
        for nut_name in loose_goal_nuts:
             for fact in state:
                  if match(fact, "at", nut_name, "*"):
                       nut_locs[nut_name] = get_parts(fact)[2]
                       break
             if nut_name not in nut_locs or nut_locs[nut_name] not in self.locations:
                  # Nut location unknown or not a valid location node
                  return float('inf')


        # 7. Calculate heuristic based on sum of costs per nut
        total_cost = 0
        for nut_name in loose_goal_nuts:
            nut_loc = nut_locs[nut_name]

            # Cost to get man *with a usable spanner* to nut_loc
            # This is the minimum cost to reach a state where (at man nut_loc) and (carrying man S) and (usable S) hold.
            cost_to_achieve_preconditions_at_nut_loc = float('inf')

            # Option 1: Man uses a spanner he is currently carrying (if any usable)
            if man_carrying_usable_spanner_exists:
                 if nut_loc in dist_man and dist_man[nut_loc] != float('inf'):
                      cost_to_achieve_preconditions_at_nut_loc = min(cost_to_achieve_preconditions_at_nut_loc, dist_man[nut_loc])

            # Option 2: Man goes to pick up a usable spanner (from ground or carried) and brings it to nut_loc
            min_cost_pickup_bring = float('inf')
            if usable_spanner_locs:
                 # Find the minimum cost to go from man_location to a usable spanner location (L_S),
                 # pick it up (+1), and then go from L_S to nut_loc.
                 for s_name, s_loc in usable_spanner_locs.items():
                      # Ensure s_loc is reachable from man_location and nut_loc is reachable from s_loc
                      if s_loc in dist_man and dist_man[s_loc] != float('inf') and \
                         nut_loc in self.dist.get(s_loc, {}) and self.dist[s_loc][nut_loc] != float('inf'):
                           cost = dist_man[s_loc] + 1 + self.dist[s_loc][nut_loc]
                           min_cost_pickup_bring = min(min_cost_pickup_bring, cost)

            cost_to_achieve_preconditions_at_nut_loc = min(cost_to_achieve_preconditions_at_nut_loc, min_cost_pickup_bring)

            # If it's impossible to get a spanner to the nut location, this nut cannot be tightened
            if cost_to_achieve_preconditions_at_nut_loc == float('inf'):
                 return float('inf') # Unreachable nut or no usable spanners available anywhere

            # Total cost for this nut: 1 (tighten) + cost to achieve preconditions at nut_loc
            total_cost += 1 + cost_to_achieve_preconditions_at_nut_loc

        return total_cost
