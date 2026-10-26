from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic # Assuming this base class exists

# Helper functions to parse PDDL facts
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
    # Ensure we don't go out of bounds if fact has fewer parts than args
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It models the process as a sequence of cycles, where each cycle
    involves the man walking to a usable spanner, picking it up, walking to a
    loose goal nut, and tightening it. The heuristic greedily selects the
    cheapest next spanner-nut pair based on the man's current location.

    # Assumptions
    - The goal is to tighten a specific set of nuts.
    - A spanner becomes unusable after one tightening action.
    - The man can only carry one spanner at a time.
    - The number of initially available usable spanners (including any carried)
      must be at least the number of loose goal nuts for the problem to be solvable.
    - The graph of locations connected by 'link' predicates is static.

    # Heuristic Initialization
    - Parses object types (man, spanner, nut, location) from task objects.
    - Identifies the name of the man.
    - Identifies the set of nuts that must be tightened in the goal state.
    - Builds a graph of locations based on 'link' predicates from static facts.
    - Computes all-pairs shortest path distances between locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Identify the locations of all spanners and nuts.
    3. Determine which spanners are usable and which nuts are loose.
    4. Determine if the man is currently carrying a spanner, and if it's usable.
    5. Filter loose nuts to include only those that are part of the goal.
    6. Check if the total number of available usable spanners (on the ground + carried)
       is sufficient for the number of loose goal nuts. If not, return a large value
       indicating likely unsolvability.
    7. Initialize the heuristic cost `h` to 0.
    8. Set the `current_man_loc` to the man's location in the state.
    9. Create sets of remaining loose goal nuts and remaining usable spanners on the ground.
    10. If the man is currently carrying a usable spanner:
        - Find the loose goal nut closest to the man's current location.
        - Add the cost (walk to nut + tighten) to `h`.
        - Update `current_man_loc` to the location of the tightened nut.
        - Remove the nut from the set of remaining loose nuts. The carried spanner is conceptually used.
        - If no loose goal nuts were found or reachable, but some exist, return a large value.
    11. While there are still loose goal nuts remaining:
        - Find the pair of (remaining usable spanner `s`, remaining loose goal nut `n`)
          that minimizes the cost of a full cycle starting from `current_man_loc`:
          `distance(current_man_loc, loc(s)) + 1 (pickup) + distance(loc(s), loc(n)) + 1 (tighten)`.
        - Add this minimum cycle cost to `h`.
        - Update `current_man_loc` to the location of the tightened nut `n`.
        - Remove `n` from the set of remaining loose nuts.
        - Remove `s` from the set of remaining usable spanners on the ground.
        - If no such pair can be found (e.g., no spanners left, but nuts remain, or locations unreachable), return a large value.
    12. Return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting static information."""
        self.goals = task.goals
        self.static_facts = task.static
        # Assuming task.objects provides object names and types like ['bob - man', 'spanner1 - spanner', ...]
        self.objects = task.objects

        # 1. Parse object types and find the man
        self.object_types = {}
        self.man_name = None
        for obj_str in self.objects:
            name, obj_type = obj_str.split(" - ")
            self.object_types[name] = obj_type
            if obj_type == 'man':
                self.man_name = name # Assuming there's exactly one man

        # 2. Identify goal nuts
        self.goal_nuts = {
            get_parts(goal)[1]
            for goal in self.goals
            if match(goal, "tightened", "*")
        }

        # 3. Build location graph and compute distances
        locations = set()
        adj = {} # Adjacency list {location: [neighbor1, neighbor2, ...]}

        # Collect all locations mentioned in static links
        for fact in self.static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                locations.add(loc1)
                locations.add(loc2)
                adj.setdefault(loc1, []).append(loc2)
                adj.setdefault(loc2, []).append(loc1)

        self.distances = {} # {(loc1, loc2): distance}

        # Compute shortest paths using BFS from each location
        for start_node in locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                for neighbor in adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1-4. Extract current state information
        man_loc = None
        spanner_locs = {} # {spanner_name: location}
        nut_locs = {} # {nut_name: location}
        usable_spanners_names = set() # names of usable spanners
        loose_nuts_names = set() # names of loose nuts
        carried_spanner_name = None # name of spanner being carried

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                if obj == self.man_name:
                    man_loc = loc
                elif self.object_types.get(obj) == 'spanner':
                    spanner_locs[obj] = loc
                elif self.object_types.get(obj) == 'nut':
                    nut_locs[obj] = loc
            elif parts[0] == "carrying":
                # Assuming only the man carries
                carried_spanner_name = parts[2]
            elif parts[0] == "usable":
                usable_spanners_names.add(parts[1])
            elif parts[0] == "loose":
                loose_nuts_names.add(parts[1])

        # 5. Filter loose nuts to include only those in the goal set
        LooseGoalNuts = {n for n in loose_nuts_names if n in self.goal_nuts}

        # Usable spanners are those marked usable AND on the ground OR being carried
        UsableSpannersOnGround = {s for s in usable_spanners_names if s in spanner_locs}
        carried_usable = (carried_spanner_name is not None) and (carried_spanner_name in usable_spanners_names)

        # 6. Check solvability based on spanner count
        if len(UsableSpannersOnGround) + (1 if carried_usable else 0) < len(LooseGoalNuts):
            # Not enough usable spanners exist in the state to tighten all goal nuts
            return 1000000 # Return a large number indicating likely unsolvable

        # 7. Initialize heuristic cost
        h = 0
        current_man_loc = man_loc
        remaining_loose_nuts = set(LooseGoalNuts)
        remaining_usable_spanners_on_ground = set(UsableSpannersOnGround) # Names

        # 10. Handle case where man is already carrying a usable spanner
        if carried_usable and remaining_loose_nuts:
            min_cost_first_nut = float('inf')
            best_first_n_name = None

            # Find the closest loose goal nut to use the carried spanner on
            for n_name in remaining_loose_nuts:
                n_loc = nut_locs.get(n_name)
                if n_loc is None: continue # Nut location unknown (shouldn't happen in valid state)

                # Cost: walk to nut + tighten (1 action)
                dist = self.distances.get((current_man_loc, n_loc), float('inf'))
                cost = dist + 1
                if cost < min_cost_first_nut:
                    min_cost_first_nut = cost
                    best_first_n_name = n_name

            if best_first_n_name is not None and min_cost_first_nut != float('inf'):
                 h += min_cost_first_nut
                 current_man_loc = nut_locs[best_first_n_name]
                 remaining_loose_nuts.remove(best_first_n_name)
                 # The carried spanner is now used up (conceptually for the heuristic)
            elif len(remaining_loose_nuts) > 0:
                 # Cannot reach any loose nut with the carried spanner
                 return 1000000 # Likely unsolvable from here

        # 11. Process remaining loose nuts needing a new spanner pickup
        while remaining_loose_nuts:
            min_cost_cycle = float('inf')
            best_s_name = None
            best_n_name = None

            # Find the best spanner and nut pair for the next cycle
            for n_name in remaining_loose_nuts:
                n_loc = nut_locs.get(n_name)
                if n_loc is None: continue # Nut location unknown

                for s_name in remaining_usable_spanners_on_ground:
                    s_loc = spanner_locs.get(s_name)
                    if s_loc is None: continue # Spanner location unknown

                    # Cost: walk from current_man_loc to spanner + pickup (1) + walk from spanner to nut + tighten (1)
                    dist_to_s = self.distances.get((current_man_loc, s_loc), float('inf'))
                    dist_s_to_n = self.distances.get((s_loc, n_loc), float('inf'))

                    if dist_to_s == float('inf') or dist_s_to_n == float('inf'):
                        cost = float('inf')
                    else:
                        cost = dist_to_s + 1 + dist_s_to_n + 1

                    if cost < min_cost_cycle:
                        min_cost_cycle = cost
                        best_s_name = s_name
                        best_n_name = n_name

            if min_cost_cycle == float('inf'):
                 # Cannot reach any remaining spanner or nut
                 return 1000000 # Return a large number indicating likely unsolvable

            h += min_cost_cycle
            current_man_loc = nut_locs[best_n_name]
            remaining_loose_nuts.remove(best_n_name)
            remaining_usable_spanners_on_ground.remove(best_s_name)

        # 12. Return total cost
        return h
