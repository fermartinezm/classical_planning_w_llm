from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class for standalone testing
# In a real planning system, this would be provided.
class Heuristic:
    def __init__(self, task):
        pass
    def __call__(self, node):
        pass

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
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It considers the need to travel to each nut's location, acquire a usable
    spanner for each tightening operation (since spanners are single-use),
    and perform the pickup and tighten actions. The travel cost is estimated
    greedily based on the current location and the locations of needed spanners
    and nuts.

    # Assumptions
    - There is only one man.
    - Spanners are single-use (become unusable after one tighten action).
    - The man can only carry one spanner at a time.
    - Location links are bidirectional.
    - The problem is solvable (enough usable spanners exist). If not enough
      usable spanners are available in the initial state to tighten all goal
      nuts, the heuristic returns infinity.

    # Heuristic Initialization
    - Build the location graph from static `link` facts.
    - Compute all-pairs shortest paths between locations using BFS.
    - Identify all nut objects and their fixed locations from the initial state.
    - Identify all goal nuts from the goal conditions.
    - Identify the man object.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the man's current location.
    2. Check if the man is currently carrying a usable spanner.
    3. Identify all goal nuts that are currently loose. If none, the heuristic is 0.
    4. Count the total number of usable spanners available in the current state
        (either at a location or being carried). If this count is less than the
       number of loose goal nuts, the problem is unsolvable from this state,
       return infinity.
    5. Initialize the heuristic cost `h` to 0.
    6. Initialize the man's current location for the heuristic calculation
       (`current_man_loc`) and whether he is currently carrying a usable spanner
       (`currently_carrying_usable`).
    7. Identify the set of usable spanner objects currently available at locations.
    8. Sort the loose goal nuts by their distance from the `current_man_loc`.
    9. Iterate through the sorted list of loose goal nuts:
       a. For the current nut at `nut_loc`:
       b. If the man is *not* currently carrying a usable spanner:
          i. Find the closest usable spanner object `s` that is currently at a location
             (`s_loc`) relative to `current_man_loc`.
          ii. Add the distance `dist(current_man_loc, s_loc)` to `h` (walk cost).
          iii. Add 1 to `h` for the `pickup_spanner` action.
          iv. Update `current_man_loc` to `s_loc`.
          v. Remove the picked-up spanner `s` from the set of available spanners at locations.
          vi. Set `currently_carrying_usable` to True.
       c. Add the distance `dist(current_man_loc, nut_loc)` to `h` (walk cost).
       d. Add 1 to `h` for the `tighten_nut` action.
       e. Update `current_man_loc` to `nut_loc`.
       f. Set `currently_carrying_usable` to False (the spanner is used up).
    10. Return the total calculated cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing
        shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state
        # Assuming task object provides access to defined objects and their types
        # If not, we would need to parse the objects section from the PDDL problem file.
        # For this implementation, we assume task.objects is available.
        self.objects = task.objects if hasattr(task, 'objects') else self._infer_objects(task)

        self.locations = {obj for obj, type in self.objects.items() if type == 'location'}
        self.nuts = {obj for obj, type in self.objects.items() if type == 'nut'}
        self.spanners = {obj for obj, type in self.objects.items() if type == 'spanner'}
        self.men = {obj for obj, type in self.objects.items() if type == 'man'}
        # Assume there is exactly one man
        self.man_name = list(self.men)[0] if self.men else None

        # Build location graph
        self.adj = {loc: [] for loc in self.locations}
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                if l1 in self.locations and l2 in self.locations:
                    self.adj.setdefault(l1, []).append(l2)
                    self.adj.setdefault(l2, []).append(l1) # Links are bidirectional

        # Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = self._bfs(start_loc)

        # Find nut locations (nuts are locatable but don't move)
        self.nut_location = {}
        for fact in initial_state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj_name = parts[1]
                loc_name = parts[2]
                if obj_name in self.nuts:
                    self.nut_location[obj_name] = loc_name

        # Identify goal nuts
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if match(g, "tightened", "*")}

    def _infer_objects(self, task):
        """
        Infers objects and their types from initial state and goals if task.objects is not available.
        This is a fallback and might not be perfect for all PDDL structures.
        """
        objects = {}
        # Collect all arguments from initial state and goals
        all_args = set()
        for fact in task.initial_state | task.goals:
             all_args.update(get_parts(fact)[1:])

        # Simple type inference based on predicates they appear in
        for arg in all_args:
            # Check initial state facts
            for fact in task.initial_state:
                parts = get_parts(fact)
                if arg in parts:
                    if parts[0] == 'at' and parts.index(arg) == 1: # (at ?obj ?loc)
                         # Try to distinguish based on other predicates
                         is_nut = any(match(f, 'loose', arg) or match(f, 'tightened', arg) for f in task.initial_state | task.goals)
                         is_spanner = any(match(f, 'usable', arg) or match(f, 'carrying', '*', arg) for f in task.initial_state | task.goals)
                         is_man = any(match(f, 'carrying', arg, '*') for f in task.initial_state | task.goals) or any(match(f, 'at', arg, '*') and arg in self.men for f in task.initial_state) # Check if already identified as man
                         if is_nut: objects[arg] = 'nut'
                         elif is_spanner: objects[arg] = 'spanner'
                         elif is_man: objects[arg] = 'man'
                         else: objects[arg] = 'locatable' # Default for objects at locations

                    elif parts[0] == 'at' and parts.index(arg) == 2: # (at ?obj ?loc)
                         objects[arg] = 'location'
                    elif parts[0] == 'link' and arg in parts[1:]: # (link ?loc1 ?loc2)
                         objects[arg] = 'location'
                    elif parts[0] in ['loose', 'tightened'] and parts.index(arg) == 1: # (loose ?nut), (tightened ?nut)
                         objects[arg] = 'nut'
                    elif parts[0] == 'usable' and parts.index(arg) == 1: # (usable ?spanner)
                         objects[arg] = 'spanner'
                    elif parts[0] == 'carrying' and parts.index(arg) == 1: # (carrying ?man ?spanner)
                         objects[arg] = 'man'
                    elif parts[0] == 'carrying' and parts.index(arg) == 2: # (carrying ?man ?spanner)
                         objects[arg] = 'spanner'

        # Add any objects mentioned in goals that weren't in initial state facts used above
        for goal in task.goals:
             parts = get_parts(goal)
             if parts[0] == 'tightened' and len(parts) > 1 and parts[1] not in objects:
                  objects[parts[1]] = 'nut' # Assume anything tightened is a nut

        # Ensure all objects from initial state facts are covered, even if type is unknown
        for fact in task.initial_state:
             parts = get_parts(fact)
             for arg in parts[1:]:
                  if arg not in objects:
                       # Could try more sophisticated inference or default
                       objects[arg] = 'object' # Default type

        # Simple check for man if not found
        if not any(t == 'man' for t in objects.values()):
             # Look for an object that is locatable but not a nut or spanner
             locatables = {obj for obj, type in objects.items() if type == 'locatable'}
             if locatables:
                  # This is a guess, pick one locatable that isn't a nut/spanner
                  potential_men = [obj for obj in locatables if obj not in self.nuts and obj not in self.spanners]
                  if potential_men:
                       objects[potential_men[0]] = 'man' # Assume the first one is the man

        return objects


    def _bfs(self, start_loc):
        """
        Performs a Breadth-First Search from a start location to find distances
        to all reachable locations.
        """
        distances = {loc: float('inf') for loc in self.locations}
        distances[start_loc] = 0
        queue = deque([start_loc])
        visited = {start_loc}

        while queue:
            current_loc = queue.popleft()

            if current_loc in self.adj: # Ensure the location exists in the graph
                for neighbor in self.adj[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_loc] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state

        # Find man's current location
        man_loc = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_loc = get_parts(fact)[2]
                break
        if man_loc is None:
             # Man must be somewhere if problem is solvable
             return float('inf') # Should not happen in valid states

        # Check if man is carrying a usable spanner
        man_carrying_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                man_carrying_spanner = get_parts(fact)[2]
                break

        man_carrying_usable = False
        if man_carrying_spanner and f'(usable {man_carrying_spanner})' in state:
            man_carrying_usable = True

        # Identify loose goal nuts
        loose_goal_nuts = {n for n in self.goal_nuts if f'(loose {n})' in state}

        # If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        # Identify usable spanners currently at locations
        usable_spanners_at_loc = {} # {spanner_obj: location}
        for fact in state:
            if match(fact, "at", "*", "*"):
                parts = get_parts(fact)
                obj_name = parts[1]
                loc_name = parts[2]
                if obj_name in self.spanners and f'(usable {obj_name})' in state:
                    usable_spanners_at_loc[obj_name] = loc_name

        # Check if enough usable spanners exist in total
        total_usable_spanners = len(usable_spanners_at_loc) + (1 if man_carrying_usable else 0)
        if total_usable_spanners < len(loose_goal_nuts):
             # Problem is unsolvable from this state
             return float('inf')

        # Heuristic calculation based on greedy path
        h = 0
        current_man_loc = man_loc
        currently_carrying_usable = man_carrying_usable
        # Need to track available spanner objects, not just locations, as locations can have multiple spanners
        available_spanners_at_loc_copy = dict(usable_spanners_at_loc) # Copy to modify

        # Sort nuts by distance from current man location
        # Using list() to create a mutable copy for sorting
        remaining_nuts = sorted(list(loose_goal_nuts), key=lambda n: self.dist[current_man_loc][self.nut_location[n]])

        for nut in remaining_nuts:
            nut_loc = self.nut_location[nut]

            # Step 1: Acquire a usable spanner if needed
            if not currently_carrying_usable:
                # Find the closest available usable spanner object at a location
                min_dist_to_spanner = float('inf')
                best_spanner_obj = None
                best_spanner_loc = None

                # Iterate through spanner objects still available at locations
                for s_obj, s_loc in available_spanners_at_loc_copy.items():
                    if self.dist[current_man_loc][s_loc] < min_dist_to_spanner:
                        min_dist_to_spanner = self.dist[current_man_loc][s_loc]
                        best_spanner_obj = s_obj
                        best_spanner_loc = s_loc

                # This case should ideally not happen if total_usable_spanners check passed,
                # but as a safeguard:
                if best_spanner_obj is None:
                     return float('inf') # Should have enough spanners based on initial check

                # Add cost to walk to spanner and pick it up
                h += min_dist_to_spanner
                h += 1 # pickup_spanner action

                # Update man's location and status
                current_man_loc = best_spanner_loc
                currently_carrying_usable = True
                # Remove the spanner object from available ones at locations
                del available_spanners_at_loc_copy[best_spanner_obj]


            # Step 2: Walk to the nut location
            h += self.dist[current_man_loc][nut_loc] # walk action(s)
            current_man_loc = nut_loc

            # Step 3: Tighten the nut
            h += 1 # tighten_nut action
            currently_carrying_usable = False # Spanner is used up

        return h

