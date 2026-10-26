from fnmatch import fnmatch
# Assuming heuristic_base is available in the environment
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not provided
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact
    if not fact or not isinstance(fact, str) or fact[0] != '(' or fact[-1] != ')':
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
    This heuristic estimates the minimum number of actions required to tighten
    all goal nuts. It considers the number of nuts to tighten, the number of
    spanners that need to be picked up, and a lower bound on the movement
    actions required to visit the locations of the nuts and necessary spanners.

    # Assumptions
    - The man can only carry one spanner at a time.
    - Spanners are consumed after one use (`tighten_nut` makes them not usable).
    - Nut locations are static (do not change during planning).
    - The man object name can be identified from the initial state.
    - Link facts define bidirectional connections between locations. (Note: This heuristic uses a simplified movement cost that doesn't require building a graph, but the assumption is relevant for plan existence).

    # Heuristic Initialization
    - Extracts the goal conditions.
    - Identifies the man object name from the initial state.
    - Extracts the static locations of all nuts from the initial state.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify Goal Nuts: Determine which nuts are specified in the goal
       and are not yet in the 'tightened' state. Let this set be `loose_goal_nuts`.
    2. Base Cost: The number of `tighten_nut` actions required is equal to
       the number of loose goal nuts (`num_nuts`). Add `num_nuts` to the heuristic.
    3. Find Current State Information:
       - Determine the man's current location (`man_loc`).
       - Check if the man is currently carrying a usable spanner (`carried_usable_spanner`).
       - Identify all usable spanners on the ground and their locations (`usable_spanners_on_ground_locs`).
    4. Spanner Acquisition Cost: The man needs a usable spanner for each nut.
       Calculate how many spanners need to be picked up from the ground. This is
       `max(0, num_nuts - (1 if carried_usable_spanner else 0))`. Add this number
       (`pickups_needed`) to the heuristic for the `pickup_spanner` actions.
       Also, check if enough usable spanners exist in total (carried + on ground)
       to tighten all required nuts. If not, the state is unsolvable, return infinity.
    5. Movement Cost: The man needs to visit the location of each loose goal nut
       and the location of each spanner he needs to pick up.
       - Identify the set of nut locations for loose goal nuts.
       - Identify the set of spanner locations for the spanners that need to be picked up
         (select the locations of the first `pickups_needed` usable spanners on the ground).
       - Combine these into a set of `all_required_locations`.
       - Calculate a lower bound on the number of `walk` actions needed to visit all
         locations in `all_required_locations` starting from `man_loc`. This lower bound
         is `max(0, len(all_required_locations) - (1 if man_loc in all_required_locations else 0))`.
       - Add this movement cost to the heuristic.
    6. Total Heuristic: The sum of the base cost (tighten), spanner acquisition cost (pickup),
       and movement cost (walks).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, man name,
        and static nut locations.
        """
        self.goals = task.goals  # Goal conditions.
        self.static = task.static # Static facts (e.g., links)

        # Identify the man object name.
        # Look for 'carrying' facts first, then 'at' facts excluding spanners/nuts, fallback to 'bob'.
        self.man_name = None
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'carrying' and len(parts) == 3:
                self.man_name = parts[1]
                break
        if self.man_name is None:
            for fact in task.initial_state:
                parts = get_parts(fact)
                if parts and parts[0] == 'at' and len(parts) == 3:
                    obj_name = parts[1]
                    # Simple check: if it's not a spanner or nut based on name pattern
                    if not (obj_name.startswith('spanner') or obj_name.startswith('nut')):
                        self.man_name = obj_name
                        break
        if self.man_name is None:
             # Fallback: Assume 'bob' based on examples if no other man found
             self.man_name = 'bob' # This might fail if man name is different

        # Extract nut locations (assumed static, from initial state)
        self.nut_locations = {}
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('nut'):
                self.nut_locations[parts[1]] = parts[2]

        # Optional: Extract link facts if a graph-based shortest path was needed
        # self.links = set()
        # for fact in task.static:
        #     parts = get_parts(fact)
        #     if parts and parts[0] == 'link' and len(parts) == 3:
        #         l1, l2 = parts[1], parts[2]
        #         self.links.add((l1, l2))
        #         self.links.add((l2, l1)) # Links are bidirectional


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify Goal Nuts: Find nuts that need tightening.
        loose_goal_nuts = {
            get_parts(goal)[1]
            for goal in self.goals
            if get_parts(goal) and get_parts(goal)[0] == 'tightened'
            and goal not in state
        }

        # If all goal nuts are tightened, the heuristic is 0.
        if not loose_goal_nuts:
            return 0

        # 2. Base Cost: Cost for tighten actions.
        num_nuts = len(loose_goal_nuts)
        h = num_nuts

        # 3. Find Current State Information:
        # Find man's current location
        man_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1] == self.man_name:
                man_loc = parts[2]
                break
        # Assert man location is found (should always be the case in a valid state)
        # If man_loc is None, something is wrong with the state representation or man identification.
        if man_loc is None:
             # This shouldn't happen in a valid planning state, but as a fallback
             # returning a large value indicates this state is likely problematic.
             return float('inf')


        # Find usable spanner carried by the man (man carries at most one)
        carried_usable_spanner = None
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'carrying' and len(parts) == 3 and parts[1] == self.man_name:
                spanner_name = parts[2]
                if "(usable " + spanner_name + ")" in state:
                    carried_usable_spanner = spanner_name
                break # Assume man carries at most one spanner

        # Find usable spanners on the ground and their locations
        usable_spanners_on_ground_locs = {} # {spanner_name: location}
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at' and len(parts) == 3 and parts[1].startswith('spanner'):
                spanner_name = parts[1]
                spanner_loc = parts[2]
                if "(usable " + spanner_name + ")" in state:
                    usable_spanners_on_ground_locs[spanner_name] = spanner_loc

        # 4. Spanner Acquisition Cost (Pickup actions):
        # Man needs num_nuts usable spanners in total.
        # He starts with 0 or 1 carried usable spanner.
        # The rest must be picked up from the ground.
        spanners_needed_from_ground = max(0, num_nuts - (1 if carried_usable_spanner else 0))

        # Check if enough usable spanners exist in total (carried + on ground)
        total_usable_spanners_available = (1 if carried_usable_spanner else 0) + len(usable_spanners_on_ground_locs)
        if num_nuts > total_usable_spanners_available:
             # Problem is unsolvable from this state regarding spanners
             return float('inf')

        pickups_needed = spanners_needed_from_ground
        h += pickups_needed # Cost of pickup actions

        # 5. Movement Cost (Walk actions):
        # The man needs to visit the locations of the nuts and the spanners he picks up.

        # Identify nut locations that need visiting
        nut_locations_to_visit = {self.nut_locations[n] for n in loose_goal_nuts if n in self.nut_locations}

        # Identify spanner locations that need visiting for pickup
        # We need to pick up 'pickups_needed' spanners. Which ones?
        # A simple heuristic can assume we pick up any 'pickups_needed' usable spanners on the ground.
        # Let's just take the locations of the first 'pickups_needed' usable spanners found.
        spanners_to_pickup_list = list(usable_spanners_on_ground_locs.keys())[:pickups_needed]
        spanner_pickup_locations = {usable_spanners_on_ground_locs[s] for s in spanners_to_pickup_list}

        # Combine all locations the man needs to visit
        all_required_locations = nut_locations_to_visit | spanner_pickup_locations

        # Calculate a lower bound on walks needed to visit all required locations
        # starting from man_loc. This is the number of required locations minus 1
        # if the man is already at one of them, otherwise it's the number of locations.
        num_required_locations = len(all_required_locations)
        if num_required_locations > 0:
            if man_loc in all_required_locations:
                h += max(0, num_required_locations - 1)
            else:
                h += num_required_locations

        # 6. Total Heuristic: Sum calculated costs.
        return h
