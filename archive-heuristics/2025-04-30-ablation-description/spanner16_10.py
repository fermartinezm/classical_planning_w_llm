from fnmatch import fnmatch
# Assume Heuristic base class is available as per example
# from heuristics.heuristic_base import Heuristic

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
    if len(parts) < len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Mock Heuristic base class for standalone testing if needed
# class Heuristic:
#     def __init__(self, task):
#         self.goals = task.goals
#         self.static = task.static
#     def __call__(self, node):
#         raise NotImplementedError

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions required to tighten
    all loose nuts specified in the goal. It sums three components: the number
    of tighten actions needed, an estimate for the walking required to reach
    nut locations, and an estimate for the spanner pickup actions needed.

    # Assumptions
    - The problem is solvable (i.e., there are enough usable spanners initially
      to tighten all loose nuts).
    - Any location is reachable from any other location (the heuristic does not
      compute actual path distances). A walk action is estimated to cost 1.
    - A spanner pickup action is estimated to cost 1, assuming a usable spanner
      is available somewhere.
    - Each 'tighten_nut' action consumes one usable spanner.
    - There is only one man object in the domain.

    # Heuristic Initialization
    - Identify the single man object.
    - Identify all spanner objects and nut objects by inspecting initial/static facts.
    - Store the static locations of all nuts.
    - Store the set of nuts that need to be tightened according to the goal state.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value `h(s)` for a state `s` is calculated as the sum of three components:

    1.  **Tightening Cost:** The number of loose nuts in state `s` that are required
        to be tightened in the goal state. Each such nut requires one `tighten_nut`
        action. This is a lower bound on the number of actions.
        `h1 = count(N | (loose N) in s and (tightened N) in goal)`

    2.  **Walking Cost:** An estimate of the effort required for the man to reach
        the locations of the loose goal nuts. For each loose goal nut `N` at location
        `L_N`, if the man is not currently at `L_N`, we add 1 to the heuristic.
        This overestimates the true walking cost if multiple loose nuts are at the
        same location not currently occupied by the man, but provides a simple
        gradient towards states where the man is co-located with loose nuts.
        `h2 = count(N | (loose N) in s and (tightened N) in goal and location(man) != location(N))`

    3.  **Spanner Acquisition Cost:** An estimate of the effort required for the man
        to acquire usable spanners. To tighten `K` nuts, the man needs `K` usable
        spanners throughout the plan. If the man is currently carrying `C` usable
        spanners, he needs to acquire at least `max(0, K - C)` additional usable
        spanners. Each acquisition (pickup) is estimated to cost 1 action.
        `h3 = max(0, count(N | (loose N) in s and (tightened N) in goal) - count(S | (carrying man S) in s and (usable S) in s))`

    The total heuristic value is `h(s) = h1 + h2 + h3`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts,
        and identifying domain objects.
        """
        self.goals = task.goals
        self.static = task.static

        all_facts = set(task.initial_state) | set(task.static)

        # Identify nuts based on loose/tightened predicates in initial/static facts
        self.all_nuts = {get_parts(f)[1] for f in all_facts if match(f, 'loose', '*') or match(f, 'tightened', '*')}

        # Get static nut locations from static facts
        self.nut_locations = {}
        for fact in self.static:
            if match(fact, 'at', '*', '*'):
                 obj, loc = get_parts(fact)[1:3]
                 if obj in self.all_nuts:
                     self.nut_locations[obj] = loc
        # Also check initial state for nut locations if not in static (shouldn't happen based on domain)
        for fact in task.initial_state:
             if match(fact, 'at', '*', '*'):
                 obj, loc = get_parts(fact)[1:3]
                 if obj in self.all_nuts and obj not in self.nut_locations:
                     self.nut_locations[obj] = loc


        # Identify spanners based on usable predicate in initial/static facts
        self.all_spanners = {get_parts(f)[1] for f in all_facts if match(f, 'usable', '*')}

        # Identify the man. Assume the single object of type man exists.
        # Try to find an object that is 'at' a location and is not a known spanner or nut.
        locatables_at_init = {get_parts(f)[1] for f in task.initial_state if match(f, 'at', '*', '*')}
        potential_men = locatables_at_init - self.all_spanners - self.all_nuts
        self.man = next(iter(potential_men), None)

        # If not found this way, try finding an object in a 'carrying' predicate
        if self.man is None:
             carried_by_man = {get_parts(f)[1] for f in all_facts if match(f, 'carrying', '*', '*')}
             self.man = next(iter(carried_by_man), None)

        # If still not found, this is an unusual initial state.
        # For typical spanner problems, the man is 'at' a location initially.
        # A robust parser would provide object types directly.
        # As a last resort, assume the first object in initial state that isn't a spanner/nut is the man.
        if self.man is None:
             all_init_objects = {p for f in task.initial_state for p in get_parts(f)[1:]} # Collect all objects mentioned
             self.man = next(iter(all_init_objects - self.all_spanners - self.all_nuts), None)


        # Identify goal nuts (nuts that must be tightened)
        self.goal_nuts = {get_parts(goal)[1] for goal in self.goals if match(goal, 'tightened', '*')}


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach the goal.
        """
        state = node.state  # Current world state (frozenset of facts)

        # 1. Identify loose nuts that are goal nuts
        loose_target_nuts = {N for N in self.goal_nuts if f'(loose {N})' in state}
        num_loose_goal_nuts = len(loose_target_nuts)

        # If all goal nuts are tightened, the heuristic is 0
        if num_loose_goal_nuts == 0:
            return 0

        # Initialize heuristic with the minimum number of tighten actions needed
        h = num_loose_goal_nuts

        # Find man's current location
        man_location = None
        for fact in state:
            if match(fact, 'at', self.man, '*'):
                man_location = get_parts(fact)[2]
                break
        # If man_location is None, the state is likely invalid or terminal (man removed?).
        # Assuming valid states where man is always located.

        # 2. Add cost related to man's location
        # Count loose goal nuts not at man's location
        num_loose_not_at_man_loc = sum(1 for N in loose_target_nuts if self.nut_locations.get(N) != man_location)
        h += num_loose_not_at_man_loc

        # 3. Add cost related to spanners
        # Count usable spanners the man is currently carrying
        num_carried_usable = sum(1 for S in self.all_spanners if f'(carrying {self.man} {S})' in state and f'(usable {S})' in state)

        # The man needs num_loose_goal_nuts usable spanners in total.
        # He starts with num_carried_usable.
        # He needs to acquire num_loose_goal_nuts - num_carried_usable more usable spanners.
        # Each acquisition (pickup) is estimated to cost 1 action.
        num_pickups_needed = max(0, num_loose_goal_nuts - num_carried_usable)
        h += num_pickups_needed

        return h
