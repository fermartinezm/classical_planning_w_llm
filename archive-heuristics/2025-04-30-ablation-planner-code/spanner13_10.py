# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

# Assume Heuristic base class is available and provides the structure __init__(self, task) and __call__(self, node)
# from heuristics.heuristic_base import Heuristic

from collections import deque
import math # For math.inf


class spannerHeuristic: # Assuming this class will be used within a framework that provides 'task' and 'node' objects
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It does this by simulating a greedy
    process where the man iteratively chooses the next loose nut that is
    cheapest to reach and tighten, considering the need to acquire a usable
    spanner if not already carrying one. The cost includes walking, picking
    up a spanner, and tightening a nut.

    # Assumptions
    - Nuts are static (do not change location).
    - Spanners become unusable after one tighten action.
    - The man can only carry one spanner at a time.
    - All actions have a cost of 1.
    - The graph of locations connected by 'link' predicates is undirected.

    # Heuristic Initialization
    - Parses the task definition to identify the man, all spanners, all nuts,
      and all locations.
    - Extracts static 'link' facts to build a graph of locations.
    - Computes all-pairs shortest paths between locations using BFS.
    - Stores the initial locations of all nuts (which are static).
    - Stores the goal facts to identify which nuts need tightening.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the man's current location from the state.
    2.  Identify the set of loose nuts that are also present in the goal
        conditions. These are the nuts that need tightening.
    3.  Identify the set of usable spanners currently in the state (either
        on the ground or carried by the man).
    4.  Check if the number of nuts to tighten exceeds the number of usable
        spanners available. If so, the goal is unreachable, return a large value (infinity).
    5.  If no nuts need tightening, the goal is reached, return 0.
    6.  Initialize the total heuristic cost to 0.
    7.  Create working sets of remaining nuts to tighten and remaining usable
        spanners.
    8.  Determine if the man is currently carrying a usable spanner and get its name if so. This spanner is available for the first nut task only. If he has one, remove it from the general pool of available spanners as it will be consumed by the first nut.
    9.  Enter a loop that continues as long as there are nuts remaining to tighten:
        a.  Add 1 to the heuristic cost (for the 'tighten_nut' action for the next nut).
        b.  Find the minimum cost to reach *any* of the remaining nuts and prepare to tighten it,
            considering the need to acquire a spanner if necessary. This cost includes walks and a potential pickup.
        c.  Iterate through each nut remaining in the `nuts_remaining` set:
            i.  Get the location of the current nut.
            ii. Calculate the cost to get the man to this nut's location with a spanner.
                -   If the man is currently carrying a usable spanner (only possible for the first nut task if he started with one): The cost is just the walk distance from the man's current location to the nut's location. The spanner is assumed to be used up after this tightening.
                -   If the man is *not* carrying a usable spanner: He must acquire one. Find the closest *available* usable spanner (from the `spanners_remaining` set) that is on the ground. Calculate the cost to walk from the man's current location to this spanner, pick it up (cost 1), and then walk from the spanner's location to the nut's location. The total cost for spanner acquisition and travel is the sum of these walk distances plus the pickup cost.
            iii. The total *preparation* cost for this specific nut in this iteration is the calculated travel/acquisition cost (excluding the tighten action cost already added in 9a).
            iv. Keep track of the minimum total *preparation* cost found across all remaining nuts and the corresponding best nut and the spanner used.
        d.  If no nut could be selected (e.g., no usable spanners left to pick up when needed, or unreachable), return a large value (infinity).
        e.  Add the minimum *preparation* cost found (`min_cost_to_prepare_next_nut`) to the total heuristic cost.
        f.  Update the state for the next iteration:
            -   Remove the selected best nut from `nuts_remaining`.
            -   Remove the spanner used (either the one carried initially or the one picked up) from `spanners_remaining`.
            -   Update the man's current location to the location of the nut that was just tightened.
            -   Set the flag indicating the man is carrying a usable spanner to False, as the spanner was used.
    10. Return the total accumulated heuristic cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and
        precomputing shortest paths.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        objects = task.objects # Objects defined in the problem

        # Extract object names by type
        self.man_name = None
        self.all_spanner_names = []
        self.all_nut_names = []
        self.all_locations = []

        for obj_str in objects:
            name, type_ = obj_str.split(' - ')
            if type_ == 'man':
                self.man_name = name
            elif type_ == 'spanner':
                self.all_spanner_names.append(name)
            elif type_ == 'nut':
                self.all_nut_names.append(name)
            elif type_ == 'location':
                self.all_locations.append(name)

        # Build the location graph from 'link' facts
        self.adj = {loc: [] for loc in self.all_locations}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                self.adj[loc1].append(loc2)
                self.adj[loc2].append(loc1) # Links are assumed undirected

        # Compute all-pairs shortest paths using BFS
        self.dist = {loc: {other_loc: math.inf for other_loc in self.all_locations} for loc in self.all_locations}
        for start_loc in self.all_locations:
            self.dist[start_loc][start_loc] = 0
            queue = deque([start_loc])
            while queue:
                u = queue.popleft()
                for v in self.adj.get(u, []): # Use .get for safety
                    if self.dist[start_loc][v] == math.inf:
                        self.dist[start_loc][v] = self.dist[start_loc][u] + 1
                        queue.append(v)

        # Store initial locations of nuts (they are static)
        self.nut_locations = {}
        # Assuming task object has initial_state which contains initial facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1] in self.all_nut_names:
                 self.nut_locations[parts[1]] = parts[2]

        # Store goal facts for easy lookup
        self.goal_facts = set(self.goals)


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of fact strings).

        # 1. Find man's current location
        current_man_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man_name:
                current_man_loc = parts[2]
                break
        if current_man_loc is None:
             # Man's location must always be defined by an 'at' predicate
             return math.inf # Should not happen in valid states

        # 2. Identify nuts to tighten (loose and in goal)
        nuts_to_tighten = {
            n_name for n_name in self.all_nut_names
            if f'(loose {n_name})' in state and f'(tightened {n_name})' in self.goal_facts
        }

        # 3. Identify usable spanners available (on ground or carried)
        usable_spanners_in_state = {
            s_name for s_name in self.all_spanner_names
            if f'(usable {s_name})' in state
        }

        # 4. Check if man is carrying a usable spanner initially
        man_starts_carrying_usable_spanner_name = next(
            (s_name for s_name in self.all_spanner_names
             if f'(carrying {self.man_name} {s_name})' in state and f'(usable {s_name})' in state),
            None
        )
        # This flag tracks if the man has a spanner *for the current nut task*.
        # It's True only for the first nut task if he started with one.
        man_has_spanner_for_current_task = man_starts_carrying_usable_spanner_name is not None


        # 5. Count nuts and spanners
        N_nuts = len(nuts_to_tighten)
        N_spanners_available = len(usable_spanners_in_state)

        # 6. If not enough spanners, goal is unreachable
        if N_nuts > N_spanners_available:
            return math.inf # Infinity

        # If no nuts need tightening, goal is reached
        if N_nuts == 0:
            return 0

        # 7. Greedy simulation
        h = 0
        nuts_remaining = set(nuts_to_tighten)
        spanners_remaining = set(usable_spanners_in_state) # Track available usable spanners by name

        # If man starts carrying a usable spanner, it's available for the first nut task,
        # but we remove it from the general pool of 'spanners_remaining'
        # because it will be consumed by the first nut it's used on.
        if man_has_spanner_for_current_task:
             spanners_remaining.discard(man_starts_carrying_usable_spanner_name)


        while nuts_remaining:
            # Add cost for the tighten action for the next nut
            h += 1

            min_cost_to_prepare_next_nut = math.inf
            best_next_nut = None
            spanner_used_for_next_nut = None # The specific spanner name used

            # Find the best next nut to tighten greedily
            for nut_name in nuts_remaining:
                L_N = self.nut_locations[nut_name]

                if man_has_spanner_for_current_task:
                    # This branch is only taken for the very first nut task if man started with a spanner
                    cost_to_reach_nut = self.dist[current_man_loc][L_N]
                    cost_for_spanner_acquisition = 0
                    spanner_to_use = man_starts_carrying_usable_spanner_name

                    current_nut_preparation_cost = cost_to_reach_nut + cost_for_spanner_acquisition

                    if current_nut_preparation_cost < min_cost_to_prepare_next_nut:
                        min_cost_to_prepare_next_nut = current_nut_preparation_cost
                        best_next_nut = nut_name
                        spanner_used_for_next_nut = spanner_to_use

                else: # Man needs to pick up a spanner (either didn't start with one, or used the first one)
                    min_spanner_pickup_travel_cost = math.inf
                    best_spanner_to_pickup = None

                    # Find the closest usable spanner on the ground among those remaining
                    usable_on_ground_remaining = {
                        s_name for s_name in spanners_remaining
                        if any(f'(at {s_name} {loc})' in state for loc in self.all_locations) # Is on the ground
                    }

                    for spanner_name in usable_on_ground_remaining:
                        # Find location of this spanner in the current state
                        L_S = next((loc for loc in self.all_locations if f'(at {spanner_name} {loc})' in state), None)
                        if L_S is None: continue # Should not happen if it's on the ground

                        # Check if distance is finite
                        if self.dist[current_man_loc][L_S] == math.inf or self.dist[L_S][L_N] == math.inf:
                            continue # Cannot reach this spanner or cannot reach nut from spanner

                        cost_to_spanner = self.dist[current_man_loc][L_S]
                        cost_from_spanner_to_nut = self.dist[L_S][L_N]
                        total_spanner_travel_cost = cost_to_spanner + 1 + cost_from_spanner_to_nut # walk + pickup + walk

                        if total_spanner_travel_cost < min_spanner_pickup_travel_cost:
                            min_spanner_pickup_travel_cost = total_spanner_travel_cost
                            best_spanner_to_pickup = spanner_name

                    if best_spanner_to_pickup is not None:
                         current_nut_preparation_cost = min_spanner_pickup_travel_cost
                         if current_nut_preparation_cost < min_cost_to_prepare_next_nut:
                            min_cost_to_prepare_next_nut = current_nut_preparation_cost
                            best_next_nut = nut_name
                            spanner_used_for_next_nut = best_spanner_to_pickup

            # If no nut could be selected (e.g., no spanners left to pick up or unreachable), goal is unreachable
            if best_next_nut is None:
                 return math.inf # Infinity

            # Add the cost to prepare for the selected nut (walks + pickup if needed)
            h += min_cost_to_prepare_next_nut

            # Update state for the next iteration
            nuts_remaining.remove(best_next_nut)
            spanners_remaining.discard(spanner_used_for_next_nut) # This spanner is used up
            current_man_loc = self.nut_locations[best_next_nut] # Man is now at the nut location
            man_has_spanner_for_current_task = False # Man used the spanner for this task, needs a new one for the next

        return h
