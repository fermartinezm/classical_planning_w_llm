import math
import sys

from heuristics.heuristic_base import Heuristic
from task import Task


class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        This heuristic estimates the cost to reach a goal state by summing
        the number of untightened goal nuts (representing the tightening
        actions needed) and the minimum cost to get the man, equipped with
        a usable spanner, to any location where an untightened goal nut is
        located. It precomputes shortest path distances between all locations
        in the domain graph.

    Assumptions:
        - There is exactly one man object. The heuristic attempts to identify
          the man object name from the initial state facts (involved in
          'carrying' or 'at' facts, not being a known nut or spanner).
          If identification fails, it defaults to the name 'bob'.
        - Nut locations are static (do not change during planning).
        - Spanners are consumed when used for tightening (become unusable).
        - Only the man can carry spanners.
        - The location graph defined by 'link' facts is static.
        - The problem is solvable only if the total number of usable spanners
          in the initial state is at least the total number of goal nuts.

    Heuristic Initialization:
        The constructor performs the following steps:
        1. Identifies all goal nuts from the task's goal state.
        2. Identifies all spanners and the man object name from the initial state facts.
        3. Extracts all unique location names from initial state and static facts.
        4. Stores the static locations of goal nuts.
        5. Builds an adjacency matrix for the location graph based on 'link' facts.
        6. Computes all-pairs shortest path distances between locations using the Floyd-Warshall algorithm.
        7. Counts the total number of usable spanners in the initial state.
        8. Determines if the problem is fundamentally unsolvable based on the count of goal nuts and initial usable spanners.

    Step-By-Step Thinking for Computing Heuristic:
        For a given state:
        1. Identify the man's current location.
        2. Identify the set of untightened goal nuts in the current state.
        3. If there are no untightened nuts, the state is a goal state, return 0.
        4. Check if the problem was determined to be unsolvable during initialization. If so, return infinity.
        5. Identify the set of usable spanners currently in the state (at locations or carried).
        6. Count the number of untightened nuts (K) and currently available usable spanners (U_current).
        7. If K is greater than U_current, the state is a dead end (not enough usable spanners remain in reachable/carried locations), return infinity.
        8. Calculate the base heuristic value as K (representing the K tightening actions needed).
        9. Determine if the man is currently carrying a usable spanner.
        10. Identify the locations of usable spanners that are currently at locations.
        11. Calculate the minimum cost to get the man, equipped with a usable spanner, to *any* location containing an untightened goal nut. This involves considering two cases:
            a. Using the spanner the man is currently carrying (if any and usable): Cost is the shortest distance from the man's current location to the nut location.
            b. Picking up a usable spanner from a location: Cost is the shortest distance from the man's current location to a spanner location, plus 1 (for the pickup action), plus the shortest distance from that spanner location to the nut location. This minimum is taken over all available usable spanners at locations and all untightened nut locations.
        12. Add this minimum cost (from step 11) to the base heuristic value (from step 8).
        13. Return the total heuristic value.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self.initial_state = task.initial_state

        # 1. Identify all goal nuts
        self.goal_nuts = set()
        for goal_string in self.goals:
             goal_fact = self.parse_fact(goal_string)
             if goal_fact[0] == 'tightened':
                 self.goal_nuts.add(goal_fact[1])

        # 2. Identify all spanners and the man object name
        self.all_spanner_names = set()
        self.all_nut_names = set(self.goal_nuts)
        self.man_name = None
        locatable_objects_in_init = set()

        for fact_string in self.initial_state:
             fact = self.parse_fact(fact_string)
             if fact[0] == 'usable':
                 self.all_spanner_names.add(fact[1])
             elif fact[0] == 'carrying':
                 self.man_name = fact[1] # Found man name via carrying
                 self.all_spanner_names.add(fact[2])
             elif fact[0] == 'at':
                 obj = fact[1]
                 locatable_objects_in_init.add(obj)

        # If man name not found via carrying, it must be the locatable object that is not a spanner or nut
        if self.man_name is None:
            potential_men = locatable_objects_in_init - self.all_spanner_names - self.all_nut_names
            if len(potential_men) == 1:
                self.man_name = potential_men.pop()
            else:
                 # Fallback: Assume 'bob' based on examples.
                 # This is fragile but necessary if man cannot be uniquely identified from initial state facts.
                 # print("Warning: Could not definitively identify man object. Assuming 'bob'.", file=sys.stderr)
                 self.man_name = 'bob'


        # 3. Extract all unique location names
        self.locations = set()
        # 4. Stores the static locations of goal nuts
        self.goal_nuts_locs = {} # nut_name -> location_name

        for fact_string in self.initial_state | self.static:
            fact = self.parse_fact(fact_string)
            if fact[0] == 'at':
                obj = fact[1]
                loc = fact[2]
                self.locations.add(loc)
                if obj in self.goal_nuts:
                    self.goal_nuts_locs[obj] = loc
            elif fact[0] == 'link':
                l1 = fact[1]
                l2 = fact[2]
                self.locations.add(l1)
                self.locations.add(l2)

        # 5. Build adjacency matrix for the location graph
        self.loc_list = sorted(list(self.locations)) # Consistent index mapping
        self.loc_to_idx = {loc: i for i, loc in enumerate(self.loc_list)}
        n = len(self.loc_list)
        self.dist = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            self.dist[i][i] = 0

        for fact_string in self.static:
            fact = self.parse_fact(fact_string)
            if fact[0] == 'link':
                l1, l2 = fact[1], fact[2]
                if l1 in self.loc_to_idx and l2 in self.loc_to_idx: # Ensure locations are known
                    i, j = self.loc_to_idx[l1], self.loc_to_idx[l2]
                    self.dist[i][j] = 1
                    self.dist[j][i] = 1


        # 6. Computes all-pairs shortest path distances using Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.dist[i][k] != math.inf and self.dist[k][j] != math.inf:
                         self.dist[i][j] = min(self.dist[i][j], self.dist[i][k] + self.dist[k][j])

        # 7. Counts the total number of usable spanners in the initial state.
        self.total_initial_usable_spanners = 0
        for fact_string in self.initial_state:
             fact = self.parse_fact(fact_string)
             if fact[0] == 'usable':
                  self.total_initial_usable_spanners += 1

        # 8. Determines if the problem is fundamentally unsolvable
        self.total_goal_nuts = len(self.goal_nuts)
        self.unsolvable = self.total_initial_usable_spanners < self.total_goal_nuts


    @staticmethod
    def parse_fact(fact_string):
        """Helper function to parse a fact string into a tuple."""
        # Remove surrounding brackets and split by spaces
        parts = fact_string[1:-1].split()
        return tuple(parts)

    def get_distance(self, loc1_name, loc2_name):
        """Lookup shortest distance between two locations."""
        if loc1_name not in self.loc_to_idx or loc2_name not in self.loc_to_idx:
            # This indicates an issue if locations from state/goals/init/static are not in self.locations
            # print(f"Warning: Distance requested for unknown location: {loc1_name} or {loc2_name}", file=sys.stderr)
            return math.inf
        i = self.loc_to_idx[loc1_name]
        j = self.loc_to_idx[loc2_name]
        return self.dist[i][j]


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.
        """
        state = node.state

        # 4. Check if the problem was determined to be unsolvable
        if self.unsolvable:
             return math.inf

        # 1. Identify the man's current location.
        man_loc = None
        for fact_string in state:
             fact = self.parse_fact(fact_string)
             if fact[0] == 'at' and fact[1] == self.man_name:
                  man_loc = fact[2]
                  break

        if man_loc is None:
             # Man is not at any location? Should not happen in valid states.
             # Return infinity as a safe bet.
             # print(f"Warning: Man location not found in state: {state}", file=sys.stderr)
             return math.inf


        # 2. Identify the set of untightened goal nuts in the current state.
        tightened_nuts_in_state = {self.parse_fact(f)[1] for f in state if self.parse_fact(f)[0] == 'tightened'}
        untightened_nuts = set(self.goal_nuts) - tightened_nuts_in_state

        # 3. If there are no untightened nuts, the state is a goal state, return 0.
        K = len(untightened_nuts)
        if K == 0:
            return 0

        # 5. Identify the set of usable spanners currently in the state (at locations or carried).
        usable_spanners_in_state = {self.parse_fact(f)[1] for f in state if self.parse_fact(f)[0] == 'usable'}

        # 9. Determine if the man is currently carrying a usable spanner.
        man_carrying_usable_spanner = False
        for fact_string in state:
             fact = self.parse_fact(fact_string)
             if fact[0] == 'carrying' and fact[1] == self.man_name and fact[2] in usable_spanners_in_state:
                  man_carrying_usable_spanner = True
                  break # Found one usable spanner being carried

        # 10. Identify the locations of usable spanners that are currently at locations.
        current_usable_spanner_locs = set()
        for fact_string in state:
             fact = self.parse_fact(fact_string)
             if fact[0] == 'at' and fact[1] in usable_spanners_in_state:
                  current_usable_spanner_locs.add(fact[2])

        # 6. Count currently available usable spanners (at locations + carried).
        usable_spanners_currently_available = len(current_usable_spanner_locs) + (1 if man_carrying_usable_spanner else 0)

        # 7. If K is greater than U_current, the state is a dead end
        if K > usable_spanners_currently_available:
             return math.inf


        # 8. Calculate the base heuristic value as K (tightening actions).
        h = K

        # 11. Calculate the minimum cost to get the man, equipped with a usable spanner,
        #     to any location containing an untightened goal nut.
        untightened_nut_locs = {self.goal_nuts_locs[nut] for nut in untightened_nuts}

        min_cost_reach_nut_with_spanner = math.inf

        for nut_loc in untightened_nut_locs:
             cost_for_this_nut_loc = math.inf

             # Option 1: Use carried spanner (if any and usable)
             if man_carrying_usable_spanner:
                  dist_to_nut = self.get_distance(man_loc, nut_loc)
                  if dist_to_nut != math.inf:
                       # Cost is just travel
                       cost_for_this_nut_loc = min(cost_for_this_nut_loc, dist_to_nut)

             # Option 2: Pick up a spanner from a location
             min_cost_pickup_and_travel = math.inf
             if current_usable_spanner_locs: # Only consider if there are usable spanners at locations
                 for spanner_loc in current_usable_spanner_locs:
                      dist_man_to_spanner = self.get_distance(man_loc, spanner_loc)
                      dist_spanner_to_nut = self.get_distance(spanner_loc, nut_loc)
                      if dist_man_to_spanner != math.inf and dist_spanner_to_nut != math.inf:
                           cost_pickup_and_travel = dist_man_to_spanner + 1 + dist_spanner_to_nut
                           min_cost_pickup_and_travel = min(min_cost_pickup_and_travel, cost_pickup_and_travel)

             cost_for_this_nut_loc = min(cost_for_this_nut_loc, min_cost_pickup_and_travel)

             min_cost_reach_nut_with_spanner = min(min_cost_reach_nut_with_spanner, cost_for_this_nut_loc)

        # If it's impossible to reach any nut location with a spanner from the current state
        if min_cost_reach_nut_with_spanner == math.inf:
             return math.inf # This state is a dead end

        # 12. Add this minimum cost to the base heuristic value.
        h += min_cost_reach_nut_with_spanner

        # 13. Return the total heuristic value.
        return h
