import heapq
import logging
from collections import deque, defaultdict

from heuristics.heuristic_base import Heuristic
from task import Operator, Task

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """Parses a PDDL fact string into a tuple (predicate, arg1, arg2, ...)."""
    # Remove surrounding brackets and split by space
    parts = fact_string.strip('()').split()
    return tuple(parts)

# BFS implementation to find shortest paths
def bfs(graph, start_node):
    """
    Performs BFS from start_node on the given graph.
    Returns a dictionary mapping reachable nodes to their distance from start_node.
    Graph is represented as an adjacency list (dict: node -> set of neighbors).
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    # Ensure start_node is in the graph keys, even if it has no neighbors
    if start_node not in graph:
        graph[start_node] = set()

    while queue:
        current_node = queue.popleft()

        # Ensure current_node is in graph keys before iterating neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class roversHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Rovers domain.

    Summary:
    The heuristic estimates the cost to reach the goal state by summing up
    estimated costs for each unachieved goal fact. For a goal requiring
    communication of data (soil, rock, or image), the heuristic checks if the
    required data has already been collected by any rover.
    - If the data is collected, the cost is estimated as 1 (communicate action)
      plus the minimum navigation cost for a rover holding the data to reach
      any waypoint visible from the lander.
    - If the data is not collected, the cost is estimated as 1 (collection action)
      plus 1 (communicate action) plus the minimum total navigation cost
      for a suitable rover to go from its current location to the collection location
      (sample waypoint or image waypoint) and then navigate from there to any
      waypoint visible from the lander.
    Navigation costs are precomputed shortest path distances for each rover
    on its traversable graph. The heuristic ignores store constraints (empty/full)
    and camera calibration state for simplicity and efficiency.

    Assumptions:
    - The heuristic assumes that if a rover is equipped for a task (soil, rock, imaging),
      it can perform that task whenever at the correct location, ignoring store
      availability or camera calibration state.
    - The heuristic assumes that navigation is the primary cost component besides
      the final action (sample/image/communicate). Each navigation step counts as 1 action.
    - The heuristic assumes that if a goal requires data from a specific waypoint
      (soil/rock sample) or objective (image), that sample exists initially or
      that objective is visible from at least one waypoint, respectively.
      Unreachable goals (due to static constraints) are assigned infinite cost.
    - The heuristic sums costs for individual unachieved goals, which might
      overestimate in cases where actions contribute to multiple goals or
      shared navigation paths exist. This is acceptable for a non-admissible
      greedy heuristic focused on minimizing expanded nodes.

    Heuristic Initialization:
    The constructor pre-processes the static facts from the task description
    to build efficient data structures:
    - Sets of rovers equipped for soil, rock, and imaging.
    - Dictionaries mapping stores to rovers, cameras to rovers, cameras to supported modes,
      and cameras to calibration targets.
    - Sets of waypoints with initial soil and rock samples.
    - Dictionary mapping objectives to sets of waypoints from which they are visible.
    - Dictionary mapping landers to their locations.
    - Set of waypoints visible from any lander location (communication points).
    - For each rover, an adjacency list representing its traversable graph
      (based on `can_traverse` facts).
    - For each rover, a dictionary of shortest path distances between all pairs
      of waypoints reachable by that rover, computed using BFS.

    Step-By-Step Thinking for Computing Heuristic:
    1. Get the current state facts and the goal facts.
    2. Parse the current state to extract dynamic information: rover locations,
       collected soil/rock/image data, communicated data.
    3. Identify the set of unachieved goal facts.
    4. Initialize the total heuristic value `h` to 0.
    5. Iterate through each unachieved goal fact:
        a. If the goal is `(communicated_soil_data w)`:
            i. Check if `(have_soil_analysis r w)` exists in the current state for any rover `r`.
            ii. If yes: Calculate the cost to communicate. This is 1 (communicate action)
                plus the minimum navigation cost for any rover holding this data
                from its current location to any communication waypoint. Add this cost to `h`.
            iii. If no: Calculate the cost to collect and communicate. This is 1 (sample action)
                 plus 1 (communicate action) plus the minimum total navigation cost
                 for a soil-equipped rover to go from its current location to waypoint `w`
                 and then from `w` to any communication waypoint. If `w` had no initial
                 soil sample or no rover is soil-equipped, the cost is infinity. Add this cost to `h`.
        b. If the goal is `(communicated_rock_data w)`:
            i. Similar logic as for soil data, using rock-specific predicates and equipment.
        c. If the goal is `(communicated_image_data o m)`:
            i. Check if `(have_image r o m)` exists in the current state for any rover `r`.
            ii. If yes: Calculate the cost to communicate. This is 1 (communicate action)
                plus the minimum navigation cost for any rover holding this data
                from its current location to any communication waypoint. Add this cost to `h`.
            iii. If no: Calculate the cost to collect and communicate. This is 1 (image action)
                 plus 1 (communicate action) plus the minimum total navigation cost
                 for an imaging-equipped rover (with a camera supporting mode `m` on board)
                 to go from its current location to any waypoint visible from objective `o`
                 and then from that waypoint to any communication waypoint. If objective `o`
                 is not visible from any waypoint or no rover is capable of taking this image,
                 the cost is infinity. Add this cost to `h`.
    6. If at any point a cost calculation results in infinity, the total heuristic is infinity.
    7. If the total heuristic `h` is 0 and there are unachieved goals, return 1 (should not happen
       with this cost logic, but as a safeguard). Otherwise, return `h`.
    """

    def __init__(self, task):
        super().__init__()
        self.goals = task.goals
        self.static = task.static
        self._parse_static_info()
        self._precompute_distances(task.facts) # Pass all facts to get all waypoints

    def _parse_static_info(self):
        """Parses static facts into useful data structures."""
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        self.store_of = {} # store -> rover
        self.on_board = {} # camera -> rover
        self.supports = defaultdict(set) # camera -> set of modes
        self.calibration_target = {} # camera -> objective
        self.visible = defaultdict(set) # waypoint -> set of visible waypoints
        self.can_traverse = defaultdict(lambda: defaultdict(set)) # rover -> waypoint -> set of traversable waypoints
        self.visible_from = defaultdict(set) # objective -> set of waypoints
        self.at_soil_sample_initial = set() # waypoint
        self.at_rock_sample_initial = set() # waypoint
        self.lander_at = {} # lander -> waypoint
        self.communication_waypoints = set() # waypoint visible from any lander

        for fact_string in self.static:
            fact = parse_fact(fact_string)
            predicate = fact[0]
            args = fact[1:]

            if predicate == 'equipped_for_soil_analysis' and len(args) == 1:
                self.equipped_for_soil.add(args[0])
            elif predicate == 'equipped_for_rock_analysis' and len(args) == 1:
                self.equipped_for_rock.add(args[0])
            elif predicate == 'equipped_for_imaging' and len(args) == 1:
                self.equipped_for_imaging.add(args[0])
            elif predicate == 'store_of' and len(args) == 2:
                self.store_of[args[0]] = args[1]
            elif predicate == 'on_board' and len(args) == 2:
                self.on_board[args[0]] = args[1]
            elif predicate == 'supports' and len(args) == 2:
                self.supports[args[0]].add(args[1])
            elif predicate == 'calibration_target' and len(args) == 2:
                self.calibration_target[args[0]] = args[1]
            elif predicate == 'visible' and len(args) == 2:
                self.visible[args[0]].add(args[1])
            elif predicate == 'can_traverse' and len(args) == 3:
                 # can_traverse ?r - rover ?x - waypoint ?y - waypoint
                self.can_traverse[args[0]][args[1]].add(args[2])
            elif predicate == 'visible_from' and len(args) == 2:
                self.visible_from[args[0]].add(args[1])
            elif predicate == 'at_soil_sample' and len(args) == 1:
                self.at_soil_sample_initial.add(args[0])
            elif predicate == 'at_rock_sample' and len(args) == 1:
                self.at_rock_sample_initial.add(args[0])
            elif predicate == 'at_lander' and len(args) == 2:
                self.lander_at[args[0]] = args[1]

        # Determine communication waypoints (visible from any lander)
        lander_locations = set(self.lander_at.values())
        for lander_loc in lander_locations:
             # A rover at waypoint X can communicate if (visible X lander_loc)
             # We need to find all X such that (visible X lander_loc) is true.
             # The 'visible' static facts are stored as waypoint -> set of visible waypoints.
             # We need the reverse mapping or iterate through all visible facts.
             # Let's iterate through all visible facts.
             for wp1, visible_wps in self.visible.items():
                 if lander_loc in visible_wps:
                     self.communication_waypoints.add(wp1)
             # Also add the lander location itself if it's a waypoint (it is)
             # and visible from itself (not explicitly stated, but common assumption)
             # The PDDL requires (visible ?x ?y) where ?y is lander location.
             # So, any waypoint ?x visible from lander location ?y is a comm waypoint.
             # The above loop covers this.

        # If no lander or no visible waypoints from lander, communication is impossible.
        if not self.communication_waypoints and self.lander_at:
             logging.warning("No communication waypoints found visible from lander locations.")


    def _precompute_distances(self, all_facts):
        """Precomputes shortest path distances for each rover."""
        self.rover_distances = {} # rover -> start_wp -> end_wp -> distance

        # Collect all waypoints from task.facts (initial state, goals, static)
        waypoints = set()
        for fact_string in all_facts:
             fact = parse_fact(fact_string)
             predicate = fact[0]
             # Waypoints appear as arguments in many predicates. Extract based on PDDL definition.
             if predicate in {'at', 'at_lander', 'have_soil_analysis', 'have_rock_analysis', 'visible_from'}:
                 if len(fact) > 2: waypoints.add(fact[2])
             elif predicate in {'can_traverse', 'visible'}:
                 if len(fact) > 1: waypoints.add(fact[1])
                 if len(fact) > 2: waypoints.add(fact[2])
             elif predicate in {'communicated_soil_data', 'communicated_rock_data', 'at_soil_sample', 'at_rock_sample'}:
                 if len(fact) > 1: waypoints.add(fact[1])
             elif predicate == 'calibrate':
                 if len(fact) > 4: waypoints.add(fact[4])
             elif predicate == 'take_image':
                 if len(fact) > 2: waypoints.add(fact[2])
             elif predicate in {'communicate_soil_data', 'communicate_rock_data'}:
                 if len(fact) > 3: waypoints.add(fact[3]) # p
                 if len(fact) > 4: waypoints.add(fact[4]) # x
                 if len(fact) > 5: waypoints.add(fact[5]) # y
             elif predicate == 'communicate_image_data':
                 if len(fact) > 6: waypoints.add(fact[6]) # x
                 if len(fact) > 7: waypoints.add(fact[7]) # y

        # Add waypoints from lander_at and communication_waypoints derived from static facts
        waypoints.update(self.lander_at.values())
        waypoints.update(self.communication_waypoints)

        # Ensure all waypoints from can_traverse static facts are included, even if they don't appear elsewhere
        for rover_graph in self.can_traverse.values():
            for wp, neighbors in rover_graph.items():
                waypoints.add(wp)
                waypoints.update(neighbors)

        # Get all rovers involved in the domain
        rovers = set(self.can_traverse.keys()) | self.equipped_for_soil | self.equipped_for_rock | self.equipped_for_imaging
        rovers |= set(self.store_of.values())
        rovers |= set(self.on_board.values())

        # Build distances for each rover
        for rover in rovers:
            self.rover_distances[rover] = {}
            rover_graph = self.can_traverse.get(rover, {}) # Get the specific rover's graph

            # Build a complete graph representation including all waypoints as nodes
            complete_rover_graph = {wp: set() for wp in waypoints}
            for wp, neighbors in rover_graph.items():
                 complete_rover_graph[wp].update(neighbors)

            # BFS from each waypoint to all others for this rover
            for start_wp in waypoints:
                 self.rover_distances[rover][start_wp] = bfs(complete_rover_graph, start_wp)

        # Handle rovers that have no can_traverse facts (cannot move)
        for rover in rovers:
             if rover not in self.rover_distances:
                  self.rover_distances[rover] = {wp: {wp: 0} for wp in waypoints} # Can only reach itself

    def _get_rover_location(self, state, rover):
        """Finds the current location of a rover in the state."""
        for fact_string in state:
            fact = parse_fact(fact_string)
            if fact[0] == 'at' and len(fact) == 3 and fact[1] == rover:
                return fact[2]
        return None # Rover location not found (should not happen in valid states)

    def _min_nav_cost(self, rover, start_wp, end_wp):
        """Gets the precomputed minimum navigation cost for a rover."""
        if start_wp is None or end_wp is None:
             return float('inf') # Cannot navigate from/to unknown location
        if start_wp == end_wp:
            return 0
        # Check if rover exists and start_wp exists in precomputed distances
        if rover in self.rover_distances and start_wp in self.rover_distances[rover]:
            # Check if end_wp is reachable from start_wp
            return self.rover_distances[rover][start_wp].get(end_wp, float('inf'))
        return float('inf') # Rover or start_wp not in precomputed data, or no path

    def _min_nav_cost_to_comm(self, rover, start_wp):
        """Finds min nav cost for a rover from start_wp to any communication waypoint."""
        min_cost = float('inf')
        if not self.communication_waypoints:
             return float('inf') # No communication points exist

        for comm_wp in self.communication_waypoints:
            cost = self._min_nav_cost(rover, start_wp, comm_wp)
            min_cost = min(min_cost, cost)
        return min_cost

    def __call__(self, node):
        """Computes the domain-dependent heuristic value for a state."""
        state = node.state
        h = 0
        unachieved_goals = self.goals - state

        # If goal is already reached, heuristic is 0
        if not unachieved_goals:
            return 0

        # Parse dynamic state information
        rover_at = {}
        have_soil = set() # (rover, waypoint)
        have_rock = set() # (rover, waypoint)
        have_image = set() # (rover, objective, mode)
        # communicated facts are goals, already in unachieved_goals check

        for fact_string in state:
            fact = parse_fact(fact_string)
            predicate = fact[0]
            args = fact[1:]

            if predicate == 'at' and len(args) == 2:
                rover_at[args[0]] = args[1]
            elif predicate == 'have_soil_analysis' and len(args) == 2:
                have_soil.add(tuple(args))
            elif predicate == 'have_rock_analysis' and len(args) == 2:
                have_rock.add(tuple(args))
            elif predicate == 'have_image' and len(args) == 3:
                have_image.add(tuple(args))

        # Track which data collection/communication tasks are needed
        needed_soil_samples = set() # waypoint
        needed_rock_samples = set() # waypoint
        needed_images = set() # (objective, mode)

        # Calculate cost for each unachieved communication goal
        for goal_fact_string in unachieved_goals:
            goal_fact = parse_fact(goal_fact_string)
            predicate = goal_fact[0]
            args = goal_fact[1:]

            if predicate == 'communicated_soil_data' and len(args) == 1:
                w = args[0]
                # Check if data is already collected by any rover
                data_collected = any((r, w) in have_soil for r in rover_at) # Check all rovers currently in state
                if data_collected:
                    # Cost to communicate: 1 (action) + min nav cost to comm point
                    min_comm_nav = float('inf')
                    for r, loc in rover_at.items():
                         if (r, w) in have_soil: # Rover has the data
                             cost = self._min_nav_cost_to_comm(r, loc)
                             min_comm_nav = min(min_comm_nav, cost)

                    if min_comm_nav == float('inf'):
                         return float('inf') # Cannot communicate
                    h += 1 + min_comm_nav
                else:
                    # Data not collected, need to sample and communicate
                    needed_soil_samples.add(w)

            elif predicate == 'communicated_rock_data' and len(args) == 1:
                w = args[0]
                # Check if data is already collected by any rover
                data_collected = any((r, w) in have_rock for r in rover_at)
                if data_collected:
                    # Cost to communicate: 1 (action) + min nav cost to comm point
                    min_comm_nav = float('inf')
                    for r, loc in rover_at.items():
                         if (r, w) in have_rock: # Rover has the data
                             cost = self._min_nav_cost_to_comm(r, loc)
                             min_comm_nav = min(min_comm_nav, cost)

                    if min_comm_nav == float('inf'):
                         return float('inf') # Cannot communicate
                    h += 1 + min_comm_nav
                else:
                    # Data not collected, need to sample and communicate
                    needed_rock_samples.add(w)

            elif predicate == 'communicated_image_data' and len(args) == 2:
                o, m = args
                # Check if data is already collected by any rover
                data_collected = any((r, o, m) in have_image for r in rover_at)
                if data_collected:
                    # Cost to communicate: 1 (action) + min nav cost to comm point
                    min_comm_nav = float('inf')
                    for r, loc in rover_at.items():
                         if (r, o, m) in have_image: # Rover has the data
                             cost = self._min_nav_cost_to_comm(r, loc)
                             min_comm_nav = min(min_comm_nav, cost)

                    if min_comm_nav == float('inf'):
                         return float('inf') # Cannot communicate
                    h += 1 + min_comm_nav
                else:
                    # Data not collected, need to image and communicate
                    needed_images.add((o, m))

            # Assuming goals are only communicated data facts.

        # Calculate cost for needed data collection + communication
        # Summing minimum costs for each needed task.

        for w in needed_soil_samples:
            # Cost to sample at w and communicate from w
            if w not in self.at_soil_sample_initial:
                 return float('inf') # Cannot sample if no sample exists initially

            min_total_cost = float('inf')
            suitable_rovers = [r for r in rover_at if r in self.equipped_for_soil]

            if not suitable_rovers:
                 return float('inf') # No rover can sample soil

            for r in suitable_rovers:
                current_loc = rover_at[r]
                nav_to_sample = self._min_nav_cost(r, current_loc, w)
                if nav_to_sample == float('inf'):
                    continue # This rover cannot reach the sample location

                nav_from_sample_to_comm = self._min_nav_cost_to_comm(r, w)
                if nav_from_sample_to_comm == float('inf'):
                    continue # Cannot communicate from the sample location

                # Cost = 1 (sample) + nav_to_sample + 1 (communicate) + nav_from_sample_to_comm
                total_cost = 1 + nav_to_sample + 1 + nav_from_sample_to_comm
                min_total_cost = min(min_total_cost, total_cost)

            if min_total_cost == float('inf'):
                 return float('inf') # No suitable rover can perform the task sequence

            h += min_total_cost

        for w in needed_rock_samples:
            # Cost to sample at w and communicate from w
            if w not in self.at_rock_sample_initial:
                 return float('inf') # Cannot sample if no sample exists initially

            min_total_cost = float('inf')
            suitable_rovers = [r for r in rover_at if r in self.equipped_for_rock]

            if not suitable_rovers:
                 return float('inf') # No rover can sample rock

            for r in suitable_rovers:
                current_loc = rover_at[r]
                nav_to_sample = self._min_nav_cost(r, current_loc, w)
                if nav_to_sample == float('inf'):
                    continue # This rover cannot reach the sample location

                nav_from_sample_to_comm = self._min_nav_cost_to_comm(r, w)
                if nav_from_sample_to_comm == float('inf'):
                    continue # Cannot communicate from the sample location

                # Cost = 1 (sample) + nav_to_sample + 1 (communicate) + nav_from_sample_to_comm
                total_cost = 1 + nav_to_sample + 1 + nav_from_sample_to_comm
                min_total_cost = min(min_total_cost, total_cost)

            if min_total_cost == float('inf'):
                 return float('inf') # No suitable rover can perform the task sequence

            h += min_total_cost

        for o, m in needed_images:
            # Cost to image objective o in mode m and communicate from image location
            image_waypoints = self.visible_from.get(o, set())
            if not image_waypoints:
                 return float('inf') # Cannot image if objective not visible from anywhere

            min_total_cost = float('inf')
            # Find rovers capable of taking this image (equipped for imaging, has camera supporting mode m)
            suitable_rovers = []
            for r in rover_at:
                 if r in self.equipped_for_imaging:
                     # Check if rover has a camera that supports this mode
                     has_suitable_camera = False
                     for cam, cam_rover in self.on_board.items():
                         if cam_rover == r and m in self.supports.get(cam, set()):
                             has_suitable_camera = True
                             break
                     if has_suitable_camera:
                         suitable_rovers.append(r)

            if not suitable_rovers:
                 return float('inf') # No rover can take this image

            for r in suitable_rovers:
                current_loc = rover_at[r]
                min_nav_to_image_wp = float('inf')
                best_image_wp = None

                # Find the best waypoint to take the image from for this rover
                for p in image_waypoints:
                    nav_cost = self._min_nav_cost(r, current_loc, p)
                    if nav_cost < min_nav_to_image_wp:
                        min_nav_to_image_wp = nav_cost
                        best_image_wp = p

                if min_nav_to_image_wp == float('inf'):
                    continue # This rover cannot reach any image waypoint

                # Cost from the chosen image waypoint to a communication point
                nav_from_image_to_comm = self._min_nav_cost_to_comm(r, best_image_wp)
                if nav_from_image_to_comm == float('inf'):
                    continue # Cannot communicate from the image location

                # Cost = 1 (image) + nav_to_image_wp + 1 (communicate) + nav_from_image_to_comm
                total_cost = 1 + min_nav_to_image_wp + 1 + nav_from_image_to_comm
                min_total_cost = min(min_total_cost, total_cost)

            if min_total_cost == float('inf'):
                 return float('inf') # No suitable rover can perform the task sequence

            h += min_total_cost

        # If h is 0, it means all unachieved goals had a calculated cost of 0.
        # This should only happen if unachieved_goals is empty.
        # If for some reason unachieved_goals is not empty but h is 0,
        # it implies a calculation error or a state where tasks are free.
        # The current logic ensures h > 0 if unachieved_goals is not empty
        # unless a cost was exactly 0 (only nav_cost can be 0, but actions add 1).
        # So, h > 0 if unachieved_goals is not empty.
        # The check `if not unachieved_goals: return 0` at the start handles the goal state.

        return h
