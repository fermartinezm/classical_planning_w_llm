import re
from collections import deque

class sokobanHeuristic:
    """
    Domain-dependent heuristic for the Sokoban planning task.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the shortest path distances for each box from its current location
        to its goal location. The distances are computed on the grid defined
        by the 'adjacent' facts in the static information, respecting the
        defined connections. This heuristic is not admissible but aims to
        guide a greedy best-first search effectively by prioritizing states
        where boxes are closer to their goal positions.

    Assumptions:
        - The problem instance uses locations named 'loc_row_col' where row
          and col are integers.
        - The goal state specifies the target location for each box using
          '(at box_name loc_row_col)' facts.
        - The 'adjacent' facts in the static information define the grid
          connectivity and are sufficient to build a graph representing
          reachable locations.
        - The heuristic is used for greedy best-first search and does not
          need to be admissible.
        - For solvable problems, all box goal locations are reachable from
          their initial locations within the defined grid.

    Heuristic Initialization:
        1. Parse all 'adjacent' facts from the static information to build
           a graph representation of the grid. The graph nodes are
           (row, col) tuples derived from location names 'loc_row_col'.
           Edges connect adjacent locations as defined by the facts.
        2. Compute all-pairs shortest paths on this grid graph using Breadth-First
           Search (BFS) starting from every location node. Store these distances
           in a dictionary mapping (start_loc_tuple, end_loc_tuple) to distance.
           If a path does not exist between two locations, the distance is
           considered infinite (or effectively unreachable).
        3. Parse the goal facts to create a mapping from each box name to its
           target location tuple (row, col).

    Step-By-Step Thinking for Computing Heuristic:
        1. Given a state (frozenset of facts), check if it is the goal state
           using the task's goal_reached method. If yes, the heuristic value is 0.
        2. If not the goal state, parse the state to find the current location
           of each box. This is done by iterating through the state facts and
           identifying facts matching the pattern '(at box_name loc_row_col)'.
        3. Initialize the total heuristic value to 0.
        4. For each box found in the current state that is also specified in the goal:
            a. Get the box's current location string from the state.
            b. Convert the current location string to its (row, col) tuple
               representation using the pre-computed mapping.
            c. Look up the box's goal location tuple using the pre-computed
               box-to-goal mapping.
            d. If the current location tuple is the same as the goal location
               tuple for this box, the box is already at its goal; add 0 to
               the total heuristic.
            e. If the current location tuple is different from the goal location
               tuple, find the shortest distance between the current location
               tuple and the goal location tuple using the pre-computed
               all-pairs shortest path distances. If a finite distance exists,
               add this distance to the total heuristic. If no path exists
               (distance is effectively infinite), the state is likely a dead end
               or part of an unsolvable path. Return a very large value (infinity)
               to prune this branch in GBFS.
        5. Return the total heuristic value.
    """

    def __init__(self, task):
        self.task = task
        self.goal_box_locations = {}
        self.location_str_to_tuple = {}
        self.location_tuple_to_str = {}
        self.graph = {}
        self.distances = {} # Stores shortest path distances between location tuples

        # 1. Parse goal facts to map boxes to goal locations
        for goal_fact in self.task.goals:
            match = re.match(r"\(at (\w+) (loc_\d+_\d+)\)", goal_fact)
            if match:
                box_name = match.group(1)
                loc_str = match.group(2)
                loc_tuple = self._parse_location_string(loc_str)
                if loc_tuple: # Only store if location string was parsed successfully
                    self.goal_box_locations[box_name] = loc_tuple

        # 2. Build graph from adjacent facts and create location mappings
        all_locations = set()
        for fact in self.task.static:
            if fact.startswith('(adjacent'):
                # Example: '(adjacent loc_4_2 loc_4_3 right)'
                parts = fact.strip('()').split()
                # Ensure fact has enough parts before accessing indices 1 and 2
                if len(parts) >= 3:
                    loc1_str = parts[1]
                    loc2_str = parts[2]
                    all_locations.add(loc1_str)
                    all_locations.add(loc2_str)

        # Create mappings and initialize graph nodes for all found locations
        for loc_str in all_locations:
             loc_tuple = self._parse_location_string(loc_str)
             if loc_tuple: # Only add to mappings/graph if parsed successfully
                 self.location_str_to_tuple[loc_str] = loc_tuple
                 self.location_tuple_to_str[loc_tuple] = loc_str
                 self.graph[loc_tuple] = [] # Initialize adjacency list

        # Add edges to the graph
        for fact in self.task.static:
            if fact.startswith('(adjacent'):
                parts = fact.strip('()').split()
                 # Ensure fact has enough parts before accessing indices 1 and 2
                if len(parts) >= 3:
                    loc1_str = parts[1]
                    loc2_str = parts[2]
                    # Ensure locations were successfully parsed and added to graph nodes
                    if loc1_str in self.location_str_to_tuple and loc2_str in self.location_str_to_tuple:
                        loc1_tuple = self.location_str_to_tuple[loc1_str]
                        loc2_tuple = self.location_str_to_tuple[loc2_str]
                        self.graph[loc1_tuple].append(loc2_tuple)
                        # Assuming adjacency is symmetric
                        self.graph[loc2_tuple].append(loc1_tuple)

        # Remove duplicates from adjacency lists (optional but clean)
        for loc_tuple in self.graph:
            self.graph[loc_tuple] = list(set(self.graph[loc_tuple]))

        # 3. Compute all-pairs shortest paths using BFS
        # Only compute for locations that are part of the graph
        for start_node in list(self.graph.keys()): # Iterate over a copy
            self._bfs(start_node)

    def _parse_location_string(self, loc_str):
        """Converts 'loc_row_col' string to (row, col) tuple."""
        parts = loc_str.split('_')
        # Basic validation based on expected format
        if len(parts) == 3 and parts[0] == 'loc' and parts[1].isdigit() and parts[2].isdigit():
             return (int(parts[1]), int(parts[2]))
        # Return None if parsing fails
        return None


    def _bfs(self, start_node):
        """Computes shortest paths from start_node to all reachable nodes."""
        # Ensure start_node is valid and in the graph
        if start_node is None or start_node not in self.graph:
            return # Cannot run BFS from an invalid or non-existent node

        q = deque([(start_node, 0)])
        visited = {start_node}
        self.distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            # Ensure current_node is still valid and has neighbors in the graph
            if current_node not in self.graph:
                 continue # Should not happen if BFS starts from graph keys

            for neighbor in self.graph.get(current_node, []): # Use .get for safety
                # Ensure neighbor is valid
                if neighbor is not None and neighbor not in visited:
                    visited.add(neighbor)
                    self.distances[(start_node, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    def __call__(self, state):
        """
        Computes the heuristic value for the given state.

        @param state A frozenset of facts representing the current state.
        @return The estimated cost (non-negative integer or infinity) to reach a goal state.
        """
        # Check if goal is reached
        if self.task.goal_reached(state):
            return 0

        current_box_locations = {}
        # robot_location = None # Robot location is not used in this heuristic

        # Parse current state to find box locations
        for fact in state:
            if fact.startswith('(at '):
                match = re.match(r"\(at (\w+) (loc_\d+_\d+)\)", fact)
                if match:
                    box_name = match.group(1)
                    loc_str = match.group(2)
                    loc_tuple = self._parse_location_string(loc_str)
                    if loc_tuple: # Only store if location string was parsed successfully
                        current_box_locations[box_name] = loc_tuple
            # elif fact.startswith('(at-robot '):
            #      match = re.match(r"\(at-robot (loc_\d+_\d+)\)", fact)
            #      if match:
            #           robot_location = self._parse_location_string(match.group(1))


        total_distance = 0

        # Sum distances for each box to its goal
        for box_name, current_loc_tuple in current_box_locations.items():
            # Only consider boxes that have a specified goal location
            if box_name in self.goal_box_locations:
                goal_loc_tuple = self.goal_box_locations[box_name]

                # Ensure both current and goal locations were parsed successfully
                if current_loc_tuple is None or goal_loc_tuple is None:
                     # This indicates a parsing issue or invalid location in state/goal
                     # Treat as unreachable
                     return float('inf')

                if current_loc_tuple != goal_loc_tuple:
                    # Look up pre-computed distance
                    # The key is (start_tuple, end_tuple)
                    distance_key = (current_loc_tuple, goal_loc_tuple)
                    distance = self.distances.get(distance_key)

                    if distance is not None:
                        total_distance += distance
                    else:
                        # No path found between current box location and its goal location
                        # This state is likely a dead end or part of an unsolvable path.
                        # Return infinity to discourage the search from exploring this branch.
                        # print(f"Warning: No path found from {current_loc_tuple} to {goal_loc_tuple} for {box_name}")
                        return float('inf')

        # If there are boxes in the goal that are not in the current state,
        # this heuristic implicitly assumes they are not relevant or will appear
        # later. Standard Sokoban problems usually have all goal boxes present
        # in the initial state.

        # If there are goal locations with no corresponding box in the state,
        # this heuristic won't capture the cost of getting a box *to* that location
        # if the box isn't currently tracked. However, the goal facts specify
        # which *specific* box goes to which location. So we only care about
        # the boxes that are currently in the state and have a goal location.

        return total_distance
