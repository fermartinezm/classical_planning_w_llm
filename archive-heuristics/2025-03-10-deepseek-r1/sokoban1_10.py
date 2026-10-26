from fnmatch import fnmatch
from collections import defaultdict, deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    return fact[1:-1].split()

def match(fact, *args):
    parts = get_parts(fact)
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))

class sokoban1Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to solve a Sokoban problem by considering the minimal pushes needed for each box to reach its goal and the player's distance to the nearest box.

    # Assumptions
    - Each box requires a number of pushes equal to the shortest path from its current position to its goal, considering the static maze layout.
    - The player needs to reach a position adjacent to a box to push it, contributing their shortest path distance to the nearest such position.
    - Adjacency is bidirectional, allowing the player to move freely between connected cells.

    # Heuristic Initialization
    - Extract static adjacency information to build a graph of reachable locations.
    - Precompute shortest paths between all pairs of locations using BFS.
    - Extract goal locations for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1. **Extract Current State Information**:
       - Identify the current location of the player and each box.
    2. **Box Distance Calculation**:
       - For each box not at its goal, compute the shortest path distance from its current location to its goal location (number of pushes needed).
       - Sum these distances to get the total estimated pushes.
    3. **Player Distance Calculation**:
       - For each non-goal box, find all adjacent cells (potential push positions).
       - Compute the shortest path from the player's current location to each of these adjacent cells.
       - Track the minimal player distance to any adjacent cell of any non-goal box.
    4. **Combine Values**:
       - The heuristic value is the sum of total pushes and the minimal player distance.
    """

    def __init__(self, task):
        """Initialize the heuristic with static data and precompute shortest paths."""
        self.box_goals = {}
        # Extract goal locations for each box
        for goal in task.goals:
            if match(goal, 'at', '*', '*'):
                parts = get_parts(goal)
                box = parts[1]
                loc = parts[2]
                self.box_goals[box] = loc

        # Build adjacency graph from static facts
        self.adjacency = defaultdict(set)
        locations = set()
        for fact in task.static:
            if match(fact, 'adjacent', '*', '*', '*'):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.adjacency[l1].add(l2)
                self.adjacency[l2].add(l1)  # Treat as undirected
                locations.add(l1)
                locations.add(l2)

        # Precompute shortest paths between all pairs using BFS
        self.shortest_path = defaultdict(dict)
        for loc in locations:
            # BFS from loc to all reachable locations
            visited = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            # Set infinity for unreachable (though Sokoban should have all connected)
            for node in locations:
                self.shortest_path[loc][node] = visited.get(node, float('inf'))

    def __call__(self, node):
        state = node.state
        current_boxes = {}
        player_loc = None

        # Extract current box positions and player location
        for fact in state:
            parts = get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'at':
                if len(parts) == 3:  # Box position
                    box = parts[1]
                    loc = parts[2]
                    current_boxes[box] = loc
                elif len(parts) == 2 and parts[0] == 'at-robot':  # Player position
                    player_loc = parts[1]

        sum_pushes = 0
        min_player_distance = float('inf')

        for box, current_loc in current_boxes.items():
            goal_loc = self.box_goals.get(box)
            if not goal_loc:
                continue  # Box has no goal (unlikely in Sokoban)
            if current_loc == goal_loc:
                continue  # Box is already at goal

            # Calculate box's push distance
            box_distance = self.shortest_path[current_loc].get(goal_loc, float('inf'))
            if box_distance == float('inf'):
                return float('inf')  # Unreachable goal
            sum_pushes += box_distance

            # Find minimal player distance to this box's adjacent cells
            adjacent_locs = self.adjacency.get(current_loc, [])
            for adj_loc in adjacent_locs:
                distance = self.shortest_path[player_loc].get(adj_loc, float('inf'))
                if distance < min_player_distance:
                    min_player_distance = distance

        # Handle case where all boxes are at goals
        if sum_pushes == 0:
            return 0

        # If no non-goal boxes but sum_pushes is 0, return 0
        if min_player_distance == float('inf'):
            min_player_distance = 0

        return sum_pushes + min_player_distance
