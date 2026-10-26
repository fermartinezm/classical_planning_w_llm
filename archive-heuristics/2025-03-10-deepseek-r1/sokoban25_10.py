<think>
Okay, I need to create a domain-dependent heuristic for the Sokoban domain. Let's start by understanding the Sokoban problem. The main goal is to push boxes to their target locations. The heuristic should estimate the number of actions needed to reach the goal, and it doesn't have to be admissible since it's for greedy best-first search.

First, I'll look at the PDDL domain file. The key predicates are at-robot (robot's location), at (box locations), adjacent (connections between locations), and clear (whether a location is empty). The actions are move and push. The robot can move to adjacent clear locations or push a box from an adjacent location into a next cell if that's clear.

The heuristic needs to consider both moving the robot and pushing the boxes. For each box, the distance from its current position to its goal is important. But also, the robot needs to reach the box to push it. So, perhaps the heuristic should sum the Manhattan distances (or some path distance) from each box to its goal, plus the robot's distance to the nearest box. But Sokoban is more complex because pushing a box requires the robot to be in the correct position relative to the box and the target direction.

Wait, but the example heuristics for other domains, like Gripper and Logistics, used domain-specific logic. For Sokoban, a common heuristic is the sum of the minimal pushes required for each box plus the minimal moves for the robot to get into position. However, since the heuristic doesn't need to be admissible, maybe we can use a more informed estimate.

Let me think. For each box, compute the shortest path from its current position to its goal, considering only the walls (static adjacency) and ignoring other boxes. Then sum these distances. But that's the "ignore other boxes" heuristic, which is often used but can be misleading. However, since the heuristic is for GBFS, which prioritizes lower heuristic values, we need something that guides the search effectively.

Alternatively, for each box, compute the Manhattan distance to its goal, and add the robot's distance to the box's current position. But the robot might need to move around to push the box. However, this might underestimate or overestimate depending on the scenario.

Another approach is to calculate the minimal number of moves required for the robot to reach each box and push it towards its goal. But this could be computationally expensive if done for each state.

Wait, the example state given has the robot at loc_4_7 and a box at loc_3_5. The goal is to get the box to loc_2_4. So the robot needs to move to the box, then push it towards the goal. The path for the box is from loc_3_5 to loc_3_4 (if pushed left), then up to loc_2_4. So two pushes. The robot's path to the box might be from loc_4_7 to loc_4_6, then loc_3_6, then loc_3_5 (but the box is there, so maybe via another route). Alternatively, maybe the robot can approach from the other side.

But how to model this in a heuristic? Maybe for each box, compute the shortest path from its current position to its goal, considering the directions it can be pushed, and then add the robot's distance to the position from which it can start pushing the box along that path.

Alternatively, use the sum of the distances each box is from its goal (using a suitable metric, like Manhattan or actual path distance if precomputed) plus the robot's distance to the nearest box that is not yet at its goal.

But how to compute the robot's distance? The robot can move through adjacent cells as long as they are clear. So the robot's path is the shortest path to a position adjacent to the box, considering the current clear cells.

So the heuristic could be:

For each box not in goal:
   distance of box to goal (as pushes)
   + robot's distance to the box's position (to push it)

But since pushing a box requires the robot to be adjacent to it in the direction opposite to the push, the robot's distance should be to the cell from which it can push the box towards the goal.

Alternatively, for each box, the minimal robot moves needed to get behind the box (in the direction needed to push it towards the goal) plus the number of pushes needed for the box.

But this might be too complex. Let's think of a simpler approach.

An often-used Sokoban heuristic is the sum of the Manhattan distances from each box to its goal, plus the minimal robot distance to any box not yet in place. But this might not account for the need to have the robot in the correct position to push.

Alternatively, for each box, compute the minimal number of pushes required (assuming other boxes are not in the way) and add the robot's distance to the starting position of the first push.

But how to compute the minimal pushes? The minimal path for the box to the goal, considering only walls. For each box, the minimal number of steps it needs to be pushed. For example, if the box is at (3,5) and the goal is at (2,4), the pushes would be left then up. Each push is one action, but the robot has to move to the correct position each time.

Alternatively, for each box, compute the shortest path from its current position to the goal, where each step is a push in a valid direction, considering the static adjacency (so that walls are accounted for). The length of this path is the minimal number of pushes. Then, for each such push, the robot needs to be in the adjacent cell opposite to the push direction. So the robot's path to the first push position is part of the heuristic.

But this requires precomputing the shortest paths for all possible box positions to their goals, which might be feasible but could be time-consuming. Alternatively, during heuristic calculation, for each box, perform a BFS to find the minimal pushes required, considering the static adjacency.

Wait, but the static adjacency is part of the problem's structure. So during the heuristic's initialization, we can precompute for each location the adjacent locations. Then, for any box position, we can compute the minimal number of steps (pushes) needed to reach the goal, considering that each push moves the box one cell in a direction where there's a path.

But how to handle this in code? Let's think. For each box, the heuristic can compute the minimal path from its current position to the goal, using BFS where each move is a push. Each push requires that the cell the box is moving into is adjacent and that the cell behind the box (from the push direction) is accessible for the robot. Wait, but this might be complicated.

Alternatively, for each box, compute the Manhattan distance to the goal, multiplied by a factor (like 2, since each push requires at least a move and a push, but that's just a rough estimate). But Manhattan might not account for walls.

Alternatively, precompute for each location the minimal number of pushes to each goal. But that's not feasible for large grids.

Another idea: use the sum of the minimal robot moves required to push each box to its goal. But how?

Alternatively, for each box, the distance from the box to its goal is the minimal number of pushes, and the robot's distance to the position needed to start pushing. For example, if the box is at B and needs to be pushed up to G, the robot needs to be south of B. So the robot's distance to that cell is part of the heuristic.

So the heuristic for a box would be (pushes needed) * 2 (assuming each push requires a move and a push action) plus the robot's distance to the initial push position.

But how to compute the pushes needed? Maybe the minimal path from the box's current position to the goal, considering the static adjacents, and each step is a possible push direction.

Alternatively, use the A* algorithm for each box's path, considering only static walls, and sum those distances. Then add the robot's current distance to the nearest box's starting push position.

But this might be computationally expensive for each state.

Wait, but the user said the heuristic should be efficiently computable. So maybe a simpler approach is needed.

Let's look at the example heuristics provided. The Gripper heuristic considered the number of balls and the trips needed. The Logistics heuristic considered the steps needed to move packages via different transport methods.

For Sokoban, perhaps the heuristic can be the sum of the Manhattan distances between each box and its goal, plus the Manhattan distance between the robot and the nearest box. But Manhattan might not account for walls. However, since the heuristic doesn't need to be admissible, maybe using Manhattan is acceptable even if it's not perfect.

Alternatively, compute the actual shortest path for the robot to each box and for each box to its goal, using BFS. But doing BFS for each box and the robot in every state could be too slow.

Wait, but the static adjacency is known. So perhaps precompute for each location, the shortest path distances to all other locations. Then, during heuristic calculation, for each box, get the precomputed distance from its current position to the goal, and for the robot, get the precomputed distance to the box's current position.

But how to precompute all pairs shortest paths? That's feasible for small grids but could be memory-intensive for large ones. However, given that Sokoban levels can vary in size, perhaps this approach isn't scalable. Alternatively, during the heuristic's initialization, build a graph structure based on the static adjacency facts, and then for any two locations, compute the shortest path on the fly using BFS.

Wait, but the static adjacency is fixed for each problem instance. So during the heuristic's __init__ method, we can build a graph where each location is a node, and edges are based on the adjacent facts. Then, for any two locations, we can compute the shortest path between them using BFS. But doing BFS for every pair during heuristic calculation might be too slow. So perhaps precompute a distance matrix during initialization.

But if the instance has many locations (like the second example with a 13x13 grid), precomputing all pairs would be O(n^2) which is 169^2=28561 entries, which is manageable. Wait, 13x13 is 169 locations. So 169^2 is about 28k entries. For each problem instance, precomputing all pairs' shortest paths using BFS for each node would take O(n*(n+m)) time, where m is the number of edges. For a grid, m is around 4n (each cell connected to up to four neighbors). So 169 * (169 + 4*169) = 169*5*169 = 142,805 steps. That's feasible for a heuristic.

So the plan is:

1. Precompute for each location, the shortest path to all other locations using BFS, considering the static adjacency. This gives the minimal number of moves between any two locations, ignoring dynamic obstacles (like other boxes and the robot's position). Since the heuristic doesn't need to be admissible, it's okay to ignore dynamic obstacles, even though this could lead to underestimating or overestimating. But since the heuristic is for GBFS, which just needs to guide the search, it's acceptable.

2. For each box, compute the minimal distance from its current position to its goal location using the precomputed distances. Sum these distances. This gives the minimal number of pushes required if there are no other boxes in the way.

3. For the robot, compute the minimal distance from its current position to each box's current position (since the robot needs to reach the box to push it). Take the minimum of these distances and add it to the sum of box distances.

4. The total heuristic value is the sum of all box distances plus the robot's minimal distance to any box.

But wait, each push requires the robot to be adjacent to the box. So for each box, the robot needs to be at a position adjacent to the box in the direction opposite to the push. But since the heuristic is supposed to be an estimate, maybe it's okay to approximate this as the robot's distance to the box's current position (since once there, it can push the box in some direction). However, the actual path for pushing may require the robot to be in a specific position relative to the box and the goal.

Alternatively, the robot's distance to the box's position is a lower bound on the steps needed to reach the box, and each box's distance to the goal is the minimal pushes needed. So the heuristic would be sum(box_dists) + min_robot_to_box.

But how to handle boxes that are already at their goals? For those, we don't add their distance. So the heuristic only considers boxes not yet in their goal positions.

Another consideration: when a box is at its goal, it's part of the static state and can't be moved again. So the heuristic should ignore such boxes.

Putting this together:

In the heuristic's __init__ method:

- Extract all static adjacency facts to build a graph structure.
- Precompute all-pairs shortest paths for locations using BFS.

In the __call__ method:

- For each box, check if it's already at its goal. If not, add the precomputed distance from its current location to the goal.
- For the robot, compute the minimal distance from its current location to each box's current location (for boxes not at their goals). If there are no such boxes, the robot's distance is 0.
- Sum the box distances and the robot's minimal distance.

This would give a heuristic that estimates the total steps as the sum of pushes needed for all boxes plus the steps the robot needs to reach the first box.

But this might not account for the fact that after pushing one box, the robot needs to go to another. However, since the heuristic is for GBFS and doesn't need to be perfect, this could still be a useful estimate.

Let's test this with the example instance 1. The initial state has the robot at loc_6_4 and box1 at loc_4_4. The goal is to move box1 to loc_2_4.

The precomputed distance from loc_4_4 to loc_2_4 is 2 (up two times). The robot's distance to loc_4_4 is from loc_6_4 to loc_4_4. The path is loc_6_4 → loc_5_4 → loc_4_4. So distance is 2. So the heuristic would be 2 (box distance) + 2 (robot distance) = 4. The actual optimal steps might be: robot moves to loc_4_4 (but wait, loc_4_4 is occupied by the box. So the robot can't move there. Oh right, the robot can only move to clear locations. So in the initial state, the robot is at loc_6_4. To reach the box at loc_4_4, the robot needs to move to loc_5_4, then to loc_4_4 (but that's where the box is). Wait, but the box is at loc_4_4, so the robot can't move there. The robot needs to move to a position adjacent to the box. So to push the box up, the robot needs to be at loc_5_4 (south of the box). Wait, the adjacent directions are up, down, left, right. So to push the box up, the robot must be south of the box (since pushing up requires the robot to be south of the box and the box moves north).

Wait, the adjacency facts in the first example show that loc_4_4 is adjacent to loc_3_4 (up) and loc_5_4 (down). So to push the box up from loc_4_4 to loc_3_4, the robot must be at loc_5_4 (south of the box). Wait, no. Wait, the 'push' action parameters are ?rloc (robot's location), ?bloc (box's location), ?floc (where the box is pushed to). The preconditions are that the robot is at ?rloc, the box is at ?bloc, ?floc is clear, and adjacent ?rloc ?bloc ?dir, adjacent ?bloc ?floc ?dir. So the robot is adjacent to the box in the same direction as the box is adjacent to the floc. So to push the box north (up), the robot must be south of the box. So for the box at loc_4_4, to push it up to loc_3_4, the robot must be at loc_5_4 (south of loc_4_4). Then the push action would move the robot to loc_4_4 (the box's old position), and the box to loc_3_4.

So in the initial state, the robot is at loc_6_4. To push the box at loc_4_4 up, the robot needs to be at loc_5_4. The path from loc_6_4 to loc_5_4 is one step (down direction). Then the robot can push the box up to loc_3_4. Then, the robot is at loc_4_4. Then, to push the box again up to loc_2_4, the robot needs to be south of loc_3_4 (i.e., loc_4_4). So the robot is already there. Push again. So total steps: move down (1), push (1), push (1). Total 3 actions. But the heuristic would compute the box's distance as 2 (from loc_4_4 to loc_2_4), and the robot's distance to the box's current position (loc_4_4) is 2 (from loc_6_4 to loc_4_4 via loc_5_4 and loc_4_4). Wait, but the robot can't move into the box's position. So the precomputed shortest path from loc_6_4 to loc_4_4 would be 2 steps (through loc_5_4). But in reality, the robot can't go to loc_4_4 because the box is there. So the precomputed adjacency-based shortest paths ignore dynamic obstacles (like boxes and the robot's position), which is correct because the static adjacency is fixed. However, the heuristic is allowed to be optimistic here, as it's not required to be admissible.

In this case, the heuristic would compute 2 (box distance) + 2 (robot distance) =4, but the actual minimal steps are 3. So the heuristic overestimates here. But since the heuristic is for GBFS, which prefers lower heuristic values, this could still work. However, in other cases, it might not be accurate. But since the user allows non-admissible heuristics, this is acceptable.

But perhaps the robot's distance should be to the position where it can push the box towards the goal. For example, for the box at loc_4_4, the robot needs to be at loc_5_4 (south) to push it up. So the robot's distance to loc_5_4 is 1 step (from loc_6_4 to loc_5_4). Then the box's distance is 2 steps (two pushes up). So the heuristic would be 2 +1=3, which matches the actual steps. But how to compute this in the heuristic?

This suggests that the heuristic should calculate, for each box, the minimal robot distance to a position from which it can push the box towards the goal. But how to determine that position?

This seems complex. Maybe a better approach is to compute for each box the minimal robot distance to any adjacent cell of the box, then add the box's distance to the goal. Since the robot can be in any adjacent cell to push the box in a direction, but the direction depends on the box's path to the goal.

Alternatively, the heuristic could consider the robot's distance to the box's current position, and the box's distance to the goal. Since pushing the box requires the robot to be adjacent, the minimal robot distance to the box's position is (current robot position to any adjacent cell of the box) + 0 (if already adjacent). But the precomputed distances are based on the static adjacency, so the distance from robot's current location to any adjacent cell of the box would be the minimal steps required to reach that cell.

But how to find the adjacent cells of the box in the current state? For each box, we can find all adjacent cells (from the static adjacency) and compute the minimal distance from the robot's current position to any of those cells. Then, for each box, sum that minimal distance plus the box's distance to the goal. Then take the minimal such sum (for the closest box) and sum with the other boxes' distances.

But this could be computationally intensive for multiple boxes. However, for each box, we need to compute the minimal robot distance to any of its adjacent cells (since the robot can push from any adjacent cell in any direction). Then, the total heuristic for that box is (robot distance to adjacent cell) + (box's distance to goal). Then, the overall heuristic is the sum over all boxes of (box's distance to goal) plus the minimal (robot distance to adjacent cell of box) for each box. Wait, but that's adding robot distances for each box, which would overcount.

Alternatively, the heuristic is the sum of all box distances plus the minimal robot distance to any box's adjacent cell. But that may not account for the robot needing to visit multiple boxes.

Hmm. Let's think of it this way: The robot needs to reach a position to push each box. For each box, the robot's distance to that box's position (or its adjacent cells) is part of the cost. But since the robot can't be in multiple places at once, the heuristic can't just sum all these distances. So perhaps the minimal robot distance to any box's adjacent cell plus the sum of the box distances.

Alternatively, for the heuristic, the robot's contribution is the minimal distance to reach any box that is not yet at its goal. This way, the heuristic assumes that once the robot reaches that box, it can start pushing it towards the goal, and the other boxes are handled afterward. This is a greedy approach and might not be accurate, but it's efficient.

So the heuristic would be:

sum(box_dists) + min_robot_dist_to_any_box_adjacent_cell

But how to compute min_robot_dist_to_any_box_adjacent_cell?

For each box not at its goal:

   get all adjacent cells to the box's current position (from static adjacency)
   compute the minimal distance from the robot's current position to any of these cells
   take the minimal such distance across all boxes

Then, the heuristic is sum(box_dists) + min_robot_dist.

This would account for the robot needing to reach a position adjacent to a box to push it. Let's test this with the first example.

Box is at loc_4_4. Adjacent cells are loc_3_4 (up), loc_5_4 (down), loc_4_3 (left), loc_4_5 (right). The robot is at loc_6_4. The minimal distance from loc_6_4 to any adjacent cell of loc_4_4:

distance to loc_5_4 (down of loc_4_4) is 2 steps (loc_6_4 → loc_5_4). Because loc_5_4 is adjacent to loc_4_4. So the robot's distance to loc_5_4 is 1 (from loc_6_4 to loc_5_4 is one step down). Then, the box's distance to goal (loc_2_4) is 2 pushes (up, up). So sum is 2 + 1 = 3, which matches the optimal steps.

So in this case, the heuristic would correctly return 3.

Another example: if the robot is already adjacent to the box, then the minimal distance is 0 (if adjacent) or 1 (if one step away). Wait, the robot's distance to the adjacent cell would be 0 if it's already there. So in the initial example, the robot is at loc_6_4. The adjacent cells of the box (loc_4_4) are loc_5_4, loc_3_4, loc_4_3, loc_4_5. The robot's distance to loc_5_4 is 1 step (since loc_6_4 → loc_5_4 is one move down). So the minimal robot distance is 1. The box's distance is 2. So total heuristic is 3.

Yes, that's correct.

So the steps for the heuristic are:

1. Precompute all-pairs shortest paths between locations using static adjacency.

2. For each box not at its goal:

   a. Compute the shortest path from its current location to the goal location (number of pushes needed, assuming no other boxes are in the way). This is the box's distance.

   b. Find all adjacent cells to the box's current location (possible positions the robot needs to be in to push the box).

   c. Compute the minimal distance from the robot's current location to any of these adjacent cells. This is the robot's contribution for this box.

3. Sum all box distances.

4. Find the minimal robot contribution across all boxes (since the robot can start with the closest box).

5. The heuristic value is the sum of all box distances plus the minimal robot contribution.

Wait, but if there are multiple boxes, the robot may need to push each one, so the sum of all box distances plus the sum of robot distances for each box. But that would overcount, as the robot can't be in multiple places at once. So instead, the heuristic should take the sum of box distances plus the sum of the minimal robot distances to each box's adjacent cells. But that's not correct, as the robot can push one box, then another, and the path for the robot between boxes is not accounted for.

Alternatively, the heuristic could be the sum of box distances plus the robot's distance to the nearest box's adjacent cell. This assumes that the robot will handle the nearest box first, then the others, but this might not account for the total steps needed. However, since the heuristic is for GBFS, which prioritizes nodes with lower heuristic values, this could still be effective.

Alternatively, for multiple boxes, the heuristic could be the sum of box distances plus the sum of the robot's minimal distances to each box's adjacent cells. But this would overcount the robot's movement. For example, if the robot has to move from box A to box B, the distance between A and B's adjacent cells should be added. But this is complex.

Given the complexity, perhaps the best approach for an efficient heuristic is to sum the box distances and add the robot's minimal distance to any box's adjacent cell. This would guide the search towards states where boxes are closer to their goals and the robot is close to a box.

So the steps in code would be:

In the constructor:

- Extract static adjacency facts to build a graph of locations.
- Precompute all-pairs shortest paths using BFS for each location.

In the __call__ method:

- For each box, check if it's at its goal. If not:

   a. Get its current location.

   b. Compute the shortest path from current location to goal (pushes needed).

   c. Find all adjacent cells (from static adjacency) to the box's current location.

   d. Compute the minimal distance from the robot's current location to any of these adjacent cells.

   e. Sum the box's distance and the robot's minimal distance to this box's adjacent cells.

   f. Add this to the total heuristic.

But wait, that would sum for each box the box's distance plus the robot's distance to it. Which would be total = sum(box_dist_i + robot_dist_i) for all boxes. But this would overcount the robot's distance, as the robot can't be at multiple boxes at the same time.

Alternatively, the heuristic should be sum(box_dist_i) + max(robot_dist_i) or sum(box_dist_i) + sum(robot_dist_i). But this could lead to overestimation.

Alternatively, for each box, the robot's distance to it is the time to reach it, and the box's distance is the time to push it. But the total would be the sum of all pushes plus the sum of all robot movements. But this is not accurate.

Alternatively, the heuristic can be the sum of the box distances plus the robot's distance to the closest box's adjacent cell. This assumes that once the robot reaches that box, the other boxes will be handled in sequence, but the robot's path between boxes is not considered. This is a greedy approach but efficient.

So the code would:

For each box not at goal:

   box_dist = precomputed_distance(box_location, goal_location)

   adjacent_cells = get_adjacent_cells(box_location)

   robot_dist = min(precomputed_distance(robot_location, adj) for adj in adjacent_cells)

   total_box_cost = box_dist + robot_dist

The heuristic is the sum of all box_dists plus the minimal robot_dist among all boxes.

Wait, no. For each box, the cost is box_dist + robot_dist. But since the robot can only handle one box at a time, the total cost would be the sum of all box_dists plus the sum of the robot_dists for each box. But this is overcounting because the robot may move from one box to another, and the distance between boxes is not considered.

This is a tricky problem. To keep the heuristic efficient, perhaps the best approach is to sum the box distances and add the robot's distance to the nearest box's adjacent cell. This would prioritize states where boxes are close to their goals and the robot is close to a box.

So the heuristic would be:

sum of box_dists + min_robot_dist_to_any_box_adjacent_cell

This is what I'll implement.

Now, how to implement this in code.

First, in the __init__ method:

- Parse the static adjacency facts to build a graph. Each location has a list of adjacent locations (regardless of direction).

Wait, the adjacency facts in the domain are directional. For example, (adjacent loc_4_2 loc_4_3 right) means loc_4_2 is adjacent to loc_4_3 to the right. But for building a graph, the adjacency is bidirectional. So for each (adjacent ?l1 ?l2 ?d), add an edge from l1 to l2 and vice versa.

Wait, no. The adjacency predicate in the Sokoban domain is directional. For example, (adjacent loc_4_2 loc_4_3 right) means that from loc_4_2, moving right leads to loc_4_3. Similarly, (adjacent loc_4_3 loc_4_2 left) is another fact. So each adjacency is stored as a directed edge. But for the purpose of movement, the robot can move in any direction as long as the adjacent cell is clear. So the graph should be built as undirected, i.e., if l1 is adjacent to l2 in any direction, there's an edge between them.

So during initialization, the code should create a dictionary where each location points to a set of adjacent locations, regardless of direction.

Then, for each static fact that starts with 'adjacent', add both directions. Wait, no. Because the static adjacency facts include both directions. For example, in the first example, there are (adjacent loc_4_2 loc_4_3 right) and (adjacent loc_4_3 loc_4_2 left). So each pair is represented as two directed edges. But for the graph, we can treat it as undirected, so adding each location to the adjacency list of the other.

So during initialization:

adjacent_graph = defaultdict(set)

for fact in static_facts:
    if fact starts with 'adjacent':
        parts = parse the fact into (adjacent l1 l2 dir)
        adjacent_graph[l1].add(l2)
        adjacent_graph[l2].add(l1)  # since movement is possible in both directions if adjacent.

Wait, no. Because the 'adjacent' facts are unidirectional in the PDDL. For example, if (adjacent A B right) exists, but not (adjacent B A left), then in the Sokoban domain, moving from B to A may not be allowed. Wait, no. Because the Sokoban domain's move action requires that the 'adjacent' from ?from to ?to in direction ?dir. So if the adjacent fact from B to A exists, then the robot can move from B to A. So the adjacency graph should be built based on all adjacency facts, but each direction is considered. So the graph is directed. However, in practice, the Sokoban levels are usually grids where adjacency is bidirectional. So perhaps the static facts include both directions. But to be safe, the code should treat the adjacency as directed.

Wait, but in the example static facts, for each adjacency in one direction, there's the reverse. For example, (adjacent loc_4_2 loc_4_3 right) and (adjacent loc_4_3 loc_4_2 left). So the graph is bidirectional. So the code can treat it as undirected.

But to be precise, the code should build a directed graph based on the adjacency facts. For example, if a location l1 is adjacent to l2 in some direction, then there's an edge from l1 to l2. The reverse may or may not be present. But according to the Sokoban domain's actions, the robot can move from ?from to ?to if there's an adjacent fact in any direction. So the adjacency graph should be directed. However, in the Sokoban domain, movement is allowed in any direction as long as the adjacent fact exists. So for example, to move from l1 to l2, there must be an adjacent fact with l1 as ?from and l2 as ?to, and the direction is irrelevant for movement (except for the action's parameter). So the adjacency graph is directed.

But for the purpose of pathfinding (robot movement), the code needs to find paths through the directed adjacency graph. For example, if there's an edge from l1 to l2 but not from l2 to l1, then the robot can move from l1 to l2 but not back. However, in Sokoban levels, adjacency is typically bidirectional. So perhaps the code can treat the graph as undirected. But to be accurate, it should use the directed edges.

This complicates the BFS for shortest paths. For example, if there's a directed edge from l1 to l2 but not the reverse, then the path from l2 to l1 would not exist, even if the robot could move in that direction via another route.

But according to the domain's move action, the robot can move from ?from to ?to only if there's an adjacent fact between them. So the adjacency graph is directed. Therefore, when precomputing shortest paths, the code must treat the graph as directed.

This complicates things. For example, in the initial example, moving from loc_6_4 to loc_5_4 is allowed because there's an adjacent fact (adjacent loc_6_4 loc_5_4 up) and (adjacent loc_5_4 loc_6_4 down). So the graph is bidirectional. But in a different level, perhaps some adjacents are one-way. So the code must account for directed edges.

Therefore, during the heuristic's initialization, the code should build a directed adjacency graph based on the static adjacency facts. Then, for each location, perform BFS to find the shortest path to all other locations, following the directed edges.

But this could lead to situations where a location is unreachable from another, even if they are physically adjacent. However, according to the problem's structure, the Sokoban levels are typically grids with bidirectional movement, but the code must handle directed edges.

This makes the code more complex. But for the purpose of this heuristic, perhaps the code can treat the adjacency as undirected, given that in practice, Sokoban levels are grids with bidirectional adjacents. So the code can build an undirected graph, adding both directions for each adjacency fact.

Alternatively, the code can process each adjacency fact and add a directed edge from l1 to l2. But when the robot can move in both directions, the adjacency facts would include both directions. So the code can proceed with a directed graph.

But given that the static facts include both directions for each adjacency (as seen in the examples), the code can safely build an undirected graph by adding both directions for each adjacency fact.

So, during initialization:

adjacent_graph = defaultdict(set)

for fact in static_facts:
    if match(fact, 'adjacent', ...):
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            l1 = parts[1]
            l2 = parts[2]
            adjacent_graph[l1].add(l2)
            adjacent_graph[l2].add(l1)  # assuming that adjacency is bidirectional

But this is an assumption. However, in the Sokoban domain, the adjacency facts are bidirectional. For example, moving right from l1 to l2 implies that moving left from l2 to l1 is allowed, and the static facts include both directions.

Therefore, building an undirected graph is acceptable.

Once the adjacency graph is built, precompute the shortest paths between all pairs of locations using BFS, treating the graph as undirected.

Now, the code for the heuristic:

First, extract the goal locations for each box from the task's goals.

In the __init__ method:

- Extract the goal_locations: a dictionary mapping each box to its goal location.

Then, for each state, in the __call__ method:

- Extract the current locations of the robot and boxes.

For each box:

   if box is not at its goal:

       box_dist = precomputed_distance(box_location, goal_location)

       adjacent_cells = adjacent_graph.get(box_location, set())

       if adjacent_cells is empty:

           # box is isolated, can't be pushed. heuristic is infinity.

       else:

           robot_dist = min( precomputed_distance(robot_location, adj) for adj in adjacent_cells )

       total_box_dist += box_dist

       min_robot_dist = min( min_robot_dist, robot_dist )

The heuristic is total_box_dist + min_robot_dist.

But if there are no boxes to move, the heuristic is 0.

Now, implementing this in code.

The code will need to:

1. In the constructor:

   a. Parse the static adjacency facts to build the adjacency graph.

   b. Precompute all-pairs shortest paths using BFS.

   c. Parse the goals to find each box's target location.

2. In the __call__ method:

   a. Extract the current state's robot location and box locations.

   b. For each box, check if it's at its goal. If not, compute its distance to the goal and the robot's distance to its adjacent cells.

   c. Sum all box distances and add the minimal robot distance to any box's adjacent cell.

But how to extract the current locations of the robot and boxes from the state?

In the state, the facts are represented as strings like '(at-robot loc_6_4)', '(at box1 loc_4_4)', etc.

So, for each fact in the state:

- If it starts with 'at-robot', the location is the second part.

- If it starts with 'at' and the first argument is a box, then it's the box's current location.

So, in code:

robot_location = None
box_locations = {}

for fact in state:
    parts = get_parts(fact)
    if parts[0] == 'at-robot':
        robot_location = parts[1]
    elif parts[0] == 'at' and parts[1].startswith('box'):
        box = parts[1]
        loc = parts[2]
        box_locations[box] = loc

Then, for each box in box_locations:

   if box is in the goal_locations and box_locations[box] == goal_locations[box]:

       skip

   else:

       compute box_dist and robot_dist.

But how to handle the goals? The goals are a set of PDDL conditions. For example, the goal could be (at box1 loc_2_4). So in the __init__ method, we need to parse the goals to extract for each box what its target location is.

In the __init__ method:

self.goal_locations = {}

for goal in task.goals:

    parts = get_parts(goal)

    if parts[0] == 'at' and len(parts) == 3:

        box = parts[1]

        loc = parts[2]

        self.goal_locations[box] = loc

So this creates a dictionary mapping each box to its goal location.

Now, for each box in box_locations:

current_loc = box_locations[box]

goal_loc = self.goal_locations.get(box, None)

if goal_loc is None:

    # this box has no goal, but according to the problem, all boxes have goals.

    continue

if current_loc == goal_loc:

    continue

else:

    box_dist = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))

    adjacent_cells = self.adjacent_graph.get(current_loc, [])

    if not adjacent_cells:

        # box is stuck, but the problem may not have such cases.

        robot_dist = float('inf')

    else:

        robot_dist = min( self.shortest_paths.get(robot_location, {}).get(adj, float('inf')) for adj in adjacent_cells )

    total_box_dist += box_dist

    if robot_dist < current_min_robot_dist:

        current_min_robot_dist = robot_dist

After processing all boxes:

if total_box_dist == 0:

    return 0  # all boxes are at goals

else:

    return total_box_dist + current_min_robot_dist

But wait, the minimal robot distance is the minimal distance from the robot's current location to any adjacent cell of any box not at its goal. So we need to compute for each box, the minimal robot distance to its adjacent cells, and then take the minimal among all those.

So for each box not at goal:

   compute the minimal robot distance to its adjacent cells

   keep track of the minimal such distance across all boxes

So, even if there are multiple boxes, the heuristic adds the minimal robot distance to any of their adjacent cells.

This makes sense because the robot can choose to go to the closest box first.

Putting it all together:

In the __call__ method:

robot_location = ... 

box_locations = ... 

total_box_dist = 0

min_robot_dist = infinity

for box, current_loc in box_locations.items():

    goal_loc = self.goal_locations.get(box, None)

    if goal_loc is None or current_loc == goal_loc:

        continue

    # Compute box distance

    box_dist = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))

    if box_dist == float('inf'):

        # Box can't reach goal, state is unsolvable. But the heuristic should return a finite value?

        # The problem states the heuristic should be finite for solvable states, but how to handle this?

        # For now, assume that the state is solvable.

        box_dist = 0  # or some large number?

    total_box_dist += box_dist

    # Compute robot's minimal distance to adjacent cells of this box

    adjacent_cells = self.adjacent_graph.get(current_loc, [])

    if not adjacent_cells:

        robot_dist = float('inf')

    else:

        robot_dist = min( self.shortest_paths.get(robot_location, {}).get(adj, float('inf')) for adj in adjacent_cells )

    if robot_dist < min_robot_dist:

        min_robot_dist = robot_dist

if total_box_dist == 0:

    return 0

else:

    if min_robot_dist == float('inf'):

        # Robot can't reach any box's adjacent cells. So the state is unsolvable.

        # But the problem says the heuristic should be finite for solvable states. So return a large number?

        return float('inf')

    else:

        return total_box_dist + min_robot_dist

But how to handle cases where the box's distance is infinity (unreachable) or the robot's distance is infinity? According to the problem statement, the heuristic should return a finite value for solvable states. So perhaps the code should assume that the state is solvable and the heuristic returns a finite value. But in practice, for unsolvable states, the heuristic can return a large number. However, the problem states that the heuristic should be finite for solvable states, but the code can return any value for unsolvable ones. But the planner may not generate such states.

So the code can proceed as above.

Now, the code needs to precompute the shortest paths between all pairs of locations. This is done in the __init__ method.

In the __init__ method:

self.adjacent_graph = defaultdict(set)

for fact in task.static:

    parts = get_parts(fact)

    if parts[0] == 'adjacent':

        l1 = parts[1]

        l2 = parts[2]

        self.adjacent_graph[l1].add(l2)

        self.adjacent_graph[l2].add(l1)  # assuming undirected

Then, precompute shortest paths using BFS for each location:

self.shortest_paths = defaultdict(dict)

locations = set(self.adjacent_graph.keys())

for start in locations:

    visited = {start: 0}

    queue = deque([start])

    while queue:

        current = queue.popleft()

        for neighbor in self.adjacent_graph[current]:

            if neighbor not in visited:

                visited[neighbor] = visited[current] + 1

                queue.append(neighbor)

    for loc, distance in visited.items():

        self.shortest_paths[start][loc] = distance

    # For locations not reachable from start, distance remains as infinity.

But this is for undirected graphs. If the graph is directed, BFS would need to follow directed edges. But since we're assuming the adjacency is undirected, this code should work.

Now, putting all this into code.

Imports needed:

from collections import defaultdict, deque
from fnmatch import fnmatch

The code will need to import these.

Now, the code for the heuristic:

class sokoban25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions. It combines the minimal pushes required for each box and the robot's distance to the nearest box's adjacent cell.

    # Assumptions
    - Boxes can only be pushed to adjacent cells if the robot can reach the required position.
    - The robot can move freely to any clear location, ignoring other boxes during pathfinding.
    - Each push action counts as one step, and robot movement is computed using shortest paths.

    # Heuristic Initialization
    - Precompute the shortest paths between all pairs of locations using BFS based on static adjacency.
    - Extract the goal locations for each box from the task's goals.
    - Build an adjacency graph from the static 'adjacent' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Compute the shortest path distance from the box's current location to its goal (number of pushes).
        b. Find all adjacent cells to the box's current location (possible robot positions for pushing).
        c. Compute the minimal distance from the robot's current location to any of these adjacent cells.
    2. Sum the box distances for all boxes not at their goals.
    3. Add the minimal robot distance to any box's adjacent cell to the sum.
    4. The total heuristic value is the sum from steps 2 and 3.
    """

    def __init__(self, task):
        """Initialize the heuristic with precomputed shortest paths and goal locations."""
        self.goal_locations = {}
        self.adjacent_graph = defaultdict(set)
        self.shortest_paths = defaultdict(dict)

        # Extract goal locations for each box
        for goal in task.goals:
            parts = self.get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        for fact in task.static:
            parts = self.get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2 = parts[1], parts[2]
                self.adjacent_graph[l1].add(l2)
                self.adjacent_graph[l2].add(l1)  # Assume adjacency is bidirectional

        # Precompute shortest paths between all locations using BFS
        locations = set(self.adjacent_graph.keys())
        for start in locations:
            visited = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacent_graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            # Update shortest paths for this start location
            for loc, dist in visited.items():
                self.shortest_paths[start][loc] = dist
            # Locations not reachable will not be in visited, so their distance remains as infinity

    def get_parts(self, fact):
        """Extract components of a PDDL fact."""
        return fact.strip('()').split()

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        robot_loc = None
        box_locs = {}

        # Extract current robot and box locations from the state
        for fact in state:
            parts = self.get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                box_locs[box] = loc

        if not robot_loc:
            return float('inf')  # Invalid state

        total_box_dist = 0
        min_robot_dist = float('inf')

        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue  # Box is already at goal

            # Compute box's distance to goal
            box_dist = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
            if box_dist == float('inf'):
                continue  # Assume the problem is solvable, skip this box's contribution

            total_box_dist += box_dist

            # Compute robot's minimal distance to adjacent cells of the box
            adjacent_cells = self.adjacent_graph.get(current_loc, [])
            if not adjacent_cells:
                continue

            current_min = min(
                (self.shortest_paths.get(robot_loc, {}).get(adj, float('inf')) for adj in adjacent_cells),
                default=float('inf')
            )
            if current_min < min_robot_dist:
                min_robot_dist = current_min

        if total_box_dist == 0:
            return 0  # All boxes are at their goals

        if min_robot_dist == float('inf'):
            # Robot cannot reach any box's adjacent cells, return a large value
            return total_box_dist * 2  # Arbitrary penalty

        return total_box_dist + min_robot_dist

Wait, but in the code for the robot's minimal distance, the line:

current_min = min( (self.shortest_paths.get(robot_loc, {}).get(adj, float('inf')) for adj in adjacent_cells), default=float('inf') )

But in Python versions before 3.4, the 'default' argument for min() is not available. So to handle cases where the generator is empty, we need to check if there are any valid distances.

But the code assumes that adjacent_cells is not empty (since it's checked). However, if adjacent_cells is empty, the code skips it. So the 'current_min' is computed for non-empty adjacent_cells.

Alternatively, handle it with a try-except block, but that's messy.

Alternatively, compute the minimum manually:

robot_dists = []
for adj in adjacent_cells:
    dist = self.shortest_paths.get(robot_loc, {}).get(adj, float('inf'))
    robot_dists.append(dist)
if robot_dists:
    current_min = min(robot_dists)
else:
    current_min = float('inf')

This is safer.

So updating the code:

for box, current_loc in box_locs.items():
    goal_loc = self.goal_locations.get(box)
    if not goal_loc or current_loc == goal_loc:
        continue

    box_dist = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
    if box_dist == float('inf'):
        continue

    total_box_dist += box_dist

    adjacent_cells = self.adjacent_graph.get(current_loc, [])
    robot_dists = []
    for adj in adjacent_cells:
        dist = self.shortest_paths.get(robot_loc, {}).get(adj, float('inf'))
        robot_dists.append(dist)
    if robot_dists:
        current_min = min(robot_dists)
        if current_min < min_robot_dist:
            min_robot_dist = current_min

This ensures that even if adjacent_cells is empty, robot_dists is empty, and current_min is not updated.

Now, putting this into the code.

Final code:
</think>

from collections import defaultdict, deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


class sokoban25Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to push all boxes to their goal positions. It combines the minimal pushes required for each box and the robot's distance to the nearest box's adjacent cell.

    # Assumptions
    - Boxes can only be pushed to adjacent cells if the robot can reach the required position.
    - The robot can move freely to any clear location, ignoring other boxes during pathfinding.
    - Each push action counts as one step, and robot movement is computed using shortest paths.

    # Heuristic Initialization
    - Precompute the shortest paths between all pairs of locations using BFS based on static adjacency.
    - Extract the goal locations for each box from the task's goals.
    - Build an adjacency graph from the static 'adjacent' facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Compute the shortest path distance from the box's current location to its goal (number of pushes).
        b. Find all adjacent cells to the box's current location (possible robot positions for pushing).
        c. Compute the minimal distance from the robot's current location to any of these adjacent cells.
    2. Sum the box distances for all boxes not at their goals.
    3. Add the minimal robot distance to any box's adjacent cell to the sum.
    4. The total heuristic value is the sum from steps 2 and 3.
    """

    def __init__(self, task):
        """Initialize the heuristic with precomputed shortest paths and goal locations."""
        self.goal_locations = {}
        self.adjacent_graph = defaultdict(set)
        self.shortest_paths = defaultdict(dict)

        # Extract goal locations for each box
        for goal in task.goals:
            parts = self._get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                box = parts[1]
                loc = parts[2]
                self.goal_locations[box] = loc

        # Build adjacency graph from static facts
        for fact in task.static:
            parts = self._get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2 = parts[1], parts[2]
                self.adjacent_graph[l1].add(l2)
                self.adjacent_graph[l2].add(l1)  # Assume bidirectional movement

        # Precompute shortest paths between all locations using BFS
        all_locations = set(self.adjacent_graph.keys())
        for start_loc in all_locations:
            visited = {start_loc: 0}
            queue = deque([start_loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacent_graph.get(current, []):
                    if neighbor not in visited:
                        visited[neighbor] = visited[current] + 1
                        queue.append(neighbor)
            # Store distances for this start location
            self.shortest_paths[start_loc] = defaultdict(lambda: float('inf'), visited)

    def _get_parts(self, fact):
        """Extract components of a PDDL fact."""
        return fact.strip('()').split()

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        robot_loc = None
        box_locs = {}

        # Extract current robot and box positions
        for fact in state:
            parts = self._get_parts(fact)
            if not parts:
                continue
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at' and parts[1].startswith('box'):
                box = parts[1]
                loc = parts[2]
                box_locs[box] = loc

        if not robot_loc:
            return float('inf')  # Invalid state

        total_box_dist = 0
        min_robot_dist = float('inf')

        for box, current_loc in box_locs.items():
            goal_loc = self.goal_locations.get(box)
            if not goal_loc or current_loc == goal_loc:
                continue  # Box is already at goal

            # Calculate box's minimal push distance to goal
            box_dist = self.shortest_paths[current_loc].get(goal_loc, float('inf'))
            if box_dist == float('inf'):
                continue  # Assume problem is solvable, skip invalid paths

            total_box_dist += box_dist

            # Calculate robot's minimal distance to box's adjacent cells
            adjacent_cells = self.adjacent_graph.get(current_loc, [])
            robot_dists = []
            for adj in adjacent_cells:
                robot_dists.append(self.shortest_paths[robot_loc].get(adj, float('inf')))
            
            if robot_dists:
                current_min = min(robot_dists)
                if current_min < min_robot_dist:
                    min_robot_dist = current_min

        # Handle case where all boxes are placed
        if total_box_dist == 0:
            return 0

        # Apply penalty if robot can't reach any box (but assume state is valid)
        if min_robot_dist == float('inf'):
            return total_box_dist * 2  # Empirical penalty factor

        return total_box_dist + min_robot_dist