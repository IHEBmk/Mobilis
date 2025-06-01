# import math
# import pulp
# import numpy as np

# # Helper: Haversine formula to compute distance between two lat/lon points
# def haversine(lon1, lat1, lon2, lat2):
#     R = 6371  # Earth radius in km
#     lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
#     dlon, dlat = lon2 - lon1, lat2 - lat1
#     a = (math.sin(dlat / 2) ** 2 +
#          math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
#     return R * 2 * math.asin(math.sqrt(a))

# # Plan route for a single day (or time chunk)
# def plan_route(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15):
#     N = len(points)

#     # Create subset of unvisited points, ensuring start point is always included
#     unvisited_points = [point for point in points if point['name'] not in visited_points and point['name']!='Start']
#     unvisited_points.insert(0, points[0])  # Always start at the same location
#     N_unvisited = len(unvisited_points)

#     # Compute travel time matrix (in minutes)
#     travel_time = np.zeros((N_unvisited, N_unvisited))
#     for i in range(N_unvisited):
#         for j in range(N_unvisited):
#             if i != j:
#                 dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
#                                  unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
#                 time = (dist / speed_kmph) * 60
#                 travel_time[i][j] = time

#     # Define integer programming model
#     model = pulp.LpProblem("Orienteering", pulp.LpMaximize)

#     x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N_unvisited) for j in range(N_unvisited) if i != j), cat="Binary")
#     u = pulp.LpVariable.dicts("u", (i for i in range(N_unvisited)), lowBound=0, upBound=N_unvisited - 1, cat="Continuous")

#     # Objective: maximize number of visits (excluding return to start)
#     model += pulp.lpSum(x[i, j] for i in range(N_unvisited) for j in range(N_unvisited) if i != j and j != 0)

#     # Constraints
#     model += pulp.lpSum(x[0, j] for j in range(1, N_unvisited)) == 1
#     model += pulp.lpSum(x[j, 0] for j in range(1, N_unvisited)) == 0

#     for i in range(1, N_unvisited):
#         model += pulp.lpSum(x[j, i] for j in range(N_unvisited) if j != i) <= 1
#         model += pulp.lpSum(x[i, j] for j in range(N_unvisited) if j != i) <= 1

#     model += pulp.lpSum(x[i, j] * (travel_time[i][j] + visit_cost_minutes)
#                         for i in range(N_unvisited) for j in range(N_unvisited) if i != j) <= time_limit_minutes

#     # Subtour elimination (MTZ)
#     for i in range(1, N_unvisited):
#         for j in range(1, N_unvisited):
#             if i != j:
#                 model += u[i] - u[j] + (N_unvisited - 1) * x[i, j] <= N_unvisited - 2

#     model.solve()

#     # Extract solution
#     solution_edges = [(i, j) for i in range(N_unvisited) for j in range(N_unvisited)
#                       if i != j and pulp.value(x[i, j]) == 1]

#     # Build ordered route
#     solution_order = [unvisited_points[0]['name']]
#     current = 0
#     while True:
#         next_nodes = [j for i, j in solution_edges if i == current]
#         if not next_nodes:
#             break
#         current = next_nodes[0]
#         solution_order.append(unvisited_points[current]['name'])

#     estimated_time = sum(travel_time[i][j] + visit_cost_minutes for i, j in solution_edges)

#     return solution_order, solution_edges, unvisited_points, estimated_time

# # Multi-day planning
# def plan_multiple_days(points, total_minutes, daily_limit_minutes, speed_kmph):
#     visited_points = []
#     all_routes = []
#     all_edges = []
#     all_estimates = []

#     num_days = math.ceil(total_minutes / daily_limit_minutes)

#     for day in range(num_days):
#         route, edges, unvisited_points, estimated_time = plan_route(
#             points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit_minutes, visited_points=visited_points
#         )

#         if len(route) <= 1:  # No progress
#             break

#         all_routes.append(route)
#         all_edges.append(edges)
#         all_estimates.append(estimated_time)
#         visited_points.extend(route[1:])  # Skip start point

#         if len(visited_points) >= len(points) - 1:
#             break

#     return all_routes, all_edges, all_estimates





# import math
# import pulp
# import numpy as np

# # Helper: Haversine formula to compute distance between two lat/lon points
# def haversine(lon1, lat1, lon2, lat2):
#     R = 6371  # Earth radius in km
#     lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
#     dlon, dlat = lon2 - lon1, lat2 - lat1
#     a = (math.sin(dlat / 2) ** 2 +
#          math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
#     return R * 2 * math.asin(math.sqrt(a))

# # Plan route for a single day (or time chunk)
# def plan_route(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15):
#     N = len(points)

#     # Create subset of unvisited points, ensuring start point is always included
#     unvisited_points = [point for point in points if point['name'] not in visited_points and point['name'] != 'Start']
#     unvisited_points.insert(0, points[0])  # Always start at the same location
#     N_unvisited = len(unvisited_points)

#     # Compute travel time matrix (in minutes)
#     travel_time = np.zeros((N_unvisited, N_unvisited))
#     for i in range(N_unvisited):
#         for j in range(N_unvisited):
#             if i != j:
#                 dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
#                                  unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
#                 time = (dist / speed_kmph) * 60
#                 travel_time[i][j] = time

#     # Define integer programming model
#     model = pulp.LpProblem("Orienteering", pulp.LpMaximize)

#     x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N_unvisited) for j in range(N_unvisited) if i != j), cat="Binary")
#     u = pulp.LpVariable.dicts("u", (i for i in range(N_unvisited)), lowBound=0, upBound=N_unvisited - 1, cat="Continuous")

#     # Objective: maximize number of visits (excluding return to start)
#     model += pulp.lpSum(x[i, j] for i in range(N_unvisited) for j in range(N_unvisited) if i != j and j != 0)

#     # Constraints
#     model += pulp.lpSum(x[0, j] for j in range(1, N_unvisited)) == 1
#     model += pulp.lpSum(x[j, 0] for j in range(1, N_unvisited)) == 0

#     for i in range(1, N_unvisited):
#         model += pulp.lpSum(x[j, i] for j in range(N_unvisited) if j != i) <= 1
#         model += pulp.lpSum(x[i, j] for j in range(N_unvisited) if j != i) <= 1

#     model += pulp.lpSum(x[i, j] * (travel_time[i][j] + visit_cost_minutes)
#                         for i in range(N_unvisited) for j in range(N_unvisited) if i != j) <= time_limit_minutes

#     # Subtour elimination (MTZ)
#     for i in range(1, N_unvisited):
#         for j in range(1, N_unvisited):
#             if i != j:
#                 model += u[i] - u[j] + (N_unvisited - 1) * x[i, j] <= N_unvisited - 2

#     model.solve()

#     # Extract solution
#     solution_edges = [(i, j) for i in range(N_unvisited) for j in range(N_unvisited)
#                       if i != j and pulp.value(x[i, j]) == 1]

#     # Build ordered route
#     solution_order = [unvisited_points[0]['name']]
#     current = 0
#     while True:
#         next_nodes = [j for i, j in solution_edges if i == current]
#         if not next_nodes:
#             break
#         current = next_nodes[0]
#         solution_order.append(unvisited_points[current]['name'])

#     estimated_time = sum(travel_time[i][j] + visit_cost_minutes for i, j in solution_edges)

#     return solution_order, solution_edges, unvisited_points, estimated_time

# # Multi-day planning until all points are visited
# def plan_multiple_days(points, daily_limit_minutes, speed_kmph):
#     visited_points = []
#     all_routes = []
#     all_edges = []
#     all_estimates = []

#     while len(visited_points) < len(points) - 1:  # Exclude start point
#         route, edges, unvisited_points, estimated_time = plan_route(
#             points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit_minutes, visited_points=visited_points
#         )

#         if len(route) <= 1:  # No progress
#             break

#         all_routes.append(route)
#         all_edges.append(edges)
#         all_estimates.append(estimated_time)
#         visited_points.extend(route[1:])  # Skip start point

#     return all_routes, all_edges, all_estimates



# import math
# import numpy as np
# from typing import List, Dict, Tuple, Set
# import heapq
# import time as time_module


# # Helper: Haversine formula to compute distance between two lat/lon points
# def haversine(lon1, lat1, lon2, lat2):
#     R = 6371  # Earth radius in km
#     lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
#     dlon, dlat = lon2 - lon1, lat2 - lat1
#     a = (math.sin(dlat / 2) ** 2 +
#          math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
#     return R * 2 * math.asin(math.sqrt(a))


# def compute_travel_time_matrix(points, speed_kmph):
#     """Precompute the travel time matrix for all points."""
#     N = len(points)
#     travel_time = np.zeros((N, N))
    
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 dist = haversine(points[i]['longitude'], points[i]['latitude'],
#                                points[j]['longitude'], points[j]['latitude'])
#                 time = (dist / speed_kmph) * 60  # Convert to minutes
#                 travel_time[i][j] = time
    
#     return travel_time


# def plan_route_no_return(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15):
#     """
#     Plan a route for a single day without returning to start using a greedy insertion algorithm.
    
#     Args:
#         points: List of point dictionaries with 'name', 'latitude', 'longitude'
#         speed_kmph: Speed in km/h
#         time_limit_minutes: Time limit for the route in minutes
#         visited_points: Set or list of already visited point names
#         visit_cost_minutes: Time spent at each visited point
        
#     Returns:
#         solution_order: List of point names in visitation order
#         solution_edges: List of tuples (i, j) representing edges in solution
#         unvisited_points: List of points considered in this solution
#         estimated_time: Estimated time for this route
#     """
#     # Start time measurement for performance tracking
#     start_time = time_module.time()
    
#     # Create subset of unvisited points, ensuring start point is always included
#     unvisited_indices = {}  # Map from original index to new index
#     unvisited_points = []
    
#     # Always include start point (assumed to be at index 0)
#     unvisited_points.append(points[0])
#     unvisited_indices[0] = 0
    
#     # Add other unvisited points
#     new_idx = 1
#     for i, point in enumerate(points):
#         if i > 0 and point['name'] not in visited_points:
#             unvisited_points.append(point)
#             unvisited_indices[i] = new_idx
#             new_idx += 1
    
#     N_unvisited = len(unvisited_points)
    
#     # Compute travel time matrix for unvisited points
#     travel_time = np.zeros((N_unvisited, N_unvisited))
#     for i in range(N_unvisited):
#         for j in range(N_unvisited):
#             if i != j:
#                 dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
#                                unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
#                 travel_time_val = (dist / speed_kmph) * 60
#                 travel_time[i][j] = travel_time_val
    
#     # Greedy insertion algorithm
#     # Initialize route with start point
#     route_indices = [0]  # Start at depot (index 0)
#     current_time = 0
    
#     while True:
#         best_insertion = None
#         best_increase = float('inf')
        
#         # Try to insert each unvisited point into the route
#         for i in range(1, N_unvisited):  # Skip start point (index 0)
#             if i not in route_indices:
#                 # Try to insert at each position in the route
#                 for pos in range(1, len(route_indices) + 1):
#                     # Consider inserting point i between points at pos-1 and pos
#                     # Special case for insertion at the beginning
                    
#                     if pos == len(route_indices):
#                         prev_idx = route_indices[-1]
#                         next_idx = None  # No need to return to start
#                     else:
#                         prev_idx = route_indices[pos-1]
#                         next_idx = route_indices[pos]
                    
#                     # Calculate time increase
#                     if next_idx is None:
#                         # For insertion at the end, we only add time from previous to new point
#                         old_time = 0  # No previous edge at the end
#                         new_time = travel_time[prev_idx][i] + visit_cost_minutes
#                     else:
#                         old_time = travel_time[prev_idx][next_idx] if pos > 0 else 0
#                         new_time = travel_time[prev_idx][i] + travel_time[i][next_idx] + visit_cost_minutes
                        
#                     time_increase = new_time - old_time
                    
#                     # Check if this is the best insertion
#                     if time_increase < best_increase:
#                         # Calculate the total time with this insertion
#                         new_route = route_indices.copy()
#                         new_route.insert(pos, i)
                        
#                         # Calculate total route time
#                         total_time = 0
#                         for j in range(len(new_route) - 1):
#                             total_time += travel_time[new_route[j]][new_route[j+1]]
                        
#                         # Add visit time for each point (except start point)
#                         total_time += (len(new_route) - 1) * visit_cost_minutes
                        
#                         # Check if we're still within the time limit
#                         if total_time <= time_limit_minutes:
#                             best_increase = time_increase
#                             best_insertion = (i, pos, total_time)
        
#         # If no feasible insertion found, break
#         if best_insertion is None:
#             break
        
#         # Insert the best point
#         point_idx, pos, new_total_time = best_insertion
#         route_indices.insert(pos, point_idx)
#         current_time = new_total_time
    
#     # Build the solution
#     solution_order = [unvisited_points[idx]['name'] for idx in route_indices]
    
#     # Build solution edges
#     solution_edges = []
#     for i in range(len(route_indices) - 1):
#         solution_edges.append((route_indices[i], route_indices[i+1]))
    
#     # Calculate estimated time (without return to start)
#     estimated_time = 0
#     for i, j in solution_edges:
#         estimated_time += travel_time[i][j]
#     estimated_time += (len(route_indices) - 1) * visit_cost_minutes  # No visit time for depot
    
#     # Check if we actually have a feasible solution with at least one visit
#     if len(solution_order) <= 1:  # Only start point
#         return [unvisited_points[0]['name']], [], unvisited_points, 0

#     # Print performance info
#     end_time = time_module.time()
#     execution_time = end_time - start_time
#     print(f"Route planning completed in {execution_time:.4f} seconds. Visiting {len(solution_order)-1} points.")
    
#     return solution_order, solution_edges, unvisited_points, estimated_time


# def plan_route_enhanced_no_return(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15, 
#                         time_buffer=0.05):
#     """
#     Enhanced route planning without return to start using a combination of greedy and randomized insertion.
    
#     Args:
#         points: List of point dictionaries
#         speed_kmph: Speed in km/h
#         time_limit_minutes: Time limit in minutes
#         visited_points: Set of already visited point names
#         visit_cost_minutes: Time at each point
#         time_buffer: Fraction of time to reserve as buffer (0.05 = 5%)
        
#     Returns:
#         Similar to plan_route function
#     """
#     # Apply time buffer to allow for uncertainties
#     effective_time_limit = time_limit_minutes * (1 - time_buffer)
    
#     # Create subset of unvisited points
#     unvisited_indices = {}
#     unvisited_points = []
    
#     # Add start point
#     unvisited_points.append(points[0])
#     unvisited_indices[0] = 0
    
#     # Add unvisited points
#     new_idx = 1
#     for i, point in enumerate(points):
#         if i > 0 and point['name'] not in visited_points:
#             unvisited_points.append(point)
#             unvisited_indices[i] = new_idx
#             new_idx += 1
    
#     N_unvisited = len(unvisited_points)
    
#     # Compute travel time matrix
#     travel_time = np.zeros((N_unvisited, N_unvisited))
#     for i in range(N_unvisited):
#         for j in range(N_unvisited):
#             if i != j:
#                 dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
#                                unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
#                 travel_time_val = (dist / speed_kmph) * 60
#                 travel_time[i][j] = travel_time_val
    
#     # Try multiple starting points and keep the best solution
#     best_route = []
#     best_time = 0
    
#     # Find k nearest neighbors to start point (k scales with problem size)
#     k = min(5, N_unvisited - 1)
#     if k > 0:
#         nearest_to_start = np.argsort(travel_time[0, 1:])[:k] + 1  # +1 because we skipped index 0
#     else:
#         nearest_to_start = []
        
#     # Try each neighbor as the first visit
#     candidates = [0] + list(nearest_to_start)  # Include start point too
    
#     for start_candidate in candidates:
#         # Initialize route with start point and first visit
#         if start_candidate == 0:
#             # Just start at depot with empty route
#             route_indices = [0]
#         else:
#             route_indices = [0, start_candidate]
        
#         # Greedy insertion phase
#         while True:
#             best_insertion = None
#             best_score = float('-inf')  # We'll use a score-based approach
            
#             # For each unvisited point
#             for i in range(1, N_unvisited):
#                 if i not in route_indices:
#                     # Try to insert at each position
#                     for pos in range(1, len(route_indices) + 1):  # Don't insert before start
#                         # Calculate insertion positions
#                         if pos == len(route_indices):
#                             prev_idx = route_indices[-1]
#                             next_idx = None  # No need to return to start
#                         else:
#                             prev_idx = route_indices[pos-1]
#                             next_idx = route_indices[pos]
                        
#                         # Calculate time increase
#                         if next_idx is None:
#                             # For insertion at the end, we only add time from previous to new point
#                             old_time = 0
#                             new_time = travel_time[prev_idx][i] + visit_cost_minutes
#                         else:
#                             old_time = travel_time[prev_idx][next_idx]
#                             new_time = travel_time[prev_idx][i] + travel_time[i][next_idx] + visit_cost_minutes
                            
#                         time_increase = new_time - old_time
                        
#                         # Calculate the score (favoring points that are closer)
#                         score = -time_increase
                        
#                         # Check if this insertion is better
#                         if score > best_score:
#                             # Verify time constraint
#                             new_route = route_indices.copy()
#                             new_route.insert(pos, i)
                            
#                             # Calculate total route time
#                             total_time = 0
#                             for j in range(len(new_route) - 1):
#                                 total_time += travel_time[new_route[j]][new_route[j+1]]
                            
#                             # Add visit times
#                             total_time += (len(new_route) - 1) * visit_cost_minutes
                            
#                             if total_time <= effective_time_limit:
#                                 best_score = score
#                                 best_insertion = (i, pos, total_time)
            
#             # If no feasible insertion found, break
#             if best_insertion is None:
#                 break
            
#             # Insert the best point
#             point_idx, pos, new_total_time = best_insertion
#             route_indices.insert(pos, point_idx)
        
#         # Calculate route quality
#         if len(route_indices) > 1:  # At least one visit
#             current_time = 0
#             for j in range(len(route_indices) - 1):
#                 current_time += travel_time[route_indices[j]][route_indices[j+1]]
            
#             # Add visit times
#             current_time += (len(route_indices) - 1) * visit_cost_minutes
            
#             # Check if this route is better
#             if len(route_indices) > len(best_route) or (len(route_indices) == len(best_route) and current_time < best_time):
#                 best_route = route_indices.copy()
#                 best_time = current_time
    
#     # If no feasible route found, return minimal solution
#     if not best_route or len(best_route) <= 1:
#         return [unvisited_points[0]['name']], [], unvisited_points, 0
    
#     # Build the final solution
#     route_indices = best_route
#     solution_order = [unvisited_points[idx]['name'] for idx in route_indices]
    
#     # Build solution edges
#     solution_edges = []
#     for i in range(len(route_indices) - 1):
#         solution_edges.append((route_indices[i], route_indices[i+1]))
    
#     # Calculate estimated time (without return to start)
#     estimated_time = 0
#     for i, j in solution_edges:
#         estimated_time += travel_time[i][j]
#     estimated_time += (len(route_indices) - 1) * visit_cost_minutes
    
#     return solution_order, solution_edges, unvisited_points, estimated_time


# def plan_multiple_days_no_return(points, daily_limit_minutes, speed_kmph):
#     """
#     Multi-day planning without requiring return to start point at end of day.
    
#     Args:
#         points: List of point dictionaries with 'name', 'latitude', 'longitude'
#         daily_limit_minutes: Time limit for each day in minutes
#         speed_kmph: Speed in km/h
        
#     Returns:
#         all_routes: List of routes, each route is a list of point names
#         all_edges: List of edge lists, each edge list is a list of (i,j) tuples
#         all_estimates: List of estimated times for each route
#     """
#     start_time = time_module.time()
#     visited_points = set()
#     all_routes = []
#     all_edges = []
#     all_estimates = []
    
#     # Exclude start point from target count
#     total_points = len(points) - 1
#     iteration = 0
    
#     print(f"Planning routes for {total_points} points with {daily_limit_minutes} minutes per day")
#     print(f"NOTE: Routes do NOT return to start point at end of day")
    
#     while len(visited_points) < total_points:
#         iteration += 1
#         print(f"Day {iteration}: Planning for {total_points - len(visited_points)} remaining points")
        
#         # Try both algorithms and use the better result
#         route1, edges1, unvisited1, time1 = plan_route_no_return(
#             points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit_minutes,
#             visited_points=visited_points, visit_cost_minutes=15
#         )
#         if route1[0]!="Start":
#             print("NO START 1")
#         route2, edges2, unvisited2, time2 = plan_route_enhanced_no_return(
#             points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit_minutes,
#             visited_points=visited_points, visit_cost_minutes=15
#         )
#         if route2[0]!="Start":
#             print("NO START 2")
#         # Choose the better route (more points or equal points with less time)
#         if len(route1) > len(route2) or (len(route1) == len(route2) and time1 < time2):
#             route, edges, unvisited_points, estimated_time = route1, edges1, unvisited1, time1
#         else:
#             route, edges, unvisited_points, estimated_time = route2, edges2, unvisited2, time2
        
#         if len(route) <= 1:  # No progress
#             print("Warning: No progress made in this iteration. Check constraints.")
#             break
            
#         # Record points visited in this route (skip start point)
#         for point_name in route[1:]:
#             if point_name != points[0]['name']:  # Skip depot
#                 visited_points.add(point_name)
        
#         all_routes.append(route)
#         all_edges.append(edges)
#         all_estimates.append(estimated_time)
        
#         print(f"Day {iteration} complete. Visited {len(route)-1} points. Time: {estimated_time:.1f} min")
#         print(f"Visited so far: {len(visited_points)}/{total_points} points")
    
#     end_time = time_module.time()
#     total_execution_time = end_time - start_time
#     print(f"Multi-day planning completed in {total_execution_time:.4f} seconds")
#     print(f"Total days required: {len(all_routes)}")
    
#     return all_routes, all_edges, all_estimates


# # Function to calculate route statistics
# def calculate_statistics(all_routes, all_estimates, points):
#     """Calculate statistics about the routes."""
#     if not all_routes:
#         return {}
    
#     stats = {
#         "total_days": len(all_routes),
#         "total_points_visited": sum(len(route) - 1 for route in all_routes),  # Only subtract start
#         "avg_points_per_day": sum(len(route) - 1 for route in all_routes) / len(all_routes),
#         "avg_time_per_day": sum(all_estimates) / len(all_estimates) if all_estimates else 0,
#         "max_time_day": max(all_estimates) if all_estimates else 0,
#         "min_time_day": min(all_estimates) if all_estimates else 0,
#     }
    
#     return stats



import math
import numpy as np
from typing import List, Dict, Tuple, Set
import heapq
import time as time_module
from datetime import datetime, timedelta


# Helper: Haversine formula to compute distance between two lat/lon points
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def compute_travel_time_matrix(points, speed_kmph):
    """Precompute the travel time matrix for all points."""
    N = len(points)
    travel_time = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = haversine(points[i]['longitude'], points[i]['latitude'],
                               points[j]['longitude'], points[j]['latitude'])
                time = (dist / speed_kmph) * 60  # Convert to minutes
                travel_time[i][j] = time
    
    return travel_time


def plan_route_no_return(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15):
    """
    Plan a route for a single day without returning to start using a greedy insertion algorithm.
    
    Args:
        points: List of point dictionaries with 'name', 'latitude', 'longitude'
        speed_kmph: Speed in km/h
        time_limit_minutes: Time limit for the route in minutes
        visited_points: Set or list of already visited point names
        visit_cost_minutes: Time spent at each visited point
        
    Returns:
        solution_order: List of point names in visitation order
        solution_edges: List of tuples (i, j) representing edges in solution
        unvisited_points: List of points considered in this solution
        estimated_time: Estimated time for this route
    """
    # Start time measurement for performance tracking
    start_time = time_module.time()
    
    # Create subset of unvisited points, ensuring start point is always included
    unvisited_indices = {}  # Map from original index to new index
    unvisited_points = []
    
    # Always include start point (assumed to be at index 0)
    unvisited_points.append(points[0])
    unvisited_indices[0] = 0
    
    # Add other unvisited points
    new_idx = 1
    for i, point in enumerate(points):
        if i > 0 and point['name'] not in visited_points:
            unvisited_points.append(point)
            unvisited_indices[i] = new_idx
            new_idx += 1
    
    N_unvisited = len(unvisited_points)
    
    # Compute travel time matrix for unvisited points
    travel_time = np.zeros((N_unvisited, N_unvisited))
    for i in range(N_unvisited):
        for j in range(N_unvisited):
            if i != j:
                dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
                               unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
                travel_time_val = (dist / speed_kmph) * 60
                travel_time[i][j] = travel_time_val
    
    # Greedy insertion algorithm
    # Initialize route with start point
    route_indices = [0]  # Start at depot (index 0)
    current_time = 0
    
    while True:
        best_insertion = None
        best_increase = float('inf')
        
        # Try to insert each unvisited point into the route
        for i in range(1, N_unvisited):  # Skip start point (index 0)
            if i not in route_indices:
                # Try to insert at each position in the route
                for pos in range(1, len(route_indices) + 1):
                    # Consider inserting point i between points at pos-1 and pos
                    # Special case for insertion at the beginning
                    
                    # Special case for insertion at the end
                    if pos == len(route_indices):
                        prev_idx = route_indices[-1]
                        next_idx = None  # No need to return to start
                    else:
                        prev_idx = route_indices[pos-1]
                        next_idx = route_indices[pos]
                    
                    # Calculate time increase
                    if next_idx is None:
                        # For insertion at the end, we only add time from previous to new point
                        old_time = 0  # No previous edge at the end
                        new_time = travel_time[prev_idx][i] + visit_cost_minutes
                    else:
                        old_time = travel_time[prev_idx][next_idx] if pos > 0 else 0
                        new_time = travel_time[prev_idx][i] + travel_time[i][next_idx] + visit_cost_minutes
                        
                    time_increase = new_time - old_time
                    
                    # Check if this is the best insertion
                    if time_increase < best_increase:
                        # Calculate the total time with this insertion
                        new_route = route_indices.copy()
                        new_route.insert(pos, i)
                        
                        # Calculate total route time
                        total_time = 0
                        for j in range(len(new_route) - 1):
                            total_time += travel_time[new_route[j]][new_route[j+1]]
                        
                        # Add visit time for each point (except start point)
                        total_time += (len(new_route) - 1) * visit_cost_minutes
                        
                        # Check if we're still within the time limit
                        if total_time <= time_limit_minutes:
                            best_increase = time_increase
                            best_insertion = (i, pos, total_time)
        
        # If no feasible insertion found, break
        if best_insertion is None:
            break
        
        # Insert the best point
        point_idx, pos, new_total_time = best_insertion
        route_indices.insert(pos, point_idx)
        current_time = new_total_time
    
    # Build the solution
    solution_order = [unvisited_points[idx]['name'] for idx in route_indices]
    
    # Build solution edges
    solution_edges = []
    for i in range(len(route_indices) - 1):
        solution_edges.append((route_indices[i], route_indices[i+1]))
    
    # Calculate estimated time (without return to start)
    estimated_time = 0
    for i, j in solution_edges:
        estimated_time += travel_time[i][j]
    estimated_time += (len(route_indices) - 1) * visit_cost_minutes  # No visit time for depot
    
    # Check if we actually have a feasible solution with at least one visit
    if len(solution_order) <= 1:  # Only start point
        return [unvisited_points[0]['name']], [], unvisited_points, 0

    # Print performance info
    end_time = time_module.time()
    execution_time = end_time - start_time
    print(f"Route planning completed in {execution_time:.4f} seconds. Visiting {len(solution_order)-1} points.")
    
    return solution_order, solution_edges, unvisited_points, estimated_time


def plan_route_enhanced_no_return(points, speed_kmph, time_limit_minutes, visited_points, visit_cost_minutes=15, 
                        time_buffer=0.05):
    """
    Enhanced route planning without return to start using a combination of greedy and randomized insertion.
    
    Args:
        points: List of point dictionaries
        speed_kmph: Speed in km/h
        time_limit_minutes: Time limit in minutes
        visited_points: Set of already visited point names
        visit_cost_minutes: Time at each point
        time_buffer: Fraction of time to reserve as buffer (0.05 = 5%)
        
    Returns:
        Similar to plan_route function
    """
    # Apply time buffer to allow for uncertainties
    effective_time_limit = time_limit_minutes * (1 - time_buffer)
    
    # Create subset of unvisited points
    unvisited_indices = {}
    unvisited_points = []
    
    # Add start point
    unvisited_points.append(points[0])
    unvisited_indices[0] = 0
    
    # Add unvisited points
    new_idx = 1
    for i, point in enumerate(points):
        if i > 0 and point['name'] not in visited_points:
            unvisited_points.append(point)
            unvisited_indices[i] = new_idx
            new_idx += 1
    
    N_unvisited = len(unvisited_points)
    
    # Compute travel time matrix
    travel_time = np.zeros((N_unvisited, N_unvisited))
    for i in range(N_unvisited):
        for j in range(N_unvisited):
            if i != j:
                dist = haversine(unvisited_points[i]['longitude'], unvisited_points[i]['latitude'],
                               unvisited_points[j]['longitude'], unvisited_points[j]['latitude'])
                travel_time_val = (dist / speed_kmph) * 60
                travel_time[i][j] = travel_time_val
    
    # Try multiple starting points and keep the best solution
    best_route = []
    best_time = 0
    
    # Find k nearest neighbors to start point (k scales with problem size)
    k = min(5, N_unvisited - 1)
    if k > 0:
        nearest_to_start = np.argsort(travel_time[0, 1:])[:k] + 1  # +1 because we skipped index 0
    else:
        nearest_to_start = []
        
    # Try each neighbor as the first visit
    candidates = [0] + list(nearest_to_start)  # Include start point too
    
    for start_candidate in candidates:
        # Initialize route with start point and first visit
        if start_candidate == 0:
            # Just start at depot with empty route
            route_indices = [0]
        else:
            route_indices = [0, start_candidate]
        
        # Greedy insertion phase
        while True:
            best_insertion = None
            best_score = float('-inf')  # We'll use a score-based approach
            
            # For each unvisited point
            for i in range(1, N_unvisited):
                if i not in route_indices:
                    # Try to insert at each position
                    for pos in range(1, len(route_indices) + 1):  # Don't insert before start
                        # Calculate insertion positions
                        if pos == len(route_indices):
                            prev_idx = route_indices[-1]
                            next_idx = None  # No need to return to start
                        else:
                            prev_idx = route_indices[pos-1]
                            next_idx = route_indices[pos]
                        
                        # Calculate time increase
                        if next_idx is None:
                            # For insertion at the end, we only add time from previous to new point
                            old_time = 0
                            new_time = travel_time[prev_idx][i] + visit_cost_minutes
                        else:
                            old_time = travel_time[prev_idx][next_idx]
                            new_time = travel_time[prev_idx][i] + travel_time[i][next_idx] + visit_cost_minutes
                            
                        time_increase = new_time - old_time
                        
                        # Calculate the score (favoring points that are closer)
                        score = -time_increase
                        
                        # Check if this insertion is better
                        if score > best_score:
                            # Verify time constraint
                            new_route = route_indices.copy()
                            new_route.insert(pos, i)
                            
                            # Calculate total route time
                            total_time = 0
                            for j in range(len(new_route) - 1):
                                total_time += travel_time[new_route[j]][new_route[j+1]]
                            
                            # Add visit times
                            total_time += (len(new_route) - 1) * visit_cost_minutes
                            
                            if total_time <= effective_time_limit:
                                best_score = score
                                best_insertion = (i, pos, total_time)
            
            # If no feasible insertion found, break
            if best_insertion is None:
                break
            
            # Insert the best point
            point_idx, pos, new_total_time = best_insertion
            route_indices.insert(pos, point_idx)
        
        # Calculate route quality
        if len(route_indices) > 1:  # At least one visit
            current_time = 0
            for j in range(len(route_indices) - 1):
                current_time += travel_time[route_indices[j]][route_indices[j+1]]
            
            # Add visit times
            current_time += (len(route_indices) - 1) * visit_cost_minutes
            
            # Check if this route is better
            if len(route_indices) > len(best_route) or (len(route_indices) == len(best_route) and current_time < best_time):
                best_route = route_indices.copy()
                best_time = current_time
    
    # If no feasible route found, return minimal solution
    if not best_route or len(best_route) <= 1:
        return [unvisited_points[0]['name']], [], unvisited_points, 0
    
    # Build the final solution
    route_indices = best_route
    solution_order = [unvisited_points[idx]['name'] for idx in route_indices]
    
    # Build solution edges
    solution_edges = []
    for i in range(len(route_indices) - 1):
        solution_edges.append((route_indices[i], route_indices[i+1]))
    
    # Calculate estimated time (without return to start)
    estimated_time = 0
    for i, j in solution_edges:
        estimated_time += travel_time[i][j]
    estimated_time += (len(route_indices) - 1) * visit_cost_minutes
    
    return solution_order, solution_edges, unvisited_points, estimated_time


def extend_schedule_map(schedule_map: Dict[str, int], start_date: str = None) -> Dict[str, int]:
    """
    Extend the schedule map with additional days, ensuring weekends have 0 minutes
    and weekdays have 480 minutes.
    
    Args:
        schedule_map: Dictionary with date strings as keys and daily limits as values
        start_date: Starting date in 'YYYY-MM-DD' format. If None, uses the last date in schedule_map
        
    Returns:
        Extended schedule map
    """
    if not schedule_map:
        # If empty, start from today
        if start_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        # Find the last date in the schedule
        dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in schedule_map.keys()]
        current_date = max(dates) + timedelta(days=1)
    
    # Add one week of days
    extended_map = schedule_map.copy()
    for i in range(7):
        date_str = current_date.strftime('%Y-%m-%d')
        # Check if it's weekend (Friday=4, Saturday=5 in Python's weekday())
        if current_date.weekday() in [4, 5]:  # Friday and Saturday
            extended_map[date_str] = 0
        else:
            extended_map[date_str] = 480
        current_date += timedelta(days=1)
    
    return extended_map


def plan_multiple_days_flexible(points, schedule_map: Dict[str, int], speed_kmph):
    """
    Multi-day planning with flexible daily limits based on dates.
    Automatically extends schedule for weekends (0 minutes) and weekdays (480 minutes).
    
    Args:
        points: List of point dictionaries with 'name', 'latitude', 'longitude'
        schedule_map: Dictionary mapping date strings ('YYYY-MM-DD') to daily time limits in minutes
        speed_kmph: Speed in km/h
        
    Returns:
        all_routes: List of routes, each route is a list of point names
        all_edges: List of edge lists, each edge list is a list of (i,j) tuples
        all_estimates: List of estimated times for each route
        schedule_used: Dictionary mapping dates to route information
    """
    start_time = time_module.time()
    visited_points = set()
    all_routes = []
    all_edges = []
    all_estimates = []
    schedule_used = {}
    
    # Exclude start point from target count
    total_points = len(points) - 1
    
    print(f"Planning routes for {total_points} points with flexible daily schedule")
    print(f"Initial schedule provided for {len(schedule_map)} days")
    print(f"NOTE: Routes do NOT return to start point at end of day")
    
    # Sort dates to process them in order
    current_schedule = schedule_map.copy()
    
    while len(visited_points) < total_points:
        # Find the next available day with time > 0
        available_dates = [date for date, limit in current_schedule.items() 
                          if date not in schedule_used and limit > 0]
        
        if not available_dates:
            # Need to extend the schedule
            print("Extending schedule for additional days...")
            current_schedule = extend_schedule_map(current_schedule)
            available_dates = [date for date, limit in current_schedule.items() 
                              if date not in schedule_used and limit > 0]
            
            if not available_dates:
                print("Error: Unable to extend schedule properly")
                break
        
        # Get the earliest available date
        current_date = min(available_dates)
        daily_limit = current_schedule[current_date]
        
        # Convert date string to datetime for day name
        date_obj = datetime.strptime(current_date, '%Y-%m-%d')
        day_name = date_obj.strftime('%A')
        
        print(f"\n{current_date} ({day_name}): Planning for {total_points - len(visited_points)} remaining points (Limit: {daily_limit} min)")
        
        if daily_limit == 0:
            # Skip days with 0 limit
            schedule_used[current_date] = {
                'route': [points[0]['name']],  # Only start point
                'estimated_time': 0,
                'points_visited': 0,
                'day_name': day_name
            }
            print(f"Skipping {current_date} ({day_name}) - No time available")
            continue
        
        # Try both algorithms and use the better result
        route1, edges1, unvisited1, time1 = plan_route_no_return(
            points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit,
            visited_points=visited_points, visit_cost_minutes=15
        )
        
        route2, edges2, unvisited2, time2 = plan_route_enhanced_no_return(
            points, speed_kmph=speed_kmph, time_limit_minutes=daily_limit,
            visited_points=visited_points, visit_cost_minutes=15
        )
        
        # Choose the better route (more points or equal points with less time)
        if len(route1) > len(route2) or (len(route1) == len(route2) and time1 < time2):
            route, edges, unvisited_points, estimated_time = route1, edges1, unvisited1, time1
        else:
            route, edges, unvisited_points, estimated_time = route2, edges2, unvisited2, time2
        
        # Record points visited in this route (skip start point)
        points_visited_today = 0
        for point_name in route[1:]:
            if point_name != points[0]['name']:  # Skip depot
                visited_points.add(point_name)
                points_visited_today += 1
        
        all_routes.append(route)
        all_edges.append(edges)
        all_estimates.append(estimated_time)
        
        # Record schedule usage
        schedule_used[current_date] = {
            'route': route,
            'estimated_time': estimated_time,
            'points_visited': points_visited_today,
            'day_name': day_name
        }
        
        print(f"{current_date} ({day_name}) complete. Visited {points_visited_today} points. Time: {estimated_time:.1f} min")
        print(f"Visited so far: {len(visited_points)}/{total_points} points")
        
        # If no progress made, extend schedule and continue
        if points_visited_today == 0:
            print("Warning: No progress made today. Extending schedule...")
            current_schedule = extend_schedule_map(current_schedule)
    
    end_time = time_module.time()
    total_execution_time = end_time - start_time
    print(f"\nMulti-day planning completed in {total_execution_time:.4f} seconds")
    print(f"Total days used: {len([d for d in schedule_used.values() if d['points_visited'] > 0])}")
    
    return all_routes, all_edges, all_estimates, schedule_used


def calculate_flexible_statistics(all_routes, all_estimates, schedule_used, points):
    """Calculate enhanced statistics about the flexible routes."""
    if not all_routes:
        return {}
    
    # Basic statistics
    active_days = [d for d in schedule_used.values() if d['points_visited'] > 0]
    total_working_days = len(active_days)
    weekend_days = len([d for d in schedule_used.values() if d['day_name'] in ['Friday', 'Saturday']])
    
    stats = {
        "total_days_scheduled": len(schedule_used),
        "total_working_days": total_working_days,
        "weekend_days_in_schedule": weekend_days,
        "total_points_visited": sum(len(route) - 1 for route in all_routes),
        "avg_points_per_working_day": sum(d['points_visited'] for d in active_days) / total_working_days if total_working_days > 0 else 0,
        "avg_time_per_working_day": sum(d['estimated_time'] for d in active_days) / total_working_days if total_working_days > 0 else 0,
        "max_time_day": max(all_estimates) if all_estimates else 0,
        "min_time_day": min([t for t in all_estimates if t > 0]) if any(t > 0 for t in all_estimates) else 0,
        "total_travel_time": sum(all_estimates),
        "schedule_efficiency": (sum(d['points_visited'] for d in active_days) / len(points)) * 100 if points else 0
    }
    
    # Day breakdown
    day_breakdown = {}
    for date, info in schedule_used.items():
        day_breakdown[date] = {
            "day_name": info['day_name'],
            "points_visited": info['points_visited'],
            "estimated_time": info['estimated_time'],
            "route_length": len(info['route'])
        }
    
    stats["day_breakdown"] = day_breakdown
    
    return stats


def extend_schedule_map(schedule_map: Dict[str, int], start_date: str = None) -> Dict[str, int]:
    """
    Extend the schedule map with additional days, ensuring weekends have 0 minutes
    and weekdays have 480 minutes.
    
    Args:
        schedule_map: Dictionary with date strings as keys and daily limits as values
        start_date: Starting date in 'YYYY-MM-DD' format. If None, uses the last date in schedule_map
        
    Returns:
        Extended schedule map
    """
    if not schedule_map:
        if start_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        # Find the last date in the schedule
        dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in schedule_map.keys()]
        current_date = max(dates) + timedelta(days=1)
    
    # Add one week of days
    extended_map = schedule_map.copy()
    for i in range(7):
        date_str = current_date.strftime('%Y-%m-%d')
        # Check if it's weekend (Friday=4, Saturday=5 in Python's weekday())
        if current_date.weekday() in [4, 5]:  # Friday and Saturday
            extended_map[date_str] = 0
        else:
            extended_map[date_str] = 480
        current_date += timedelta(days=1)
    
    return extended_map
def create_schedule_from_deadline(start_date: str, deadline_date: str, 
                                weekday_minutes: int = 480, weekend_minutes: int = 0) -> Dict[str, int]:
    """
    Create a schedule mapping from start date to deadline date with default time limits.
    
    Args:
        start_date: Starting date in 'YYYY-MM-DD' format
        deadline_date: Deadline date in 'YYYY-MM-DD' format
        weekday_minutes: Default minutes for weekdays (Monday-Thursday, Sunday)
        weekend_minutes: Default minutes for weekends (Friday, Saturday)
        
    Returns:
        Dictionary mapping date strings to daily time limits
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        deadline = datetime.strptime(deadline_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. Error: {e}")
    
    if start > deadline:
        raise ValueError("Start date must be before or equal to deadline date")
    
    schedule_map = {}
    current_date = start
    
    while current_date <= deadline:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Check if it's weekend (Friday=4, Saturday=5 in Python's weekday())
        if current_date.weekday() in [4, 5]:  # Friday and Saturday
            schedule_map[date_str] = weekend_minutes
        else:
            schedule_map[date_str] = weekday_minutes
            
        current_date += timedelta(days=1)
    
    return schedule_map
def count_valid_days(start_date, end_date):
    count = 0
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() not in (4, 5):  # Skip Friday (4) and Saturday (5)
            count += 1
        current_date += timedelta(days=1)
    return count
