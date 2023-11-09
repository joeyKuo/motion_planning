"""
Probabilistic Road Map (PRM) Planner
author: Atsushi Sakai (@Atsushi_twi)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# parameter
N_SAMPLE = 1000  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length
global distance_array
distance_array=[]
global runtime_array
runtime_array=[]


show_animation = True


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," +\
               str(self.cost) + "," + str(self.parent_index)


def prm_planning(start_x, start_y, goal_x, goal_y,
                 obstacle_x_list, obstacle_y_list, robot_radius, *, rng=None):
    """
    Run probabilistic road map planning
    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: (Optional) Random generator
    :return:
    """
    global start_time
    start_time=time.time()
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)

    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,
                                       robot_radius,
                                       obstacle_x_list, obstacle_y_list,
                                       obstacle_kd_tree, rng)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    road_map = generate_road_map(sample_x, sample_y,
                                 robot_radius, obstacle_kd_tree)

    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Road map generation
    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):

        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    #  plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]
    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)

    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            end_time=time.time()
            
            runtime_array.append(end_time-start_time)
            #print("runt time: %f second" % (end_time-start_time))
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index
    
    distance=0   
    
    for i in range(1,len(rx)):
     dist_x=rx[i]-rx[i-1]
     dist_y=ry[i]-ry[i-1]
     distance=distance+math.sqrt(dist_x*dist_x+dist_y*dist_y)
    
    #print("distance"+str(distance))
    
    distance_array.append(distance)
    
    return rx, ry


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover
    
    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]            

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    max_x = max(ox)
    max_y = max(oy)
    min_x = min(ox)
    min_y = min(oy)

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        dist, index = obstacle_kd_tree.query([tx, ty])

        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def main(rng=None):
    print(__file__ + " start!!")

    # start and goal position
    sx = 0.0  # [m]
    sy = 0.0  # [m]
    gx = 6.0  # [m]
    gy = 10.0  # [m]
    robot_size = 0.8  # [m]

    ox = []
    oy = []

    for i in np.arange(-2,15,1):
        ox.append(i)
        oy.append(-2.0)
    for i in np.arange(-2,15,1):
        ox.append(15.1)
        oy.append(i)
    for i in np.arange(-2,15,1):
        ox.append(i)
        oy.append(15.1)
    for i in np.arange(-2,15,1):
        ox.append(-2.0)
        oy.append(i)

    xc = 5 #x-co of circle (center)
    yc = 5 #y-co of circle (center)
    r = 1 #radius of circle
    
    for i in range(360):
        if i<180:
         for j in np.arange(-math.sin(i),math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y)
        else:
            for j in np.arange(math.sin(i),-math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y) 
    
    xc = 3 #x-co of circle (center)
    yc = 6 #y-co of circle (center)
    r = 2 #radius of circle
    
    for i in range(360):
        if i<180:
         for j in np.arange(-math.sin(i),math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y)
        else:
            for j in np.arange(math.sin(i),-math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y) 

    xc = 3 #x-co of circle (center)
    yc = 8 #y-co of circle (center)
    r = 2 #radius of circle
    
    for i in range(360):
        if i<180:
         for j in np.arange(-math.sin(i),math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y)
        else:
            for j in np.arange(math.sin(i),-math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y) 

    xc = 3 #x-co of circle (center)
    yc = 10 #y-co of circle (center)
    r = 2 #radius of circle
    
    for i in range(360):
        if i<180:
         for j in np.arange(-math.sin(i),math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y)
        else:
            for j in np.arange(math.sin(i),-math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y) 

    xc = 7 #x-co of circle (center)
    yc = 5 #y-co of circle (center)
    r = 2 #radius of circle
    
    for i in range(360):
        if i<180:
         for j in np.arange(-math.sin(i),math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y)
        else:
            for j in np.arange(math.sin(i),-math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y) 

    xc = 9 #x-co of circle (center)
    yc = 5 #y-co of circle (center)
    r = 2 #radius of circle
    
    for i in range(360):
        if i<180:
         for j in np.arange(-math.sin(i),math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y)
        else:
            for j in np.arange(math.sin(i),-math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y) 

    xc = 8 #x-co of circle (center)
    yc = 10 #y-co of circle (center)
    r = 1 #radius of circle
    
    for i in range(360):
        if i<180:
         for j in np.arange(-math.sin(i),math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y)
        else:
            for j in np.arange(math.sin(i),-math.sin(i),0.1):
             y = yc + r*j
             x = xc+ r*math.cos(i)
             ox.append(x)
             oy.append(y) 

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_size, rng=rng)

    assert rx, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(1)
        #plt.show()
        #plt.clf()
        #plt.close('all')
    
    
        
    
    #print(distance_array)
    #print(runtime_array)
    
    

    



if __name__ == '__main__':
  for i in range(10):
    main()
    plt.clf()
    plt.cla()
    plt.close('all')
    print("travel distance is:" +str(distance_array))
    print("runtime is"+str(runtime_array)+"second")

