# https://stackoverflow.com/questions/69596239/how-to-avoid-memory-leak-when-dealing-with-kmeans-for-example-in-this-code-i-am - error kmeans
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import pyransac3d as pyrsc
from sklearn.cluster import KMeans, DBSCAN
import random
import math

# Set the number of threads for OpenMP
os.environ["OMP_NUM_THREADS"] = '2'

# Define the path to the folder containing the CSV files
csv_folder_path = "C:/Users/annae/PycharmProjects/Lab2_ofc"


def clustering(points, num_clusters):
    # Convert the points to a NumPy array
    clust = np.array(points)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, n_jobs=2).fit(clust)
    # Get the cluster labels and centroids
    lab = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return lab, centroids


def read_csv(file_name):
    # Construct the full path to the file
    path = os.path.join(csv_folder_path, file_name)
    # Open the file and create a CSV reader object
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Iterate over the rows in the file and yield a tuple of the x, y, and z coordinates
        for x_csv, y_csv, z_csv in reader:
            yield float(x_csv), float(y_csv), float(z_csv)


def load_cloud(file_name, offset=0):
    # Create an empty list to hold the points
    cloud = []
    # Iterate over the rows in the file and append a tuple of the x, y, and z coordinates to the list
    for x_csv, y_csv, z_csv in read_csv(file_name):
        cloud.append((x_csv, y_csv + offset, z_csv))
    return cloud


def load_cloud_points():
    # Create an empty list to hold the clouds
    clouds_from_csv = []
    # Load each of the three CSV files and append the resulting clouds to the list
    clouds_from_csv.extend(load_cloud('horizontal_surface.xyz'))
    clouds_from_csv.extend(load_cloud('vertical_surface.xyz'))
    clouds_from_csv.extend(load_cloud('cylinder_surface.xyz'))
    return clouds_from_csv


def ransac(points, minimum_dist=0.5, iterations=50):
    global a, b, c, d
    inliers = []
    outliers = []
    new_in = []
    new_out = []
    new_sum_dist = 0
    sum_dist = 0
    divider = iterations

    while iterations:
        iterations -= 1

        # select 3 random points
        x1 = random.choice(points)
        x2 = random.choice(points)
        x3 = random.choice(points)

        a = (x2[1] - x1[1]) * (x3[2] - x1[2]) - (x2[2] - x1[2]) * (x3[1] - x1[1])
        b = (x2[2] - x1[2]) * (x3[0] - x1[0]) - (x2[0] - x1[0]) * (x3[2] - x1[2])
        c = (x2[0] - x1[0]) * (x3[1] - x1[1]) - (x2[1] - x1[1]) * (x3[0] - x1[0])
        d = -(a * x1[0] + b * x1[1] + c * x1[2])

        divider = max(0.1, np.sqrt(a * a + b * b + c * c))  # calculate divider to distance calculations

        for point in points:
            # calculate the distance between point and plane
            distance = math.fabs(a * point[0] + b * point[1] + c * point[2] + d) / divider
            new_sum_dist = new_sum_dist + distance

            if distance <= minimum_dist:   # add the point to inliers if it's close enough to the plane
                new_in.append(point)
            else:
                new_out.append(point)

        # check if in these iterations we have more inliers than before if its true save results
        if len(new_in) > len(inliers):
            inliers.clear()
            outliers.clear()
            inliers = new_in
            outliers = new_out
            sum_dist = new_sum_dist

    distance_mean = sum_dist / divider
    return inliers, outliers, a, b, c, d, distance_mean


def print_plane_info(normal_x, normal_y, normal_z, distance_mean):
    print('Normal vector is:', end=' ')
    print(f'{normal_x} {normal_y} {normal_z}')
    print(f'Mean distance between points and plane: {distance_mean}')

    if distance_mean < 2:
        print('A plane', end=' ')
        if abs(normal_z) > abs(normal_x) and abs(normal_z) > abs(normal_y):
            print('is probably horizontal')
        else:
            print('is probably vertical')
    else:
        print('Not a plane')
    print('----------------------------------')


cloud_points = load_cloud_points()  # load previously generated point plane files
x, y, z = zip(*cloud_points)  # unzip 3 axis variables from cloud_points

plt.figure()  # plot cloud_points
plt.title('Points')
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# find disjoint point clouds using k-means algorithms for k=3
X = np.array(cloud_points)
clust = KMeans(n_init=10, n_clusters=3)
clust.fit(X)
index_pr = clust.predict(X)

yellow = index_pr == 0
blue = index_pr == 1
green = index_pr == 2

c1 = X[yellow]
c2 = X[blue]
c3 = X[green]

plt.figure()    # plot the results
plt.title('Kmeans')
ax = plt.axes(projection='3d')
ax.scatter3D(X[yellow, 0], X[yellow, 1], X[yellow, 2], color="yellow")
ax.scatter3D(X[blue, 0], X[blue, 1], X[blue, 2], color="blue")
ax.scatter3D(X[green, 0], X[green, 1], X[green, 2], color="green")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# A matching test using the RANSAC algorithm
pa = pb = pc = pd = 0
inliers_1, outliers_1, pa1, pb1, pc1, pd1, dist_mean1 = ransac(c1, 0.5, 50)
x_inliers1, y_inliers1, z_inliers1 = zip(*inliers_1)

inliers_2, outliers_2, pa2, pb2, pc2, pd2, dist_mean2 = ransac(c2, 0.5, 50)
x_inliers2, y_inliers2, z_inliers2 = zip(*inliers_2)

inliers_3, outliers_3, pa3, pb3, pc3, pd3, dist_mean3 = ransac(c3, 0.5, 50)
x_inliers3, y_inliers3, z_inliers3 = zip(*inliers_3)

plt.figure()
plt.title('RANSAC')
ax = plt.axes(projection='3d')
ax.scatter3D(x_inliers1, y_inliers1, z_inliers1, color='yellow')
ax.scatter3D(x_inliers2, y_inliers2, z_inliers2, color='blue')
ax.scatter3D(x_inliers3, y_inliers3, z_inliers3, color='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print_plane_info(pa1, pb1, pc1, dist_mean1)
print_plane_info(pa2, pb2, pc2, dist_mean2)
print_plane_info(pa3, pb3, pc3, dist_mean3)

# DBSCAN algorithm
DD = np.array(cloud_points)
dbscan = DBSCAN(eps=2, min_samples=3).fit(DD)
labels = dbscan.labels_

# translation of cluster index to color
yellow = labels == 0
blue = labels == 1
green = labels == 2

# separate each cloud to new variable
c1 = DD[yellow]
c2 = DD[blue]
c3 = DD[green]

# create meshgrid to plot a plane
mx = np.linspace(-20, 20, 10)
my = np.linspace(-20, 20, 10)
mx, my = np.meshgrid(mx, my)

p1 = pyrsc.Plane()
eq_1, in_1 = p1.fit(c1, thresh=0.05, minPoints=100, maxIteration=1000)
eq1 = eq_1[0] * mx + eq_1[1] * my + eq_1[2] + eq_1[3]

p2 = pyrsc.Plane()
eq_2, in_2 = p2.fit(c2, thresh=0.05, minPoints=100, maxIteration=1000)
eq2 = eq_2[0] * mx + eq_2[1] * my + eq_2[2] + eq_2[3]

p3 = pyrsc.Plane()
eq_3, in_3 = p3.fit(c3, thresh=0.05, minPoints=100, maxIteration=1000)
eq3 = eq_3[0] * mx + eq_3[1] * my + eq_3[2] + eq_3[3]

plt.figure()
plt.title('DBSCAN and pyransac')
ax = plt.axes(projection='3d')
ax.scatter3D(DD[yellow, 0], DD[yellow, 1], DD[yellow, 2], color="yellow")
ax.scatter3D(DD[blue, 0], DD[blue, 1], DD[blue, 2], color="blue")
ax.scatter3D(DD[green, 0], DD[green, 1], DD[green, 2], color="green")
ax.plot_surface(mx, my, eq1)
ax.plot_surface(mx, my, eq2)
ax.plot_surface(mx, my, eq3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
