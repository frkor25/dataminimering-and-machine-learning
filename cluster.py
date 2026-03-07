

def calculate_centroid(points: list) -> tuple:
    """
    Calculate the centroid of a cluster of points in n-dimensional space.

    Parameters:
    points (list of tuples): A list of coordinates representing the points in the cluster.
                            Works with any dimensionality (2D, 3D, 4D, etc.)

    Returns:
    tuple: The coordinates of the centroid with the same dimensionality as input.
    """
    if not points:
        return ()

    count_number_of_points = len(points)
    number_of_dimensions = len(points[0])
    
    # Calculate the mean for each dimension
    centroid = tuple(
        sum(point[dim] for point in points) / count_number_of_points
        for dim in range(number_of_dimensions)
    )

    return centroid

def calculate_medoid(points: list) -> tuple:
    """
    Calculate the medoid of a cluster of points in n-dimensional space.

    Parameters:
    points (list of tuples): A list of coordinates representing the points in the cluster.
                            Works with any dimensionality (2D, 3D, 4D, etc.)                       

    Returns:
    tuple: The coordinates of the medoid with the same dimensionality as input.
    """
    if not points:
        return ()

    # Calculate the distance from each point to every other point
    all_distances = []
    for i in range(len(points)):
        total_distance = sum(
            sum((points[i][dim] - points[j][dim]) ** 2 for dim in range(len(points[0])))
            for j in range(len(points))
        )
        all_distances.append((total_distance, points[i]))

    # Find the point with the smallest total distance to all other points
    min_distance = all_distances[0][0]
    medoid = all_distances[0][1]
    for distance, point in all_distances[1:]:
        if distance < min_distance:
            min_distance = distance
            medoid = point

    return medoid

def simplified_silhouette_coefficient(clusters: list) -> float:
    """
    Calculate the simplified silhouette coefficient for a clustering.

    Parameters:
    clusters (list): A list of clusters, where each cluster is a list of points.

    Returns:
    float: The average simplified silhouette score.
    """

    if not clusters:
        return 0

    # Calculate centroids of each cluster (can be substituted with calculate_medoid if desired)
    centroids = [calculate_centroid(cluster) for cluster in clusters]

    silhouettes = []

    for cluster_index in range(len(clusters)):
        cluster = clusters[cluster_index]

        for point in cluster:

            # Distance to own centroid (a)
            own_centroid = centroids[cluster_index]
            a = sum((point[dim] - own_centroid[dim]) ** 2 for dim in range(len(point)))

            # Distance to nearest other centroid (b)
            b = float("inf")
            for i in range(len(centroids)):
                if i != cluster_index:
                    distance = sum((point[dim] - centroids[i][dim]) ** 2 for dim in range(len(point)))
                    if distance < b:
                        b = distance

            # Calculate silhouette value
            s = (b - a) / max(a, b)
            silhouettes.append(s)

    return sum(silhouettes) / len(silhouettes)
