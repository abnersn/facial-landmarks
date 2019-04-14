import numpy as np


class ShapeModel:

    def __rotate(self, matrix, theta):
        """Rotates a shape by a given angle in radians along the coordinates
        system's center.

        Args:
            matrix: A numpy matrix that will be rotated.
            theta: The angle (in radians) of the rotation

        Returns:
            The rotated matrix
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])
        result = np.dot(rotation_matrix, np.transpose(matrix))
        return np.transpose(result)

    def __root_mean_square(self, matrix):
        """Calculates the root mean squared distance of a set of points

        Args:
            matrix: A numpy matrix that contains the points.

        Returns:
            A float with the root mean squared distance of the given matrix.
        """
        square = np.power(matrix, 2)
        square_mean = np.mean(square)
        return np.sqrt(square_mean)

    def __scale_rms(self, matrix):
        """Scales a given shape so that the root mean squared distance of its
        points be equal to 1.

        Args:
            matrix: A numpy matrix that will be scaled.

        Returns:
            The scaled matrix
        """
        rms = self.__root_mean_square(matrix)
        return matrix / rms

    def __translate_mean(self, shape):
        """Translates the shape to the center of the coordinates system.

        Args:
            shape: The shape to be translated.

        Returns:
            The translated shape.
        """
        mean = np.mean(shape, axis=0)
        return shape - mean

    def __find_theta(self, matrix_a, matrix_b):
        """Calculates the angle by which matrix_b should be rotated in order to
        minimize its squared distance in relation to matrix_a. This is the
        analytical solution for the rotation step of the Procrustes analysis.

        Args:
            matrix_a: A numpy matrix that contains a set of points.
            matrix_b: A numpy matrix that contains a set of points.

        Returns:
            The optimal angle in radians.
        """
        x_values_a = matrix_a[:, 0]
        x_values_b = matrix_b[:, 0]

        y_values_a = matrix_a[:, 1]
        y_values_b = matrix_b[:, 1]

        numerator = np.sum(y_values_a * x_values_b - x_values_a * y_values_b)
        denominator = np.sum(x_values_a * x_values_b + y_values_a * y_values_b)

        theta = np.arctan(numerator / denominator)
        return theta

    def __normalize(self, dataset):
        """Performs Procrustes analysis over a given dataset.

        Args: A dataset as a python dictionary in which keys are image filenames
        and values are its corresponding vector of tuples that represents annotated
        shapes.

        Returns: A new dictionary with image filenames as keys and a Procrustes
        transformed numpy array.
        """
        new_dataset = {}
        reference_key = next(iter(dataset))
        new_dataset[reference_key] = np.array(dataset[reference_key])
        new_dataset[reference_key] = self.__translate_mean(
            new_dataset[reference_key])
        new_dataset[reference_key] = self.__scale_rms(
            new_dataset[reference_key])

        for image_file, points_list in dataset.items():
            current_mean = self.__mean_of_faces(new_dataset)
            shape = np.array(points_list)
            translated_shape = self.__translate_mean(shape)
            scaled_shape = self.__scale_rms(translated_shape)
            found_theta = self.__find_theta(current_mean, shape)
            rotated_shape = self.__rotate(scaled_shape, found_theta)
            new_dataset[image_file] = rotated_shape

        return new_dataset

    def __mean_of_faces(self, dataset):
        """Calculates the average face of a dataset of shapes.

        Args:
            dataset: A dictionary containing the shapes of the faces.
        Returns:
            A set of points that forms the shape of the average face
        """
        shape = dataset[next(iter(dataset))].shape
        sum_of_samples = np.zeros(shape)
        for points in dataset.values():
            sample = np.array(points)
            sum_of_samples += sample
        return sum_of_samples / len(dataset)

    def __covariance(self, dataset):
        """Calculates the covariance matrix of the data.

        Args:
            dataset: A dictionary containing the shapes to which the covariance
                    will be calculated.
        Returns:
            The covariance matrix.
        """
        shape = dataset[next(iter(dataset))].shape
        sum_of_samples = np.zeros([shape[0], shape[0]])
        for points in dataset.values():
            sample = np.array(points)
            sum_of_samples += np.dot((sample - self.base_shape),
                                     np.transpose(sample - self.base_shape))
        return sum_of_samples / (len(dataset) - 1)

    def __train_model(self, dataset):
        """Performs the Principal Component Analysis over a
        high dimensional dataset.

        Args:
            dataset: A dictionary containing the data to which the PCA shall be
                    applied.
            number_of_params: Number of parameters (eigenvalues and eigenvectors)
                            desired for the output.
        Returns:
            The eigenvalues and eigenvectors of the covariance matrix that
            corresponds to the result of the PCA.
        """
        covariance_matrix = self.__covariance(dataset)

        e_values, e_vectors = np.linalg.eig(covariance_matrix)
        e_values = e_values[0:self.number_of_params]
        e_vectors = e_vectors[:, 0:self.number_of_params]

        return [e_vectors, e_values]

    def deform(self, parameters):
        return self.base_shape + np.dot(self.vectors, parameters)

    def retrieve_parameters(self, shape):
        difference = shape - self.base_shape
        return np.dot(self.vectors.T, difference)

    def retrieve_parameters_faulty(self, shape, falty_points):
        falty_base = np.delete(self.base_shape, falty_points, axis=0)
        faulty_vector = np.delete(self.vectors, falty_points, axis=0)
        difference = shape - falty_base
        return np.dot(faulty_vector.T, difference)

    def __init__(self, number_of_params, dataset):
        self.number_of_params = number_of_params
        self.base_shape = self.__mean_of_faces(dataset)
        [v, e] = self.__train_model(dataset)
        self.vectors = v
        self.eigen = e
