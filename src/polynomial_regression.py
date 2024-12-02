import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the error handling for numpy to raise exceptions for better debugging
np.seterr(all='raise')

# Factor used for plotting range
factor = 2.0


class LinearModel:
    """
    A simple linear regression model that can fit polynomial features.
    """

    def __init__(self, theta=None):
        # Initialize the weights (theta) to None
        self.theta = theta

    def fit(self, X, y):
        """
        Fit the linear model to the data.

        Args:
            X: np.ndarray, shape (n_samples, n_features)
                The input features.
            y: np.ndarray, shape (n_samples,)
                The target values.
        """
        # Compute the optimal theta using the normal equation
        # theta = (X^T X)^{-1} X^T y -> comes from solving for theta the gradient of the cost function J
        self.theta = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        """
        Predict using the linear model.

        Args:
            X: np.ndarray, shape (n_samples, n_features)
                The input features.

        Returns:
            y_pred: np.ndarray, shape (n_samples,)
                The predicted values.
        """
        return X @ self.theta

    @staticmethod
    def create_polynomial_features(X, degree):
        """
        Create polynomial features up to a given degree.

        Args:
            X: np.ndarray, shape (n_samples, 1)
                The original input features.
            degree: int
                The degree of the polynomial.

        Returns:
            X_poly: np.ndarray, shape (n_samples, degree + 1)
                The polynomial features.
        """
        n_samples = X.shape[0]
        X_poly = np.ones((n_samples, degree + 1))
        for i in range(1, degree + 1):
            X_poly[:, i] = np.power(X[:, 0], i)
        return X_poly

    @staticmethod
    def create_sin_features(X, degree):
        """
        Create features that include a sine transformation and polynomial terms.

        Args:
            X: np.ndarray, shape (n_samples, 1)
                The original input features.
            degree: int
                The degree of the polynomial terms.

        Returns:
            X_features: np.ndarray, shape (n_samples, degree + 2)
                The features including sine and polynomial terms.
        """
        n_samples = X.shape[0]
        X_features = np.ones((n_samples, degree + 2))
        X_features[:, 0] = np.sin(X[:, 0])  # Sine term
        for i in range(1, degree + 1):
            # we skip column 1, because they're already 1s
            X_features[:, i + 1] = X[:, 0] ** i
        return X_features


def load_dataset(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path: str
            The path to the CSV file.

    Returns:
        X: np.ndarray, shape (n_samples, 1)
            The input features.
        y: np.ndarray, shape (n_samples,)
            The target values.
    """
    data = pd.read_csv(file_path)
    X = data[['x']].values
    y = data[['y']].values
    return X, y


def plot_model(X_train, y_train, degrees, use_sin=False, filename='plot.png'):
    """
    Fit models with different degrees and plot the results.

    Args:
        X_train: np.ndarray, shape (n_samples, 1)
            The training input features.
        y_train: np.ndarray, shape (n_samples,)
            The training target values.
        degrees: list of int
            The degrees of the polynomial features to use.
        use_sin: bool
            Whether to include a sine term in the features.
        filename: str
            The name of the file to save the plot.
    """
    # Generate points for plotting the prediction curves
    X_plot = np.linspace(-factor * np.pi, factor * np.pi, 1000).reshape(-1, 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training Data')

    for degree in degrees:
        model = LinearModel()
        if use_sin:
            # Create features including sine term
            X_train_features = model.create_sin_features(X_train, degree)
            X_plot_features = model.create_sin_features(X_plot, degree)
        else:
            # Create polynomial features
            X_train_features = model.create_polynomial_features(
                X_train, degree)
            X_plot_features = model.create_polynomial_features(X_plot, degree)

        # Fit the model and make predictions
        model.fit(X_train_features, y_train)
        y_plot = model.predict(X_plot_features)

        # Plot the model's predictions
        plt.plot(X_plot, y_plot, label=f'Degree {degree}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Regression' + (' with Sine Term' if use_sin else ''))
    plt.legend()
    plt.savefig(filename)
    plt.show()


def main():
    # Load the datasets
    X_train_large, y_train_large = load_dataset('train.csv')
    X_train_small, y_train_small = load_dataset('small.csv')

    # Degrees to test
    degrees = [1, 2, 3, 5, 10, 20]

    # Plot models on the large dataset without sine term
    plot_model(X_train_large, y_train_large, degrees,
               use_sin=False, filename='large_poly.png')

    # Plot models on the large dataset with sine term
    plot_model(X_train_large, y_train_large, degrees,
               use_sin=True, filename='large_sin.png')

    # Plot models on the small dataset without sine term
    plot_model(X_train_small, y_train_small, degrees,
               use_sin=False, filename='small_poly.png')

    # Plot models on the small dataset with sine term
    plot_model(X_train_small, y_train_small, degrees,
               use_sin=True, filename='small_sin.png')


if __name__ == '__main__':
    main()
