"""

This module evaluates multiple outlier detection algorithms on converted datasets
with various parameter configurations and generates comprehensive visualizations.

Key Features:
  - Support for 20+ outlier detection algorithms
  - Parameterized algorithm evaluation with timeout handling
  - Detailed execution tracking and metrics calculation
  - Interactive visualization generation with Plotly
  - Progress tracking and error resilience

Organization:
  1. Imports & Dependencies
  2. Configuration & Constants
  3. Utility Functions (Data Processing, I/O, Visualization)
  4. Models Evaluation Class
  5. Dataset Configuration
  6. Main Execution Pipeline
"""

from scipy.io import arff
import pandas as pd
import numpy as np
import warnings
import random
import traceback
from datetime import datetime
import os
import re
import platform
import signal
import multiprocessing.pool
import functools
import json
import math
import gc
import csv

from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import kaleido
import cv2

from pyod.models.knn import KNN
from pyod.models.cof import COF
from pyod.models.lof import LOF
from pyod.models.kde import KDE
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.abod import ABOD
from script_aux.loci import run_loci, LOCI, LOCIMatrix
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.sos import SOS
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.sod import SOD
from pyod.models.rod import ROD
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.deep_svdd import DeepSVDD

# Suppress all library warnings for cleaner output
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1: Configuration & Constants
# =============================================================================

# Maximum execution time in seconds (24 hours)
TIMEOUT_EXECUTION = 86400

# Default outlier rate for limiting outliers in dataset (percentage)
DEFAULT_OUTLIER_RATE = 5

# Number of batches to execute for each parameter
DEFAULT_BATCH_SIZE = 1


# =============================================================================
# SECTION 2: Utility Functions - Time & Execution
# =============================================================================

def get_execution_duration(start_time: datetime) -> str:
    """
    Calculate and print execution duration from start time.
    
    Args:
        start_time: datetime object representing start of execution
        
    Returns:
        String representation of duration (HH:MM:SS format)
    """
    end_time = datetime.now()
    duration = end_time - start_time
    print(f'Duration: {duration}')
    return str(duration)


def handler_timeout(signum, frame):
    """
    Signal handler for timeout interruption.
    
    Args:
        signum: Signal number
        frame: Current stack frame
        
    Raises:
        Exception: End of time timeout
    """
    print('Execution timeout!')
    raise Exception('end of time')


def timeout(max_timeout: int):
    """
    Timeout decorator using ThreadPool to enforce maximum execution time.
    
    Parameters in seconds. Raises TimeoutError if execution exceeds max_timeout.
    
    Args:
        max_timeout: Maximum execution time in seconds
        
    Returns:
        Decorator function
    """
    def timeout_decorator(item):
        """Wrap the original function with timeout logic"""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Execute function with timeout using ThreadPool"""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # Raises TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator


# =============================================================================
# SECTION 3: Utility Functions - Data Processing
# =============================================================================

def limit_outlier_count(df: pd.DataFrame, pct: float = DEFAULT_OUTLIER_RATE) -> None:
    """
    Limit outlier instances to specified percentage of dataset.
    
    Randomly removes outlier instances if percentage exceeds threshold.
    Modifies DataFrame in-place.
    
    Args:
        df: DataFrame with 'outlier' column containing 'yes'/'no' labels
        pct: Maximum outlier percentage allowed (default: 5%)
    """
    outliers = df[df['outlier'] == 'yes']
    count_outliers = len(outliers)
    index_removed = []
    index_outliers = list(outliers.index)
    num_instances = len(df)
    
    # Check if outlier percentage exceeds threshold
    if count_outliers / num_instances > pct / 100:
        ideal_num_outliers = int(num_instances * pct / 100)
        
        # Remove random outliers until target count reached
        while count_outliers > ideal_num_outliers:
            random_index = random.choice(index_outliers)
            if random_index not in index_removed:
                index_removed.append(random_index)
                df.drop(index=random_index, inplace=True)
                count_outliers -= 1
    
    df.reset_index(drop=True, inplace=True)


def order_by_descending_values(data: dict) -> dict:
    """
    Sort dictionary by values in descending order.
    
    Args:
        data: Dictionary with numeric values
        
    Returns:
        Dictionary sorted by values (descending) with values wrapped in lists
    """
    return {str(k): [v] for k, v in sorted(data.items(), key=lambda item: -item[1])}


def clean_dataset_name(dataset_name: str) -> str:
    """
    Clean dataset filename by removing extensions.
    
    Args:
        dataset_name: Raw dataset filename
        
    Returns:
        Cleaned dataset name without file extensions
    """
    return dataset_name.replace('.arff', '').replace('.csv', '')


# =============================================================================
# SECTION 4: Utility Functions - File I/O
# =============================================================================

def save_execution_metrics(datas: list, path: str = '') -> None:
    """
    Save execution time metrics to CSV file.
    
    Appends data to file or creates new file if doesn't exist.
    
    Args:
        datas: List of tuples (algorithm, parameter, dataset, time_microseconds)
        path: Output directory path
    """
    file = os.sep.join([path, 'dataset_execution.csv'])
    columns = ['algorithm', 'parameter', 'dataset', 'time_execution (microseg)']
    
    # Add header if file doesn't exist
    if not os.path.exists(file):
        datas = [columns] + datas
    
    # Append data to file
    with open(file, 'a') as writer:
        for d in datas:
            writer.write(';'.join([str(i) for i in d]) + '\n')
        writer.close()


def save_detailed_execution(datas: list, conversion_method: str, path: str = '') -> None:
    """
    Save detailed execution results (per-instance predictions) to CSV file.
    
    Args:
        datas: List of tuples with execution details
        conversion_method: Dataset conversion method name
        path: Output directory path
    """
    file = os.sep.join([path, f'{conversion_method}_detail_execution.csv'])
    columns = ['algorithm', 'parameter', 'point', 'index', 'correct', 
               'dataset', 'type', 'score', 'ranking']
    
    # Add header and remove duplicates if file doesn't exist
    if not os.path.exists(file):
        datas = [columns] + sorted(list(set(datas)), key=lambda x: (x[1], x[6], x[3]))
    
    # Append data to file
    with open(file, 'a') as writer:
        for d in datas:
            writer.write(';'.join([str(i) for i in d]) + '\n')
        writer.close()


def save_control_execution(datas: list, dataset: str, path: str = '') -> None:
    """
    Save execution control information (algorithm completion tracking) to CSV.
    
    Args:
        datas: List of tuples (dataset, algorithm, count_dict)
        dataset: Dataset name (for tracking purposes)
        path: Output directory path
    """
    file = os.sep.join([path, 'control_execution.csv'])
    columns = ['dataset', 'algorithm', 'count']
    
    # Add header if file doesn't exist
    if not os.path.exists(file):
        datas = [columns] + datas
    
    # Append data to file
    with open(file, 'a') as writer:
        for d in datas:
            writer.write(';'.join([str(i) for i in d]) + '\n')
        writer.close()


def load_control_execution(path: str = '') -> pd.DataFrame:
    """
    Load execution control information from CSV file.
    
    Args:
        path: Directory containing control_execution.csv
        
    Returns:
        DataFrame with control data or None if file doesn't exist
    """
    file = os.sep.join([path, 'dataset_execution.csv'])
    if os.path.exists(file):
        return pd.read_csv(file, sep=';')
    return None


def check_algorithm_executed(dataset: str, algorithm: str, 
                            execution_df: pd.DataFrame) -> dict:
    """
    Check if algorithm has been previously executed for a dataset.
    
    Args:
        dataset: Dataset name
        algorithm: Algorithm name
        execution_df: Control DataFrame from load_control_execution()
        
    Returns:
        Dictionary of execution counts or empty dict if not executed
    """
    if execution_df is not None:
        filtro = execution_df[(execution_df.dataset == dataset) & 
                           (execution_df.algorithm == algorithm)]
        if len(filtro) > 0:
            return True
    return False


def load_dataframe_from_file(path: str = '') -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        path: Full path to CSV file
        
    Returns:
        Loaded DataFrame or None if file doesn't exist
    """
    if os.path.exists(path):
        return pd.read_csv(path, sep=';')
    return None


# =============================================================================
# SECTION 5: Utility Functions - Visualization
# =============================================================================

def plot_line_result(result: dict, title: str = '', method_type: str = '', 
                    output_file: str = None) -> None:
    """
    Generate line plot showing algorithm performance ranking.
    
    Creates Plotly figure showing ordered results with positions ranked from best
    to worst performing.
    
    Args:
        result: Dictionary with performance metrics (values will be ordered)
        title: Plot title
        method_type: Method type label (e.g., 'Outliers', 'Inliers')
        output_file: Output PNG file path (if None, displays in browser)
    """
    # Sort results by descending values
    data = order_by_descending_values(result)
    
    # Create ranked labels
    count = 0
    order_data = []
    for key in data.keys():
        count += 1
        order_data.append((f"{count}º ({key})", data[key][0]))
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame(order_data, columns=['position (points)', 'counter'])
    
    # Create line plot
    fig = go.Figure(
        data=[
            go.Scatter(x=df_plot['position (points)'],
                      y=df_plot['counter'],
                      mode='lines',
                      line_color='indigo'),
        ],
    )
    
    fig.update_layout(
        title=f"{title} {method_type}",
        yaxis_title='counter',
        xaxis_title='position (points)',
        legend_title="Legend Title",
    )
    
    # Export or display
    if output_file is None:
        fig.show()
    else:
        fig.write_image(f"..\\..\\results\\conversion_methods\\{output_file}.png")


def plot_scatter_results(df_display: pd.DataFrame, model: str, 
                        method_type: str, parameter_name: str, 
                        output_file: str = None) -> None:
    """
    Generate scatter plot showing algorithm predictions per point and parameter.
    
    Color-codes by correctness (blue=correct, red=incorrect).
    
    Args:
        df_display: DataFrame with columns ['ponto', 'param', 'acerto']
        model: Algorithm name
        method_type: Type of analysis (e.g., 'Outliers', 'Inliers')
        parameter_name: Parameter being varied
        output_file: Output PNG file path (if None, displays in browser)
    """
    # Encode correctness as colors (1=correct, 0=incorrect)
    marker_color = df_display['acerto'].apply(lambda x: 1 if x else 0)
    
    # Create scatter plot
    fig = go.Figure(
        data=[go.Scatter(x=df_display['ponto'],
                        y=df_display['param'],
                        line=dict(width=0.02),
                        mode='markers',
                        hovertext=df_display['ponto'],
                        marker=dict(color=marker_color,
                                   opacity=1,
                                   colorscale='Bluered_r',
                                   size=3))],
        layout_title_text=f"{method_type} ({model})"
    )
    
    fig.update_layout(
        yaxis_title=parameter_name,
        xaxis_title='index point',
        legend_title="Legend Title",
    )
    
    # Export or display
    if output_file is None:
        fig.show()
    else:
        fig.write_image(f"..\\..\\results\\conversion_methods\\{output_file}.png", 
                       engine="kaleido")


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing parameter-point combinations with default (incorrect) values.
    
    Ensures complete matrix of all parameter values across all points.
    
    Args:
        df: DataFrame with columns ['param', 'ponto', 'acerto', 'tipo']
        
    Returns:
        Complete DataFrame with filled missing values
    """
    params = df['param'].unique()
    points = df['ponto'].unique()
    faltantes = []
    
    # Find missing combinations
    for pnt in points:
        for p in params:
            if len(df[(df['param'] == p) & (df['ponto'] == pnt)]) == 0:
                faltantes.append((p, pnt + 1, False, 'O'))
    
    # Concatenate missing values
    df = pd.concat([df, pd.DataFrame(faltantes, 
                                    columns=['param', 'ponto', 'acerto', 'tipo'])])
    
    # Sort and reset index
    df = df.sort_values(by=['ponto', 'param'], ascending=True)
    df = df.astype({'ponto': 'str'})
    df.reset_index(drop=True, inplace=True)
    
    return df


def plot_ground_truth_map(df: pd.DataFrame, title: str = 'Outlier', 
                         dataset_type: str = '', y_axis_col: str = None, 
                         is_binary: bool = True, output_file: str = None,
                         width: int = 800, height: int = 500) -> None:
    """
    Generate heatmap showing ground truth accuracy across points and datasets.
    
    Uses Plotly scatter plot with custom coloring to show per-instance accuracy.
    
    Args:
        df: DataFrame with columns ['point', 'result', 'dataset']
        title: Plot title
        dataset_type: Type of dataset (for labeling)
        y_axis_col: Column name for y-axis
        is_binary: Whether results are binary or continuous
        output_file: Output PNG file path
        width: Figure width in pixels
        height: Figure height in pixels
    """
    # Clean dataset names
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('.arff', '').replace('.csv', ''))
    
    # Convert result to numeric
    df['result'] = df['result'].apply(lambda x: 1 if x else 0)
    
    # Create scatter plot
    fig = px.scatter(df, x="point", y="dataset", color="result",
                    color_continuous_scale=[(0, "red"), (0.5, "green"), (1, "blue")],
                    width=width, height=height)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{dataset_type.capitalize()} / {title.capitalize()} / {'Binary' if is_binary else 'Percent'}",
            x=(0.25 if title == 'Outlier' else 0.1)
        ),
        coloraxis_colorbar=dict(title="Accuracy"),
        coloraxis_showscale=False,
        yaxis=dict(title='', type="category"),
        xaxis=dict(title='', type="category"),
        font=dict(family="Courier New, monospace", size=10)
    )
    
    # Export or display
    if output_file is None:
        fig.show()
    else:
        fig.write_image(f"..\\..\\results\\conversion_methods\\{output_file}.png", 
                       engine="kaleido")


def crop_image(dataset_type: str, data_type: str, plot_type: str) -> None:
    """
    Crop generated PNG images to remove whitespace.
    
    Args:
        dataset_type: Dataset type name
        data_type: Data type ('O' for outlier, 'I' for inlier)
        plot_type: Plot type identifier
    """
    img_path = f'..\\..\\results\\conversion_methods\\{dataset_type}_{data_type}_{plot_type}.png'
    img = cv2.imread(img_path)
    
    # Define crop margins by dataset type
    crop_map = {
        'synthetic': (5, 40),
        'literature': (20, 40),
        'odds': (25, 40),
        'semantic': (180, 80),
    }
    
    # Crop image
    if data_type == 'O':
        margins = crop_map.get(dataset_type, (0, 0))
        cropped_image = img[0:img.shape[0], margins[0]:img.shape[1] - margins[1]]
    else:
        cropped_image = img[0:img.shape[0], 0:img.shape[1]]
    
    # Save cropped image
    cv2.imwrite(img_path, cropped_image)


def join_images(dataset_type: str, plot_type: str) -> None:
    """
    Join two images (outlier and inlier) side by side horizontally.
    
    Args:
        dataset_type: Dataset type name
        plot_type: Plot type identifier
    """
    # Crop images first
    crop_image(dataset_type, 'O', plot_type)
    crop_image(dataset_type, 'I', plot_type)
    
    # Load images
    img1 = cv2.imread(f'..\\..\\results\\conversion_methods\\{dataset_type}_I_{plot_type}.png')
    img2 = cv2.imread(f'..\\..\\results\\conversion_methods\\{dataset_type}_O_{plot_type}.png')
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create combined image
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1, :3] = img1
    vis[:h2, w1:w1 + w2, :3] = img2
    
    # Save combined image
    cv2.imwrite(f"..\\..\\results\\conversion_methods\\join_{dataset_type}_{plot_type}.jpg", vis)


# =============================================================================
# SECTION 6: Models Evaluation Class
# =============================================================================

class OutlierDetectionModels:
    """
    Orchestrates evaluation of multiple outlier detection algorithms on datasets.
    
    Manages algorithm configuration, execution, result tracking, and visualization.
    """
    
    def __init__(self, df: pd.DataFrame, dataset_name: str, execution_df: pd.DataFrame = None):
        """
        Initialize Models evaluator with dataset and algorithm configuration.
        
        Args:
            df: Preprocessed DataFrame with features and 'outlier' column
            dataset_name: Name of the dataset
            execution_df: DataFrame for execution control tracking
        """
        self.df = df
        self.dataset_name = dataset_name
        self.plot = True
        self.execution_df = execution_df
        
        if df is not None and len(df) > 0:
            # Load parameter ranges for each algorithm
            self.list_k, self.min_pts, self.r, self.bins, self.list_cblof = \
                self._load_algorithm_parameters(df)
            
            # Extract ground truth indices
            self.ground_truth_outlier_index = df[df.outlier == 'yes'].index
            self.ground_truth_inlier_index = df[df.outlier == 'no'].index
            
            # Prepare feature matrix and labels
            self.X = df.values[:, :-1].astype(np.number)
            self.Y = df['outlier'].apply(lambda x: 0 if x == 'no' else 1).values[:]
            self.count_outliers = len(df[df.outlier == 'yes'])
            
            # Configure available algorithms and their parameters
            self._configure_algorithms()
            
            # Result storage
            self.dispersao = {}
            self.controle_outlier_index = {}
    
    def _load_algorithm_parameters(self, df: pd.DataFrame) -> tuple:
        """
        Load and compute optimal parameter ranges for all algorithms.
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            Tuple of (list_k, min_pts, r, bins, list_cblof) parameter ranges
        """
        n_samples = len(df)
        
        # K values for distance-based algorithms (1 to 10% of dataset)
        k_max = round(n_samples * 0.1)
        k_min = 1
        list_k = self._create_linspace_list(k_min, k_max, 10)
        
        # CBLOF cluster count (9 to 10% of dataset)
        k_max_cblof = round(n_samples * 0.1)
        k_min_cblof = 9
        if k_max_cblof < k_min_cblof:
            k_max_cblof = n_samples
        list_cblof = self._create_linspace_list(k_min_cblof, k_max_cblof, 10)
        
        # Minimum points for density-based algorithms (10 to 50)
        min_pts = self._create_linspace_list(10, 50, 10)
        
        # Radius values for LOCI (1 to 25)
        r = self._create_linspace_list(1, 25, 10)
        
        # Bins for HBOS
        min_bin = 10
        max_bin = math.sqrt(n_samples)
        if n_samples > 400:
            bins = self._create_linspace_list(min_bin, int(max_bin), 10)
        else:
            bins = list(range(min_bin, 20))
        
        return list_k, min_pts, r, bins, list_cblof
    
    def _create_linspace_list(self, start: float, stop: float, 
                             num_terms: int) -> list:
        """
        Create evenly spaced list of values (avoiding duplicates).
        
        Args:
            start: Starting value
            stop: Stopping value
            num_terms: Number of terms to generate
            
        Returns:
            List of evenly spaced integer values
        """
        values = [int(n) for n in list(np.linspace(start, stop, num_terms))]
        
        # Handle duplicate values
        if len(set(values)) != num_terms:
            interval = round((stop - start) / (num_terms - 1))
            if interval == 0:
                interval = 1
            values = [x for x in range(int(start), int(stop), interval)]
            if len(values) < num_terms and start != stop:
                values.append(values[-1] + interval)
        
        return values
    
    def _configure_algorithms(self) -> None:
        """Configure all available outlier detection algorithms."""
        self.modelos = {
            'KNN': {'alg': KNN, 'batch': 1, 'param': 'K', 'values': self.list_k},
            'LOF': {'alg': LocalOutlierFactor, 'batch': 1, 'param': 'MntPnt', 
                   'values': self.min_pts},
            'KDE': {'alg': KDE, 'batch': 1, 'param': 'H', 'values': self.list_k},
            'COF': {'alg': COF, 'batch': 1, 'param': 'K', 'values': self.min_pts},
            'iForest': {'alg': IForest, 'batch': 1, 'param': 'nº iter', 
                       'values': list(range(1, 11))},
            'INNE': {'alg': INNE, 'batch': 1, 'param': 'nº iter', 
                    'values': list(range(1, 11))},
            'ABOD': {'alg': ABOD, 'batch': 1, 'param': 'nº iter', 'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            'OCSVM': {'alg': OCSVM, 'batch': 1, 'param': 'nu',
                     'values': [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]},
            'ECOD': {'alg': ECOD, 'batch': 1, 'param': 'nº iter', 'values': [1]},
            'COPOD': {'alg': COPOD, 'batch': 1, 'param': 'nº iter', 'values': [1]},
            'HBOS': {'alg': HBOS, 'batch': 1, 'param': 'bins', 'values': self.bins},
            'SOS': {'alg': SOS, 'batch': 1, 'param': 'K', 'values': self.list_k},
            'PCA': {'alg': PCA, 'batch': 1, 'param': 'n_components',
                   'values': [0.05, 0.10, 0.20, 0.25, 0.35, 0.5, 0.6, 0.75, 0.90, None]},
            'MCD': {'alg': MCD, 'batch': 1, 'param': 'nº iter', 'values': [1]},
            'SOD': {'alg': SOD, 'batch': 1, 'param': 'K',
                   'values': [5, 10, 15, 20, 25, 30, 35, 40, 50, 55]},
            'CBLOF': {'alg': CBLOF, 'batch': 1, 'param': 'n_clusters',
                     'values': self.list_cblof},
            'LOCI': {'alg': LOCI, 'batch': 1, 'param': 'r', 'values': self.r},
            'VAE': {'alg': VAE, 'batch': 1, 'param': 'beta',
                   'values': self._create_linspace_list(1, 150, 10)},
            'MO_GAAL': {'alg': MO_GAAL, 'batch': 1, 'param': 'nº iter',
                       'values': list(range(1, 11))},
            'SO_GAAL': {'alg': SO_GAAL, 'batch': 1, 'param': 'nº iter',
                       'values': list(range(1, 11))},
        }
    
    def _get_ranking_position(self, scores: list, element_index: int) -> int:
        """
        Get ranking position of element in sorted scores (descending).
        
        Args:
            scores: List of numeric scores
            element_index: Index of element in original list
            
        Returns:
            Position in sorted (descending) list (1-indexed)
            
        Raises:
            ValueError: If element index is out of range
        """
        if not (0 <= element_index < len(scores)):
            raise ValueError("Position out of list range")
        
        # Get element value
        item = scores[element_index]
        
        # Sort scores in descending order
        sorted_scores = sorted(scores, reverse=True)
        
        # Return 1-indexed position
        return sorted_scores.index(item) + 1
    
    def print_dataset_summary(self) -> None:
        """Print statistical summary of dataset."""
        num_instances = len(self.df)
        num_features = len(self.df.columns) - 1
        num_outliers = len(self.df[self.df['outlier'] == 'yes'])
        
        print(f'Dataset: {self.dataset_name}')
        print(f'Number of Instances: {num_instances}')
        print(f'Number of Features: {num_features}')
        print(f'Number of Outliers: {num_outliers} ({round(num_outliers/num_instances*100, 2)}%)\n')
    
    def print_algorithm_summary(self, algorithm: str, accuracy_outlier: pd.DataFrame,
                               accuracy_inlier: pd.DataFrame) -> None:
        """
        Print performance summary for an algorithm.
        
        Args:
            algorithm: Algorithm name
            accuracy_outlier: DataFrame with outlier prediction accuracy
            accuracy_inlier: DataFrame with inlier prediction accuracy
        """
        num_correct_outliers = len(accuracy_outlier[accuracy_outlier['acerto']])
        num_correct_inliers = len(accuracy_inlier[accuracy_inlier['acerto']])
        
        print(f'\tAlgorithm: {algorithm}')
        print(f'\tParameters: {self.modelos[algorithm]["values"]}')
        print(f'\tOutlier Accuracy: {num_correct_outliers}/{len(accuracy_outlier)} '
              f'({round(num_correct_outliers/len(accuracy_outlier)*100, 2)}%)')
        print(f'\tInlier Accuracy: {num_correct_inliers}/{len(accuracy_inlier)} '
              f'({round(num_correct_inliers/len(accuracy_inlier)*100, 2)}%)\n')
    
    @timeout(TIMEOUT_EXECUTION)
    def _execute_algorithm(self, algorithm: str, 
                          contador: dict) -> tuple:
        """
        Execute algorithm with all parameter values and collect results.
        
        Args:
            algorithm: Algorithm name
            contador: Counter dictionary tracking correct predictions
            
        Returns:
            Tuple of (execution_times, contador, dispersao) results
        """
        execution_times = []
        dispersao = {algorithm: []}
        
        print(f"Executing: {algorithm}")
        
        # Iterate over parameter values
        for param_value in self.modelos[algorithm]['values']:
            print(f"  Parameter: {param_value}")
            
            for batch in range(self.modelos[algorithm]['batch']):
                start_time = datetime.now()
                
                # Fit model and get anomaly scores based on algorithm type
                distances = self._fit_and_score_model(algorithm, param_value)
                
                # Find top-k outliers based on scores
                outlier_index, inlier_index, distance_largest = \
                    self._identify_outliers_inliers(algorithm, distances)
                
                # Update counter with correct predictions
                for idx in outlier_index:
                    if idx in self.ground_truth_outlier_index:
                        contador[idx] += 1.0 / self.modelos[algorithm]['batch']
                
                for idx in inlier_index:
                    if idx in self.ground_truth_inlier_index:
                        contador[idx] += 1.0 / self.modelos[algorithm]['batch']
                
                # Record execution time
                execution_times.append((algorithm, param_value, self.dataset_name,
                                      (datetime.now() - start_time).microseconds))
                
                # Record detailed results for visualization
                for idx in outlier_index:
                    correct = idx in self.ground_truth_outlier_index
                    type_label = 'O' if idx in self.ground_truth_outlier_index else 'I'
                    ranking = self._get_ranking_position(distance_largest, idx)
                    dispersao[algorithm].append((param_value, idx, correct, type_label,
                                               round(distance_largest[idx], 5), ranking))
                
                for idx in inlier_index:
                    correct = idx in self.ground_truth_inlier_index
                    type_label = 'I' if idx in self.ground_truth_inlier_index else 'O'
                    ranking = self._get_ranking_position(distance_largest, idx)
                    dispersao[algorithm].append((param_value, idx, correct, type_label,
                                               round(distance_largest[idx], 5), ranking))
        
        return execution_times, contador, dispersao
    
    def _fit_and_score_model(self, algorithm: str, param_value) -> np.ndarray:
        """
        Fit algorithm model and return anomaly scores.
        
        Args:
            algorithm: Algorithm name
            param_value: Parameter value to use
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        # Models that don't require parameters
        if algorithm in ['iForest', 'INNE', 'ABOD', 'ECOD', 'COPOD', 'MCD', 'ROD']:
            model = self.modelos[algorithm]['alg']()
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # Deep learning models with specific configurations
        elif algorithm in ['SO_GAAL', 'MO_GAAL', 'DeepSVDD']:
            if algorithm == 'DeepSVDD':
                model = self.modelos[algorithm]['alg'](n_features=64, epochs=20)
            else:
                model = self.modelos[algorithm]['alg'](stop_epochs=5)
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # KDE without parameters
        elif algorithm == 'KDE':
            model = self.modelos[algorithm]['alg']()
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # LOF with n_neighbors parameter
        elif algorithm == 'LOF':
            model = self.modelos[algorithm]['alg'](n_neighbors=param_value)
            model.fit_predict(self.X)
            return model.negative_outlier_factor_
        
        # HBOS with n_bins parameter
        elif algorithm == 'HBOS':
            model = self.modelos[algorithm]['alg'](n_bins=param_value)
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # CBLOF with n_clusters parameter
        elif algorithm == 'CBLOF':
            model = self.modelos[algorithm]['alg'](n_clusters=param_value)
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # LOCI with custom implementation
        elif algorithm == 'LOCI':
            column_names = [c for c in self.df.columns if c != 'outlier']
            X_tuples = np.array([tuple(self.df.loc[i, column_names]) 
                               for i in range(len(self.df))])
            loci_i = LOCIMatrix(X_tuples, alpha=0.5, k=param_value)
            max_distance = loci_i._get_max_distance(X_tuples)
            return run_loci(X_tuples, max_dist=param_value/100 * max_distance).outlier_indices
        # LOCI with k parameter
        #elif algorithm == 'LOCI':
        #    model = self.modelos[algorithm]['alg'](k=param_value)
        #    model.fit_predict(self.X)
        #    return model.decision_function(self.X)
        
        # OCSVM with nu parameter
        elif algorithm == 'OCSVM':
            model = self.modelos[algorithm]['alg'](nu=param_value)
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # SOS with perplexity parameter
        elif algorithm == 'SOS':
            model = self.modelos[algorithm]['alg'](perplexity=param_value)
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # PCA with n_components parameter
        elif algorithm == 'PCA':
            model = self.modelos[algorithm]['alg'](n_components=param_value)
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # SOD with n_neighbors and ref_set parameters
        elif algorithm == 'SOD':
            ref_set = min(10, param_value - 1) if param_value > 10 else param_value - 1
            model = self.modelos[algorithm]['alg'](n_neighbors=param_value, 
                                                  ref_set=ref_set)
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # VAE with custom encoder/decoder architecture
        elif algorithm == 'VAE':
            encoder_size = self.X[0].size // 2 * 2
            encoder_neurons = [encoder_size]
            while encoder_size / 2 > 1:
                encoder_size = int(encoder_size / 2)
                encoder_neurons.append(encoder_size)
                if len(encoder_neurons) >= 3:
                    break
            while len(encoder_neurons) < 3:
                encoder_neurons.append(encoder_neurons[-1])
            
            decoder_neurons = list(reversed(encoder_neurons))
            model = self.modelos[algorithm]['alg'](
                encoder_neuron_list=encoder_neurons,
                decoder_neuron_list=decoder_neurons,
                epoch_num=20,
                verbose=0
            )
            model.fit_predict(self.X)
            return model.decision_function(self.X)
        
        # Default: distance-based algorithms with n_neighbors parameter
        else:
            param_value = max(2, param_value)  # Ensure minimum parameter value
            model = self.modelos[algorithm]['alg'](n_neighbors=param_value)
            model.fit(self.X)
            return model.decision_function(self.X)
    
    def _identify_outliers_inliers(self, algorithm: str, 
                                  distances: np.ndarray) -> tuple:
        """
        Identify top-k outliers and inliers based on anomaly scores.
        
        Args:
            algorithm: Algorithm name (affects score interpretation)
            distances: Array of anomaly scores
            
        Returns:
            Tuple of (outlier_indices, inlier_indices, distance_largest)
        """
        # Extract maximum distance per instance
        distance_largest = []
        
        if algorithm in ['iForest', 'CD']:
            # For forest methods, use probability predictions
            for dist in distances:
                distance_largest.append(dist[1] if isinstance(dist, (list, np.ndarray)) else dist)
            top_outliers = sorted(distance_largest, reverse=True)[:self.count_outliers]
        elif algorithm == 'LOCI':
            # LOCI returns indices directly
            return distances, set(self.df.index).difference(set(distances)), []
        else:
            # For most algorithms
            for dist in distances:
                distance_largest.append(dist)
            top_outliers = sorted(distance_largest, 
                                reverse=(algorithm != 'LOF'))[:self.count_outliers]
        
        # Identify outlier indices
        if algorithm == 'LOF':
            outlier_index = np.where(distance_largest <= max(top_outliers))[0]
        elif algorithm == 'LOCI':
            outlier_index = distances
        else:
            outlier_index = np.where(distance_largest >= min(top_outliers))[0]
        
        # Identify inlier indices
        if algorithm == 'LOF':
            inlier_index = np.where(distance_largest > max(top_outliers))[0]
        elif algorithm == 'LOCI':
            inlier_index = set(self.df.index).difference(set(distances))
        else:
            inlier_index = np.where(distance_largest < min(top_outliers))[0]
        
        return outlier_index, inlier_index, distance_largest
    
    def _plot_algorithm_results(self, algorithm: str, contador: dict,
                               dispersao: dict) -> None:
        """
        Generate and save visualizations for algorithm results.
        
        Args:
            algorithm: Algorithm name
            contador: Counter dictionary
            dispersao: Dispersao dictionary with detailed results
        """
        if not self.plot:
            return
        
        # Create summary by correctness
        contadores_dataset = {
            algorithm: {str(k): [v] for k, v in sorted(contador.items(), 
                                                       key=lambda x: -x[1])}
        }
        
        # Plot overall results
        plot_line_result(contador, algorithm, '(Outliers + Inliers)',
                        output_file=f"{self.dataset_name}_{algorithm}_1")
        
        # Plot by outlier/inlier
        self._plot_outlier_inlier_results(algorithm, contadores_dataset)
    
    def _plot_outlier_inlier_results(self, algorithm: str, 
                                    contadores_dataset: dict) -> None:
        """
        Generate separate plots for outlier and inlier results.
        
        Args:
            algorithm: Algorithm name
            contadores_dataset: Dictionary with ordered results
        """
        if algorithm not in contadores_dataset:
            return
        
        # Extract results
        results_dict = {int(i): contadores_dataset[algorithm][str(i)][0] 
                       for i in contadores_dataset[algorithm].keys()}
        
        # Separate outliers and inliers
        outliers_results = {i: results_dict[i] 
                           for i in self.ground_truth_outlier_index 
                           if i in results_dict}
        inliers_results = {i: results_dict[i] 
                          for i in self.ground_truth_inlier_index 
                          if i in results_dict}
        
        # Best and worst predictions
        best_outliers = {k: v for k, v in sorted(outliers_results.items(), 
                                                 key=lambda x: -x[1]) if v > 0}
        best_inliers = {k: v for k, v in sorted(inliers_results.items(), 
                                               key=lambda x: -x[1]) if v > 0}
        worst_outliers = {k: v for k, v in sorted(outliers_results.items(), 
                                                  key=lambda x: x[1]) if v == 0}
        worst_inliers = {k: v for k, v in sorted(inliers_results.items(), 
                                                 key=lambda x: x[1]) if v == 0}
        
        # Create ordered results
        outliers_order = {**best_outliers, **worst_outliers}
        inliers_order = {**best_inliers, **worst_inliers}
        
        # Plot results
        if self.plot:
            plot_line_result(outliers_order, algorithm, '(Outliers)',
                            output_file=f"{self.dataset_name}_{algorithm}_2")
            plot_line_result(inliers_order, algorithm, '(Inliers)',
                            output_file=f"{self.dataset_name}_{algorithm}_3")
    
    def _generate_detailed_plots(self, algorithm: str, 
                                dispersao: dict) -> list:
        """
        Generate detailed scatter plots showing accuracy per point and parameter.
        
        Args:
            algorithm: Algorithm name
            dispersao: Dictionary with detailed execution results
            
        Returns:
            List of detailed result tuples for CSV export
        """
        datas = []
        
        if algorithm not in dispersao:
            return datas
        
        # Process outliers
        outlier_results = []
        sequence_points = self.ground_truth_outlier_index.tolist()
        
        for result in dispersao[algorithm]:
            if result[1] in self.ground_truth_outlier_index:
                outlier_idx = sequence_points.index(result[1])
                outlier_results.append((algorithm, result[0], result[1] + 1, 
                                      outlier_idx + 1, result[2], 
                                      self.dataset_name, result[3], 
                                      result[4], result[5]))
        
        df_outliers = pd.DataFrame(outlier_results,
                                  columns=['alg', 'param', 'ponto', 'index', 
                                          'acerto', 'dataset', 'tipo', 'score', 'ranking'])
        df_outliers = fill_missing_values(df_outliers)
        datas.extend(outlier_results)
        
        if self.plot:
            plot_scatter_results(df_outliers, algorithm, 'Outliers',
                               self.modelos[algorithm]['param'],
                               output_file=f"{self.dataset_name}_{algorithm}_4")
        
        # Process inliers
        inlier_results = []
        sequence_points = self.ground_truth_inlier_index.tolist()
        
        for result in dispersao[algorithm]:
            if result[1] in self.ground_truth_inlier_index:
                inlier_idx = sequence_points.index(result[1])
                inlier_results.append((algorithm, result[0], result[1], 
                                     inlier_idx + 1, result[2], 
                                     self.dataset_name, result[3], 
                                     result[4], result[5]))
        
        df_inliers = pd.DataFrame(inlier_results,
                                 columns=['alg', 'param', 'ponto', 'index', 
                                         'acerto', 'dataset', 'tipo', 'score', 'ranking'])
        df_inliers = fill_missing_values(df_inliers)
        datas.extend(inlier_results)
        
        if self.plot and len(df_inliers) > 0:
            plot_scatter_results(df_inliers, algorithm, 'Inliers',
                               self.modelos[algorithm]['param'],
                               output_file=f"{self.dataset_name}_{algorithm}_5")
        
        return datas
    
    def evaluate_dataset(self, conversion_method: str, output_path: str) -> None:
        """
        Execute evaluation of all algorithms on the dataset.
        
        Main evaluation pipeline that:
        1. Iterates over all configured algorithms
        2. Executes each with timeout handling
        3. Collects results and generates visualizations
        4. Saves metrics and execution logs
        
        Args:
            conversion_method: Dataset conversion method name
            output_path: Directory for saving results
        """
        for algorithm in self.modelos.keys():
            execution_times = []
            control_execution = []
            
            # Initialize result counters
            contador = {i: 0 for i in range(len(self.df))}
            self.controle_outlier_index[algorithm] = {}
            
            print(f"Evaluating: {algorithm}")
            
            try:
                # Check if dataset already processed
                if check_algorithm_executed(self.dataset_name, algorithm, self.execution_df):
                    print(f"→ Skipping already processed dataset: {self.dataset_name} with {algorithm}")
                    continue
        
                # Execute algorithm with timeout
                executions, contador, dispersao = self._execute_algorithm(algorithm, contador)
                execution_times.extend(executions)
                control_execution.append([self.dataset_name, algorithm, contador])
                
            except TimeoutError:
                print(f'Timeout: {algorithm} on {self.dataset_name}')
                execution_times.append((algorithm, '', self.dataset_name, -1))
            except Exception as e:
                print(f'Error: {algorithm} on {self.dataset_name}')
                print(traceback.format_exc())
                continue
            
            # Generate visualizations and results
            self._plot_algorithm_results(algorithm, contador, dispersao)
            detailed_data = self._generate_detailed_plots(algorithm, dispersao)
            
            # Save results to files
            save_execution_metrics(execution_times, path=output_path)
            save_detailed_execution(detailed_data, conversion_method.split(os.sep)[-1], 
                                   path=output_path)
            save_control_execution(control_execution, self.dataset_name, path=output_path)
            
            # Cleanup
            del dispersao
            del detailed_data
            gc.collect()


# =============================================================================
# SECTION 7: Dataset Configuration
# =============================================================================

DATASET_GROUPS = {
    'binary': [
        'autism.csv',
        'autistic.csv',
        'banknote.csv',
        'blood_transfusion.csv',
        'cervical_cancer.csv',
        'connectionist.csv',
        'haberman.csv',
        'heart.csv',
        'kidney.csv',  
        'mammographic.csv',
        'musk.csv',
        'ozone.csv',
        'parkinson.csv',
        'phishing.csv',
        'rice.csv',
        'wbc.csv'
    ],
    'non_binary': [
        'cardiotocography.csv',
        'digits.csv',
        'ecoli.csv',
        'glass.csv',
        'heart_disease.csv',
        'hepatitis_c_egyptian.csv',
        'image_segmentation.csv',
        'landsat_satellite.csv',
        'letters.csv',
        'mice_protein.csv',
        'obesity.csv',
        'students_knowledge.csv',
        'vehicle_silhouettes.csv',
        'vertebral.csv',
        'waveform.csv',
        'wholesale_customer.csv',
        'wine.csv',
        'wine_quality.csv',
        'yeast.csv',
    ],
}


# =============================================================================
# SECTION 8: Main Execution Pipeline
# =============================================================================

def main():
    """
    Main execution pipeline.
    
    Orchestrates complete workflow:
    1. Load dataset configuration
    2. Iterate over dataset groups
    3. Execute algorithm evaluation
    4. Generate comprehensive results
    """
    print("\n" + "="*70)
    print("Outlier Detection Algorithm Evaluation Pipeline")
    print("="*70 + "\n")
    
    # Dataset processing configurations
    dataset_configs = [
        {'tipo': r'binary\converted\BIN', 'files': DATASET_GROUPS['binary']},
        #{'tipo': r'binary\converted\BINDOWN', 'files': DATASET_GROUPS['binary']},
        #{'tipo': r'non_binary\converted\EXC', 'files': DATASET_GROUPS['non_binary']},
        #{'tipo': r'non_binary\converted\EXCDOWN', 'files': DATASET_GROUPS['non_binary']},
        #{'tipo': r'non_binary\converted\GRO', 'files': DATASET_GROUPS['non_binary']},
        #{'tipo': r'non_binary\converted\GRODOWN', 'files': DATASET_GROUPS['non_binary']},
    ]
    
    # Process each configuration
    for config in dataset_configs:
        conversion_method = config['tipo']
        datasets_list = config['files']
        
        # Initialize control tracking
        execution_df = load_control_execution(
            path=f"..\\..\\results\\conversion_methods\\{conversion_method.replace('converted/', '')}"
        )
        
        # Load datasets from directory
        dataset_dir = os.path.join("..\\..\\datasets\\conversion_methods", conversion_method)
        
        if not os.path.exists(dataset_dir):
            print(f"⚠ Directory not found: {dataset_dir}")
            continue
        
        dataset_files = [f for f in os.listdir(dataset_dir) 
                        if f.endswith(('.csv', '.arff'))]
        
        # Process each dataset
        for dataset_file in dataset_files:            
            file_path = os.path.join(dataset_dir, dataset_file)
            
            print(f"\n{'='*70}")
            print(f"Processing: {dataset_file}")
            print(f"{'='*70}")
            
            start_time = datetime.now()
            
            try:
                # Load dataset (CSV or ARFF format)
                if dataset_file.endswith('.csv'):
                    df = pd.read_csv(file_path, sep=';')
                elif dataset_file.endswith('.arff'):
                    data = arff.loadarff(file_path)
                    df = pd.DataFrame(data[0])
                    df['outlier'] = df['outlier'].apply(lambda x: x.decode() 
                                                       if isinstance(x, bytes) else x)
                else:
                    continue
                
                # Remove ID column if present
                if 'id' in df.columns:
                    df.drop(columns=['id'], inplace=True)
                
                # Limit outlier count to default rate
                limit_outlier_count(df, pct=DEFAULT_OUTLIER_RATE)
                
                # Initialize and execute evaluation
                models = OutlierDetectionModels(df, dataset_file, execution_df)
                models.print_dataset_summary()
                models.plot = False  # Disable intermediate plotting
                
                # Evaluate all algorithms
                output_dir = f"..\\..\\results\\conversion_methods\\{conversion_method.replace('converted/', '')}"
                os.makedirs(output_dir, exist_ok=True)
                
                models.evaluate_dataset(conversion_method, output_path=output_dir)
                
                print(f"✓ Completed: {dataset_file}")
                
            except Exception as e:
                print(f"✗ Error processing {dataset_file}")
                print(traceback.format_exc())
            
            finally:
                get_execution_duration(start_time)
                if models is not None:
                    del models
                gc.collect()
    
    print("\n" + "="*70)
    print("Pipeline Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()