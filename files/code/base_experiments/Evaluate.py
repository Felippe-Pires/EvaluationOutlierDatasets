#!/usr/bin/env python
# coding: utf-8
"""
Refactored Evaluate.py

- Reorganized imports and helper functions
- Added English docstrings and inline comments
- Fixed minor bug (use self.df instead of global df inside Models.evaluate_dataset)
- Kept original behavior and external API unchanged
"""

from datetime import datetime
import os
import random
import math
import gc
import warnings
import traceback
import re
import platform
import functools
import multiprocessing.pool
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

from scipy.io import arff
import scipy.io
from numpy import load

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import kaleido
import cv2

# PyOD models and local helpers
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.knn import KNN
from pyod.models.cof import COF
from pyod.models.kde import KDE
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
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

# custom LOCI implementation (kept as in original project)
from script_aux.loci import run_loci, LOCI, LOCIMatrix

# suppress noisy warnings
warnings.filterwarnings("ignore")


# -------------------------
# Utility helpers
# -------------------------
def duration(start_time: datetime) -> None:
    """Print the elapsed time since start_time."""
    print("Duration: {}".format(datetime.now() - start_time))


def timeout(max_timeout: int):
    """
    Timeout decorator that runs the decorated function in a thread pool and
    raises a TimeoutError if it doesn't return within max_timeout seconds.
    """
    def timeout_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(func, args, kwargs)
            return async_result.get(max_timeout)
        return wrapper
    return timeout_decorator


def decrease_order(data: Dict[Any, int]) -> Dict[str, List[int]]:
    """
    Return a dictionary with keys converted to strings and values wrapped in lists,
    sorted by value descending.
    """
    return {str(k): [v] for k, v in sorted(data.items(), key=lambda item: -item[1])}


def save_csv_rows(file_path: str, rows: List[List[Any]]) -> None:
    """Append rows to a CSV-like file using ';' as separator. Create file with header if missing."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    is_new = not os.path.exists(file_path)
    mode = 'a'
    with open(file_path, mode, newline='') as f:
        for r in rows:
            f.write(';'.join([str(i) for i in r]) + '\n')


def get_file_by_dataframe(path: str = '') -> Optional[pd.DataFrame]:
    """Read a semicolon-separated CSV file into a DataFrame, return None if missing."""
    if os.path.exists(path):
        return pd.read_csv(path, sep=';')
    return None


# -------------------------
# File / execution record helpers
# -------------------------
def save_execution(records: List[List[Any]], path: str = '') -> None:
    """Save a list of execution summary rows to results/dataset_execution.csv (append)."""
    file = os.path.join(path, 'dataset_execution.csv')
    if not os.path.exists(file):
        # write header row first if creating file
        header = ['algorithm', 'parameter', 'dataset', 'time_execution (microseg)']
        records = [header] + records
    save_csv_rows(file, records)


def save_detail_execution(records: List[List[Any]], type_name: str, path: str = '') -> None:
    """Save detailed execution rows to results/{type}_detail_execution.csv (append)."""
    file = os.path.join(path, f'{type_name}_detail_execution.csv')
    if not os.path.exists(file):
        header = ['algorithm', 'parameter', 'point', 'index', 'correct', 'dataset', 'type', 'score', 'ranking']
        # deduplicate and sort for initial write
        records = [header] + sorted(list(set(records)), key=lambda x: (x[1], x[6], x[3]))
    save_csv_rows(file, records)


def save_control_execution(records: List[List[Any]], dataset: str, path: str = '') -> None:
    """Save control execution rows to results/control_execution.csv (append)."""
    file = os.path.join(path, 'control_execution.csv')
    if not os.path.exists(file):
        header = ['dataset', 'algorithm', 'count']
        records = [header] + records
    save_csv_rows(file, records)


def init_control(path: str = '') -> Optional[pd.DataFrame]:
    """Load the control_execution.csv if it exists and return DataFrame, otherwise None."""
    file = os.path.join(path, 'control_execution.csv')
    if os.path.exists(file):
        return pd.read_csv(file, sep=';')
    return None


def check_executed(dataset: str, algorithm: str, df_control: Optional[pd.DataFrame]) -> Dict:
    """
    If control dataframe contains a record for (dataset, algorithm) return the evaluated
    'count' data structure (eval'd). Otherwise return empty dict.
    """
    if df_control is not None:
        filtro = df_control[(df_control.dataset == dataset) & (df_control.algorithm == algorithm)]
        if len(filtro) > 0:
            val = filtro['count'].values[0]
            return eval(val) if val != '' and val is not None else {}
    return {}


# -------------------------
# Small dataframe helpers
# -------------------------
def limit_amount_outliers(df: pd.DataFrame, pct: float = 5.0) -> None:
    """
    Limit the number of rows labeled as outlier to `pct` percent by randomly dropping
    excess outlier rows in-place. Assumes outlier column contains 'yes'/'no'.
    """
    outliers = df[df['outlier'] == 'yes']
    num_outliers = len(outliers)
    num_rows = len(df)
    if num_rows == 0:
        return
    max_allowed = int(num_rows * pct / 100.0)
    if num_outliers <= max_allowed:
        df.reset_index(drop=True, inplace=True)
        return

    to_remove = set(random.sample(list(outliers.index), k=(num_outliers - max_allowed)))
    df.drop(index=to_remove, inplace=True)
    df.reset_index(drop=True, inplace=True)


def valores_faltantes(df: pd.DataFrame, ordem: bool = False) -> pd.DataFrame:
    """
    Ensure that combinations of 'param' and 'ponto' appear in the dataframe by
    adding rows with default values when missing. Returns a new DataFrame.
    """
    params = df['param'].unique()
    points = df['ponto'].unique()
    faltantes = []
    for pnt in points:
        for p in params:
            if len(df[(df['param'] == p) & (df['ponto'] == pnt)]) == 0:
                faltantes.append((p, pnt + 1, False, 'O'))
    df = pd.concat([df, pd.DataFrame(faltantes, columns=['param', 'ponto', 'acerto', 'tipo'])], ignore_index=True)
    df = df.sort_values(by=['ponto', 'param'], ascending=True)
    df = df.astype({'ponto': 'str'})
    df.reset_index(drop=True, inplace=True)
    return df


def remove_tick_partial(tick: str) -> str:
    """Short helper to clean ticks/names using regex (kept from original)."""
    return re.sub(r'([A-Za-z0-9_]*_)(v0[2-5])', r'\2', tick)


# -------------------------
# Plotting helpers (Plotly + OpenCV)
# -------------------------
def plot_result(result: Dict[int, int], title: str = '', tipo: str = '', nome_arquivo: Optional[str] = None) -> None:
    """
    Plot a simple line of counts by position using Plotly. If nome_arquivo is provided,
    save to results/{nome_arquivo}.png; otherwise show the figure.
    """
    data = decrease_order(result)
    order_data = []
    count = 0
    for key in data.keys():
        count += 1
        order_data.append((f"{count}º ({key})", data[key][0]))

    df_plot = pd.DataFrame(order_data, columns=['position (points)', 'counter'])
    fig = go.Figure(data=[go.Scatter(x=df_plot['position (points)'], y=df_plot['counter'], mode='lines', line_color='indigo')])
    fig.update_layout(title=title + ' ' + tipo, yaxis_title='counter', xaxis_title='position (points)')

    if nome_arquivo is None:
        fig.show()
    else:
        out = os.path.join('../results', f"{nome_arquivo}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.write_image(out)


def plot_scatter_model(df_disp: pd.DataFrame, model: str, tipo: str, parametro: str, nome_arquivo: Optional[str] = None) -> None:
    """
    Plot a scatter of model scores (point index vs score parameter).
    Marker color encodes correct/incorrect classification.
    """
    markercolor = df_disp['acerto'].apply(lambda x: 1 if x else 0)
    fig = go.Figure(data=[go.Scatter(x=df_disp['ponto'], y=df_disp['param'], mode='markers',
                                     marker=dict(color=markercolor, colorscale='Bluered_r', size=3),
                                     hovertext=df_disp['ponto'])],
                    layout_title_text=f"{tipo} ({model})")
    fig.update_layout(yaxis_title=parametro, xaxis_title='index point')

    if nome_arquivo is None:
        fig.show()
    else:
        out = os.path.join('../results', f"{nome_arquivo}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.write_image(out, engine="kaleido")


def plot_map_ground_truth(df: pd.DataFrame, title_prefix: str = 'Outlier', tipo_dataset: str = '', y_axix: Optional[str] = None,
                          is_binary: bool = True, nome_arquivo: Optional[str] = None, width: int = 800, height: int = 500) -> None:
    """
    Create a colored scatter map that shows points x datasets colored by result or percentage.
    df must contain columns: 'point', 'dataset', 'result' (for binary) or appropriate numeric 'result' (for percent).
    """
    df = df.copy()
    df['dataset'] = df['dataset'].apply(lambda x: x.replace('.arff', '').replace('.csv', ''))
    if is_binary:
        df['result'] = df['result'].apply(lambda x: 1 if x else 0)
        fig1 = px.scatter(df, x="point", y="dataset", color="result",
                          color_continuous_scale=[(0, "red"), (0.5, "green"), (1, "blue")],
                          width=width, height=height)
        title_text = f"{tipo_dataset.capitalize()} / {title_prefix.capitalize()} / Binary"
    else:
        fig1 = px.scatter(df, x="point", y="dataset", color="result",
                          color_continuous_scale=[(0, "red"), (0.5, "green"), (1, "blue")],
                          width=width, height=height)
        title_text = f"{tipo_dataset.capitalize()} / {title_prefix.capitalize()} / Percent"

    fig1.update_layout(title=dict(text=title_text, x=0.25),
                       coloraxis_colorbar=dict(title="Accuracy"),
                       coloraxis_showscale=False,
                       yaxis=dict(title='', type="category"),
                       xaxis=dict(title='', type="category"),
                       font=dict(family="Courier New, monospace", size=10))

    if nome_arquivo is None:
        fig1.show()
    else:
        out = os.path.join(r'..\..\results\instances_detected', f"{nome_arquivo}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            fig1.write_image(out, engine="kaleido")
        except Exception as e:
            print(f"Error saving image {out}: {e}")


def crop_plot(tipo_dataset: str, tipo_dado: str, tipo_plot: str) -> None:
    """Crop saved images in results/ to remove padding for map visualizations."""
    path = os.path.join('../results', f'{tipo_dataset}_{tipo_dado}_{tipo_plot}.png')
    if not os.path.exists(path):
        return
    img = cv2.imread(path)

    map_crop = {
        'literature': (20, 40),
        'odds': (25, 40),
        'semantic': (180, 80),
        'ADBench': (180, 80),
        'real': (180, 80),
    }

    left, right = map_crop.get(tipo_dataset, (0, 0))
    if tipo_dado == 'O':
        cropped_image = img[0:img.shape[0], left:img.shape[1] - right]
    else:
        cropped_image = img[0:img.shape[0], 0:img.shape[1]]
    cv2.imwrite(path, cropped_image)


def join_plot(tipo_dataset: str, tipo_plot: str) -> None:
    """Join the I and O plots horizontally into a single image."""
    crop_plot(tipo_dataset, 'O', tipo_plot)
    crop_plot(tipo_dataset, 'I', tipo_plot)

    path_I = os.path.join(r'..\..\results\instances_detected', f'{tipo_dataset}_I_{tipo_plot}.png')
    path_O = os.path.join(r'..\..\results\instances_detected', f'{tipo_dataset}_O_{tipo_plot}.png')
    if not (os.path.exists(path_I) and os.path.exists(path_O)):
        return

    img1 = cv2.imread(path_I)
    img2 = cv2.imread(path_O)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1, :3] = img1
    vis[:h2, w1:w1 + w2, :3] = img2
    out = os.path.join(r'..\..\results\instances_detected', f"join_{tipo_dataset}_{tipo_plot}.jpg")
    cv2.imwrite(out, vis)


# -------------------------
# Core Models / Evaluation
# -------------------------
class Models:
    """
    Encapsulates model configuration, execution and plotting for a single dataset.
    The dataset DataFrame must have an 'outlier' column with 'yes'/'no' values and
    features in all other columns.
    """

    def __init__(self, df: Optional[pd.DataFrame], dataset_name: Optional[str]):
        self.df = df.copy(deep=True) if df is not None else None
        self.dataset_name = dataset_name
        self.plot = True
        if self.df is not None:
            # initialize parameters for model grids
            self.list_k, self.min_pts, self.r, self.bins, self.list_cblof = self.load_param_models(self.df)

            # ground truth indices
            self.ground_truth_outlier_index = self.df[self.df.outlier == 'yes'].index
            self.ground_truth_inlier_index = self.df[self.df.outlier == 'no'].index

            # feature matrix and binary labels
            self.X = self.df.values[:, :-1].astype(np.number)
            self.Y = self.df['outlier'].apply(lambda x: 0 if x == 'no' else 1).values[:]
            self.count_outliers = len(self.ground_truth_outlier_index)

            # configure models and hyperparameter grids
            self.modelos = {
                'KNN': {'alg': KNN, 'batch': 1, 'param': 'K', 'values': self.list_k},
                'LOF': {'alg': LocalOutlierFactor, 'batch': 1, 'param': 'MntPnt', 'values': self.min_pts},
                'KDE': {'alg': KDE, 'batch': 1, 'param': 'H', 'values': self.list_k},
                'COF': {'alg': COF, 'batch': 1, 'param': 'K', 'values': self.min_pts},
                'iForest': {'alg': IForest, 'batch': 1, 'param': 'nº iter', 'values': list(range(1, 11))},
                'INNE': {'alg': INNE, 'batch': 1, 'param': 'nº iter', 'values': list(range(1, 11))},
                'ABOD': {'alg': ABOD, 'batch': 1, 'param': 'nº iter', 'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                'OCSVM': {'alg': OCSVM, 'batch': 1, 'param': 'nu', 'values': [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5]},
                'ECOD': {'alg': ECOD, 'batch': 1, 'param': 'nº iter', 'values': [1]},
                'COPOD': {'alg': COPOD, 'batch': 1, 'param': 'nº iter', 'values': [1]},
                'HBOS': {'alg': HBOS, 'batch': 1, 'param': 'bins', 'values': self.bins},
                'SOS': {'alg': SOS, 'batch': 1, 'param': 'K', 'values': self.list_k},
                'PCA': {'alg': PCA, 'batch': 1, 'param': 'n_components', 'values': [0.05, 0.10, 0.20, 0.25, 0.35, 0.5, 0.6, 0.75, 0.90, None]},
                'MCD': {'alg': MCD, 'batch': 1, 'param': 'nº iter', 'values': [1]},
                'SOD': {'alg': SOD, 'batch': 1, 'param': 'K', 'values': [5, 10, 15, 20, 25, 30, 35, 40, 50, 55]},
                'ROD': {'alg': ROD, 'batch': 1, 'param': 'nº iter', 'values': [1]},
            }

            # storage for visualization and counters
            self.dispersao: Dict[str, List[Tuple]] = {}
            self.controle_outlier_index: Dict[str, Dict] = {}

    # -------------------------
    # Parameter grid helpers
    # -------------------------
    def get_list_values(self, start: int, stop: int, num_terms: int, integer: bool = True) -> List:
        """Return num_terms evenly spaced values between start and stop."""
        if integer:
            return [int(n) for n in list(np.linspace(start, stop, num_terms))]
        return [float(n) for n in list(np.linspace(start, stop, num_terms))]

    def load_param_models(self, df: pd.DataFrame):
        """
        Build reasonable hyperparameter grids for the dataset size.
        Returns: list_k, min_pts, r, bins, list_cblof
        """
        n = len(df)

        # K candidates for k-nearest methods
        k_max = max(2, round(n * 0.1))
        k_min = 1
        num_terms = 10
        list_k = [int(nv) for nv in list(np.linspace(k_min, k_max, num_terms))]
        if len(set(list_k)) != num_terms:
            interval = max(1, round((k_max - 1) / (num_terms - 1)))
            list_k = [x for x in range(k_min, k_max, interval)]
            if len(list_k) < num_terms:
                list_k.append(list_k[-1] + interval)

        # CBLOF grid
        k_max_cb = max(10, round(n * 0.1))
        k_min_cb = 9
        list_cblof = [int(nv) for nv in list(np.linspace(k_min_cb, k_max_cb, num_terms))]
        if len(set(list_cblof)) != num_terms:
            interval = max(1, round((k_max_cb - 1) / (num_terms - 1)))
            list_cblof = [x for x in range(k_min_cb, k_max_cb, interval)]
            if len(list_cblof) < num_terms:
                list_cblof.append(list_cblof[-1] + interval)

        # min_pts for LOF-like methods
        min_pts = [int(nv) for nv in list(np.linspace(10, 50, num_terms))]
        if len(set(min_pts)) != num_terms:
            interval = max(1, round((50 - 10) / num_terms))
            min_pts = [x for x in range(10, 50, interval)]
            if len(min_pts) < num_terms:
                min_pts.append(min_pts[-1] + interval)

        # r parameter grid
        r = [int(nv) for nv in list(np.linspace(1, 25, num_terms))]
        if len(set(r)) != num_terms:
            interval = max(1, round((25 - 1) / num_terms))
            r = list(range(1, 26, interval))[:num_terms]

        # bins for histogram-based detectors
        if n > 400:
            max_bin = int(math.sqrt(n))
            interval = max(1, round((max_bin - 10) / (num_terms - 1)))
            bins = [int(x) for x in range(10, max_bin, interval)] + [int(max_bin)]
        else:
            bins = list(range(10, 20))

        return list_k, min_pts, r, bins, list_cblof

    # -------------------------
    # Scoring / ranking helpers
    # -------------------------
    def ranking_scores(self, scores: List[float], element: int) -> int:
        """
        Return 1-based rank position of the score at index element within scores
        (sorted descending). Raises ValueError for out-of-range element.
        """
        if not (0 <= element < len(scores)):
            raise ValueError("Element index out of range")
        item = scores[element]
        lista_ordenada = sorted(scores, reverse=True)
        return lista_ordenada.index(item) + 1

    # -------------------------
    # Evaluation orchestration
    # -------------------------
    def resume_dataset(self) -> None:
        """Print a short summary of the dataset (counts and dimensions)."""
        if self.df is None:
            print("No dataset loaded")
            return
        num_regs = len(self.df)
        print(f"Dataset: {self.dataset_name}")
        print(f"Nº de Registros: {num_regs}")
        print(f"Nº de Dimensões: {len(self.df.columns) - 1} ({', '.join(list(self.df.columns[:-1]))})")
        num_outliers = len(self.df[self.df['outlier'] == 'yes'])
        print(f"Nº de Outliers: {num_outliers} ({round(num_outliers / max(1, num_regs) * 100, 2)}%)")
        print("\n")

    def resume_result(self, modelo: str, acerto_outlier: pd.DataFrame, acerto_inlier: pd.DataFrame) -> None:
        """Print a brief summary of results for a given model."""
        num_acerto_outliers = len(acerto_outlier[acerto_outlier['acerto']])
        num_acerto_inliers = len(acerto_inlier[acerto_inlier['acerto']])
        print(f"\tAlgoritmo: {modelo}")
        print(f"\tParâmetros: {self.modelos[modelo]['values']}")
        print(f"\tNº de Acertos de Outliers: {num_acerto_outliers} de {len(acerto_outlier)} ({round(num_acerto_outliers / max(1, len(acerto_outlier)) * 100, 2)}%)")
        print(f"\tNº de Acertos de Inliers: {num_acerto_inliers} de {len(acerto_inlier)} ({round(num_acerto_inliers / max(1, len(acerto_inlier)) * 100, 2)}%)")
        print("\n")

    def evaluate_dataset(self, tipo_dataset: str, path_output: str, df_control_global: Optional[pd.DataFrame] = None) -> None:
        """
        Evaluate all configured algorithms for this dataset. Results and plots are saved
        under the provided path_output (folder).
        """
        if self.df is None:
            return

        for key in self.modelos.keys():
            time_execution = []
            control_execution = []

            # initialize per-index counter
            contador = {i: 0 for i in range(len(self.df))}
            self.controle_outlier_index[key] = {}

            # check control file to skip executed combos
            contagem = check_executed(self.dataset_name, key, df_control_global)
            dispersao = {key: []}
            if len(contagem) == 0:
                print(key)
                try:
                    executions, contador, dispersao = self.execution_model(key, contador)
                    time_execution += executions
                    control_execution.append([self.dataset_name, key, contador])
                except TimeoutError:
                    print('Timeout')
                    print(f'Algorithm: {key}, Dataset: {self.dataset_name}')
                    time_execution += [(key, '', self.dataset_name, -1)]
            else:
                # reuse previous counts if control indicates done
                contador = contagem

            # prepare and save visualizations
            contadores_dataset = {key: decrease_order(contador)}
            self.plot_outlier_inlier(key, contadores_dataset)
            if self.plot:
                plot_result(contador, key, '(Outliers + Inliers)', nome_arquivo='_'.join([self.dataset_name or 'ds', key, '1']))

            datas = self.plot_color_map(key, dispersao)
            save_detail_execution(datas, tipo_dataset, path=path_output)

            # cleanup
            del contadores_dataset
            del dispersao
            del datas
            gc.collect()

            save_execution(time_execution, path=os.path.join('../results'))
            save_control_execution(control_execution, dataset=self.dataset_name, path=os.path.join('../results'))

    @timeout(86400)
    def execution_model(self, key: str, contador: Dict[int, float]) -> Tuple[List[Tuple], Dict[int, float], Dict[str, List[Tuple]]]:
        """
        Execute a single model 'key' across its parameter grid. Returns:
        (list of execution summaries, updated contador, dispersao entries)
        """
        executions: List[Tuple] = []
        dispersao = {key: []}

        for k in self.modelos[key]['values']:
            print(f"Parameter: {k}")
            param = k
            # adjust trivial k=1 for some methods
            if k == 1 and key not in ('iForest', 'ABOD'):
                k += 1

            for b in range(0, self.modelos[key]['batch']):
                start_time = datetime.now()

                # Instantiate and run the algorithm. Many branches follow original behavior.
                if key in ['iForest', 'INNE', 'ABOD', 'ECOD', 'COPOD', 'MCD', 'ROD']:
                    model = self.modelos[key]['alg']()
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                    probabilidade = getattr(model, "predict_proba", lambda X: None)(self.X)
                elif key in ['SO_GAAL', 'MO_GAAL', 'DeepSVDD']:
                    if key == 'DeepSVDD':
                        model = self.modelos[key]['alg'](epochs=20)
                    else:
                        model = self.modelos[key]['alg'](stop_epochs=5)
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                    probabilidade = getattr(model, "predict_proba", lambda X: None)(self.X)
                elif key == 'KDE':
                    model = self.modelos[key]['alg']()
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                elif key == 'LOF':
                    model = self.modelos[key]['alg'](n_neighbors=k)
                    model.fit_predict(self.X)
                    distances = model.negative_outlier_factor_
                elif key == 'HBOS':
                    model = self.modelos[key]['alg'](n_bins=k)
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                elif key == 'OCSVM':
                    model = self.modelos[key]['alg'](nu=k)
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                elif key == 'SOS':
                    model = self.modelos[key]['alg'](perplexity=k)
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                elif key == 'PCA':
                    model = self.modelos[key]['alg'](n_components=k)
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                elif key == 'SOD':
                    ref_set = 10
                    if k <= ref_set:
                        ref_set = k - 1
                    model = self.modelos[key]['alg'](n_neighbors=k, ref_set=ref_set)
                    model.fit_predict(self.X)
                    distances = model.decision_function(self.X)
                elif key == 'LOCI':
                    # LOCI uses a custom matrix approach in the project's script_aux
                    colunas = [c for c in self.df.columns if c != 'outlier']
                    numeros = [tuple(row[c] for c in colunas) for _, row in self.df.iterrows()]
                    self.X = np.array(numeros)
                    loci_i = LOCIMatrix(self.X, alpha=0.5, k=k)
                    max_distance = loci_i._get_max_distance(self.X)
                    loci_res = run_loci(self.X, max_dist=k / 100 * max_distance)
                    outlier_indices = loci_res.outlier_indices
                else:
                    # default nearest-neighbors style detectors
                    model = self.modelos[key]['alg'](n_neighbors=k)
                    model.fit(self.X)
                    distances = model.decision_function(self.X)

                # Aggregate distances to a single score per instance
                distance_largest = []
                if key in ('iForest', 'CD'):
                    # for probabilistic models, probability may be returned
                    distance_largest = [prob[1] for prob in probabilidade] if probabilidade is not None else []
                    top_outliers = sorted(distance_largest, reverse=True)[:self.count_outliers]
                elif key == 'LOCI':
                    # LOCI handled separately
                    top_outliers = []
                else:
                    for dist in distances:
                        # some detectors return arrays per instance; take max
                        distance_largest.append(dist.max() if hasattr(dist, '__iter__') else dist)
                    top_outliers = sorted(distance_largest, reverse=(key != 'LOF'))[:self.count_outliers]

                # determine predicted outlier/inlier indices
                if key == 'LOF':
                    outlier_index = np.where(np.array(distance_largest) <= max(top_outliers))[0]
                elif key == 'LOCI':
                    outlier_index = list(outlier_indices)
                else:
                    outlier_index = np.where(np.array(distance_largest) >= min(top_outliers))[0]

                # save the choice for later inspection
                self.controle_outlier_index.setdefault(key, {})[k] = outlier_index
                outlier_index_values = self.df.loc[outlier_index]

                # increment counters for correctly detected outliers
                for i, _ in outlier_index_values.iterrows():
                    if i in self.ground_truth_outlier_index:
                        contador[i] += float(1 / self.modelos[key]['batch'])

                # compute inliers similarly
                if key == 'LOF':
                    inlier_index = np.where(np.array(distance_largest) > max(top_outliers))[0]
                elif key == 'LOCI':
                    inlier_index = set(self.df.index).difference(set(outlier_indices))
                else:
                    inlier_index = np.where(np.array(distance_largest) < min(top_outliers))[0]

                inlier_index_values = self.df.loc[inlier_index]
                for i, _ in inlier_index_values.iterrows():
                    if i in self.ground_truth_inlier_index:
                        contador[i] += float(1 / self.modelos[key]['batch'])

                executions.append((key, k, self.dataset_name, (datetime.now() - start_time).microseconds))

            # Build dispersao entries for plotting (outliers and inliers)
            for o in outlier_index:
                if o in self.ground_truth_outlier_index:
                    dispersao[key].append((k, o, True, 'O', round(distance_largest[o], 5), self.ranking_scores(distance_largest, o)))
                else:
                    dispersao[key].append((k, o, False, 'I', round(distance_largest[o], 5), self.ranking_scores(distance_largest, o)))

            for o in inlier_index:
                if o in self.ground_truth_inlier_index:
                    dispersao[key].append((k, o, True, 'I', round(distance_largest[o], 5), self.ranking_scores(distance_largest, o)))
                else:
                    dispersao[key].append((k, o, False, 'O', round(distance_largest[o], 5), self.ranking_scores(distance_largest, o)))

        return executions, contador, dispersao

    # -------------------------
    # Plotting / aggregation helpers
    # -------------------------
    def plot_outlier_inlier(self, model: str, contadores_dataset: Dict[str, Dict[str, List[int]]]) -> None:
        """Compute and plot best/worst outliers and inliers for a given model."""
        if model not in contadores_dataset:
            return

        # convert values to simple counters
        contadores_dataset[model] = {i: contadores_dataset[model][i][0] for i in contadores_dataset[model].keys()}

        outliers_count = {i: contadores_dataset[model][str(i)] for i in self.ground_truth_outlier_index}
        inliers_count = {i: contadores_dataset[model][str(i)] for i in self.ground_truth_inlier_index}

        best_outliers = {k: v for k, v in sorted(outliers_count.items(), key=lambda item: -item[1]) if v > 0}
        best_inliers = {k: v for k, v in sorted(inliers_count.items(), key=lambda item: -item[1]) if v > 0}
        worst_outliers = {k: v for k, v in sorted(outliers_count.items(), key=lambda item: item[1]) if v == 0}
        worst_inliers = {k: v for k, v in sorted(inliers_count.items(), key=lambda item: item[1]) if v == 0}

        outliers_order = dict(best_outliers)
        outliers_order.update(worst_outliers)
        inliers_order = dict(best_inliers)
        inliers_order.update(worst_inliers)

        if self.plot:
            plot_result(outliers_order, model, '(Outliers)', nome_arquivo='_'.join([self.dataset_name or 'ds', model, '2']))
            plot_result(inliers_order, model, '(Inliers)', nome_arquivo='_'.join([self.dataset_name or 'ds', model, '3']))

    def plot_color_map(self, model: str, dispersao: Dict[str, List[Tuple]]) -> List[Tuple]:
        """
        Produce scatter data for outliers and inliers to be used by save_detail_execution.
        Returns a list of tuples representing rows to be saved.
        """
        datas: List[Tuple] = []
        if model not in dispersao:
            return datas

        # Outliers (ground-truth)
        outliers_points = []
        seq_out = self.ground_truth_outlier_index.tolist()
        for x in dispersao[model]:
            if x[1] in self.ground_truth_outlier_index:
                outliers_points.append((model, x[0], x[1] + 1, seq_out.index(x[1]) + 1, x[2], self.dataset_name, x[3], x[4], x[5]))
        df_disp_out = pd.DataFrame(outliers_points, columns=['alg', 'param', 'ponto', 'index', 'acerto', 'dataset', 'tipo', 'score', 'ranking'])
        df_disp_out = valores_faltantes(df_disp_out, True)
        datas += outliers_points
        if self.plot:
            plot_scatter_model(df_disp_out, model, 'Outliers', self.modelos[model]['param'], nome_arquivo='_'.join([self.dataset_name or 'ds', model, '4']))

        # Inliers (ground-truth)
        inliers_points = []
        seq_in = self.ground_truth_inlier_index.tolist()
        for x in dispersao[model]:
            if x[1] in self.ground_truth_inlier_index:
                inliers_points.append((model, x[0], x[1], seq_in.index(x[1]) + 1, x[2], self.dataset_name, x[3], x[4], x[5]))
        df_disp_in = pd.DataFrame(inliers_points, columns=['alg', 'param', 'ponto', 'index', 'acerto', 'dataset', 'tipo', 'score', 'ranking'])
        df_disp_in = valores_faltantes(df_disp_in, False)
        datas += inliers_points
        if self.plot:
            plot_scatter_model(df_disp_in, model, 'Inliers', self.modelos[model]['param'], nome_arquivo='_'.join([self.dataset_name or 'ds', model, '5']))

        if len(df_disp_out) > 0:
            self.resume_result(model, df_disp_out, df_disp_in)

        return datas
    
    def plot_binary_plot(self, dataset_type, point_type):
        df = get_file_by_dataframe(path=os.sep.join([r'..\..\results', '{}_detail_execution.csv'.format(dataset_type)]))
        if df is not None:
            # Separates into two lists to allow visualization of minority classes, because the majority class is printed first.
            list_correct = []
            list_incorrect = []
            datasets = df['dataset'].unique().tolist()
            datasets = sorted(datasets, key=lambda x: files_dataset[dataset_type].index(x) if x in files_dataset[dataset_type] else -1)
            for dataset in datasets:
                if point_type == 'O':
                    outliers = df[(df['dataset'] == dataset) & (df['type'] == 'O')]
                    num_outliers = outliers['index'].unique()
                    total = round(len(outliers[(outliers['correct']) & (outliers['dataset'] == dataset)]['index'].unique())/len(num_outliers)*100, 2)
                    total_str = '      (' + str(total) + '%)'
                    for out in num_outliers:
                        if len(outliers[(outliers['index'] == out) & (outliers['correct'])]) > 0:
                            list_correct.append([dataset + total_str, out, True])
                        else:
                            list_incorrect.append([dataset + total_str, out, False])
                if point_type == 'I':
                    inliers = df[(df['dataset'] == dataset) & (df['type'] == 'I')]
                    num_inliers = inliers['index'].unique()
                    total = round(len(inliers[(inliers['correct'])]['index'].unique()) / len(num_inliers) * 100, 2)
                    total_str = ' (' + str(total) + '%)'
                    for in_ in num_inliers:
                        if len(inliers[(inliers['index'] == in_) & (inliers['correct'])]) > 0:
                            list_correct.append([dataset + total_str, in_, True])
                        else:
                            list_incorrect.append([dataset + total_str, in_, False])

            list_result = []
            if len(list_correct) > len(list_incorrect):
                list_result = list_correct + list_incorrect
            else:
                list_result = list_incorrect + list_correct
            height = 700 + len(datasets) * 4
            plot_map_ground_truth(pd.DataFrame(list_result, columns=['dataset', 'point', 'result']),
                                 title_prefix='Outlier' if point_type == 'O' else 'Inlier' if point_type == 'I' else 'Outlier + Inlier',
                                 tipo_dataset=dataset_type.split('_')[0], y_axix='dataset',
                                 nome_arquivo=(dataset_type + '_' + point_type + '_' + 'binary').replace('.arff',
                                                                                                      '').replace(
                                     '.csv', ''), is_binary=True, height=height)

    def plot_percent_plot(self, dataset_type, point_type):
        df = get_file_by_dataframe(path=os.sep.join([r'..\..\results', '{}_detail_execution.csv'.format(dataset_type)]))
        if df is not None:
            count_list = []
            datasets = df['dataset'].unique().tolist()
            datasets = sorted(datasets, key=lambda x: files_dataset[dataset_type].index(x) if x in files_dataset[dataset_type] else -1)
            for dataset in datasets:
                if point_type == 'O':
                    outliers = df[(df['dataset'] == dataset) & (df['type'] == 'O')]
                    num_outliers = outliers['index'].unique()
                    total = round(len(outliers[(outliers['correct'])])/len(outliers)*100, 2)
                    total_str = '      (' + str(total) + '%)'
                    for out in num_outliers:
                        num_execution = outliers[(outliers['index'] == out)]
                        num_corrects = num_execution[num_execution['correct']]
                        count_list.append([dataset + total_str, out, len(num_corrects)/len(num_execution)])

                if point_type == 'I':
                    inliers = df[(df['dataset'] == dataset) & (df['type'] == 'I')]
                    num_inliers = inliers['index'].unique()
                    total = round(len(inliers[(inliers['correct'])]) / len(inliers) * 100, 2)
                    total_str = ' (' + str(total) + '%)'
                    for in_ in num_inliers:
                        num_execution = inliers[(inliers['index'] == in_)]
                        num_corrects = num_execution[num_execution['correct']]
                        count_list.append([dataset + total_str, in_, len(num_corrects) / len(num_execution)])

            height = 700 + len(datasets) * 4
            plot_map_ground_truth(pd.DataFrame(count_list, columns=['dataset', 'point', 'result']),
                                 title_prefix='Outlier' if point_type == 'O' else 'Inlier' if point_type == 'I' else 'Outlier + Inlier',
                                 tipo_dataset=dataset_type.split('_')[0], y_axix='dataset',
                                 nome_arquivo=(dataset_type + '_' + point_type + '_' + 'percent').replace('.arff',
                                                                                                      '').replace(
                                     '.csv', ''), is_binary=False, height=height)


# -------------------------
# Dataset configuration (unchanged, kept as constant)
# -------------------------
files_dataset = {
    'semantic': r'..\..\datasets\evaluation\processed\semantic',
    'literature': r'..\..\datasets\evaluation\processed\literature',
    'odds': r'..\..\datasets\evaluation\processed\odds',
    'ADBench': r'..\..\datasets\evaluation\processed\ADBench',
    'real': r'..\..\datasets\evaluation\processed\real',
}

# -------------------------
# Main execution loop (kept behavior from original script)
# -------------------------
all_dataset_binary: Dict = {}
all_dataset_percent: Dict = {}

list_files = [
    {'tipo': 'literature', 'files': os.listdir(files_dataset['literature'])},
    {'tipo': 'semantic', 'files': os.listdir(files_dataset['semantic'])},
    {'tipo': 'odds', 'files': os.listdir(files_dataset['odds'])},
    {'tipo': 'ADBench', 'files': os.listdir(files_dataset['ADBench'])},
    {'tipo': 'real', 'files': os.listdir(files_dataset['real'])},
]

# load control file if exists
df_control = init_control(path='..\..\results')

for files in list_files:
    for file in files['files']:
        try:
            # load dataset (arff or csv)
            dataset_path = os.path.join(r'..\..\datasets\processed', files['tipo'], file)
            if file.lower().endswith('.csv'):
                df = pd.read_csv(dataset_path, sep=',')
            else:
                data = arff.loadarff(dataset_path)
                df = pd.DataFrame(data[0])
                # decode bytes if present in nominal columns
                for col in df.select_dtypes([object]).columns:
                    df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
                if 'outlier' in df.columns and df['outlier'].dtype == object:
                    df['outlier'] = df['outlier'].apply(lambda x: x if isinstance(x, str) else str(x))

            if 'id' in df.columns:
                df.drop(columns=['id'], inplace=True)

            # limit outliers to 5% as in original script
            limit_amount_outliers(df, pct=5)

            start_time = datetime.now()
            print(start_time)
            models = Models(df, file)
            models.resume_dataset()
            models.plot = False
            models.evaluate_dataset(files['tipo'], path_output='../results', df_control_global=df_control)
            duration(start_time)
        except Exception as e:
            print('ERROR processing file:', file)
            print(traceback.format_exc())
        finally:
            try:
                del models
            except Exception:
                pass
            gc.collect()
        print('----------------\n')

    # after finishing a dataset group, produce summary plots (original behavior)
    models = Models(None, None)
    # directory to save plots with instances detected
    os.makedirs(r'..\..\results\instances_detected', exist_ok=True)
    models.plot_binary_plot(files['tipo'], point_type='O')
    models.plot_binary_plot(files['tipo'], point_type='I')
    models.plot_percent_plot(files['tipo'], point_type='O')
    models.plot_percent_plot(files['tipo'], point_type='I')
    join_plot(files['tipo'], 'percent')
    join_plot(files['tipo'], 'binary')