import polars as pl
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class MLPerformanceEvaluator:
    def __init__(self, original_df: pl.DataFrame, anonymized_df: pl.DataFrame, target_column: str):
        """
        Инициализация оценщика производительности ML
        
        :param original_df: Исходный DataFrame
        :param anonymized_df: Обезличенный DataFrame
        :param target_column: Целевая переменная для ML оценки
        """
        self.original_df = original_df
        self.anonymized_df = anonymized_df
        self.target_column = target_column
        
        # Проверка данных
        # if original_df.columns != anonymized_df.columns:
        #     raise ValueError("Колонки в исходном и обезличенном датасетах не совпадают")
        if target_column not in original_df.columns:
            raise ValueError(f"Целевая колонка '{target_column}' отсутствует в датасете")
        if target_column not in anonymized_df.columns:
            raise ValueError(f"Целевая колонка '{target_column}' отсутствует в датасете")
        
        # Определяем типы признаков
        self.numeric_features_orig = self._get_numeric_features(part = 'orig')
        self.categorical_features_orig = self._get_categorical_features(part = 'orig')

        self.numeric_features_anon = self._get_numeric_features(part = 'anon')
        self.categorical_features_anon = self._get_categorical_features(part = 'anon')

    def _get_numeric_features(self, part : str = 'orig') -> list:
        """Возвращает список числовых признаков"""
        if part == 'orig':
            return [col for col in self.original_df.columns 
                    if col != self.target_column and 
                    self.original_df[col].dtype in (pl.Int64, pl.Float64)]
        else:
            return [col for col in self.anonymized_df.columns 
                    if col != self.target_column and 
                    self.anonymized_df[col].dtype in (pl.Int64, pl.Float64)]

    def _get_categorical_features(self, part : str = 'orig') -> list:
        """Возвращает список категориальных признаков"""
        if part == 'orig':
            return [col for col in self.original_df.columns 
                    if col != self.target_column and 
                    self.original_df[col].dtype in (pl.Utf8, pl.Categorical, pl.Boolean)]
        else:
            return [col for col in self.anonymized_df.columns 
                    if col != self.target_column and 
                    self.anonymized_df[col].dtype in (pl.Utf8, pl.Categorical, pl.Boolean)]

    def _create_preprocessor(self, part : str = 'orig') -> ColumnTransformer:
        """Создает предварительный обработчик для признаков"""
        transformers = []
        if part == 'orig':
            if self.numeric_features_orig:
                transformers.append(('num', StandardScaler(), self.numeric_features_orig))
                
            if self.categorical_features_orig:
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features_orig))
            #print(transformers)
        else:
            if self.numeric_features_anon:
                transformers.append(('num', StandardScaler(), self.numeric_features_anon))
                
            if self.categorical_features_anon:
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features_anon))
            #print(transformers)

        return ColumnTransformer(transformers)        

    def evaluate_ml_performance(self, model_type: str = "auto", test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Оценка влияния обезличивания на производительность ML-моделей
        
        :param model_type: Тип модели ('auto', 'logistic', 'linear', 'random_forest')
        :param test_size: Доля тестовых данных
        :param random_state: Seed для воспроизводимости
        :return: Результаты оценки
        """
        # Определяем тип задачи
        if model_type == "auto":
            if self.original_df[self.target_column].dtype in (pl.Boolean, pl.Categorical, pl.Utf8):
                model_type = "classification"
            else:
                model_type = "regression"
        
        # Создаем предварительный обработчик
        preprocessor_orig = self._create_preprocessor(part='orig')
        preprocessor_anon = self._create_preprocessor(part='anon')

        # Выбираем модель
        if model_type == "classification":
            model_orig = LogisticRegression(max_iter=1000, random_state=random_state)
            model_anon = LogisticRegression(max_iter=1000, random_state=random_state)
            metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
        elif model_type == "regression":
            model_orig = LinearRegression()
            model_anon = LinearRegression()
            metrics = ["r2", "mse"]
        elif model_type == "random_forest":
            if self.original_df[self.target_column].dtype in (pl.Int64, pl.Boolean, pl.Categorical, pl.Utf8):
                model_orig = RandomForestClassifier(random_state=random_state)
                model_anon = RandomForestClassifier(random_state=random_state)
                metrics = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1']
            else:
                model_orig = RandomForestRegressor(random_state=random_state)
                model_anon = RandomForestRegressor(random_state=random_state)
                metrics = ["r2", "mse"]
        else:
            raise ValueError("Неподдерживаемый тип модели. Допустимые значения: 'auto', 'classification', 'regression', 'random_forest'")
        
        # Создаем пайплайн с предобработкой и моделью
        pipeline_orig = Pipeline(steps=[
            ('preprocessor', preprocessor_orig),
            ('model', model_orig)
        ])
        pipeline_anon = Pipeline(steps=[
            ('preprocessor', preprocessor_anon),
            ('model', model_anon)
        ])

        # Подготовка данных
        X_orig = self.original_df.drop(self.target_column)
        y_orig = self.original_df[self.target_column].to_numpy()

        y_lbl_enc_orig = LabelEncoder()
        y_orig = y_lbl_enc_orig.fit_transform(y_orig)
        
        X_anon = self.anonymized_df.drop(self.target_column)
        y_anon = self.anonymized_df[self.target_column].to_numpy()

        y_lbl_enc_anon = LabelEncoder()
        y_anon = y_lbl_enc_anon.fit_transform(y_anon)
        
        # Разделение данных
        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X_orig, y_orig, test_size=test_size, random_state=random_state
        )
        
        X_anon_train, X_anon_test, y_anon_train, y_anon_test = train_test_split(
            X_anon, y_anon, test_size=test_size, random_state=random_state
        )
        
        # Обучение и оценка на исходных данных
        pipeline_orig.fit(X_orig_train, y_orig_train)
        orig_pred = pipeline_orig.predict(X_orig_test)
        #print(orig_pred)
        #print(y_orig_test)
        print("Обучение модели на исходных данных завершено")

        # Обучение и оценка на обезличенных данных
        pipeline_anon.fit(X_anon_train, y_anon_train)
        anon_pred = pipeline_anon.predict(X_anon_test)

        print("  Обучение модели на обезличенных данных завершено")
        # Расчет метрик
        results = {
            "model_type": model_type,
            "test_size": test_size,
            "random_state": random_state,
            "metrics": {}  
        } 

        for metric in metrics:
            match metric:
                case "accuracy": 
                    results["metrics"]["accuracy_orig"] = accuracy_score(y_orig_test, orig_pred)
                    results["metrics"]["accuracy_anon"] = accuracy_score(y_anon_test, anon_pred)
                case "precision": 
                    if len(np.unique(y_orig_test)) == 2:
                        results["metrics"]["precision_orig"] = precision_score(y_orig_test, orig_pred)
                        results["metrics"]["precision_anon"] = precision_score(y_anon_test, anon_pred)
                    else:
                        results["metrics"]["precision_orig"] = precision_score(y_orig_test, orig_pred, average='weighted')
                        results["metrics"]["precision_anon"] = precision_score(y_anon_test, anon_pred, average='weighted')
                case "recall": 
                    if len(np.unique(y_orig_test)) == 2:
                        results["metrics"]["recall_orig"] = recall_score(y_orig_test, orig_pred)
                        results["metrics"]["recall_anon"] = recall_score(y_anon_test, anon_pred)
                    else:
                        results["metrics"]["recall_orig"] = recall_score(y_orig_test, orig_pred, average='weighted')
                        results["metrics"]["recall_anon"] = recall_score(y_anon_test, anon_pred, average='weighted')
                case "roc_auc": 
                    # Для ROC AUC нужны вероятности классов
                    orig_proba = pipeline_orig.predict_proba(X_orig_test)
                    anon_proba = pipeline_anon.predict_proba(X_anon_test)
                    if len(np.unique(y_orig_test)) == 2:
                        results["metrics"]["roc_auc_orig"] = roc_auc_score(y_orig_test, orig_proba[:, 1])
                        results["metrics"]["roc_auc_anon"] = roc_auc_score(y_anon_test, anon_proba[:, 1])
                    else:
                        results["metrics"]["roc_auc_orig"] = roc_auc_score(y_orig_test, orig_proba, multi_class='ovr', average='weighted')
                        results["metrics"]["roc_auc_anon"] = roc_auc_score(y_anon_test, anon_proba, multi_class='ovr', average='weighted')
                case "f1": 
                    results["metrics"]["f1_orig"] = f1_score(y_orig_test, orig_pred, average='weighted')
                    results["metrics"]["f1_anon"] = f1_score(y_anon_test, anon_pred, average='weighted')
                case "r2": 
                    results["metrics"]["r2_orig"] = r2_score(y_orig_test, orig_pred)
                    results["metrics"]["r2_anon"] = r2_score(y_anon_test, anon_pred)
                case "mse": 
                    results["metrics"]["mse_orig"] = mean_squared_error(y_orig_test, orig_pred)
                    results["metrics"]["mse_anon"] = mean_squared_error(y_anon_test, anon_pred)
                case _: raise ValueError("Неподдерживаемая метрика")

            # if metric == "accuracy":
            #     results["metrics"]["accuracy_orig"] = accuracy_score(y_orig_test, orig_pred)
            #     results["metrics"]["accuracy_anon"] = accuracy_score(y_anon_test, anon_pred)
            # elif metric == "f1":
            #     results["metrics"]["f1_orig"] = f1_score(y_orig_test, orig_pred, average='weighted')
            #     results["metrics"]["f1_anon"] = f1_score(y_anon_test, anon_pred, average='weighted')
            # elif metric == "r2":
            #     results["metrics"]["r2_orig"] = r2_score(y_orig_test, orig_pred)
            #     results["metrics"]["r2_anon"] = r2_score(y_anon_test, anon_pred)
            # elif metric == "mse":
            #     results["metrics"]["mse_orig"] = mean_squared_error(y_orig_test, orig_pred)
            #     results["metrics"]["mse_anon"] = mean_squared_error(y_anon_test, anon_pred)
        
        # Расчет разницы в производительности
        for key in list(results["metrics"].keys()):
            if key.endswith("_orig"):
                base_key = key.replace("_orig", "")
                anon_key = f"{base_key}_anon"
                diff_key = f"{base_key}_diff"
                
                results["metrics"][diff_key] = abs(results["metrics"][key] - results["metrics"][anon_key])
        
        print(f"Тип задачи: {results['model_type']}")
        print("\nМетрики производительности:")
        for metric, value in results["metrics"].items():
            if metric.endswith("_orig"):
                print(f"  Оригинальные данные ({metric.replace('_orig', '')}): {value:.4f}")
            elif metric.endswith("_anon"):
                print(f"    Обезличенные данные ({metric.replace('_anon', '')}): {value:.4f}")
            elif metric.endswith("_diff"):
                print(f"  Разница ({metric.replace('_diff', '')}): {value:.4f}")

        return results
    
class ShowStatistic:
    """
    Класс для сравнения ключевых статистических характеристик
    исходного и обезличенного наборов данных.
    """
    def __init__(self, original_df: pl.DataFrame, anonymized_df: pl.DataFrame):
        self.original_df = original_df
        self.anonymized_df = anonymized_df

    def compare_statistics(self) -> pl.DataFrame:
        """
        Сравнивает по каждому признаку:
          - среднее значение (mean)
          - стандартное отклонение (std)
          - процент пропущенных значений (missing_rate)
          - количество уникальных значений (unique_count)
        Возвращает Polars DataFrame с результатами.
        """
        rows = []
        
        # Определяем все уникальные колонки из обоих DataFrame
        all_columns = sorted(list(set(self.original_df.columns) | set(self.anonymized_df.columns)))
        
        n_orig = self.original_df.height
        n_anon = self.anonymized_df.height # Возможно, размер может отличаться

        for col in all_columns:
            orig_series = self.original_df[col] if col in self.original_df.columns else None
            anon_series = self.anonymized_df[col] if col in self.anonymized_df.columns else None

            orig_mean = None
            anon_mean = None
            orig_std = None
            anon_std = None
            orig_missing = None
            anon_missing = None
            orig_unique = None
            anon_unique = None

            if orig_series is not None:
                # Среднее и стандартное отклонение — только для числовых признаков
                if orig_series.dtype in (pl.Int64, pl.Float64):
                    orig_mean = orig_series.mean()
                    orig_std = orig_series.std()
                # Доля пропусков
                orig_missing = orig_series.is_null().sum() / n_orig
                # Количество уникальных значений
                orig_unique = orig_series.n_unique()
            else:
                # Если колонка отсутствует в original_df, можно указать, что она удалена или NaN
                orig_mean = "N/A (removed)"
                orig_std = "N/A (removed)"
                orig_missing = "N/A (removed)"
                orig_unique = "N/A (removed)"

            if anon_series is not None:
                if anon_series.dtype in (pl.Int64, pl.Float64):
                    anon_mean = anon_series.mean()
                    anon_std = anon_series.std()
                anon_missing = anon_series.is_null().sum() / n_anon
                anon_unique = anon_series.n_unique()
            else:
                # Если колонка отсутствует в anonymized_df, это означает, что она была удалена
                anon_mean = "N/A (added)"
                anon_std = "N/A (added)"
                anon_missing = "N/A (added)"
                anon_unique = "N/A (added)"
            
            rows.append({
                "column": col,
                "orig_mean": orig_mean,
                "anon_mean": anon_mean,
                "orig_std": orig_std,
                "anon_std": anon_std,
                "orig_missing_rate": orig_missing,
                "anon_missing_rate": anon_missing,
                "orig_unique_count": orig_unique,
                "anon_unique_count": anon_unique,
            })

        return pl.DataFrame(rows)