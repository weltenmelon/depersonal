import polars as pl
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Union, Tuple, Dict, List
import math

def generalize(df: pl.DataFrame, col_name: str, bins: list) -> pl.DataFrame:
    """Обобщает значения в интервалы."""
    labels = [f"{bins[i]+1}-{bins[i+1]}" for i in range(len(bins)-1)]
    labels = [f"<{bins[0]}"] + labels + [f">{bins[-1]}"]
    return df.with_columns(
        pl.col(col_name).cut(bins, labels=labels).cast(pl.Utf8)
    )

def suppress_rare_categories(df: pl.DataFrame, column: str, threshold: float = 0.05) -> pl.DataFrame:
    """Заменяет редкие категории на 'Other'."""
    #counts = df.group_by(column).agg(pl.len().alias("count"))
    counts = df[column].value_counts()
    min_count = threshold * df.height
    rare = counts.filter(pl.col("count") < min_count)[column].to_list()
    return df.with_columns(
        pl.when(pl.col(column).is_in(rare))
        .then(pl.lit("Other"))  # Используйте pl.lit для строкового литерала
        .otherwise(pl.col(column))
        .alias(column)
    )

def drop_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Удаляет указанный столбец из DataFrame (метод изменения состава).
    Внимание: Удаление необратимо без внешних данных!
    """
    if column not in df.columns:
        raise ValueError(f"Столбец {column} отсутствует в DataFrame")
    return df.drop(column)

class EnteringIdentifiers:
    def __init__(self, column: str):
        """
        Инициализирует объект для обезличивания данных.
        :param column: Название колонки для обезличивания.
        """
        self.original_column = column
        self.new_column = f"id_{uuid.uuid4().hex[:6]}"  # Уникальное имя новой колонки
        self.mapping_table: Optional[pl.DataFrame] = None

    def transform(
        self,
        df: pl.DataFrame,
        existing_mapping_path: Optional[Union[str, Path]] = None
    ) -> pl.DataFrame:
        """Обезличивает данные, сохраняя старые идентификаторы."""
        # Загрузка существующей таблицы
        existing_mapping = self._load_existing_mapping(existing_mapping_path)
        existing_map = (
            dict(zip(existing_mapping[self.original_column], existing_mapping["identifier"]))
            if existing_mapping is not None
            else {}
        )

        # Извлечение уникальных значений из новых данных
        new_unique = df.get_column(self.original_column).unique().to_list()
        #print("new_unique",new_unique)

        # Разделение на существующие и новые
        new_values = [value for value in new_unique if value not in existing_map]
        #print("new_values", new_values)
        # Генерация новых ID только для новых значений
        new_identifiers = [uuid.uuid4().hex for _ in new_values]
        #print("new_identifiers", new_identifiers)

        # Создание новой таблицы соответствий только для новых значений
        new_mapping_df = pl.DataFrame({
            "identifier": new_identifiers,
            self.original_column: new_values,
            "target_column": [self.new_column] * len(new_values)
        })

        # Объединение с существующей таблицей
        if existing_mapping is not None:
            updated_mapping = pl.concat([existing_mapping, new_mapping_df], how="diagonal").unique(
                subset=[self.original_column], keep="first"
            )
        else:
            updated_mapping = new_mapping_df

        # Сохранение таблицы
        self.mapping_table = updated_mapping

        # Замена значений в данных
        full_value_map = {**existing_map, **dict(zip(new_values, new_identifiers))}
        return df.with_columns(
            pl.col(self.original_column).replace(full_value_map)
        ).rename({self.original_column: self.new_column})

    def _load_existing_mapping(self, path: Optional[Union[str, Path]]) -> Optional[pl.DataFrame]:
        """Загружает существующую таблицу соответствий с валидацией"""
        if path is None:
            return None
            
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл {path} не найден")
            
        mapping = (
            pl.read_csv(path) if path.suffix == ".csv" else
            pl.read_parquet(path) if path.suffix == ".parquet" else
            pl.read_excel(path)
        )
        
        # Валидация структуры
        required_columns = {"identifier", "target_column", self.original_column}
        if not required_columns.issubset(mapping.columns):
            raise ValueError("Неверный формат таблицы соответствий")
            
        return mapping

    def save_mapping(self, path: Union[str, Path], mode: str = "append", format: str = "csv"):
        """Сохраняет таблицу с режимами перезаписи/добавления"""
        if self.mapping_table is None:
            raise ValueError("Сначала выполните transform()")

        path = Path(path)
        if mode == "append" and path.exists():
            existing = (
                pl.read_csv(path) if format == "csv" else
                pl.read_parquet(path) if format == "parquet" else
                pl.read_excel(path)
            )
            updated = pl.concat([existing, self.mapping_table]).unique(
                subset=[self.original_column], 
                keep="first"  # Сохраняем старые записи
            )
        else:
            updated = self.mapping_table

        # Сохранение в нужном формате
        if format == "csv":
            updated.write_csv(path)
        elif format == "parquet":
            updated.write_parquet(path)
        elif format == "excel":
            with pl.Config(tbl_rows=-1):
                self.mapping_table.write_excel(path)
        else:
            raise ValueError("Неподдерживаемый формат")

    @staticmethod
    def restore(
        df: pl.DataFrame,
        column: str,
        mapping_path: Union[str, Path],
        mapping_format: str = "csv"
    ) -> pl.DataFrame:
        # Загрузка таблицы соответствий

        match mapping_format:
            case "csv": mapping = pl.read_csv(mapping_path)
            case "parquet": mapping = pl.read_parquet(mapping_path)
            case "excel": mapping = pl.read_excel(mapping_path)
            case _: raise ValueError("Неподдерживаемый формат")
        
        # Проверка наличия колонки в DataFrame
        if column not in df.columns:
            raise ValueError(f"Колонка '{column}' отсутствует в DataFrame")
    
        # Проверка структуры таблицы
        original_col_name = EnteringIdentifiers._get_original_col_name(mapping)
        required_columns = {"identifier", "target_column",original_col_name}
        if not required_columns.issubset(mapping.columns):
            raise ValueError("Неверный формат таблицы соответствий")
        
        # Проверка соответствия target_column
        target_column_in_mapping = mapping["target_column"][0]  # Берём первое значение
        if target_column_in_mapping != column:
            raise ValueError(
                f"Колонка в таблице соответствий ({target_column_in_mapping}) "
                f"не совпадает с переданной колонкой ({column})"
            )
            
        # Создание словаря для замены
        reverse_map = dict(zip(mapping["identifier"], mapping[original_col_name]))
        
        # Восстановление данных
        return df.with_columns(
            pl.col(column).replace(reverse_map)
        ).rename({column: original_col_name})

    @staticmethod
    def _get_original_col_name(mapping_df: pl.DataFrame) -> str:
        return [col for col in mapping_df.columns if col not in {"identifier", "target_column"}][0]

class ShuffleTransformer:
    def __init__(self, column: str):
        self.column = column
        self.shuffle_seed: Optional[int] = None
        self.original_indices: Optional[pl.Series] = None

    def shuffle(self, df: pl.DataFrame, seed: Optional[int] = None) -> pl.DataFrame:
        """Перемешивает значения только в указанной колонке"""
        # Генерируем seed и сохраняем оригинальные индексы
        self.shuffle_seed = seed if seed is not None else np.random.randint(0, 2**32)
        self.original_indices = df.get_column(self.column).to_list()

        # Создаем копию колонки и перемешиваем
        shuffled_values = (
            df.get_column(self.column)
            .sample(fraction=1.0, shuffle=True, seed=self.shuffle_seed)
            .to_list()
        )

        return df.with_columns(
            pl.Series(name=self.column, values=shuffled_values)
        )

    def unshuffle(self, df: pl.DataFrame) -> pl.DataFrame:
        """Восстанавливает оригинальный порядок в колонке"""
        if self.shuffle_seed is None or self.original_indices is None:
            raise ValueError("Сначала выполните shuffle()")

        # Восстанавливаем оригинальные значения
        return df.with_columns(
            pl.Series(name=self.column, values=self.original_indices)
        )
    
    def _get_seed(self):
        return self.shuffle_seed

class DecomposeTransformer:
    def __init__(self, part1_columns: List[str], part2_columns: List[str]):
        self.part1_columns = part1_columns
        self.part2_columns = part2_columns
        self.part1_id = f"id_part1_{uuid.uuid4().hex[:6]}"
        self.part2_id = f"id_part2_{uuid.uuid4().hex[:6]}"
        self.mapping: Optional[pl.DataFrame] = None

    def decompose(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Декомпозирует данные на две части"""
        ids = [uuid.uuid4().hex for _ in range(df.height)]
        ids2 = [uuid.uuid4().hex for _ in range(df.height)]
        
        part1 = (
            df.select(self.part1_columns)
            .with_columns(pl.Series(self.part1_id, ids)))
        
        part2 = (
            df.select(self.part2_columns)
            .with_columns(pl.Series(self.part2_id, ids2)))
        
        self.mapping = pl.DataFrame({
            self.part1_id: ids,
            self.part2_id: ids2
        })
        
        return part1, part2

    def save_mapping(self, path: Union[str, Path], format: str = "csv"):
        """Сохраняет таблицу соответствий"""
        if self.mapping is None:
            raise ValueError("Сначала выполните decompose()")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        match format:
            case "csv": self.mapping.write_csv(path)
            case "parquet": self.mapping.write_parquet(path)
            case "excel": self.mapping.write_excel(path)
            case _: raise ValueError("Неподдерживаемый формат")

    @classmethod
    def load_mapping(cls, path: Union[str, Path], format: str = "csv") -> pl.DataFrame:
        """Загружает таблицу соответствий"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл {path} не найден")
            
        return (
            pl.read_csv(path) if format == "csv" else
            pl.read_parquet(path) if format == "parquet" else
            pl.read_excel(path)
        )

    @classmethod
    def restore(
        cls,
        part1: pl.DataFrame,
        part2: pl.DataFrame,
        mapping: pl.DataFrame
    ) -> pl.DataFrame:
        """Восстанавливает данные из частей"""
        # Автоматическое определение ID-столбцов
        part1_id = mapping.columns[0]
        part2_id = mapping.columns[1]
        
        return (
            part1.join(mapping, on=part1_id)
            .join(part2, on=part2_id)
            .drop([part1_id, part2_id])
        )

class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 sensitivity: Union[float, dict] = 1.0):
        """
        :param epsilon: Параметр приватности (ε)
        :param delta: Вероятность утечки (только для Гаусса)
        :param sensitivity: Чувствительность данных
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    # Лапласовский механизм (для числовых данных)
    def add_laplace_noise(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Добавляет шум по распределению Лапласа"""
        return df.with_columns([
            self._apply_laplace(col, df)
            for col in columns
        ])

    def _apply_laplace(self, column: str, df: pl.DataFrame) -> pl.Expr:
        scale = self._get_scale(column) / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=df.height)
        return self._apply_noise(column, df, noise)

    # Гауссовский механизм
    def add_gaussian_noise(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Добавляет шум по нормальному распределению"""
        return df.with_columns([
            self._apply_gaussian(col, df)
            for col in columns
        ])

    def _apply_gaussian(self, column: str, df: pl.DataFrame) -> pl.Expr:
        sigma = math.sqrt(2 * math.log(1.25/self.delta)) * self._get_scale(column) / self.epsilon
        noise = np.random.normal(loc=0, scale=sigma, size=df.height)
        return self._apply_noise(column, df, noise)
    
    def _calculate_probabilities(self, values: List[str], utility_fn: Callable) -> Dict[str, str]:
        scores = {v: math.exp(self.epsilon * utility_fn(v)/(2*self._get_scale())) for v in values}
        total = sum(scores.values())
        return {v: np.random.choice(list(scores.keys()), p=[s/total for s in scores.values()]) 
                for v in values}

    # Общие вспомогательные методы
    def _get_scale(self, column: str = None) -> float:
        if isinstance(self.sensitivity, dict):
            return self.sensitivity.get(column, 1.0)
        return self.sensitivity

    def _apply_noise(self, column: str, df: pl.DataFrame, noise: np.ndarray) -> pl.Expr:
        return (
            pl.col(column) + pl.Series(noise)
            .round(0)
            .cast(df.schema[column])
            .alias(column)
        )

    def calculate_sensitivity(self, df: pl.DataFrame, column: str, 
                             method: str = 'range') -> float:
        """Авторасчет чувствительности"""
        if method == 'range':
            return df[column].max() - df[column].min()
        elif method == 'max':
            return df[column].max()
        elif method == 'std':
            return df[column].std()
        else:
            raise ValueError("Недопустимый метод расчета чувствительности")
    
