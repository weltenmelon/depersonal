import polars as pl
    
class KAnonymity:
    def __init__(self, quasi_columns: list):
        self.quasi_columns = quasi_columns

    def analyze_k(self, df: pl.DataFrame) -> tuple[int, pl.DataFrame]:
        """
        Возвращает:
        - Минимальное k (уровень анонимности данных)
        - Группы с размером < текущего k (для анализа уязвимостей)
        - удовлетворяет ли датасет k анонимности
        """
        # Группируем по квази-идентификаторам и считаем размер групп
        groups = df.group_by(self.quasi_columns).agg(pl.len().alias("group_size"))
        
        # Находим минимальный размер группы
        min_k = groups["group_size"].min()
        
        # Фильтруем группы, которые не удовлетворяют даже минимальному k
        weak_groups = groups.filter(pl.col("group_size") == min_k)
        
        return min_k, weak_groups

    def get_current_k(self, df: pl.DataFrame) -> int:
        """Возвращает текущий уровень k-анонимности."""
        return df.group_by(self.quasi_columns).agg(pl.len().alias("count"))["count"].min()
    
    def check_k(self, k: int, df: pl.DataFrame) -> bool:
        """Проверяет, удовлетворяет ли DataFrame k-анонимности."""
        min_groups = df.group_by(self.quasi_columns).agg(pl.len().alias("count")).min()
        return min_groups >= k
    
class LDiversity:
    def __init__(self, quasi_columns: list, sensitive_column: str):
        self.quasi_columns = quasi_columns
        self.sensitive_column = sensitive_column

    def analyze_l(self, df: pl.DataFrame) -> tuple[int, pl.DataFrame]:
        """
        Возвращает:
        - Минимальное l (уровень разнообразия)
        - Группы с недостаточным разнообразием
        """
        groups = self._calculate_diversity(df)
        min_l = groups["unique_count"].min()
        weak_groups = groups.filter(pl.col("unique_count") == min_l)
        return min_l, weak_groups

    def get_current_l(self, df: pl.DataFrame) -> int:
        """Возвращает текущий уровень l-разнообразия"""
        return self._calculate_diversity(df)["unique_count"].min()
    
    def check_l(self, l: int, df: pl.DataFrame) -> bool:
        """Проверяет удовлетворяет ли DataFrame l-разнообразию"""
        return self.get_current_l(df) >= l

    def _calculate_diversity(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.group_by(self.quasi_columns).agg(
            pl.col(self.sensitive_column).n_unique().alias("unique_count")
        )
    
class TCloseness:
    def __init__(self, quasi_columns: list, sensitive_column: str):
        self.quasi_columns = quasi_columns
        self.sensitive_column = sensitive_column

    def _global_distribution(self, df: pl.DataFrame) -> dict:
        # Вычисляем глобальное распределение значений
        counts = df[self.sensitive_column].value_counts(normalize=True)
        return {row[0]: row[1] for row in counts.iter_rows()}  # {value: proportion}

    def _group_distribution_from_series(self, group_series: pl.DataFrame, global_dist: dict) -> float:
        # Вычисляем распределение в группе
        local_counts = group_series.value_counts(normalize=True)
        local_dist = {row[0]: row[1] for row in local_counts.iter_rows()}

        # Вычисляем L1-расстояние между распределениями
        total_diff = 0.0
        for value, p_global in global_dist.items():
            p_local = local_dist.get(value, 0.0)
            total_diff += abs(p_local - p_global)
            
        return total_diff / 2  # Нормализация расстояния
    
    def analyze_t(self, df: pl.DataFrame) -> tuple[float, pl.DataFrame]:
        # Глобальное распределение чувствительного атрибута
        global_dist = self._global_distribution(df)

        result_df = df.group_by(self.quasi_columns).agg(
            pl.col(self.sensitive_column).map_elements(
                lambda group_series: self._group_distribution_from_series(group_series, global_dist),
                return_dtype=pl.Float64
            ).alias("t_distance")
        )

        # Находим максимальное расстояние
        max_t = result_df["t_distance"].max()
        if max_t is None:
             # Обработка случая, если result_df пустой
             return 0.0, result_df.clear()
        
        weak_groups = result_df.filter(pl.col("t_distance") == max_t)

        return max_t, weak_groups

    def get_current_t(self, df: pl.DataFrame) -> float:
        return self.analyze_t(df)[0]

    def check_t(self, threshold: float, df: pl.DataFrame) -> bool:
        return self.get_current_t(df) <= threshold