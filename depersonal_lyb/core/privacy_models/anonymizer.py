import numpy as np
import pandas as pd
import re
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class DifferentialPrivacyProcessor:
    """
    Класс для обработки данных с использованием методов дифференциальной приватности.
    
    Аргументы:
        epsilon_numeric (float): Параметр ε для числовых данных.
        epsilon_categorical (float): Параметр ε для категориальных данных.
        keep_domain (bool): Сохранять ли домен при анонимизации электронной почты.
    """
    def __init__(self, epsilon_numeric=1.0, epsilon_categorical=1.0, keep_domain=True):
        self.epsilon_numeric = epsilon_numeric
        self.epsilon_categorical = epsilon_categorical
        self.keep_domain = keep_domain

    def add_numeric_noise(self, data_column, method="range"):
        """
        Добавление шума к числовым данным.
        
        Аргументы:
            data_column (pd.Series): Столбец числовых данных.
            method (str): Метод расчета чувствительности ("range", "std", "user_defined").
        
        Возвращает:
            pd.Series: Столбец с добавленным шумом.
        """
        if method == "range":
            sensitivity = data_column.max() - data_column.min()
        elif method == "std":
            sensitivity = data_column.std()
        elif isinstance(method, (int, float)) and method > 0:
            sensitivity = method
        else:
            raise ValueError("Неподдерживаемый метод для числовой чувствительности.")

        scale = sensitivity / self.epsilon_numeric
        noise = np.random.laplace(loc=0, scale=scale, size=data_column.shape)
        return data_column + noise

    def randomize_categories(self, data_column):
        """
        Рандомизация категориальных данных.
        
        Аргументы:
            data_column (pd.Series): Столбец категориальных данных.
        
        Возвращает:
            pd.Series: Столбец с рандомизированными значениями.
        """
        unique_categories = data_column.dropna().unique()
        num_categories = len(unique_categories)
        probabilities = np.exp(self.epsilon_categorical) / (np.exp(self.epsilon_categorical) + num_categories - 1)
        randomized_column = []

        for value in data_column:
            if np.random.rand() < probabilities:
                randomized_column.append(value)
            else:
                randomized_column.append(np.random.choice(unique_categories))

        return pd.Series(randomized_column)

    def extract_city_from_address(self, address_column):
        """
        Извлечение города из полного адреса.
        
        Аргументы:
            address_column (pd.Series): Столбец с полными адресами.
        
        Возвращает:
            pd.Series: Столбец, содержащий только города.
        """
        city_pattern = re.compile(r"г\.\s*[А-Яа-яЁё\s\-]+")
        city_column = address_column.apply(lambda x: city_pattern.search(x).group(0) if isinstance(x, str) and city_pattern.search(x) else "")
        return city_column

    def generate_synthetic_email(self, email):
        """
        Генерация синтетического почтового адреса.
        
        Аргументы:
            email (str): Исходный почтовый адрес.
        
        Возвращает:
            str: Синтетический почтовый адрес.
        """
        local_part = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), size=8))

        if '@' in email:
            _, domain = email.split('@')
            if self.keep_domain:
                return f"{local_part}@{domain}"
            else:
                random_domain = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=5)) + ".com"
                return f"{local_part}@{random_domain}"
        return f"{local_part}@random.com"

    def randomize_email(self, email_column):
        """
        Анонимизация электронных почтовых адресов.
        
        Аргументы:
            email_column (pd.Series): Столбец с почтовыми адресами.
        
        Возвращает:
            pd.Series: Столбец с анонимизированными адресами.
        """
        return email_column.apply(self.generate_synthetic_email)

    def anonymize_names(self, data_column):
        """
        Анонимизация столбца с ФИО путем перемешивания фамилий, имен и отчеств между строками.
        
        Аргументы:
            data_column (pd.Series): Столбец с ФИО.
        
        Возвращает:
            pd.Series: Обработанный столбец с перемешанными данными.
        """
        name_parts = data_column.str.split(expand=True).fillna("")
        last_names = name_parts[0].tolist()
        first_names = name_parts[1].tolist() if name_parts.shape[1] > 1 else [""] * len(data_column)
        patronymics = name_parts[2].tolist() if name_parts.shape[1] > 2 else [""] * len(data_column)

        np.random.shuffle(last_names)
        np.random.shuffle(first_names)
        np.random.shuffle(patronymics)

        randomized_names = [
            " ".join(filter(None, [last_names[i], first_names[i], patronymics[i]]))
            for i in range(len(data_column))
        ]
        return pd.Series(randomized_names)

    def add_date_noise(self, date_column):
        """
        Добавление шума к датам с учетом формата.
        
        Аргументы:
            date_column (pd.Series): Столбец с датами.
        
        Возвращает:
            pd.Series: Столбец с зашумленными датами.
        """
        def process_date(date_str):
            try:
                if re.match(r"^\d{4}$", date_str):
                    year = int(date_str)
                    noise = int(np.random.laplace(0, 1 / self.epsilon_numeric))
                    return str(year + noise)
                elif re.match(r"^\d{2}\.\d{2}\.\d{4}$", date_str):
                    print('process_date 2')

                    day, month, year = map(int, date_str.split('.'))
                    day += int(np.random.laplace(0, 1 / self.epsilon_numeric))
                    month += int(np.random.laplace(0, 1 / self.epsilon_numeric))
                    year += int(np.random.laplace(0, 1 / self.epsilon_numeric))
                    return datetime(year, month, day).strftime("%d.%m.%Y")
                else:
                    return date_str
            except Exception:
                return date_str

        return date_column.astype(str).apply(process_date)

    def process(self, data, numeric_columns=None, categorical_columns=None, name_columns=None, address_columns=None, email_columns=None, date_columns=None, numeric_method="range"):
        """
        Применение обработки данных с дифференциальной приватностью.
        
        Аргументы:
            data (pd.DataFrame): DataFrame с данными.
            numeric_columns (list): Список числовых столбцов для обработки.
            categorical_columns (list): Список категориальных столбцов для обработки.
            name_columns (list): Список столбцов с ФИО для обработки.
            address_columns (list): Список столбцов с адресами для обработки.
            email_columns (list): Список столбцов с почтовыми адресами для обработки.
            date_columns (list): Список столбцов с датами для обработки.
            numeric_method (str): Метод расчета чувствительности для числовых столбцов.
        
        Возвращает:
            pd.DataFrame: DataFrame с обработанными данными.
        """
        data_processed = data.copy()

        if numeric_columns:
            for col in numeric_columns:
                if col in data_processed.columns:
                    data_processed[col] = self.add_numeric_noise(data_processed[col], method=numeric_method)

        if categorical_columns:
            for col in categorical_columns:
                if col in data_processed.columns:
                    data_processed[col] = self.randomize_categories(data_processed[col])

        if name_columns:
            for col in name_columns:
                if col in data_processed.columns:
                    data_processed[col] = self.anonymize_names(data_processed[col])

        if address_columns:
            for col in address_columns:
                if col in data_processed.columns:
                    data_processed[col] = self.extract_city_from_address(data_processed[col])

        if email_columns:
            for col in email_columns:
                if col in data_processed.columns:
                    data_processed[col] = self.randomize_email(data_processed[col])

        if date_columns:
            for col in date_columns:
                if col in data_processed.columns:
                    data_processed[col] = self.add_date_noise(data_processed[col])

        return data_processed


class AnonimyzerWithIDs:
    """
    Класс для обработки данных с использованием методов введения идентификаторов.
    
    Аргументы:
        anonymize_columns (bool): Если True, переименовывает колонки в x1, x2, ...
        le_all (dict): Все модели LabelEncoder записанные в словарь {'название колонки': модель_LE}
        scaler (MinMaxScaler): Модель для нормализации числовых данных
        categorical_cols (list): Колонки с категориальными данными
        numeric_cols (list): Колонки с числовыми данными
        anon_cols_match_dict (dict): Cловарь с индентификаторами названий колонок
        le_cat_cols_match_table (dict): Cловарь с индентификаторами значений для каждой обработанной колонки
        models_return (bool): Если True, сохраняет обученные модели преобразователей
    """
    def __init__(self, anonymize_columns=True, models_return = False):
        self.anonymize_columns = anonymize_columns
        self.le_all = None
        self.scaler = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.anon_cols_match_dict = dict()
        self.le_cat_cols_match_table = dict()
        self.models_return = models_return

    def _label_encoder_ft(self, column):
        """
        Преобразует категориальные данные .
        
        Args:
            column (pd.Series): Исходная колонка.
        
        Returns:
            pd.Series: Преобразованная колонка.
        """
        le = LabelEncoder()
        processed_column = le.fit_transform(column.astype(str))

        keys = le.classes_
        values = le.transform(le.classes_)
        dictionary = dict(zip(keys, values))
        self.le_cat_cols_match_table[column.name] = dictionary

        if self.models_return:
            self.le_all[column.name] = le

        return processed_column

    def preprocess_dataset(self, data, categorical_cols=None, numeric_cols=None):
        """
        Преобразует датасет с нормализацией, обработкой категориальных данных, анонимизацией колонок и сохранением таблицы соответствий идентификаторов .
        
        Args:
            data (pd.DataFrame): Исходный датасет.
            categorical_cols (list): Категориальыне колонки которые нужно обработать
            numeric_cols (list): Числовые колонки которые нужно обработать
        Returns:
            pd.DataFrame: Преобразованный датасет.
        """
        processed_data = data.copy()
        
        # Обработка категориальных данных
        if categorical_cols:
            self.categorical_cols = categorical_cols
        else:
            self.categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
                  
        if len(self.categorical_cols) > 0:
            if self.models_return:
                self.le_all = dict.fromkeys(self.categorical_cols)

            self.le_cat_cols_match_table = dict.fromkeys(self.categorical_cols)

            for col in self.categorical_cols:
                processed_data[col] = self._label_encoder_ft(processed_data[col])

        # # Нормализация числовых данных
        if numeric_cols and type(numeric_cols)==list:
            self.numeric_cols = numeric_cols
        else:
            self.numeric_cols = processed_data.select_dtypes(include=[np.number]).columns

        if len(self.numeric_cols) > 0:
            scaler = MinMaxScaler()
            processed_data[self.numeric_cols] = scaler.fit_transform(processed_data[self.numeric_cols].values)

            for key in self.le_cat_cols_match_table.keys():
                if key in set(self.numeric_cols):
                    zero_matrix = np.zeros((len(self.le_cat_cols_match_table[key]), len(self.numeric_cols)))
                    zero_matrix.T[processed_data[self.numeric_cols].columns.get_loc(key)] = np.array(list(self.le_cat_cols_match_table[key].values()))
                    zero_matrix = scaler.transform(zero_matrix)
                    i = 0
                    for key2 in self.le_cat_cols_match_table[key].keys():
                        self.le_cat_cols_match_table[key][key2] = zero_matrix[i][processed_data[self.numeric_cols].columns.get_loc(key)]
                        i+=1

            if self.models_return:
                self.scaler = scaler
            
        # Анонимизация колонок и сохранение таблицы индентификаторов
        if self.anonymize_columns:
            old_cols = data.columns
            new_cols = [f'X{i}' for i in range(processed_data.shape[1])]
            self.anon_cols_match_dict = dict([(new_cols[i], old_cols[i]) for i in range(processed_data.shape[1])])
            processed_data.columns = new_cols

            df_match_dict = pd.DataFrame(list(self.anon_cols_match_dict.items()), columns=['Оригинальные значения', 'Введенные идентификаторы'])
            df_match_dict.to_excel(f'columns matching.xlsx')
        
        # Сохранение таблицы индентификаторов категориальных данных
        if self.le_cat_cols_match_table:
            with pd.ExcelWriter('categorical matching.xlsx') as writer:  
                for key in self.le_cat_cols_match_table.keys():
                    cat_cols_df_match_dict = pd.DataFrame(list(self.le_cat_cols_match_table[key].items()), columns=['Оригинальные значения', 'Введенные идентификаторы'])
                    cat_cols_df_match_dict.to_excel(writer, sheet_name=f'{key} ids')

        return processed_data
        
    def _reverse_transform(self, reversed_data):
        """
        Преобразует обработанную таблицу с исходными названиями колонок в исходную таблицу.
        
        Args:
            reversed_data (pd.DataFrame): Обработанная таблица с исходными названиями колонок.
        
        Returns:
            pd.DataFrame: Восстановленная таблица.
        """
        try:
            for le in self.le_all.keys():
                check_is_fitted(self.le_all[le])
            try:
                check_is_fitted(self.scaler)

                if len(self.numeric_cols) > 0:
                    reversed_data[self.numeric_cols] = self.scaler.inverse_transform(reversed_data[self.numeric_cols])

                if len(self.categorical_cols) > 0:
                    for col in self.categorical_cols:
                        reversed_data[col] = self.le_all[col].inverse_transform(round(reversed_data[col]).astype(int))
            except NotFittedError as exc:
                print(f"Модель для преобразования числовых данных не обучена \n{exc}")
        except NotFittedError as exc:
            print(f"Модель для преобразования категориальных переменных не обучена \n{exc}")
        return reversed_data

    def reverse_prepocess(self, data):
        """
        Преобразует обработанные данные в исходную таблицу .
        
        Args:
            data (pd.DataFrame): Обработанная таблица.
        
        Returns:
            pd.DataFrame: Восстановленная таблица.
        """
        if self.anonymize_columns:
            if len(self.anon_cols_match_dict) == 0:
                print("Словарь для преобразования названий колонок не заполнен, хотя предполагается, что колонки были переименованы")
            else:
                reversed_data = data.copy()
                reversed_data = reversed_data.rename(columns = self.anon_cols_match_dict)
                reversed_data = self._reverse_transform(reversed_data)
        else:
            reversed_data = self._reverse_transform(data.copy())
        return reversed_data