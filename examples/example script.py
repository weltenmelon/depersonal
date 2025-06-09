import polars as pl
from depersonal_lib.core.utils import MLPerformanceEvaluator
from depersonal_lib.core.transformations import EnteringIdentifiers
from depersonal_lib.core.transformations import generalize, suppress_rare_categories, drop_column
from depersonal_lib.core.privacy_models import KAnonymity
from pathlib import Path

marital_dict = {'Married-AF-spouse': 'Previously-married', 
                'Married-spouse-absent': 'Previously-married', 
                'Widowed': 'Previously-married', 
                'Separated': 'Previously-married', 
                'Divorced': 'Previously-married', 
                'Married-civ-spouse': 'Married'}

race_dict = {'Amer-Indian-Eskimo': 'Non-white', 
                'Asian-Pac-Islander': 'Non-white', 
                'Other': 'Non-white',
                'Black': 'Non-white'}

script_dir = Path(__file__).parent.absolute()
file_path = script_dir / 'adult.csv'

data = pl.read_csv(file_path)
depersonalized_data = data.clone()

kan_orig = KAnonymity(['age', 'education', 'marital-status', 'race', 'gender','native-country'])
print('Уровень k-анонимности в исходных данных: ', kan_orig.analyze_k(data)[0])
print('Количество уязвимых групп : ',kan_orig.analyze_k(data)[1].shape[0])

depersonalized_data = drop_column(depersonalized_data, "education")
depersonalized_data = generalize(depersonalized_data, 'age', bins=[i for i in range(0,110,10)])
depersonalized_data = suppress_rare_categories(depersonalized_data, column="native-country", threshold=0.01)
depersonalized_data = depersonalized_data.with_columns(pl.col("marital-status").replace(marital_dict))
depersonalized_data = depersonalized_data.with_columns(pl.col("race").replace(race_dict))
transformer = EnteringIdentifiers("occupation")
depersonalized_data = transformer.transform(depersonalized_data)
transformer.save_mapping("adult_mapping.csv", format="csv")

kan_anon = KAnonymity(['age', 'marital-status', 'race', 'gender','native-country'])
min_k_anon, weak_anon = kan_anon.analyze_k(depersonalized_data)
print('Уровень k-анонимности в преобразованных данных:  ', min_k_anon)
print('Количество уязвимых групп : ', weak_anon.shape[0])

evaluator = MLPerformanceEvaluator(original_df=data, anonymized_df=depersonalized_data, target_column="income")
results = evaluator.evaluate_ml_performance(model_type="auto")

input()