import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Загрузка данных
dataset = pd.read_csv('AmesHousing.csv')

# Числовые признаки (кроме SalePrice)
numeric_features = dataset.select_dtypes(include=['float64', 'int64']).drop(columns=['SalePrice'])

# Выбранные категориальные признаки
selected_categorical = [
    'Neighborhood',
    'MS Zoning',
    'House Style',
    'Exterior 1st',
    'Garage Type',
    'Exter Qual',
    'Kitchen Qual',
    'Sale Condition'
]

# Берем только выбранные категориальные признаки
categorical_features_selected = dataset[selected_categorical]

# One-hot encoding выбранных категориальных признаков
categorical_encoded_selected = pd.get_dummies(categorical_features_selected, drop_first=True)

# Удаление сильно коррелирующих признаков (>0.85)
num_corr = numeric_features.corr()
high_corr = set()
for i in range(len(num_corr.columns)):
    for j in range(i):
        if abs(num_corr.iloc[i, j]) > 0.85:
            colname = num_corr.columns[i]
            high_corr.add(colname)

# Убираем сильно коррелирующие признаки из числовых
numeric_features_reduced = numeric_features.drop(columns=high_corr)

# Объединяем reduced числовые + выбранные категориальные
selected_full_features = pd.concat([numeric_features_reduced, categorical_encoded_selected], axis=1)

# Заполняем пропуски медианой
selected_full_features = selected_full_features.fillna(selected_full_features.median())

# Нормализация
scaler_selected = StandardScaler()
normalized_selected_data = scaler_selected.fit_transform(selected_full_features)
normalized_selected_df = pd.DataFrame(normalized_selected_data, columns=selected_full_features.columns)

# PCA для 3D графика
pca_selected = PCA(n_components=2)
reduced_selected_data = pca_selected.fit_transform(normalized_selected_df)

x_pca_selected = reduced_selected_data[:, 0]
y_pca_selected = reduced_selected_data[:, 1]
z_target_selected = dataset['SalePrice']

# 3D-график
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_pca_selected, y_pca_selected, z_target_selected, c=z_target_selected, cmap='viridis')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('SalePrice')
plt.colorbar(scatter, label='SalePrice')
plt.title('3D PCA with Numeric + Selected Categorical Features')
plt.show()

# Разделение на train/test
X_selected = normalized_selected_df
y_selected = dataset['SalePrice']
X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
    X_selected, y_selected, test_size=0.2, random_state=42)

# Линейная регрессия
model_selected = LinearRegression()
model_selected.fit(X_train_selected, y_train_selected)
y_pred_selected = model_selected.predict(X_test_selected)
rmse_linear_selected = np.sqrt(mean_squared_error(y_test_selected, y_pred_selected))
print(f'RMSE (Linear Regression): {rmse_linear_selected:.2f}')

# Lasso регрессия
lasso_selected = Lasso(alpha=0.1, max_iter=20000)
lasso_selected.fit(X_train_selected, y_train_selected)
y_pred_lasso_selected = lasso_selected.predict(X_test_selected)
rmse_lasso_selected = np.sqrt(mean_squared_error(y_test_selected, y_pred_lasso_selected))
print(f'RMSE (Lasso): {rmse_lasso_selected:.2f}')

# График зависимости RMSE от alpha
alphas_selected = np.logspace(-3, 1, 10)
rmse_scores_selected = []

for alpha in alphas_selected:
    lasso_temp = Lasso(alpha=alpha, max_iter=20000, tol=0.01)
    lasso_temp.fit(X_train_selected, y_train_selected)
    y_pred = lasso_temp.predict(X_test_selected)
    rmse = np.sqrt(mean_squared_error(y_test_selected, y_pred))
    rmse_scores_selected.append(rmse)

plt.figure(figsize=(10, 6))
plt.plot(alphas_selected, rmse_scores_selected, marker='o')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('RMSE')
plt.title('RMSE vs Regularization Strength\n(Numeric + Selected Categorical, reduced)')
plt.grid(True)
plt.show()

# Добавляем автоматический вывод лучшего alpha
best_alpha_idx = np.argmin(rmse_scores_selected)
best_alpha = alphas_selected[best_alpha_idx]
best_rmse = rmse_scores_selected[best_alpha_idx]

print(f'\nЛучшее alpha: {best_alpha:.4f}, соответствующее RMSE: {best_rmse:.2f}')

# Определение важных признаков
lasso_selected = Lasso(alpha=0.1, max_iter=20000)
lasso_selected.fit(X_train_selected, y_train_selected)
coefficients_selected = pd.Series(lasso_selected.coef_, index=X_train_selected.columns)
important_features_selected = coefficients_selected[coefficients_selected != 0].sort_values(ascending=False)

# Вывод топ 10 признаков (для анализа)
print("\nТОП-10 признаков по влиянию:")
print(important_features_selected.head(10))

# Для отчета по ТЗ — только 1 признак
most_important_feature = important_features_selected.idxmax()
most_important_value = important_features_selected.max()

print(f'\nНаиболее важный признак: {most_important_feature}, влияние ≈ {most_important_value:.2f} $')
#22 мая сделано