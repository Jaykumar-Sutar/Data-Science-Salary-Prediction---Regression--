### Data Understanding

The dataset utilized for this project tracks global compensation distributions across the technology sector, enabling multi-variable analysis of industry pay structures.

* **Dataset Source:** Data Science Job Salaries Dataset
* **Core Variables Evaluated:** Experience Level, Employment Type, Job Title, Remote Work Ratio, Company Size, and Employee Location.
* **Target Variable:** `salary_in_usd` — The continuous numeric column representing the total standardized annual compensation target [float/int].

### Type of Machine Learning Task:
It is a **Regression problem** where, given a set of historical workplace features, the objective is to predict standardized numerical compensation metrics.

### Performance Metric:
Since this is a continuous numeric regression problem, model performance is evaluated using **Root Mean Squared Error (RMSE)** and **R-squared (\(R^2\)) Score** to measure variance explanation.
