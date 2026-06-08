# Model Validation Report

## Dataset

NASA CMAPSS FD001

Training Engines: 50
Window Size: 30 Cycles

---

## Random Forest Results

MAE: 11.96
RMSE: 15.87
R²: 0.851

Interpretation:

- Average prediction error ≈ 12 cycles
- Strong predictive performance
- Suitable baseline model

---

## XGBoost Results

MAE: 10.24
RMSE: 13.94
R²: 0.885

Interpretation:

- Average prediction error ≈ 10 cycles
- Best performing model
- Explains 88.5% of RUL variation

---

## Model Comparison

XGBoost outperformed Random Forest on:

✓ MAE
✓ RMSE
✓ R²

Selected Final Model:
XGBoost Regressor

---

## Conclusion

The XGBoost model provides reliable Remaining Useful Life predictions and can be used for maintenance planning and fleet health monitoring.