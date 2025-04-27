
# Informer - Time Series Forecasting

This project implements the **Informer** deep learning architecture from scratch to perform **long-sequence time series forecasting** on real-world energy data (ETT-small dataset).

---

## 📦 Project Structure

```
informer_timeseries/
│
├── data/
│   └── ETT-small.csv
│
├── model/
│   ├── attention.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── informer.py
│   └── __init__.py
│
├── runs/
│   ├── training/
│   ├── evaluation/
│   └── testing/
│
├── scripts/
│   └── replot_training_loss.py
│
├── train.py
├── evaluate.py
├── test.py
└── README.md
```

---

## 📈 Training Details

- Dataset: **ETT-small** (Electricity Transformer Temperature)
- Input sequence length: `96`
- Label length: `48`
- Prediction length: `24`
- Model size: `d_model=512`, `n_heads=8`, `e_layers=3`
- Training epochs: `50`
- Optimizer: `Adam`
- Learning rate scheduler: `StepLR (step_size=10, gamma=0.5)`
- Device: `GPU (NVIDIA GeForce RTX 3050)`

---

## 🔥 Results

| Metric | Score |
|:-------|:------|
| Test MAE | 1.0684 |
| Test RMSE | 1.4173 |

✅ Successfully learned seasonality and trend patterns from time series data.

---

## 📊 Outputs

- 📈 Training Loss Curve: `/runs/training/training_loss_plot.png`
- 📈 Training vs Validation Loss Curve: `/runs/training/training_vs_validation_loss_plot.png`
- 📄 Evaluation MAE/RMSE: `/runs/evaluation/evaluation_results.txt`
- 📈 Evaluation Sample Predictions: `/runs/evaluation/sample_predictions.png`
- 📈 Final Forecasts: `/runs/testing/forecast_plot.png`

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train model:

```bash
python train.py
```

3. Evaluate on test set:

```bash
python evaluate.py
```

4. Forecast future data:

```bash
python test.py
```

---

## 🛠 Future Improvements

- Add MinMax normalization preprocessing
- Train longer with fine-tuned learning rates

- ---

## 📚 Reference

This project is an implementation inspired by the research paper:

> **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**  
> Authors: Haoyiet al.  
> [Link to Paper (arXiv)](https://arxiv.org/abs/2012.07436)

Please refer to the original paper for theoretical foundations and complete architectural insights.

---

- Test other attention mechanisms
- Try larger d_model and deeper encoder layers

---

✅ Built completely from scratch, including attention, encoder, decoder, and training loops!
