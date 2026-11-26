# House Price Prediction

A Streamlit app that estimates a house price from property details using a
trained scikit-learn model.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep `house_price_model.pkl` and `data.csv` in the repository root alongside
`app.py`. The model was saved with scikit-learn 1.6.1, so that version is pinned
in `requirements.txt`.

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub. Include `app.py`, `requirements.txt`,
   `data.csv`, and `house_price_model.pkl`.
2. The model file is about 33 MB. Upload it with Git/GitHub Desktop, or use
   Git LFS; the GitHub web uploader has a smaller per-file limit. Community
   Cloud supports Git LFS.
3. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and choose
   **Create app**.
4. Select this repository, its branch, and `app.py` as the entrypoint. Python
   3.12 is a suitable choice for the pinned scikit-learn version.
5. Choose **Deploy**.
