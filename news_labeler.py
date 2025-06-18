import streamlit as st
import pandas as pd
import os

CSV_PATH = "manual_check_dataset.csv"
LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ Файл {CSV_PATH} не найден.")
        st.stop()
    return pd.read_csv(CSV_PATH)

def save_data(df):
    df.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()

st.set_page_config(page_title="News Verifier", layout="wide")
st.title("📰 Ручная проверка новостей AG News")

df = load_data()

unlabeled = df[df["checked_label"].isna() | (df["checked_label"] == "")]
if len(unlabeled) == 0:
    st.success("✅ Все новости размечены!")
    st.stop()

row = unlabeled.iloc[0]
idx = row.name

st.markdown(f"### Новость {idx + 1} из {len(df)}")
st.text_area("📝 Текст новости", row["text"], height=200)

st.markdown("#### Выберите правильную категорию:")
cols = st.columns(4)

for i, label in LABELS.items():
    if cols[i].button(label):
        df.at[idx, "checked_label"] = i
        save_data(df)
        st.rerun()

if st.button("Пропустить"):
    df.at[idx, "checked_label"] = "skipped"
    save_data(df)
    st.rerun()

st.markdown("---")
done = len(df[df["checked_label"].notna() & (df["checked_label"] != "")])
st.markdown(f"📊 Размечено: **{done} / {len(df)}**")