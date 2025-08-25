import streamlit as st
import pandas as pd

st.set_page_config(page_title="אפליקציית Streamlit בסיסית", page_icon="✨", layout="centered")

st.title("✨ דמו Streamlit פשוט")
st.write(
    "זו אפליקציה לדוגמה שנבנתה עם **Streamlit**. "
    "היא כוללת מחשבון קטן, רשימת משימות, והצגת קובץ CSV עם גרף."
)

# --- Sidebar ---
st.sidebar.header("הגדרות")
name = st.sidebar.text_input("שם להציג בכותרת", value="חבר/ה")
st.sidebar.write("👈 אפשר לשנות את השם כאן")

st.subheader(f"שלום {name}! 👋")

# --- Section 1: Simple Calculator ---
st.markdown("### 🧮 מחשבון פשוט")
col1, col2 = st.columns(2)
with col1:
    a = st.number_input("מספר ראשון (a)", value=1.0)
with col2:
    b = st.number_input("מספר שני (b)", value=2.0)
st.write(f"**a + b = {a + b}**")
st.write(f"a × b = {a * b}")

st.divider()

# --- Section 2: Todo list using session_state ---
st.markdown("### ✅ רשימת משימות (Session State)")
if "todos" not in st.session_state:
    st.session_state.todos = []

new_todo = st.text_input("הוסף משימה חדשה")
add = st.button("הוסף")
if add and new_todo.strip():
    st.session_state.todos.append({"text": new_todo.strip(), "done": False})

# תצוגת משימות עם אפשרות סימון והסרה
to_remove = []
for i, todo in enumerate(st.session_state.todos):
    cols = st.columns([0.1, 0.75, 0.15])
    with cols[0]:
        done = st.checkbox("בוצע", value=todo["done"], key=f"todo_{i}")
        st.session_state.todos[i]["done"] = done
    with cols[1]:
        st.write(("~~" + todo["text"] + "~~") if done else todo["text"])
    with cols[2]:
        if st.button("🗑️ מחיקה", key=f"del_{i}"):
            to_remove.append(i)
# מחיקה לאחר הלולאה
for idx in sorted(to_remove, reverse=True):
    st.session_state.todos.pop(idx)

st.divider()

# --- Section 3: CSV Uploader + simple chart ---
st.markdown("### 📈 העלאת קובץ CSV והצגה")
st.caption("אפשר לנסות עם הקובץ לדוגמה 'sample.csv' המצורף לפרויקט.")
uploaded = st.file_uploader("בחר/י קובץ CSV", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

df = None
if uploaded is not None:
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.error(f"שגיאה בקריאת הקובץ: {e}")

if df is not None:
    st.write("**תצוגת נתונים:**")
    st.dataframe(df, use_container_width=True)

    # בחירת עמודה מספרית לגרף
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.write("**גרף עמודה מספרית:**")
        col = st.selectbox("בחר/י עמודה מספרית לגרף עמודות", options=numeric_cols)
        st.bar_chart(df[col])
    else:
        st.info("לא נמצאו עמודות מספריות לשרטוט גרף.")

st.divider()
st.caption("נבנה עם ❤️ ב-Streamlit. קוד לדוגמה כולל cache עם st.cache_data ו-Session State.")
