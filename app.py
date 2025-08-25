import io
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="דמו Streamlit (קובץ בלבד)", page_icon="📄", layout="wide")

# =======================
# Utilities
# =======================
def load_csv_file(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """קריאת CSV בבטחה (UTF-8/Windows-1255), כולל ניקוי בסיסי."""
    if uploaded_file is None:
        return None
    data = uploaded_file.read()
    for enc in ("utf-8", "utf-8-sig", "windows-1255", "iso-8859-8"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            continue
    raise ValueError("לא ניתן לקרוא את ה-CSV. בדוק/י קידוד או מבנה הקובץ.")

def ensure_numeric(series: pd.Series) -> pd.Series:
    """המרה בטוחה למספרים; כשלי המרה -> NaN -> 0, כדי למנוע קריסה בגרפים."""
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(0.0).astype(float)

def guess_numeric_columns(df: pd.DataFrame) -> list:
    """איתור עמודות שניתנות להמרה למספר (גם אם הטיפוס המקורי object)."""
    numeric_cols = []
    for c in df.columns:
        try:
            _ = pd.to_numeric(df[c], errors="coerce")
            numeric_cols.append(c)
        except Exception:
            pass
    return numeric_cols

def make_bar_chart(series: pd.Series, label: str = "Value"):
    """גרף עמודות יציב עם Altair (כולל תמיכה מלאה בערכי 0)."""
    ser = ensure_numeric(series)
    plot_df = pd.DataFrame({"Index": ser.index.astype(str), label: ser.values})
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Index:N", title="Index"),
            y=alt.Y(f"{label}:Q", title=label),
        )
        .properties(height=320)
    )
    return chart

def clean_dataframe(df: pd.DataFrame, strip_whitespace: bool = True) -> pd.DataFrame:
    """ניקוי קל: הסרת רווחים בשמות עמודות ובתאי טקסט."""
    if df is None:
        return None
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    if strip_whitespace:
        for c in df2.columns:
            if pd.api.types.is_object_dtype(df2[c]):
                df2[c] = df2[c].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    return df2

# =======================
# Sidebar
# =======================
st.sidebar.title("⚙️ הגדרות")
user_name = st.sidebar.text_input("שם להציג", value="חבר/ה")
st.sidebar.caption("👈 אפשר לשנות את השם כאן")

# אפשרות להשתמש בנתוני דוגמה
use_sample = st.sidebar.toggle("השתמש/י בנתוני דוגמה", value=False)

# העלאת קובץ (האפשרות היחידה לטעינת נתונים מצד המשתמש)
uploaded = st.sidebar.file_uploader("בחר/י קובץ CSV", type=["csv"])

# =======================
# Header
# =======================
st.title("📄 דמו Streamlit — העלאת קובץ בלבד")
st.subheader(f"שלום {user_name}! 👋")
st.write(
    "האפליקציה תומכת בטעינת CSV (דרך Sidebar) או שימוש בנתוני דוגמה. "
    "כוללת ניהול משימות, תצוגת נתונים, סטטיסטיקות מהירות, הורדת CSV, "
    "וגרף עמודות יציב גם כאשר יש ערכי 0 או טיפוסים מעורבים."
)

# =======================
# Tabs
# =======================
tab_tasks, tab_data, tab_chart, tab_tools = st.tabs(
    ["✅ משימות", "📄 נתונים", "📊 גרף", "🛠️ כלים"]
)

# =======================
# Tasks Tab
# =======================
with tab_tasks:
    st.markdown("### ✅ רשימת משימות (Session State)")
    if "todos" not in st.session_state:
        st.session_state.todos = []
    new_todo = st.text_input("הוסף משימה חדשה", key="todo_input")
    cols_add = st.columns([0.2, 0.8])
    with cols_add[0]:
        if st.button("➕ הוסף"):
            if new_todo and new_todo.strip():
                st.session_state.todos.append({"text": new_todo.strip(), "done": False})
                st.success("נוסף!")
            else:
                st.warning("רשום/י טקסט למשימה.")
    st.divider()

    to_remove = []
    for i, todo in enumerate(st.session_state.todos):
        c1, c2, c3 = st.columns([0.08, 0.77, 0.15])
        with c1:
            st.session_state.todos[i]["done"] = st.checkbox("בוצע", value=todo["done"], key=f"todo_done_{i}")
        with c2:
            st.write(("~~" + todo["text"] + "~~") if st.session_state.todos[i]["done"] else todo["text"])
        with c3:
            if st.button("🗑️ מחיקה", key=f"todo_del_{i}"):
                to_remove.append(i)
    for idx in reversed(to_remove):
        st.session_state.todos.pop(idx)

# =======================
# Data Tab
# =======================
with tab_data:
    st.markdown("### 📄 טעינת נתונים")
    st.caption("טען/י CSV באמצעות ה־Sidebar או השתמש/י בנתוני דוגמה.")

    df = None
    if use_sample:
        # דוגמה כוללת 0 להדגמת התיקון
        df = pd.DataFrame(
            {
                "Category": ["A", "B", "C", "D"],
                "Value": [10, 0, 7, 15],
                "Note": ["אלפא", "בטא", "גאמא", "דלתא"],
            }
        )
    elif uploaded is not None:
        try:
            df = load_csv_file(uploaded)
        except Exception as e:
            st.error(f"שגיאה בקריאת הקובץ: {e}")

    if df is not None:
        df = clean_dataframe(df)
        st.session_state.df = df.copy()
        st.success(f"נטענו {df.shape[0]} שורות ו-{df.shape[1]} עמודות.")
        st.write("**תצוגה:**")
        st.dataframe(df, use_container_width=True)

        st.markdown("#### 🧮 סטטיסטיקות מהירות")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        st.markdown("#### 💾 הורדת הנתונים (CSV)")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("הורד CSV מעודכן", data=csv_bytes, file_name="data_clean.csv", mime="text/csv")
    else:
        st.info("טרם נטענו נתונים. בחר/י CSV ב־Sidebar או הפעיל/י נתוני דוגמה.")

# =======================
# Chart Tab
# =======================
with tab_chart:
    st.markdown("### 📊 גרף עמודות יציב (כולל תמיכה בערכי 0)")
    if "df" not in st.session_state:
        st.info("ראשית, טען/י נתונים בלשונית 'נתונים'.")
    else:
        df = st.session_state.df.copy()
        numeric_cols = guess_numeric_columns(df)
        if not numeric_cols:
            st.warning("לא נמצאו עמודות שניתן להמיר למספריות.")
        else:
            st.write("בחר/י עמודה מספרית לתצוגה:")
            col = st.selectbox("עמודה מספרית", options=numeric_cols, index=0)
            label = st.text_input("תווית ציר-Y (שם הסדרה)", value=col or "Value")

            st.markdown("בחר/י עמודת קטגוריה (אופציונלי):")
            cat_col_options = ["(Index)"] + df.columns.tolist()
            cat_col = st.selectbox("קטגוריה", options=cat_col_options, index=0)

            plot_series = ensure_numeric(df[col])

            if cat_col == "(Index)":
                chart = make_bar_chart(plot_series, label=label)
            else:
                grouped = (
                    pd.DataFrame({"cat": df[cat_col].astype(str), "val": plot_series})
                    .groupby("cat", as_index=False)["val"].sum()
                )
                chart = (
                    alt.Chart(grouped)
                    .mark_bar()
                    .encode(
                        x=alt.X("cat:N", title=cat_col),
                        y=alt.Y("val:Q", title=label),
                    )
                    .properties(height=340)
                )

            st.altair_chart(chart, use_container_width=True)

# =======================
# Tools Tab
# =======================
with tab_tools:
    st.markdown("### 🛠️ כלים נוספים")
    st.write("- הצגת טיפוסים ו-Nulls לכל עמודה\n- המרת עמודות נבחרות למספרים (coerce→NaN→0)")

    if "df" in st.session_state:
        df = st.session_state.df.copy()
        st.markdown("#### טיפוסים ו-Nulls")
        info_df = pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum()})
        st.dataframe(info_df, use_container_width=True)

        st.markdown("#### המרת עמודות למספריים (coerce→NaN→0)")
        cols_to_convert = st.multiselect(
            "בחר/י עמודות להמרה",
            options=df.columns.tolist(),
            default=[c for c in df.columns if c.lower() in ("value", "amount", "price")],
        )
        if st.button("המר/י למספריים"):
            for c in cols_to_convert:
                df[c] = ensure_numeric(df[c])
            st.session_state.df = df
            st.success("העמודות הומרו בהצלחה. עבר/י ללשונית 'גרף' כדי להמחיש.")
            st.dataframe(df.head(30), use_container_width=True)
    else:
        st.info("ראשית, טען/י נתונים בלשונית 'נתונים'.")
