import io
import re
import os
import altair as alt
import pandas as pd
import streamlit as st
import sqlite3
import tempfile

# ========= Optional models (not required) =========
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

st.set_page_config(page_title="🤖 דמו Streamlit — נתונים + AI", page_icon="🤖", layout="wide")

# =======================
# Utilities
# =======================
def load_csv_bytes(file_obj) -> pd.DataFrame:
    """קריאת CSV מתוך UploadedFile (כמה קידודים נפוצים, כולל עברית)."""
    if file_obj is None:
        return None
    data = file_obj.read()
    for enc in ("utf-8", "utf-8-sig", "windows-1255", "iso-8859-8"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            continue
    raise ValueError("לא ניתן לקרוא את ה-CSV. בדוק/י קידוד או מבנה הקובץ.")

def load_excel_bytes(file_obj) -> dict:
    """
    קריאת Excel מתוך UploadedFile והחזרה של מילון {sheet_name: DataFrame}.
    מצריך openpyxl ל-xlsx.
    """
    if file_obj is None:
        return None
    data = file_obj.read()
    xls = pd.ExcelFile(io.BytesIO(data))
    sheets = {}
    for sheet in xls.sheet_names:
        try:
            sheets[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except Exception:
            pass
    if not sheets:
        raise ValueError("לא נמצאו גיליונות קריאים בקובץ האקסל.")
    return sheets

# --------- SQLite helpers: לא טוענים הכל מראש ---------
def _sqlite_bytes_to_tempfile(db_bytes: bytes) -> str:
    """שומר בייטים של DB לקובץ זמני ומחזיר את הנתיב."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        tmp.write(db_bytes)
        return tmp.name

def get_sqlite_table_names(db_bytes: bytes) -> list:
    """מחזיר רשימת טבלאות מתוך ה-DB בלי לטעון אותן."""
    path = _sqlite_bytes_to_tempfile(db_bytes)
    try:
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        rows = cur.fetchall()
        return [r[0] for r in rows]
    finally:
        try:
            con.close()
        except Exception:
            pass
        try:
            os.remove(path)
        except Exception:
            pass

def read_sqlite_table(db_bytes: bytes, table: str, limit: int = 100000) -> pd.DataFrame:
    """קורא טבלה בודדת לפי שם, עם LIMIT בטיחותי (ברירת מחדל 100k)."""
    path = _sqlite_bytes_to_tempfile(db_bytes)
    try:
        con = sqlite3.connect(path)
        q = f"SELECT * FROM '{table}'"
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q, con)
        return df
    finally:
        try:
            con.close()
        except Exception:
            pass
        try:
            os.remove(path)
        except Exception:
            pass

def ensure_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(0.0).astype(float)

def guess_numeric_columns(df: pd.DataFrame) -> list:
    out = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c], errors="coerce")
            out.append(c)
        except Exception:
            pass
    return out

def sanitize_field_name(name: str, default: str = "Value"):
    if not name:
        return default, default
    raw = str(name).strip()
    if raw == "":
        return default, default
    safe = re.sub(r"[^\w]", "_", raw, flags=re.UNICODE)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if safe == "":
        safe = default
    return safe, raw

def make_bar_chart(series: pd.Series, label: str = "Value"):
    ser = ensure_numeric(series)
    safe_label, raw_label = sanitize_field_name(label or "Value", default="Value")
    plot_df = pd.DataFrame({"Index": ser.index.astype(str), safe_label: ser.values})
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(field="Index", type="nominal", title="Index"),
            y=alt.Y(field=safe_label, type="quantitative", title=raw_label),
        )
        .properties(height=320)
    )
    return chart

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    for c in df2.columns:
        if pd.api.types.is_object_dtype(df2[c]):
            df2[c] = df2[c].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    return df2

# ---------- AI helpers (no external services) ----------
_HE_STOPWORDS = {
    "אני","אתה","את","הוא","היא","אנחנו","אתם","אתן","הם","הן","של","עם","על","אל","עד","אם",
    "יש","אין","מאוד","גם","אבל","או","כי","לא","כן","זה","זו","בין","וכן","כך","כדי","היו","היה",
}
_EN_STOPWORDS = {
    "i","you","he","she","it","we","they","the","a","an","of","to","in","on","and","or","but","is","are",
    "was","were","be","been","very","so","that","this","these","those","as","at","by","for","from","with","without"
}

def _tokenize_sentences(text: str):
    split = re.split(r'(?<=[\.\!\?\…])\s+|\n+', text.strip())
    return [s.strip() for s in split if s.strip()]

def _word_tokens(text: str):
    return re.findall(r"[A-Za-zא-ת]+", text.lower())

def _is_hebrew(text: str) -> bool:
    return bool(re.search(r"[א-ת]", text))

def summarize_text_light(text: str, max_sentences: int = 5):
    if not text or not text.strip():
        return ""
    sentences = _tokenize_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()
    heb = _is_hebrew(text)
    stop = _HE_STOPWORDS if heb else _EN_STOPWORDS
    words = _word_tokens(text)
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    scores = []
    for s in sentences:
        score = 0
        for w in _word_tokens(s):
            if w in stop:
                continue
            score += freq.get(w, 0)
        scores.append(score)
    ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:max_sentences]
    ranked_idx = sorted(ranked_idx)
    summary = " ".join(sentences[i] for i in ranked_idx)
    return summary.strip()

def sentiment_simple(text: str):
    pos_he = {"מעולה","טוב","מצוין","יפה","מדהים","שמחה","חיובי","חזק","איכותי","קל","מהיר"}
    neg_he = {"רע","גרוע","בעייתי","עצוב","שלילי","קשה","איטי","זוועה","מבאס","נורא","כישלון"}
    pos_en = {"good","great","excellent","nice","amazing","happy","positive","strong","quality","easy","fast"}
    neg_en = {"bad","terrible","problem","sad","negative","hard","slow","awful","worse","failure","annoying"}
    tokens = _word_tokens(text)
    if _is_hebrew(text):
        p = sum(t in pos_he for t in tokens)
        n = sum(t in neg_he for t in tokens)
    else:
        p = sum(t in pos_en for t in tokens)
        n = sum(t in neg_en for t in tokens)
    score = p - n
    label = "חיובי" if score > 0 else ("שלילי" if score < 0 else "ניטרלי")
    conf = min(1.0, max(0.0, 0.5 + abs(score) / 10))
    return {"label": label, "score": conf, "p": p, "n": n}

@st.cache_resource(show_spinner=False)
def get_sentiment_pipeline():
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return pipeline("sentiment-analysis")
    except Exception:
        return None

# =======================
# Sidebar
# =======================
st.sidebar.title("⚙️ הגדרות")
user_name = st.sidebar.text_input("שם להציג", value="חבר/ה")
use_sample = st.sidebar.toggle("השתמש/י בנתוני דוגמה", value=False)
st.sidebar.caption("⬇️ העלאת קובץ מתבצעת בלשונית 'נתונים'.")

# =======================
# Header
# =======================
st.title("🤖 דמו Streamlit — נתונים + AI קליל")
st.subheader(f"שלום {user_name}! 👋")
st.write(
    "האפליקציה תומכת ב: CSV, Excel (.xlsx/.xls), ו-SQLite DB (.db/.sqlite). "
    "ב-DB: מוצגת רשימת טבלאות לבחירה, ואז נטענת רק הטבלה שנבחרה. יש גם לשונית AI."
)

# =======================
# Tabs
# =======================
tab_tasks, tab_data, tab_chart, tab_tools, tab_ai = st.tabs(
    ["✅ משימות", "📄 נתונים", "📊 גרף", "🛠️ כלים", "🤖 AI"]
)

# =======================
# Tasks Tab
# =======================
with tab_tasks:
    st.markdown("### ✅ רשימת משימות (Session State)")
    if "todos" not in st.session_state:
        st.session_state.todos = []
    new_todo = st.text_input("הוסף משימה חדשה", key="todo_input")
    add_col, _ = st.columns([0.2, 0.8])
    if add_col.button("➕ הוסף"):
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
    st.markdown("### 📄 טעינת נתונים (CSV / Excel / SQLite-DB)")
    st.caption("בחר/י קובץ להעלאה, או השתמש/י בנתוני הדוגמה (מתג ב-Sidebar).")

    df = None

    # נתוני דוגמה (אופציונלי)
    if use_sample:
        df = pd.DataFrame(
            {"Category": ["A","B","C","D"], "Value": [10,0,7,15], "Note": ["אלפא","בטא","גאמא","דלתא"]}
        )

    uploaded = st.file_uploader(
        "בחר/י קובץ: CSV / XLSX / XLS / DB / SQLITE",
        type=["csv", "xlsx", "xls", "db", "sqlite"],
        accept_multiple_files=False
    )

    excel_sheets = None
    chosen_name = None

    # --- SQLite state ---
    if "db_bytes" not in st.session_state:
        st.session_state.db_bytes = None
    if "db_tables" not in st.session_state:
        st.session_state.db_tables = []
    if "db_selected_table" not in st.session_state:
        st.session_state.db_selected_table = None

    if (df is None) and (uploaded is not None):
        name_lower = (uploaded.name or "").lower()
        try:
            if name_lower.endswith(".csv"):
                df = load_csv_bytes(uploaded)
                chosen_name = uploaded.name

            elif name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
                # טוענים גיליונות ומאפשרים לבחור
                excel_sheets = load_excel_bytes(uploaded)
                if excel_sheets:
                    sheet = st.selectbox("בחר/י גיליון להצגה", options=list(excel_sheets.keys()))
                    df = excel_sheets.get(sheet)
                    chosen_name = f"{uploaded.name} — {sheet}"

            elif name_lower.endswith(".db") or name_lower.endswith(".sqlite"):
                # שומרים את ה-DB בבייטים ל-Session State פעם אחת
                db_raw = uploaded.read()
                st.session_state.db_bytes = db_raw
                # מקבלים שמות טבלאות ומציגים לבחירה
                st.session_state.db_tables = get_sqlite_table_names(st.session_state.db_bytes)
                if not st.session_state.db_tables:
                    st.error("לא נמצאו טבלאות ב-DB.")
                else:
                    st.markdown("#### טבלאות שנמצאו ב-DB")
                    st.write(st.session_state.db_tables)
                    st.session_state.db_selected_table = st.selectbox(
                        "בחר/י טבלה להצגה",
                        options=st.session_state.db_tables,
                        index=0
                    )
                    limit = st.number_input("LIMIT לתצוגה (לשמירה על זיכרון)", min_value=1000, max_value=1_000_000, step=1000, value=100000)
                    if st.button("טען טבלה"):
                        try:
                            df = read_sqlite_table(st.session_state.db_bytes, st.session_state.db_selected_table, int(limit))
                            chosen_name = f"{uploaded.name} — {st.session_state.db_selected_table}"
                        except Exception as e:
                            st.error(f"שגיאה בקריאת הטבלה: {e}")

                    # אופציונלי: שאילתת SQL חופשית בטוחה עם LIMIT
                    with st.expander("🧪 שאילתת SQL (אופציונלי)"):
                        st.caption("אפשר לכתוב SELECT על ה-DB (נוסיף LIMIT אם חסר).")
                        sql = st.text_area("שאילתא (למשל: SELECT * FROM my_table WHERE ...)", height=120)
                        if st.button("הרץ שאילתא"):
                            if sql.strip():
                                q = sql.strip().rstrip(";")
                                if "limit" not in q.lower():
                                    q += " LIMIT 100000"
                                try:
                                    # קובץ זמני מהרשימה ב-Session
                                    tmp_path = _sqlite_bytes_to_tempfile(st.session_state.db_bytes)
                                    try:
                                        con = sqlite3.connect(tmp_path)
                                        df_query = pd.read_sql_query(q, con)
                                        df = df_query
                                        chosen_name = f"SQL query"
                                        st.success("השאילתא רצה בהצלחה.")
                                    finally:
                                        try:
                                            con.close()
                                        except Exception:
                                            pass
                                        try:
                                            os.remove(tmp_path)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    st.error(f"שגיאה בהרצת השאילתא: {e}")

            else:
                st.error("סוג קובץ לא נתמך.")

        except Exception as e:
            st.error(f"שגיאה בקריאת הקובץ: {e}")

    if df is not None:
        df = clean_dataframe(df)
        st.session_state.df = df.copy()
        st.success(f"נטענו {df.shape[0]} שורות ו-{df.shape[1]} עמודות." + (f" ({chosen_name})" if chosen_name else ""))
        st.write("**תצוגה:**")
        st.dataframe(df, use_container_width=True)

        st.markdown("#### 🧮 סטטיסטיקות מהירות")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

        st.markdown("#### 💾 הורדת הנתונים (CSV)")
        st.download_button(
            "הורד CSV מעודכן",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="data_clean.csv",
            mime="text/csv",
        )
    else:
        st.info("טרם נטענו נתונים. העלה/י קובץ או הפעל/י נתוני דוגמה.")

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
            col = st.selectbox("בחר/י עמודה מספרית", options=numeric_cols, index=0)
            user_label = st.text_input("תווית ציר-Y (שם הסדרה)", value=col or "Value")
            safe_label, raw_label = sanitize_field_name(user_label or "Value", default="Value")

            cat_opts = ["(Index)"] + df.columns.tolist()
            cat_col = st.selectbox("בחר/י עמודת קטגוריה (לא חובה)", options=cat_opts, index=0)

            series = ensure_numeric(df[col])

            if cat_col == "(Index)":
                chart = make_bar_chart(series, label=user_label or "Value")
            else:
                grouped = (
                    pd.DataFrame({"cat": df[cat_col].astype(str), safe_label: series})
                    .groupby("cat", as_index=False)[safe_label].sum()
                )
                chart = (
                    alt.Chart(grouped)
                    .mark_bar()
                    .encode(
                        x=alt.X(field="cat", type="nominal", title=cat_col),
                        y=alt.Y(field=safe_label, type="quantitative", title=raw_label),
                    )
                    .properties(height=340)
                )
            st.altair_chart(chart, use_container_width=True)

# =======================
# Tools Tab
# =======================
with tab_tools:
    st.markdown("### 🛠️ כלים")
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
            st.success("העמודות הומרו. עבר/י ללשונית 'גרף' כדי להמחיש.")
            st.dataframe(df.head(30), use_container_width=True)
    else:
        st.info("ראשית, טען/י נתונים בלשונית 'נתונים'.")

# =======================
# AI Tab
# =======================
with tab_ai:
    st.markdown("## 🤖 כלים חכמים (ללא אינטגרציה חיצונית)")
    st.caption("סיכום טקסט וניתוח סנטימנט. אם מותקן transformers — יופעל מודל סנטימנט 'אמיתי'; אחרת, ניתוח קליל מובנה.")

    st.markdown("### ✂️ סיכום טקסט")
    text_to_sum = st.text_area("הדבק/י או כתוב/י טקסט לסיכום", height=160, placeholder="הכנס/י טקסט בעברית או באנגלית...")
    max_sents = st.slider("מספר משפטים בסיכום", min_value=2, max_value=10, value=5, step=1)
    if st.button("סכם/י"):
        if text_to_sum.strip():
            summary = summarize_text_light(text_to_sum, max_sentences=max_sents)
            st.write("**סיכום:**")
            st.write(summary)
        else:
            st.warning("נא להזין טקסט.")

    st.divider()
    st.markdown("### 🙂 ניתוח סנטימנט")
    sent_text = st.text_area("טקסט לניתוח רגש", height=120, placeholder="כתוב/י משפט להערכת חיובי/שלילי/ניטרלי")
    use_transformers = st.toggle("השתמש/י ב-transformers אם זמין", value=TRANSFORMERS_AVAILABLE)
    if st.button("נתח/י"):
        if not sent_text.strip():
            st.warning("נא להזין טקסט.")
        else:
            if use_transformers and TRANSFORMERS_AVAILABLE:
                nlp = get_sentiment_pipeline()
                if nlp is None:
                    st.info("לא נמצא מודל זמין, משתמשים בניתוח הקליל.")
                    res = sentiment_simple(sent_text)
                    st.write(f"**תוצאה:** {res['label']} (בטחון {res['score']:.2f})")
                else:
                    out = nlp(sent_text)[0]
                    label = out.get("label", "")
                    score = out.get("score", 0.0)
                    st.write(f"**תוצאה (transformers):** {label} (בטחון {score:.2f})")
            else:
                res = sentiment_simple(sent_text)
                st.write(f"**תוצאה:** {res['label']} (בטחון {res['score']:.2f})")
                st.caption("טיפ: להתנסות במודל אמיתי, הוסף/י לחבילות: transformers, torch (אופציונלי).")
