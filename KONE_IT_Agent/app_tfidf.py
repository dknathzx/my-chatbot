import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# YOUR TRAINING DATA
tickets_train = [
    "my laptop is not turning on",
    "vpn not connecting from home",
    "forgot my password please reset",
    "wifi very slow",
    "teams not working",
    "printer not working",
    "internet connection dropped",
    "screen went blank",
    "sap showing error",
    "password locked out",
    "laptop battery not charging",
    "monitor flickering",
    "cannot access sharepoint",
    "account locked out",
    "network very slow today",
    "need access to new system",
    "excel file corrupted",
    "keyboard not responding",
    "outlook not opening",
    "remote desktop not working",
]

labels_train = [
    "Hardware", "Network", "Access", "Network", "Software",
    "Hardware", "Network", "Hardware", "Software", "Access",
    "Hardware", "Hardware", "Software", "Access", "Network",
    "Access", "Software", "Hardware", "Software", "Network",
]

# TRAIN YOUR MODEL
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('classifier', MultinomialNB()),
])
model.fit(tickets_train, labels_train)

# PRIORITIES AND TEAMS
priorities = {
    "Hardware": "P3",
    "Network": "P2",
    "Software": "P3",
    "Access": "P4",
}

teams = {
    "Hardware": "Hardware Support Team",
    "Network": "Network Operations Team",
    "Software": "Software Support Team",
    "Access": "IT Access Management Team",
}

# STREAMLIT UI
st.set_page_config(page_title="KONE IT Agent", layout="wide")
st.title("🤖 KONE IT Domain AI Agent")
st.subheader("Powered by YOUR TF-IDF ML Model | Zero External AI")
st.divider()

st.subheader("🎯 Type Any IT Ticket — Watch TF-IDF Work Live!")
user_ticket = st.text_input(
    "Type ticket here:",
    placeholder="e.g. My laptop is not turning on"
)

if user_ticket:
    category = model.predict([user_ticket])[0]
    confidence = round(
        max(model.predict_proba([user_ticket])[0]) * 100, 2
    )
    priority = priorities.get(category, "P4")
    team = teams.get(category, "IT Support")

    st.success("✅ TF-IDF ML Model Processed Instantly!")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Category", category)
    col2.metric("Priority", priority)
    col3.metric("Assigned Team", team)
    col4.metric("Confidence", f"{confidence}%")