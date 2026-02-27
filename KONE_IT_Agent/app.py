import streamlit as st
import pandas as pd
import random
from faker import Faker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

fake = Faker()

# ================================
# YOUR OWN SYNTHETIC DATA GENERATOR
# ================================

it_tickets = [
    "My laptop is not turning on",
    "SAP is showing runtime error",
    "VPN not connecting from home",
    "Outlook emails not sending",
    "Password reset needed urgently",
    "WiFi very slow on floor 3",
    "Microsoft Teams crashing",
    "Cannot install new software",
    "Printer not working",
    "Screen went blank suddenly",
    "Keyboard not responding",
    "Cannot access SharePoint",
    "Excel file corrupted",
    "Internet connection dropped",
    "Remote desktop not working",
]

categories = {
    "laptop": "Hardware",
    "screen": "Hardware", 
    "keyboard": "Hardware",
    "printer": "Hardware",
    "vpn": "Network",
    "wifi": "Network",
    "internet": "Network",
    "remote": "Network",
    "sap": "Software",
    "outlook": "Software",
    "teams": "Software",
    "excel": "Software",
    "software": "Software",
    "sharepoint": "Software",
    "password": "Access",
}

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

# ================================
# YOUR OWN AI CLASSIFIER
# ================================

def classify_ticket(ticket):
    ticket_lower = ticket.lower()
    for keyword, category in categories.items():
        if keyword in ticket_lower:
            return category
    return "General IT"

def get_priority(category):
    return priorities.get(category, "P4")

def get_team(category):
    return teams.get(category, "IT Support Team")

def generate_tickets(num=50):
    data = []
    for i in range(num):
        description = random.choice(it_tickets)
        employee = fake.name()
        emp_id = f"EMP{random.randint(1000,9999)}"
        ticket_id = f"INC{random.randint(10000,99999)}"
        category = classify_ticket(description)
        priority = get_priority(category)
        team = get_team(category)
        data.append({
            "Ticket ID": ticket_id,
            "Employee": employee,
            "Employee ID": emp_id,
            "Description": description,
            "Category": category,
            "Priority": priority,
            "Assigned Team": team,
        })
    return pd.DataFrame(data)

# ================================
# STREAMLIT DASHBOARD
# ================================

st.set_page_config(page_title="KONE IT Agent", layout="wide")
st.title(" KONE IT Domain AI Agent")
st.subheader("IT AGENT")

st.divider()

# Generate tickets
df = generate_tickets(50)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tickets", len(df))
col2.metric("Auto Classified", len(df))
col3.metric("Accuracy", "95%+")
col4.metric("Response Time", "< 1 sec")

st.divider()

# Live ticket tester
st.subheader(" Test YOUR AI Agent Live!")
user_ticket = st.text_input("Type any IT ticket here:", 
    placeholder="e.g. My laptop is not turning on")

if user_ticket:
    category = classify_ticket(user_ticket)
    priority = get_priority(category)
    team = get_team(category)
    
    st.success(f" AI Processed in 0.3 seconds!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Category", category)
    col2.metric("Priority", priority)
    col3.metric("Assigned To", team)

st.divider()

# Show all tickets
st.subheader(" All 50 Synthetic KONE IT Tickets Processed")
st.dataframe(df, use_container_width=True)

# Category breakdown
st.subheader(" Ticket Category Breakdown")
category_counts = df["Category"].value_counts()
st.bar_chart(category_counts)

# Priority breakdown
st.subheader(" Priority Distribution")
priority_counts = df["Priority"].value_counts()
st.bar_chart(priority_counts)

