import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from collections import deque
import plotly.graph_objects as go
import random
import string
import streamlit.components.v1 as components

# Set wide layout
st.set_page_config(layout="wide")

# Parameters
DEFAULT_BETA = 0.0929
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.5
DEFAULT_MEAL_TIMES = [8.0, 13.0, 19.0]
SAMPLING_INTERVAL = 7
CLASSIFIER_DELAY = 2
PROBABILITY_DELAY = CLASSIFIER_DELAY + 1

# Scaling factors
GENDER_SCALE = {"Male": 1.0, "Female": 0.84, "Other": 0.92}
AGE_SCALE_STEEPNESS = 0.1
REFERENCE_AGE = 50
DIET_SCALE = {"Vegetarian": 1.1, "Non-Vegetarian": 1.0}
ACTIVITY_SCALE = {"Sedentary": 0.941, "Intermediate": 1.0, "Active": 1.05}
CONDITION_MEDICATION_SCALE = {"Reduce": 0.9, "Increase": 1.1, "No change": 1.0}

# Sleep schedule
SLEEP_START = 23.0
SLEEP_END = 7.5

# Bowel sound probability distribution
BOWEL_SOUND_TYPES = ["non-BS", "SB", "MB", "CRS", "HS"]
PROB_DISTRIBUTION = {
    "Above 0.9": {"non-BS": 0.05, "SB": 0.40, "MB": 0.30, "CRS": 0.20, "HS": 0.05},
    "0.85-0.9": {"non-BS": 0.10, "SB": 0.35, "MB": 0.30, "CRS": 0.20, "HS": 0.05},
    "0.7-0.85": {"non-BS": 0.20, "SB": 0.30, "MB": 0.25, "CRS": 0.20, "HS": 0.05},
    "Below 0.7": {"non-BS": 0.30, "SB": 0.25, "MB": 0.20, "CRS": 0.15, "HS": 0.10}
}

# Alert color mapping
ALERT_COLORS = {
    "Red alert": "#ff0000",
    "High alert": "#ff6347",
    "Medium alert": "#ffff99",
    "Low alert": "#90ee90"
}

# Helper functions
def hours_to_hhmmss(hours):
    seconds = hours * 3600
    return str(timedelta(seconds=int(seconds))).zfill(8)

def hhmmss_to_hours(hhmmss):
    h, m, s = map(int, hhmmss.split(":"))
    return h + m / 60 + s / 3600

# JavaScript for scroll position
components.html("""
<script>
    function saveScrollPosition() {
        const scrollPos = window.scrollY;
        window.parent.postMessage({type: "save_scroll", scrollPos: scrollPos}, "*");
    }
    window.addEventListener("message", (event) => {
        if (event.data.type === "restore_scroll") {
            window.scrollTo(0, event.data.scrollPos);
        }
    });
    document.addEventListener("DOMContentLoaded", () => {
        const buttons = document.querySelectorAll("button");
        buttons.forEach(button => {
            button.addEventListener("click", saveScrollPosition);
        });
    });
</script>
""", height=0)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Synthetic_User_Data_Model"
if "demo_subpage" not in st.session_state:
    st.session_state.demo_subpage = "patient_selection"
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "card_color" not in st.session_state:
    st.session_state.card_color = "rgba(230, 230, 230, 0.75)"  # Updated to lighter grey with 75% transparency
if "last_final_alert" not in st.session_state:
    st.session_state.last_final_alert = {}
if "scroll_position" not in st.session_state:
    st.session_state.scroll_position = 0
if "all_patients" not in st.session_state:
    default_last_defaecation = hhmmss_to_hours("06:00:00")
    baseline_last_defaecation = hhmmss_to_hours("08:30:00")
    st.session_state.all_patients = {
        "Baseline Patient": {"patient_id": "A123456", "Gender": "Male", "Age": 50, "Diet": "Non-Vegetarian", "Activity": "Intermediate", "Prior Conditions": "No change", "Medication": "No change", "last_defaecation": baseline_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "Mary Johnson": {"patient_id": "B789012", "Gender": "Female", "Age": 45, "Diet": "Non-Vegetarian", "Activity": "Intermediate", "Prior Conditions": "Increase", "Medication": "Increase", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "Alex Taylor": {"patient_id": "C345678", "Gender": "Other", "Age": 60, "Diet": "Non-Vegetarian", "Activity": "Sedentary", "Prior Conditions": "Reduce", "Medication": "Reduce", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "David Brown": {"patient_id": "D901234", "Gender": "Male", "Age": 25, "Diet": "Vegetarian", "Activity": "Active", "Prior Conditions": "No change", "Medication": "Increase", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "Sarah Davis": {"patient_id": "E567890", "Gender": "Female", "Age": 50, "Diet": "Non-Vegetarian", "Activity": "Intermediate", "Prior Conditions": "Increase", "Medication": "No change", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "Michael Wilson": {"patient_id": "F234567", "Gender": "Male", "Age": 40, "Diet": "Non-Vegetarian", "Activity": "Intermediate", "Prior Conditions": "No change", "Medication": "No change", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "Emma Moore": {"patient_id": "G890123", "Gender": "Female", "Age": 55, "Diet": "Non-Vegetarian", "Activity": "Sedentary", "Prior Conditions": "Reduce", "Medication": "Reduce", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "James Lee": {"patient_id": "H456789", "Gender": "Other", "Age": 35, "Diet": "Vegetarian", "Activity": "Active", "Prior Conditions": "Increase", "Medication": "Increase", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "Lisa Clark": {"patient_id": "I012345", "Gender": "Female", "Age": 28, "Diet": "Non-Vegetarian", "Activity": "Intermediate", "Prior Conditions": "No change", "Medication": "No change", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES},
        "Robert Harris": {"patient_id": "J678901", "Gender": "Male", "Age": 62, "Diet": "Non-Vegetarian", "Activity": "Sedentary", "Prior Conditions": "Reduce", "Medication": "Reduce", "last_defaecation": default_last_defaecation, "meal_times": DEFAULT_MEAL_TIMES}
    }
if "patients" not in st.session_state:
    st.session_state.patients = {
        "Baseline Patient": st.session_state.all_patients["Baseline Patient"]
    }
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {
        patient_name: pd.DataFrame(columns=[
            "Gender", "Age", "Diet", "Activity", "Prior Conditions", "Medication",
            "Elapsed Time", "Time Since Last Meal", "Sound Type", "Probability Output", "Time Stamp",
            "Alert Level"
        ]) for patient_name in st.session_state.patients.keys()
    }
for patient_name in st.session_state.patients.keys():
    if patient_name not in st.session_state.last_final_alert:
        st.session_state.last_final_alert[patient_name] = None

# Alert parameters
if "alert_threshold_red" not in st.session_state:
    st.session_state.alert_threshold_red = 0.9
if "alert_threshold_high" not in st.session_state:
    st.session_state.alert_threshold_high = 0.85
if "alert_threshold_medium" not in st.session_state:
    st.session_state.alert_threshold_medium = 0.7
if "consecutive_red_count" not in st.session_state:
    st.session_state.consecutive_red_count = 5

if "consecutive_probs" not in st.session_state:
    st.session_state.consecutive_probs = {patient: deque(maxlen=st.session_state.consecutive_red_count) for patient in st.session_state.patients.keys()}
else:
    for patient in st.session_state.consecutive_probs:
        current_probs = list(st.session_state.consecutive_probs[patient])
        st.session_state.consecutive_probs[patient] = deque(current_probs, maxlen=st.session_state.consecutive_red_count)

if "plot_counter" not in st.session_state:
    st.session_state.plot_counter = 0
if "num_meals" not in st.session_state:
    st.session_state.num_meals = 3
if "meal_times" not in st.session_state:
    st.session_state.meal_times = DEFAULT_MEAL_TIMES

# Restore scroll position
if st.session_state.scroll_position > 0:
    components.html(f"""
    <script>
        window.parent.postMessage({{type: "restore_scroll", scrollPos: {st.session_state.scroll_position}}}, "*");
    </script>
    """, height=0)
    st.session_state.scroll_position = 0

# Helper functions
def generate_patient_id(existing_ids):
    while True:
        prefix = random.choice(string.ascii_uppercase)
        number = "".join(random.choices(string.digits, k=6))
        patient_id = f"{prefix}{number}"
        if patient_id not in existing_ids:
            return patient_id

def calculate_scaling_factor(patient, gender_scale_male, gender_scale_female, gender_scale_other, age_scale_steepness, diet_scale_veg, diet_scale_nonveg, activity_scale_sedentary, activity_scale_intermediate, activity_scale_active, condition_medication_scale_reduce, condition_medication_scale_increase, condition_medication_scale_nochange):
    gender_scales = {"Male": gender_scale_male, "Female": gender_scale_female, "Other": gender_scale_other}
    gender_scale = gender_scales[patient["Gender"]]
    age = patient["Age"]
    age_scale = 1 - 0.5 * np.tanh(age_scale_steepness * (age - REFERENCE_AGE))
    diet_scales = {"Vegetarian": diet_scale_veg, "Non-Vegetarian": diet_scale_nonveg}
    diet_scale = diet_scales[patient["Diet"]]
    activity_scales = {"Sedentary": activity_scale_sedentary, "Intermediate": activity_scale_intermediate, "Active": activity_scale_active}
    activity_scale = activity_scales[patient["Activity"]]
    condition_scale = {"Reduce": condition_medication_scale_reduce, "Increase": condition_medication_scale_increase, "No change": condition_medication_scale_nochange}[patient["Prior Conditions"]]
    medication_scale = {"Reduce": condition_medication_scale_reduce, "Increase": condition_medication_scale_increase, "No change": condition_medication_scale_nochange}[patient["Medication"]]
    return gender_scale * age_scale * diet_scale * activity_scale * condition_scale * medication_scale

def hazard_function(t, meal_times, beta, alpha, gamma, S_i):
    h = S_i * beta
    for meal_time in meal_times:
        h += alpha * np.exp(-gamma * max(t - meal_time, 0))
    return h

def cumulative_hazard(t, meal_times, beta, alpha, gamma, S_i):
    H = S_i * beta * t
    for meal_time in meal_times:
        H += (alpha / gamma) * (1 - np.exp(-gamma * max(t - meal_time, 0)))
    return H

def get_last_meal(elapsed_time, meal_times):
    past_meals = [mt for mt in meal_times if mt <= elapsed_time]
    if not past_meals:
        return "00:00:00"
    last_meal = max(past_meals)
    return hours_to_hhmmss(elapsed_time - last_meal)

def is_sleeping(clock_time):
    clock_time = clock_time % 24
    if SLEEP_START > SLEEP_END:
        return clock_time >= SLEEP_START or clock_time <= SLEEP_END
    return SLEEP_START <= clock_time <= SLEEP_END

def get_alert(prob, patient_name, clock_time):
    st.session_state.consecutive_probs[patient_name].append(prob)
    red_count = sum(1 for p in st.session_state.consecutive_probs[patient_name] if p > st.session_state.alert_threshold_red)
    if red_count >= st.session_state.consecutive_red_count:
        alert = "Red alert"
    elif prob > st.session_state.alert_threshold_high:
        alert = "High alert"
    elif st.session_state.alert_threshold_medium <= prob <= st.session_state.alert_threshold_high:
        alert = "Medium alert"
    else:
        alert = "Low alert"
    if is_sleeping(clock_time):
        if alert == "Red alert":
            alert = "High alert"
        elif alert == "High alert":
            alert = "Medium alert"
        elif alert == "Medium alert":
            alert = "Low alert"
    return alert

def assign_bowel_sound(prob):
    if prob > 0.9:
        dist_key = "Above 0.9"
    elif 0.85 < prob <= 0.9:
        dist_key = "0.85-0.9"
    elif 0.7 <= prob <= 0.85:
        dist_key = "0.7-0.85"
    else:
        dist_key = "Below 0.7"
    probabilities = PROB_DISTRIBUTION[dist_key]
    return np.random.choice(
        BOWEL_SOUND_TYPES,
        p=[probabilities[sound] for sound in BOWEL_SOUND_TYPES]
    )

def get_card_color(patient_name, patient_data_dict):
    if patient_name not in patient_data_dict or patient_data_dict[patient_name].empty:
        return "rgba(230, 230, 230, 0.75)"  # Updated to lighter grey with 75% transparency
    latest_alert = patient_data_dict[patient_name][patient_data_dict[patient_name]["Alert Level"] != "Processing..."]["Alert Level"]
    if not latest_alert.empty:
        return ALERT_COLORS.get(latest_alert.iloc[-1], "rgba(230, 230, 230, 0.75)")  # Updated fallback
    alert_level = st.session_state.last_final_alert.get(patient_name)
    return ALERT_COLORS.get(alert_level, "rgba(230, 230, 230, 0.75)")  # Updated fallback

def hours_to_clock_time(hours, base_time=0.0):
    total_hours = (base_time + hours) % 24
    seconds = total_hours * 3600
    return str(timedelta(seconds=int(seconds))).zfill(8)

def create_cdf_plot(current_time, patient_data, demo_active, last_defaecation_time, meal_times, simulation_start_time, S_i, beta=DEFAULT_BETA, alpha=DEFAULT_ALPHA, gamma=DEFAULT_GAMMA):
    extended_meal_times = meal_times + [mt + 24 for mt in meal_times]
    adjusted_meal_times = [mt - last_defaecation_time for mt in extended_meal_times if last_defaecation_time <= mt <= (last_defaecation_time + 25)]
    t = np.linspace(0, 25, 500)
    cdf = [1 - np.exp(-cumulative_hazard(ti, adjusted_meal_times, beta, alpha, gamma, S_i)) for ti in t]
    plot_times = [last_defaecation_time + ti for ti in t]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_times, y=cdf, mode='lines', name='Probability', line=dict(color='blue')))
    
    plot_start = last_defaecation_time
    plot_end = last_defaecation_time + 25
    sleep_start_day1 = last_defaecation_time - (last_defaecation_time % 24) + SLEEP_START if SLEEP_START >= last_defaecation_time % 24 else last_defaecation_time - (last_defaecation_time % 24) + 24 + SLEEP_START
    if sleep_start_day1 < plot_start:
        sleep_start_day1 += 24
    sleep_end_day1 = sleep_start_day1 - SLEEP_START + 24 + SLEEP_END
    if sleep_start_day1 < plot_end and sleep_end_day1 > plot_start:
        start = max(plot_start, sleep_start_day1)
        end = min(plot_end, sleep_end_day1)
        if end > start:
            fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1, fillcolor="gray", opacity=0.2, layer="below", line_width=0)
    
    sleep_start_day2 = sleep_start_day1 + 24
    sleep_end_day2 = sleep_end_day1 + 24
    if sleep_start_day2 < plot_end and sleep_end_day2 > plot_start:
        start = max(plot_start, sleep_start_day2)
        end = min(plot_end, sleep_end_day2)
        if end > start:
            fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1, fillcolor="gray", opacity=0.2, layer="below", line_width=0)
    
    if demo_active and current_time is not None and patient_data is not None:
        times = [hhmmss_to_hours(row["Elapsed Time"]) for _, row in patient_data.iterrows()]
        plot_times_data = [last_defaecation_time + ti for ti in times]
        probs = patient_data["Probability Output"].apply(lambda x: x if isinstance(x, float) else np.nan).tolist()
        fig.add_trace(go.Scatter(x=plot_times_data, y=probs, mode='markers', name='Sampled Probabilities', marker=dict(color='blue', size=10)))
        current_clock_time = (last_defaecation_time + current_time)
        window = 0.01
        fig.update_xaxes(range=[current_clock_time - window, current_clock_time + window])
        fig.update_yaxes(range=[0, 1])
    else:
        fig.update_xaxes(range=[last_defaecation_time, last_defaecation_time + 25])
        fig.update_yaxes(range=[0, 1])
    
    fig.update_layout(
        title="Cumulative Distribution Function (CDF) - Clock Time (Sleep Periods Highlighted)",
        xaxis_title="Time of Day (hours)",
        yaxis_title="Probability",
        showlegend=True,
        template="plotly_white"
    )
    
    tick_vals = [last_defaecation_time + i for i in range(0, 26, 2)]
    tick_text = [f"{str(timedelta(seconds=int((val % 24) * 3600))).zfill(8)[:5]}{' (+1 day)' if val >= 24 else ''}" for val in tick_vals]
    fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text)
    return fig

def create_adjustable_cdf_plot(beta, alpha, gamma, meal_times, gender, age, gender_scale_male, gender_scale_female, gender_scale_other, age_scale_steepness, diet_scale_veg, diet_scale_nonveg, activity_scale_sedentary, activity_scale_intermediate, activity_scale_active, condition_medication_scale_reduce, condition_medication_scale_increase, condition_medication_scale_nochange):
    example_patient = {
        "Gender": gender,
        "Age": age,
        "Diet": "Non-Vegetarian",
        "Activity": "Intermediate",
        "Prior Conditions": "No change",
        "Medication": "No change"
    }
    S_i = calculate_scaling_factor(
        example_patient,
        gender_scale_male, gender_scale_female, gender_scale_other,
        age_scale_steepness,
        diet_scale_veg, diet_scale_nonveg,
        activity_scale_sedentary, activity_scale_intermediate, activity_scale_active,
        condition_medication_scale_reduce, condition_medication_scale_increase, condition_medication_scale_nochange
    )
    t = np.linspace(0, 24, 500)
    cdf = [1 - np.exp(-cumulative_hazard(ti, meal_times, beta, alpha, gamma, S_i)) for ti in t]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=cdf, mode='lines', name='Probability', line=dict(color='blue')))
    fig.update_layout(
        title=f"Cumulative Distribution Function (CDF) for {age}-Year-Old {gender}",
        xaxis_title="Time (hours)",
        yaxis_title="Probability",
        showlegend=True,
        template="plotly_white",
        xaxis=dict(range=[0, 24]),
        yaxis=dict(range=[0, 1])
    )
    return fig

# Streamlit app
st.title("Bowel Movement Alert System Demo")

# Custom CSS (Updated .summary-card to support transparency)
st.markdown("""
    <style>
    .stApp { max-width: 1400px; margin: 0 auto; }
    .patient-card { border: 2px solid rgba(200, 200, 200, 0.5); border-radius: 10px; padding: 10px; margin: 5px; text-align: center; transition: background-color 0.2s; color: #1C2526; }
    .patient-card:hover { filter: brightness(90%); }
    .patient-card h3 { margin: 0; font-size: 18px; }
    .patient-card p { margin: 3px 0; font-size: 14px; }
    .tag { background-color: white; border: 1px solid #d3d3d3; border-radius: 5px; padding: 2px 6px; margin: 2px; display: inline-block; font-size: 12px; }
    .summary-card { border: 2px solid rgba(211, 211, 211, 0.5); border-radius: 10px; padding: 10px; margin: 5px 0; background-color: rgba(255, 255, 255, 0.9); transition: background-color 0.3s ease; } /* Added transition and base background */
    .summary-card p { margin: 3px 0; font-size: 16px; }
    .content-container { width: 100%; max-width: 100%; }
    .custom-table { width: 100%; max-width: 100%; border-collapse: collapse; overflow-x: auto; display: block; box-sizing: border-box; }
    .custom-table th, .custom-table td { padding: 4px; text-align: center; white-space: normal; border: 1px solid #ddd; box-sizing: border-box; font-size: 12px; }
    .nav-buttons { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .sub-nav-buttons { display: flex; justify-content: center; margin-bottom: 10px; gap: 10px; }
    .compact-section { margin-bottom: 5px; }
    .compact-section h3 { margin: 5px 0; }
    </style>
""", unsafe_allow_html=True)

# Navigation
st.markdown('<div class="nav-buttons">', unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Synthetic User Data Model", disabled=(st.session_state.page == "Synthetic_User_Data_Model")):
        st.session_state.page = "Synthetic_User_Data_Model"
        st.rerun()
with col2:
    if st.button("Demo", disabled=(st.session_state.page == "demo")):
        st.session_state.page = "demo"
        st.session_state.demo_subpage = "patient_selection"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Page: Parameter Adjustment
if st.session_state.page == "Synthetic_User_Data_Model":
    st.header("Synthetic User Data Model")
    st.write("Adjust parameters and scaling factors to see their effect on the probability over time.")
    
    st.subheader("Probability Model")
    st.latex(r"h_i(t) = S_i \cdot \beta_0 + \sum_m \alpha_m e^{-\gamma (t - t_{i,m})}")
    st.latex(r"H_i(t) = S_i \cdot \beta_0 \cdot t + \sum_m \frac{\alpha_m}{\gamma} \left( 1 - e^{-\gamma (t - t_{i,m})} \right)")
    st.latex(r"F_i(t) = 1 - e^{-H_i(t)}")
    st.latex(r"S_i = S_{\text{gender}} \cdot S_{\text{age}} \cdot S_{\text{diet}} \cdot S_{\text{activity}} \cdot S_{\text{condition}} \cdot S_{\text{medication}}")
    st.latex(r"S_{\text{age}} = 1 - 0.5 \cdot \tanh(k \cdot (\text{age} - 50))")
    
    st.subheader("Example Patient Attributes")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox(r"Gender", ["Male", "Female", "Other"], key="param_gender")
    with col2:
        age = st.slider(r"Age", 20, 80, 50, step=1)
    
    st.write(f"The example patient is a {age}-year-old {gender}, non-vegetarian, with intermediate activity, no change in prior conditions, and no change in medication.")
    
    st.subheader("Model Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        beta = st.slider(r"$\beta_0$: Baseline Hazard Rate", 0.01, 1.0, DEFAULT_BETA, step=0.01)
    with col2:
        alpha = st.slider(r"$\alpha_m$: Meal Effect Magnitude", 0.01, 2.0, DEFAULT_ALPHA, step=0.01)
    with col3:
        gamma = st.slider(r"$\gamma$: Decay Rate", 0.1, 1.0, DEFAULT_GAMMA, step=0.1)
    
    st.subheader("Scaling Factors")
    col1, col2 = st.columns(2)
    with col1:
        gender_scale_male = st.slider(r"$S_{\text{gender, male}}$", 0.5, 1.5, GENDER_SCALE["Male"], step=0.01)
        gender_scale_female = st.slider(r"$S_{\text{gender, female}}$", 0.5, 1.5, GENDER_SCALE["Female"], step=0.01)
        gender_scale_other = st.slider(r"$S_{\text{gender, other}}$", 0.5, 1.5, GENDER_SCALE["Other"], step=0.01)
        age_scale_steepness = st.slider(r"$k$: Age Scale Steepness", 0.01, 1.0, AGE_SCALE_STEEPNESS, step=0.01)
        diet_scale_veg = st.slider(r"$S_{\text{diet, veg}}$", 0.5, 1.5, DIET_SCALE["Vegetarian"], step=0.01)
        diet_scale_nonveg = st.slider(r"$S_{\text{diet, nonveg}}$", 0.5, 1.5, DIET_SCALE["Non-Vegetarian"], step=0.01)
    with col2:
        activity_scale_sedentary = st.slider(r"$S_{\text{activity, sedentary}}$", 0.5, 1.5, ACTIVITY_SCALE["Sedentary"], step=0.01)
        activity_scale_intermediate = st.slider(r"$S_{\text{activity, intermediate}}$", 0.5, 1.5, ACTIVITY_SCALE["Intermediate"], step=0.01)
        activity_scale_active = st.slider(r"$S_{\text{activity, active}}$", 0.5, 1.5, ACTIVITY_SCALE["Active"], step=0.01)
        condition_medication_scale_reduce = st.slider(r"$S_{\text{condition/medication, reduce}}$", 0.5, 1.5, CONDITION_MEDICATION_SCALE["Reduce"], step=0.01)
        condition_medication_scale_increase = st.slider(r"$S_{\text{condition/medication, increase}}$", 0.5, 1.5, CONDITION_MEDICATION_SCALE["Increase"], step=0.01)
        condition_medication_scale_nochange = st.slider(r"$S_{\text{condition/medication, no change}}$", 0.5, 1.5, CONDITION_MEDICATION_SCALE["No change"], step=0.01)
    
    st.markdown('<div class="compact-section">', unsafe_allow_html=True)
    st.subheader(r"Meal Times $t_{i,m}$")
    num_meals = st.number_input("Number of Meals", min_value=0, max_value=10, value=st.session_state.num_meals, step=1)
    if num_meals != st.session_state.num_meals:
        st.session_state.num_meals = num_meals
        if num_meals == 0:
            st.session_state.meal_times = []
        else:
            current_meal_times = st.session_state.meal_times
            if len(current_meal_times) > num_meals:
                st.session_state.meal_times = current_meal_times[:num_meals]
            elif len(current_meal_times) < num_meals:
                additional_meals = num_meals - len(current_meal_times)
                default_meal_times = [8.0, 13.0, 19.0, 6.0, 17.0, 10.0, 15.0, 20.0, 12.0, 22.0]
                new_meal_times = current_meal_times + default_meal_times[:additional_meals]
                if len(new_meal_times) < num_meals:
                    new_meal_times += [0.0] * (num_meals - len(new_meal_times))
                st.session_state.meal_times = new_meal_times[:num_meals]
    
    meal_times = []
    if num_meals > 0:
        cols = st.columns(min(num_meals, 5))
        for i in range(num_meals):
            with cols[i % 5]:
                meal_time = st.number_input(
                    f"Meal {i + 1}",
                    min_value=0.0,
                    max_value=24.0,
                    value=st.session_state.meal_times[i] if i < len(st.session_state.meal_times) else 0.0,
                    step=0.5,
                    key=f"meal_time_{i}"
                )
                meal_times.append(meal_time)
        st.session_state.meal_times = meal_times
    else:
        st.session_state.meal_times = []
        st.write("No meals specified. Using only the baseline hazard rate.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    meal_times = sorted(st.session_state.meal_times)
    
    try:
        st.plotly_chart(
            create_adjustable_cdf_plot(
                beta, alpha, gamma, meal_times,
                gender, age,
                gender_scale_male, gender_scale_female, gender_scale_other,
                age_scale_steepness,
                diet_scale_veg, diet_scale_nonveg,
                activity_scale_sedentary, activity_scale_intermediate, activity_scale_active,
                condition_medication_scale_reduce, condition_medication_scale_increase, condition_medication_scale_nochange
            ),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")

# Page: Demo
elif st.session_state.page == "demo":
    st.header("Demo")
    
    st.markdown('<div class="sub-nav-buttons">', unsafe_allow_html=True)
    if st.button("Select Patient", disabled=(st.session_state.demo_subpage == "patient_selection")):
        st.session_state.demo_subpage = "patient_selection"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.demo_subpage == "patient_selection":
        st.subheader("Select a Patient")
        search_query = st.text_input("Search Patients by Name or ID", placeholder="Enter name or patient ID")
        
        with st.expander("Search by Tags"):
            st.subheader("Filter by Attributes")
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_genders = st.multiselect("Gender", ["Male", "Female", "Other"], key="filter_gender")
                selected_diets = st.multiselect("Diet", ["Vegetarian", "Non-Vegetarian"], key="filter_diet")
            with col2:
                selected_activities = st.multiselect("Activity", ["Sedentary", "Intermediate", "Active"], key="filter_activity")
                selected_conditions = st.multiselect("Prior Conditions", ["Reduce", "Increase", "No change"], key="filter_conditions")
            with col3:
                selected_medications = st.multiselect("Medication", ["Reduce", "Increase", "No change"], key="filter_medication")
        
        with st.expander("Sort Options"):
            st.subheader("Sort Patients")
            col1, col2 = st.columns(2)
            with col1:
                sort_by_age = st.selectbox("Sort by Age", ["None", "Increasing", "Decreasing"], key="sort_age")
            with col2:
                sort_by_prob = st.selectbox("Sort by Latest Probability", ["None", "Increasing", "Decreasing"], key="sort_prob")
        
        filtered_patients = {
            name: data for name, data in st.session_state.patients.items()
            if search_query.lower() in name.lower() or search_query.lower() in data["patient_id"].lower()
        }
        
        if selected_genders:
            filtered_patients = {name: data for name, data in filtered_patients.items() if data["Gender"] in selected_genders}
        if selected_diets:
            filtered_patients = {name: data for name, data in filtered_patients.items() if data["Diet"] in selected_diets}
        if selected_activities:
            filtered_patients = {name: data for name, data in filtered_patients.items() if data["Activity"] in selected_activities}
        if selected_conditions:
            filtered_patients = {name: data for name, data in filtered_patients.items() if data["Prior Conditions"] in selected_conditions}
        if selected_medications:
            filtered_patients = {name: data for name, data in filtered_patients.items() if data["Medication"] in selected_medications}
        
        if sort_by_age != "None":
            filtered_patients = dict(sorted(filtered_patients.items(), key=lambda x: x[1]["Age"], reverse=(sort_by_age == "Decreasing")))
        
        if sort_by_prob != "None":
            def get_latest_prob(patient_name):
                if patient_name in st.session_state.patient_data and not st.session_state.patient_data[patient_name].empty:
                    return st.session_state.patient_data[patient_name]["Probability Output"].iloc[-1]
                return float('-inf') if sort_by_prob == "Increasing" else float('inf')
            filtered_patients = dict(sorted(filtered_patients.items(), key=lambda x: get_latest_prob(x[0]), reverse=(sort_by_prob == "Decreasing")))
        
        if not filtered_patients:
            st.warning("No patients match the search criteria. Showing all patients or add a new one.")
            filtered_patients = st.session_state.patients
        
        with st.expander("Adjust Alert Processing"):
            st.subheader("Alert Processing Parameters")
            st.write("Alerts are downgraded during sleep periods (11:00 PM to 7:30 AM).")
            col1, col2 = st.columns(2)
            with col1:
                new_red = st.slider("Red Alert Threshold", 0.0, 1.0, st.session_state.alert_threshold_red, step=0.01, key="alert_red")
                if new_red != st.session_state.alert_threshold_red:
                    st.session_state.alert_threshold_red = new_red
                    st.session_state.alert_threshold_high = min(st.session_state.alert_threshold_high, new_red)
                    st.session_state.alert_threshold_medium = min(st.session_state.alert_threshold_medium, st.session_state.alert_threshold_high)
                new_high = st.slider("High Alert Threshold", 0.0, st.session_state.alert_threshold_red, st.session_state.alert_threshold_high, step=0.01, key="alert_high")
                if new_high != st.session_state.alert_threshold_high:
                    st.session_state.alert_threshold_high = new_high
                    st.session_state.alert_threshold_medium = min(st.session_state.alert_threshold_medium, new_high)
            with col2:
                new_medium = st.slider("Medium Alert Threshold", 0.0, st.session_state.alert_threshold_high, st.session_state.alert_threshold_medium, step=0.01, key="alert_medium")
                if new_medium != st.session_state.alert_threshold_medium:
                    st.session_state.alert_threshold_medium = new_medium
                new_consecutive = st.slider("Consecutive Red Alerts for Red Alert", 1, 10, st.session_state.consecutive_red_count, step=1, key="alert_consecutive")
                if new_consecutive != st.session_state.consecutive_red_count:
                    st.session_state.consecutive_red_count = new_consecutive
                    for patient in st.session_state.consecutive_probs:
                        current_probs = list(st.session_state.consecutive_probs[patient])
                        st.session_state.consecutive_probs[patient] = deque(current_probs, maxlen=new_consecutive)
            
            if st.button("Reprocess Alerts with New Thresholds"):
                for patient_name in st.session_state.patient_data.keys():
                    if not st.session_state.patient_data[patient_name].empty:
                        st.session_state.consecutive_probs[patient_name].clear()
                        new_alerts = []
                        for idx, row in st.session_state.patient_data[patient_name].iterrows():
                            prob = row["Probability Output"]
                            elapsed_time = hhmmss_to_hours(row["Elapsed Time"])
                            clock_time = (st.session_state.patients[patient_name]["last_defaecation"] + elapsed_time) % 24
                            new_alerts.append(get_alert(prob, patient_name, clock_time))
                        st.session_state.patient_data[patient_name]["Alert Level"] = new_alerts
                        if new_alerts:
                            st.session_state.last_final_alert[patient_name] = new_alerts[-1]
                            st.session_state.card_color = ALERT_COLORS.get(new_alerts[-1], "rgba(230, 230, 230, 0.75)")  # Updated to lighter grey
                st.success("Alerts reprocessed for all patients!")
                st.session_state.scroll_position = 0
                st.rerun()
        
        with st.expander("Add New Patient"):
            st.subheader("Add New Patient")
            new_patient_name = st.text_input("Patient Name", placeholder="Enter full name")
            new_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="new_gender")
            new_age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
            new_diet = st.selectbox("Diet", ["Vegetarian", "Non-Vegetarian"], key="new_diet")
            new_activity = st.selectbox("Activity", ["Sedentary", "Intermediate", "Active"], key="new_activity")
            new_conditions = st.selectbox("Prior Conditions", ["Reduce", "Increase", "No change"], key="new_conditions")
            new_medication = st.selectbox("Medication", ["Reduce", "Increase", "No change"], key="new_medication")
            last_defaecation_input = st.text_input("Last Defaecation Time (HH:MM:SS)", value="06:00:00")
            meal_times_input = st.text_input("Meal Times (comma-separated hours)", value="8,13,19")
            
            try:
                last_defaecation = hhmmss_to_hours(last_defaecation_input)
            except ValueError:
                st.error("Invalid last defaecation time format. Using default 06:00:00.")
                last_defaecation = hhmmss_to_hours("06:00:00")
            
            try:
                meal_times = [float(x.strip()) for x in meal_times_input.split(",")]
                meal_times = [mt for mt in meal_times if 0 <= mt <= 24]
                if len(meal_times) != 3:
                    st.error("Please enter exactly 3 meal times. Using default meal times.")
                    meal_times = DEFAULT_MEAL_TIMES
            except ValueError:
                st.error("Invalid meal times format. Using default meal times (8, 13, 19).")
                meal_times = DEFAULT_MEAL_TIMES
            
            if st.button("Add Patient"):
                if not new_patient_name.strip():
                    st.error("Patient name cannot be empty.")
                elif new_patient_name in st.session_state.all_patients:
                    st.error("Patient name already exists. Please use a unique name.")
                else:
                    existing_ids = [data["patient_id"] for data in st.session_state.all_patients.values()]
                    new_patient_id = generate_patient_id(existing_ids)
                    st.session_state.all_patients[new_patient_name] = {
                        "patient_id": new_patient_id,
                        "Gender": new_gender,
                        "Age": new_age,
                        "Diet": new_diet,
                        "Activity": new_activity,
                        "Prior Conditions": new_conditions,
                        "Medication": new_medication,
                        "last_defaecation": last_defaecation,
                        "meal_times": meal_times
                    }
                    st.session_state.patients[new_patient_name] = st.session_state.all_patients[new_patient_name]
                    st.session_state.patient_data[new_patient_name] = pd.DataFrame(columns=[
                        "Gender", "Age", "Diet", "Activity", "Prior Conditions", "Medication",
                        "Elapsed Time", "Time Since Last Meal", "Sound Type", "Probability Output", "Time Stamp",
                        "Alert Level"
                    ])
                    st.session_state.consecutive_probs[new_patient_name] = deque(maxlen=st.session_state.consecutive_red_count)
                    st.session_state.last_final_alert[new_patient_name] = None
                    st.success(f"Patient '{new_patient_name}' (ID: {new_patient_id}) added successfully!")
                    st.session_state.scroll_position = 0
                    st.rerun()
        
        with st.expander("Add Existing Patient by ID"):
            st.subheader("Add Patient by ID")
            current_ids = [data["patient_id"] for data in st.session_state.patients.values()]
            available_ids = [data["patient_id"] for name, data in st.session_state.all_patients.items() if data["patient_id"] not in current_ids]
            if not available_ids:
                st.warning("No additional patients available to add.")
            else:
                selected_id = st.selectbox("Select Patient ID", available_ids, key="existing_patient_id")
                if st.button("Add Existing Patient"):
                    for name, data in st.session_state.all_patients.items():
                        if data["patient_id"] == selected_id:
                            st.session_state.patients[name] = data
                            st.session_state.patient_data[name] = pd.DataFrame(columns=[
                                "Gender", "Age", "Diet", "Activity", "Prior Conditions", "Medication",
                                "Elapsed Time", "Time Since Last Meal", "Sound Type", "Probability Output", "Time Stamp",
                                "Alert Level"
                            ])
                            st.session_state.consecutive_probs[name] = deque(maxlen=st.session_state.consecutive_red_count)
                            st.session_state.last_final_alert[name] = None
                            st.success(f"Patient '{name}' (ID: {selected_id}) added to the system!")
                            st.session_state.scroll_position = 0
                            st.rerun()
                            break
        
        with st.expander("Remove Patient"):
            st.subheader("Remove a Patient")
            patient_to_remove = st.selectbox("Select Patient to Remove", list(st.session_state.patients.keys()), key="remove_patient")
            if st.button("Remove Patient"):
                if patient_to_remove == st.session_state.selected_patient:
                    st.error("Cannot remove the currently selected patient.")
                elif len(st.session_state.patients) <= 1:
                    st.error("Cannot remove the last patient.")
                else:
                    del st.session_state.patients[patient_to_remove]
                    del st.session_state.patient_data[patient_to_remove]
                    del st.session_state.consecutive_probs[patient_to_remove]
                    if patient_to_remove in st.session_state.last_final_alert:
                        del st.session_state.last_final_alert[patient_to_remove]
                    st.success(f"Patient '{patient_to_remove}' removed successfully!")
                    st.session_state.scroll_position = 0
                    st.rerun()
        
        cols = st.columns(3)
        for idx, (patient_name, patient_data) in enumerate(filtered_patients.items()):
            col = cols[idx % 3]
            with col:
                card_color = get_card_color(patient_name, st.session_state.patient_data)
                st.markdown(f"""
                    <div class="patient-card" style="background-color: {card_color};">
                        <h3>{patient_name}</h3>
                        <div>
                            <span class="tag">{patient_data['Gender']}</span>
                            <span class="tag">{patient_data['Age']} years</span>
                            <span class="tag">{patient_data['Diet']}</span>
                            <span class="tag">{patient_data['Activity']}</span>
                            <span class="tag">{patient_data['Prior Conditions']}</span>
                        </div>
                        <p>Last Defaecation: {hours_to_hhmmss(patient_data['last_defaecation'])}</p>
                    </div>
                """, unsafe_allow_html=True)
                if col.button("Select", key=f"select_{patient_name}", use_container_width=True):
                    st.session_state.selected_patient = patient_name
                    st.session_state.demo_subpage = "simulation"
                    st.session_state.card_color = "rgba(230, 230, 230, 0.75)"  # Updated to lighter grey
                    st.rerun()
    
    elif st.session_state.demo_subpage == "simulation":
        if st.session_state.selected_patient not in st.session_state.patients:
            st.error("Selected patient not found. Returning to patient selection.")
            st.session_state.demo_subpage = "patient_selection"
            st.rerun()
        else:
            patient = st.session_state.patients[st.session_state.selected_patient]
            last_defaecation_time = patient["last_defaecation"]
            meal_times = patient["meal_times"]
            S_i = calculate_scaling_factor(
                patient,
                GENDER_SCALE["Male"], GENDER_SCALE["Female"], GENDER_SCALE["Other"],
                AGE_SCALE_STEEPNESS,
                DIET_SCALE["Vegetarian"], DIET_SCALE["Non-Vegetarian"],
                ACTIVITY_SCALE["Sedentary"], ACTIVITY_SCALE["Intermediate"], ACTIVITY_SCALE["Active"],
                CONDITION_MEDICATION_SCALE["Reduce"], CONDITION_MEDICATION_SCALE["Increase"], CONDITION_MEDICATION_SCALE["No change"]
            )
            
            st.subheader(f"Simulation for {st.session_state.selected_patient}")
            
            card_color = st.session_state.card_color
            end_time = (last_defaecation_time + 1) % 24
            card_placeholder = st.empty()
            with card_placeholder.container():
                st.markdown(f"""
                    <div class="summary-card" style="background-color: {card_color};">
                        <h3>Patient Profile</h3>
                        <p>Patient ID: {patient['patient_id']}</p>
                        <div>
                            <span class="tag">{patient['Gender']}</span>
                            <span class="tag">{patient['Age']} years</span>
                            <span class="tag">{patient['Diet']}</span>
                            <span class="tag">{patient['Activity']}</span>
                            <span class="tag">{patient['Prior Conditions']}</span>
                            <span class="tag">{patient['Medication']}</span>
                        </div>
                        <p>Scaling Factor (S_i): {S_i:.3f}</p>
                        <p>Last Defaecation: {hours_to_hhmmss(last_defaecation_time)}</p>
                        <p>Meal Times: {', '.join([hours_to_hhmmss(mt) for mt in meal_times])}</p>
                        <p>Simulation End Time: {hours_to_hhmmss(end_time)} (next day)</p>
                    </div>
                """, unsafe_allow_html=True)
            
            simulation_start_input = st.text_input("Simulation Start Time (HH:MM:SS)", value="10:00:00")
            try:
                simulation_start_time = hhmmss_to_hours(simulation_start_input)
                simulation_end_time = (last_defaecation_time + 25) % 24
                if simulation_start_time < last_defaecation_time:
                    st.error(f"Simulation start time must be after last defaecation time. Using last defaecation time.")
                    simulation_start_time = last_defaecation_time
                elif simulation_start_time >= 24 and simulation_start_time > simulation_end_time:
                    st.error(f"Simulation start time must be before the end time. Using last defaecation time.")
                    simulation_start_time = last_defaecation_time
            except ValueError:
                st.error("Invalid simulation start time format. Using last defaecation time.")
                simulation_start_time = last_defaecation_time
            
            duration = st.number_input("Demo Duration (seconds)", min_value=10, max_value=3600, value=60, step=10)
            
            plot_placeholder = st.empty()
            table_placeholder = st.empty()
            
            def render_table(df):
                if df.empty:
                    return ""
                html = '<table class="custom-table"><thead><tr>'
                for col in df.columns:
                    html += f'<th>{col}</th>'
                html += '</tr></thead><tbody>'
                for idx, row in df.iterrows():
                    html += '<tr>'
                    for col_idx, (col, val) in enumerate(row.items()):
                        if col == "Alert Level":
                            if val == "Processing...":
                                style = 'style="background-color: transparent; color: #000000;"'
                            else:
                                color = ALERT_COLORS.get(val, "#ffffff")
                                text_color = '#000000' if val != 'Medium alert' else '#000000'
                                style = f'style="background-color: {color}; color: {text_color};"'
                        else:
                            style = ''
                        html += f'<td {style}>{val}</td>'
                    html += '</tr>'
                html += '</tbody></table>'
                return html
            
            @st.fragment
            def update_demo_content(current_time, patient_data, demo_active, plot_key):
                st.session_state.card_color = get_card_color(st.session_state.selected_patient, patient_data)
                with card_placeholder.container():
                    st.markdown(f"""
                        <div class="summary-card" style="background-color: {st.session_state.card_color};">
                            <h3>Patient Profile</h3>
                            <p>Patient ID: {patient['patient_id']}</p>
                            <div>
                                <span class="tag">{patient['Gender']}</span>
                                <span class="tag">{patient['Age']} years</span>
                                <span class="tag">{patient['Diet']}</span>
                                <span class="tag">{patient['Activity']}</span>
                                <span class="tag">{patient['Prior Conditions']}</span>
                                <span class="tag">{patient['Medication']}</span>
                            </div>
                            <p>Scaling Factor (S_i): {S_i:.3f}</p>
                            <p>Last Defaecation: {hours_to_hhmmss(last_defaecation_time)}</p>
                            <p>Meal Times: {', '.join([hours_to_hhmmss(mt) for mt in meal_times])}</p>
                            <p>Simulation End Time: {hours_to_hhmmss(end_time)} (next day)</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="content-container">', unsafe_allow_html=True)
                    try:
                        chart = create_cdf_plot(
                            current_time,
                            patient_data,
                            demo_active,
                            last_defaecation_time,
                            meal_times,
                            simulation_start_time,
                            S_i
                        )
                        plot_placeholder.plotly_chart(chart, use_container_width=True, key=plot_key)
                    except Exception as e:
                        st.error(f"Error rendering Plotly chart: {str(e)}")
                    table_html = render_table(patient_data)
                    table_placeholder.markdown(table_html, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            update_demo_content(None, st.session_state.patient_data[st.session_state.selected_patient], demo_active=False, plot_key="initial_plot")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Demo"):
                    st.session_state.patient_data[st.session_state.selected_patient] = pd.DataFrame(columns=[
                        "Gender", "Age", "Diet", "Activity", "Prior Conditions", "Medication",
                        "Elapsed Time", "Time Since Last Meal", "Sound Type", "Probability Output", "Time Stamp",
                        "Alert Level"
                    ])
                    st.session_state.consecutive_probs[st.session_state.selected_patient].clear()
                    current_time = simulation_start_time - last_defaecation_time
                    
                    initial_time_seconds = simulation_start_time * 3600
                    time_stamp = str(timedelta(seconds=int(initial_time_seconds))).zfill(8)
                    
                    start_clock = time.time()
                    
                    while (time.time() - start_clock) < duration and current_time <= 25.0:
                        loop_start = time.time()
                        
                        extended_meal_times = meal_times + [mt + 24 for mt in meal_times]
                        adjusted_meal_times = [mt - last_defaecation_time for mt in extended_meal_times if last_defaecation_time <= mt <= (last_defaecation_time + 25)]
                        
                        cum_hazard = cumulative_hazard(current_time, adjusted_meal_times, DEFAULT_BETA, DEFAULT_ALPHA, DEFAULT_GAMMA, S_i)
                        cdf = 1 - np.exp(-cum_hazard)
                        elapsed_time_str = hours_to_hhmmss(current_time)
                        time_since_last_meal = get_last_meal(current_time, adjusted_meal_times)
                        clock_time = (last_defaecation_time + current_time) % 24
                        alert_level = get_alert(cdf, st.session_state.selected_patient, clock_time)
                        
                        elapsed_simulation_time = current_time - (simulation_start_time - last_defaecation_time)
                        time_stamp_seconds = (simulation_start_time * 3600) + (elapsed_simulation_time * 3600)
                        time_stamp = str(timedelta(seconds=int(time_stamp_seconds))).zfill(8)
                        if (last_defaecation_time + current_time) >= 24:
                            time_stamp += " (+1 day)"
                        
                        new_row = {
                            "Gender": patient["Gender"],
                            "Age": patient["Age"],
                            "Diet": patient["Diet"],
                            "Activity": patient["Activity"],
                            "Prior Conditions": patient["Prior Conditions"],
                            "Medication": patient["Medication"],
                            "Elapsed Time": elapsed_time_str,
                            "Time Since Last Meal": time_since_last_meal,
                            "Sound Type": "Processing...",
                            "Probability Output": "Processing...",
                            "Time Stamp": time_stamp,
                            "Alert Level": "Processing..."
                        }
                        st.session_state.patient_data[st.session_state.selected_patient] = pd.concat(
                            [pd.DataFrame([new_row]), st.session_state.patient_data[st.session_state.selected_patient]],
                            ignore_index=True
                        )
                        
                        st.session_state.plot_counter += 1
                        update_demo_content(
                            current_time,
                            st.session_state.patient_data[st.session_state.selected_patient],
                            demo_active=True,
                            plot_key=f"demo_plot_{st.session_state.plot_counter}"
                        )
                        
                        time.sleep(CLASSIFIER_DELAY)
                        
                        st.session_state.patient_data[st.session_state.selected_patient].iloc[0, st.session_state.patient_data[st.session_state.selected_patient].columns.get_loc("Sound Type")] = assign_bowel_sound(cdf)
                        st.session_state.plot_counter += 1
                        update_demo_content(
                            current_time,
                            st.session_state.patient_data[st.session_state.selected_patient],
                            demo_active=True,
                            plot_key=f"demo_plot_{st.session_state.plot_counter}"
                        )
                        
                        time.sleep(PROBABILITY_DELAY - CLASSIFIER_DELAY)
                        
                        st.session_state.patient_data[st.session_state.selected_patient].iloc[0, st.session_state.patient_data[st.session_state.selected_patient].columns.get_loc("Probability Output")] = round(cdf, 4)
                        st.session_state.patient_data[st.session_state.selected_patient].iloc[0, st.session_state.patient_data[st.session_state.selected_patient].columns.get_loc("Alert Level")] = alert_level
                        st.session_state.last_final_alert[st.session_state.selected_patient] = alert_level
                        st.session_state.card_color = ALERT_COLORS.get(alert_level, "rgba(230, 230, 230, 0.75)")  # Updated to lighter grey
                        
                        st.session_state.plot_counter += 1
                        update_demo_content(
                            current_time,
                            st.session_state.patient_data[st.session_state.selected_patient],
                            demo_active=True,
                            plot_key=f"demo_plot_{st.session_state.plot_counter}"
                        )
                        
                        current_time += SAMPLING_INTERVAL / 3600
                        elapsed = time.time() - loop_start
                        time.sleep(max(0, SAMPLING_INTERVAL - elapsed - PROBABILITY_DELAY))
            
            with col2:
                if st.button("Reset Data"):
                    st.session_state.patient_data[st.session_state.selected_patient] = pd.DataFrame(columns=[
                        "Gender", "Age", "Diet", "Activity", "Prior Conditions", "Medication",
                        "Elapsed Time", "Time Since Last Meal", "Sound Type", "Probability Output", "Time Stamp",
                        "Alert Level"
                    ])
                    st.session_state.consecutive_probs[st.session_state.selected_patient].clear()
                    st.session_state.card_color = "rgba(230, 230, 230, 0.75)"  # Updated to lighter grey
                    st.session_state.last_final_alert[st.session_state.selected_patient] = None
                    update_demo_content(None, st.session_state.patient_data[st.session_state.selected_patient], demo_active=False, plot_key="reset_plot")
                    st.success(f"Patient data cleared!")
            
            if not st.session_state.patient_data[st.session_state.selected_patient].empty:
                csv = st.session_state.patient_data[st.session_state.selected_patient].to_csv(index=False)
                st.download_button(
                    label="Download Patient Data as CSV",
                    data=csv,
                    file_name=f"{st.session_state.selected_patient}_data.csv",
                    mime="text/csv"
                )
