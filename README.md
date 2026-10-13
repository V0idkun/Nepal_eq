# Nepal Earthquake Building Damage Predictor

A full-stack machine learning application designed to predict building damage severity levels following an earthquake. By analyzing structural features and geographical data, this tool aids urban planners, first responders, and risk assessment teams in identifying vulnerable structures.

**Live Demo:** [https://nepaleq-qtburnextmaelfzxczagdm.streamlit.app/](https://nepaleq-qtburnextmaelfzxczagdm.streamlit.app/)

## Features

* **Damage Grade Prediction:** Classifies the expected severity of building damage based on architectural and structural inputs.
* **Interactive UI:** A user-friendly Streamlit interface allowing users to manually input building features (e.g., foundation type, roof type, land surface condition) or upload batch data files (CSV, DB, PDF) for analysis.
* **Real-time Inference:** Rapid classification using an optimized Scikit-learn machine learning pipeline.
* **Full-Stack Architecture:** Integrates backend database logging (PostgreSQL/MySQL) and an API layer (FastAPI) to handle prediction requests securely and efficiently.

## Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Frontend:** Streamlit
* **Database:** PostgreSQL / MySQL

## Input Features (Prediction Parameters)

The model evaluates structures based on several key inputs, including but not limited to:
* Building Age
* Foundation Type (e.g., Mud, Other)
* Ground Floor Type
* Land Surface Condition (e.g., Flat)
* Roof Type (e.g., Bamboo/Timber-Light roof)
* Superstructure Material (e.g., Mud Mortar Stone)
* Plan Configuration & Position
