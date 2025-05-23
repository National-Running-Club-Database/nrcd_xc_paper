import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Cross Country Performance Analysis\n",
                "\n",
                "This notebook analyzes cross country running performance data, including:\n",
                "- Gender distribution\n",
                "- Performance trends over time\n",
                "- Geographic distribution of meets\n",
                "- Performance normalization and adjustment for course conditions"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from scipy import stats\n",
                "from datetime import datetime\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n",
                "from sklearn.cluster import KMeans, AgglomerativeClustering\n",
                "from sklearn.metrics import r2_score\n",
                "import scipy.cluster.hierarchy as sch\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "import os"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Loading and Preprocessing"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def load_data():\n",
                "    \"\"\"Load all CSV files from the data directory.\"\"\"\n",
                "    directory_path = '../data/nrcd/data'\n",
                "    \n",
                "    # Load each CSV file into a pandas DataFrame\n",
                "    team_df = pd.read_csv(os.path.join(directory_path, 'team.csv'))\n",
                "    athlete_df = pd.read_csv(os.path.join(directory_path, 'athlete.csv'))\n",
                "    sport_df = pd.read_csv(os.path.join(directory_path, 'sport.csv'))\n",
                "    running_event_df = pd.read_csv(os.path.join(directory_path, 'running_event.csv'))\n",
                "    meet_df = pd.read_csv(os.path.join(directory_path, 'meet.csv'))\n",
                "    result_df = pd.read_csv(os.path.join(directory_path, 'result.csv'))\n",
                "    course_details_df = pd.read_csv(os.path.join(directory_path, 'course_details.csv'))\n",
                "    athlete_team_association_df = pd.read_csv(os.path.join(directory_path, 'athlete_team_association.csv'))\n",
                "    \n",
                "    # Convert date columns to datetime\n",
                "    meet_df['start_date'] = pd.to_datetime(meet_df['start_date'], format='%Y-%m-%d', errors='coerce')\n",
                "    meet_df['end_date'] = pd.to_datetime(meet_df['end_date'], format='%Y-%m-%d', errors='coerce')\n",
                "    \n",
                "    return (team_df, athlete_df, sport_df, running_event_df, meet_df, \n",
                "            result_df, course_details_df, athlete_team_association_df)\n",
                "\n",
                "# Load the data\n",
                "team_df, athlete_df, sport_df, running_event_df, meet_df, result_df, course_details_df, athlete_team_association_df = load_data()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Merging and Initial Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def merge_dataframes(athlete_df, result_df, meet_df):\n",
                "    \"\"\"Merge the dataframes for analysis.\"\"\"\n",
                "    # Merge athlete and result data\n",
                "    athlete_result_df = pd.merge(athlete_df, result_df, on='athlete_id')\n",
                "    \n",
                "    # Merge the result with meet data\n",
                "    full_df = pd.merge(athlete_result_df, meet_df, on='meet_id')\n",
                "    \n",
                "    return full_df\n",
                "\n",
                "# Merge dataframes\n",
                "full_df = merge_dataframes(athlete_df, result_df, meet_df)\n",
                "\n",
                "# Analyze gender distribution\n",
                "gender_counts = athlete_df['gender'].value_counts()\n",
                "print(f\"Number of men: {gender_counts.get('M', 0)}\")\n",
                "print(f\"Number of women: {gender_counts.get('F', 0)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cross Country Performance Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def analyze_cross_country_performance(sport_df, meet_df, result_df, athlete_df, running_event_df):\n",
                "    \"\"\"Analyze cross country performance data.\"\"\"\n",
                "    # Get cross country sport ID\n",
                "    cross_country_sport_id = sport_df[sport_df['sport_name'].str.contains('Cross Country', case=False, na=False)]['sport_id'].values[0]\n",
                "    \n",
                "    # Define date range\n",
                "    start_date = datetime(2024, 8, 1)\n",
                "    end_date = datetime(2025, 5, 1)\n",
                "    \n",
                "    # Filter meets by date range and sport\n",
                "    cross_country_meets = meet_df[\n",
                "        (meet_df['sport_id'] == cross_country_sport_id) &\n",
                "        (meet_df['start_date'] >= start_date) &\n",
                "        (meet_df['start_date'] <= end_date)\n",
                "    ]\n",
                "    \n",
                "    # Filter results for cross country meets\n",
                "    cross_country_results = result_df[result_df['meet_id'].isin(cross_country_meets['meet_id'])]\n",
                "    \n",
                "    # Merge with athlete data\n",
                "    cross_country_athletes = pd.merge(cross_country_results, athlete_df, on='athlete_id')\n",
                "    \n",
                "    # Merge with running event data to get event names\n",
                "    cross_country_athletes = pd.merge(cross_country_athletes, running_event_df[['running_event_id', 'event_name']], \n",
                "                                    on='running_event_id', how='left')\n",
                "    \n",
                "    # Merge with meet data to get start_date\n",
                "    cross_country_athletes = pd.merge(cross_country_athletes, cross_country_meets[['meet_id', 'start_date']], \n",
                "                                    on='meet_id', how='left')\n",
                "    \n",
                "    return cross_country_athletes\n",
                "\n",
                "# Analyze cross country performance\n",
                "cross_country_athletes = analyze_cross_country_performance(sport_df, meet_df, result_df, athlete_df, running_event_df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Time Processing and Normalization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def parse_time(time_str):\n",
                "    \"\"\"Convert time string to seconds.\"\"\"\n",
                "    parts = time_str.split(\":\")\n",
                "    if len(parts) == 3:\n",
                "        hours, minutes, seconds = map(float, parts)\n",
                "        return hours * 3600 + minutes * 60 + seconds\n",
                "    elif len(parts) == 2:\n",
                "        minutes, seconds = map(float, parts)\n",
                "        return minutes * 60 + seconds\n",
                "    else:\n",
                "        return float(time_str)\n",
                "\n",
                "def format_parsed_time(seconds):\n",
                "    \"\"\"Convert seconds to formatted time string.\"\"\"\n",
                "    hours = int(seconds // 3600)\n",
                "    minutes = int((seconds % 3600) // 60)\n",
                "    seconds = seconds % 60\n",
                "    if hours > 0:\n",
                "        return f\"{hours}:{minutes:02d}:{seconds:05.2f}\"\n",
                "    elif minutes > 0:\n",
                "        return f\"{minutes}:{seconds:05.2f}\"\n",
                "    else:\n",
                "        return f\"{seconds:05.2f}\"\n",
                "\n",
                "def adjust_time_for_race(event_name, time, course_details, gender):\n",
                "    \"\"\"Adjust race time based on course details and conditions.\"\"\"\n",
                "    event_dist = None\n",
                "    if event_name.endswith('m'):\n",
                "        event_dist = float(event_name.replace('m', '').strip())\n",
                "    elif event_name.endswith('mi'):\n",
                "        event_dist = float(event_name.replace('mi', '').strip()) * 1609.34\n",
                "    \n",
                "    hill_slow_time = 0\n",
                "    hill_speed_time = 0\n",
                "    \n",
                "    if event_dist:\n",
                "        if pd.notna(course_details.get('elevation_gain')) or pd.notna(course_details.get('elevation_loss')):\n",
                "            if pd.notna(course_details.get('elevation_gain')):\n",
                "                grade_increase_percentage = course_details['elevation_gain'] / event_dist * 100\n",
                "                hill_slow_time = time * 0.04 * grade_increase_percentage\n",
                "            if pd.notna(course_details.get('elevation_loss')):\n",
                "                grade_decrease_percentage = course_details['elevation_loss'] / event_dist * 100\n",
                "                hill_speed_time = time * 0.0267 * grade_decrease_percentage\n",
                "            time += hill_speed_time - hill_slow_time\n",
                "\n",
                "        if pd.notna(course_details.get('estimated_course_distance')):\n",
                "            factor = event_dist / course_details['estimated_course_distance']\n",
                "            if gender == 'F':\n",
                "                time *= factor ** 1.055\n",
                "            elif gender == 'M':\n",
                "                time *= factor ** 1.08\n",
                "\n",
                "    if pd.notna(course_details.get('temperature')) and pd.notna(course_details.get('dew_point')):\n",
                "        weather_factor = course_details['temperature'] + course_details['dew_point']\n",
                "        if weather_factor > 100:\n",
                "            percent_increase = 0.0015 * (weather_factor - 100) ** 2\n",
                "            time *= 1 - (percent_increase / 100)\n",
                "\n",
                "    return time\n",
                "\n",
                "def normalize_and_adjust_time(row):\n",
                "    \"\"\"Normalize and adjust time for a given row of data.\"\"\"\n",
                "    parsed_time = parse_time(row['result_time'])\n",
                "    \n",
                "    if pd.notna(row.get('course_details_id')):\n",
                "        course_details = {\n",
                "            'elevation_gain': row.get('elevation_gain'),\n",
                "            'elevation_loss': row.get('elevation_loss'),\n",
                "            'estimated_course_distance': row.get('estimated_course_distance'),\n",
                "            'temperature': row.get('temperature'),\n",
                "            'dew_point': row.get('dew_point')\n",
                "        }\n",
                "        adjusted_time = adjust_time_for_race(row['event_name'], parsed_time, course_details, row['gender'])\n",
                "    else:\n",
                "        adjusted_time = parsed_time\n",
                "\n",
                "    event_dist = None\n",
                "    if row['event_name'].endswith('m'):\n",
                "        event_dist = float(row['event_name'].replace('m', '').strip())\n",
                "    elif row['event_name'].endswith('mi'):\n",
                "        event_dist = float(row['event_name'].replace('mi', '').strip()) * 1609.34\n",
                "\n",
                "    if event_dist:\n",
                "        if row['gender'] == 'F':\n",
                "            normalized_time = adjusted_time * (6000 / event_dist)**1.08\n",
                "        elif row['gender'] == 'M':\n",
                "            normalized_time = adjusted_time * (8000 / event_dist)**1.055\n",
                "        else:\n",
                "            normalized_time = adjusted_time\n",
                "    else:\n",
                "        normalized_time = adjusted_time\n",
                "        \n",
                "    return format_parsed_time(normalized_time)\n",
                "\n",
                "# Process and normalize times\n",
                "cross_country_athletes['final_adjusted_time'] = cross_country_athletes.apply(normalize_and_adjust_time, axis=1)\n",
                "cross_country_athletes['final_parsed_adjusted_time'] = cross_country_athletes['final_adjusted_time'].apply(parse_time)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Performance Visualization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def plot_performance_over_time(data, gender):\n",
                "    \"\"\"Plot performance over time for a specific gender.\"\"\"\n",
                "    plt.figure(figsize=(12, 6))\n",
                "    sns.lineplot(data=data, x='start_date', y='final_parsed_adjusted_time', \n",
                "                hue='athlete_id', marker='o', palette='viridis', legend=None)\n",
                "    plt.title(f'Athlete Performance Over Season ({gender}-{\"8K\" if gender == \"M\" else \"6K\"})')\n",
                "    plt.xlabel('Date')\n",
                "    plt.ylabel('Adjusted Time (seconds)')\n",
                "    plt.xticks(rotation=45)\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "# Separate male and female athletes\n",
                "male_athletes = cross_country_athletes[cross_country_athletes['gender'] == 'M']\n",
                "female_athletes = cross_country_athletes[cross_country_athletes['gender'] == 'F']\n",
                "\n",
                "# Plot performance over time\n",
                "plot_performance_over_time(male_athletes, 'Men')\n",
                "plot_performance_over_time(female_athletes, 'Women')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Improvement Rate Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def calculate_improvement_rate(data):\n",
                "    \"\"\"Calculate improvement rate using linear regression.\"\"\"\n",
                "    slopes = []\n",
                "    for athlete_id in data['athlete_id'].unique():\n",
                "        athlete_df = data[data['athlete_id'] == athlete_id].sort_values('start_date')\n",
                "        if len(athlete_df) > 1:\n",
                "            X = np.array(athlete_df['start_date'].map(pd.Timestamp.toordinal)).reshape(-1, 1)\n",
                "            y = athlete_df['final_parsed_adjusted_time'].values\n",
                "            \n",
                "            model = LinearRegression()\n",
                "            model.fit(X, y)\n",
                "            slope = model.coef_[0]\n",
                "            slopes.append(slope)\n",
                "    \n",
                "    return np.mean(slopes)\n",
                "\n",
                "# Calculate improvement rates\n",
                "male_improvement = calculate_improvement_rate(male_athletes)\n",
                "female_improvement = calculate_improvement_rate(female_athletes)\n",
                "\n",
                "print(f\"Average Improvement Rate (Men): {male_improvement:.2f} seconds per day\")\n",
                "print(f\"Average Improvement Rate (Women): {female_improvement:.2f} seconds per day\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## State Name Normalization for Geographic Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Map full state names and abbreviations to abbreviations\n",
                "state_abbreviations = {\n",
                "    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',\n",
                "    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',\n",
                "    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',\n",
                "    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',\n",
                "    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',\n",
                "    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',\n",
                "    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',\n",
                "    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',\n",
                "    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',\n",
                "    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',\n",
                "    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',\n",
                "    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',\n",
                "    'Wisconsin': 'WI', 'Wyoming': 'WY'\n",
                "}\n",
                "# Create a new dictionary with reverse mappings\n",
                "state_abbreviations_with_reverse = state_abbreviations.copy()\n",
                "for abbr in state_abbreviations.values():\n",
                "    state_abbreviations_with_reverse[abbr] = abbr\n",
                "\n",
                "def normalize_state(state):\n",
                "    return state_abbreviations_with_reverse.get(str(state).strip(), str(state).strip())\n",
                "\n",
                "meet_df['normalized_state'] = meet_df['meet_state'].apply(normalize_state)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Geographic Distribution Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_state_choropleth(meet_df):\n",
                "    \"\"\"Create a choropleth map of meets by state.\"\"\"\n",
                "    results_by_state = meet_df['normalized_state'].value_counts().to_dict()\n",
                "    state_data = pd.DataFrame(list(results_by_state.items()),\n",
                "                            columns=['State', 'Results'])\n",
                "    fig = go.Figure(data=go.Choropleth(\n",
                "        locations=state_data['State'],\n",
                "        z=state_data['Results'],\n",
                "        locationmode='USA-states',\n",
                "        colorscale='Viridis',\n",
                "        colorbar_title=\"Number of Meets\",\n",
                "    ))\n",
                "    fig.update_layout(\n",
                "        title_text='Number of Cross Country Meets by State',\n",
                "        geo=dict(\n",
                "            scope='usa',\n",
                "            projection=go.layout.geo.Projection(type='albers usa'),\n",
                "        ),\n",
                "    )\n",
                "    fig.show()\n",
                "\n",
                "# Create state choropleth\n",
                "create_state_choropleth(meet_df)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('scripts/analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1) 