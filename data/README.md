# NRCD Data Structure

This document describes the structure and fields of the data files in the National Running Club Database.

## File Organization

The database is organized into the following CSV files:

### Core Data Files
- `athlete.csv`: Contains athlete information
- `team.csv`: Contains team information
- `meet.csv`: Contains meet/event information
- `result.csv`: Contains individual race results
- `course_details.csv`: Contains course information for events
- `sport.csv`: Contains sport types (Cross Country, Indoor Track, Outdoor Track, Road Race, Trail Race)
- `running_event.csv`: Contains specific event types (e.g., 5K, 10K, Mile, etc.)

### Association Files
- `athlete_team_association.csv`: Links athletes to their teams

### Derived/Export Files
- `joined.csv`: Combined dataset
- `export_same_format_as_input.csv`: Exported data in input format

## Data Relationships

The database follows a relational structure where:
- Athletes are associated with teams through `athlete_team_association.csv`
- Results link to athletes, meets, and specific running events
- Meets are associated with sports and may have course details
- Teams participate in meets and have athletes

## Data Fields

### athlete.csv
- `athlete_id`: Unique identifier for the athlete
- `first_name`: Athlete's first name
- `last_name`: Athlete's last name
- `gender`: Athlete's gender (M/F)
- `athlete_logo`: URL or path to athlete's logo/image
- `hometown`: Athlete's hometown
- `strava`: Athlete's Strava profile link

### team.csv
- `team_id`: Unique identifier for the team
- `team_name`: Name of the team
- `region`: Geographic region (Pacific, Heartland, Great Plains, Southeast, Great Lakes, Mid-Atlantic, Northeast)
- `city`: Team's city location
- `state`: Team's state location
- `team_logo`: URL or path to team's logo
- `team_photo`: URL or path to team's photo
- `website`: Team's website URL
- `instagram`: Team's Instagram handle

### meet.csv
- `meet_id`: Unique identifier for the meet
- `sport_id`: Reference to sport.csv
- `meet_name`: Name of the meet
- `start_date`: Meet start date (YYYY-MM-DD)
- `end_date`: Meet end date (YYYY-MM-DD)
- `meet_city`: City where meet was held
- `meet_state`: State where meet was held
- `elevation`: Elevation of meet location
- `external_result_link`: Link to external results
- `photo_link`: Link to meet photos
- `nirca_only`: Boolean indicating if meet is NIRCA-only
- `regionals`: Boolean indicating if meet is a regional championship
- `nationals`: Boolean indicating if meet is a national championship
- `track_distance`: Track distance in meters (for track meets)
- `banked_track`: Boolean indicating if track is banked
- `approved`: Boolean indicating if meet is approved

### result.csv
- `result_id`: Unique identifier for the result
- `team_id`: Reference to team.csv
- `athlete_id`: Primary athlete for the result
- `grade`: Primary athlete's grade level
- `athlete_id_2`: Second athlete (for relay events)
- `grade_2`: Second athlete's grade level
- `athlete_id_3`: Third athlete (for relay events)
- `grade_3`: Third athlete's grade level
- `athlete_id_4`: Fourth athlete (for relay events)
- `grade_4`: Fourth athlete's grade level
- `running_event_id`: Reference to running_event.csv
- `event_type`: Type of event (e.g., Championship, Freshman/Sophomore, Junior/Senior/Grad)
- `result_time`: Athlete's result (time in HH:MM:SS for running events, distance in meters for field events)
- `relay_split`: First relay split time (for relay events)
- `relay_split_2`: Second relay split time (for relay events)
- `relay_split_3`: Third relay split time (for relay events)
- `wind`: Wind speed in meters per second (for track events)
- `meet_id`: Reference to meet.csv
- `approved`: Boolean indicating if result is approved
- `user_id`: ID of user who entered the result (reference to users table, not present in this dataset)

### course_details.csv
- `course_details_id`: Unique identifier for the course details
- `meet_id`: Reference to meet.csv
- `running_event_id`: Reference to running_event.csv
- `event_type`: Type of event (e.g., Championship, Freshman/Sophomore, Junior/Senior/Grad)
- `gender`: Gender category (M/F)
- `elevation_gain`: Course elevation gain in meters
- `elevation_loss`: Course elevation loss in meters
- `estimated_course_distance`: Estimated course distance in meters
- `date_of_event`: Date of the event (YYYY-MM-DD)
- `time_of_event`: Time of the event (HH:MM:SS)
- `temperature`: Temperature in Fahrenheit
- `real_feel`: Real feel temperature in Fahrenheit
- `dew_point`: Dew point in Fahrenheit
- `humidity`: Humidity percentage
- `weather_conditions`: Weather conditions (e.g., Clear, Clouds, Rain)
- `weather_description`: Detailed weather description
- `aqi`: Air Quality Index
- `aqi_co`: Carbon Monoxide AQI
- `aqi_no`: Nitric Oxide AQI
- `aqi_no2`: Nitrogen Dioxide AQI
- `aqi_o3`: Ozone AQI
- `aqi_so2`: Sulfur Dioxide AQI
- `aqi_pm2_5`: PM2.5 AQI
- `aqi_pm10`: PM10 AQI
- `aqi_nh3`: Ammonia AQI
- `approved`: Boolean indicating if course details are approved

### sport.csv
- `sport_id`: Unique identifier for the sport
- `sport_name`: Name of the sport (Cross Country, Indoor Track, Outdoor Track, Road Race, Trail Race)

### running_event.csv
- `running_event_id`: Unique identifier for the event
- `event_name`: Name of the event (ex: 55m, 4x100m, SMR, High Jump, 8K, Marathon)

### athlete_team_association.csv
- `athlete_id`: Reference to athlete.csv
- `team_id`: Reference to team.csv
- `current_grade`: Athlete's current grade level

## Data Format
- Files are stored in CSV format
- UTF-8 encoding
- First row contains field names
- Dates are in ISO format (YYYY-MM-DD)
- Times are in 24-hour format (HH:MM:SS)
