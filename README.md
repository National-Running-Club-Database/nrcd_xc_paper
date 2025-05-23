# JSS Paper

This repository contains the code and data for the JSS (Journal of Sports Science) paper analysis.

## Data Sources

The primary data source is the National Running Club Database (NRCD):
- Location: `data/nrcd/data`
- Source: https://github.com/National-Running-Club-Database/data
- Note: Do not edit the data files directly as they are sourced from another repository

## Updating the Data

To update the data from the source repository:

1. Navigate to the data directory:
   ```bash
   cd data/nrcd
   ```

2. Fetch the latest changes from the source repository:
   ```bash
   git fetch origin
   ```

3. Reset to the latest version:
   ```bash
   git reset --hard origin/main
   ```

Note: This will overwrite any local changes in the data directory. Make sure to commit any important changes before updating.


