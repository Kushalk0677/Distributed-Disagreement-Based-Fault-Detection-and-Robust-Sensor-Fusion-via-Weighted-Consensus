# data/raw/ — Place your dataset files here

| File                          | Dataset                        | Download from                                                    |
|-------------------------------|--------------------------------|------------------------------------------------------------------|
| `BerkeleyLab.txt`             | Intel Berkeley Research Lab    | http://db.csail.mit.edu/labdata/labdata.html                    |
| `AirQualityUCI.csv`           | UCI Air Quality                | https://archive.ics.uci.edu/ml/datasets/Air+Quality             |
| `smart_city_sensor_data.csv`  | Smart City IoT (Kaggle)        | https://www.kaggle.com/datasets/emirhanai/smart-city-sensor-data |
| `sensor_maintenance_data.csv` | Sensor Maintenance (Kaggle)    | https://www.kaggle.com/datasets/dnkumawat/sensor-maintenance     |

## File format notes

### BerkeleyLab.txt
Space-separated, **no header**. Columns:
```
date  time  epoch  moteid  temperature  humidity  light  voltage
```
Example: `2004-02-28 00:59:16.02 3 1 19.98 37.09 45.08 2.69`

### AirQualityUCI.csv
Semicolon-separated, comma as decimal, header on row 1.
Missing values encoded as `-200`.

### smart_city_sensor_data.csv
UTF-8 (BOM), comma-separated, Turkish column headers (auto-mapped by loader).

### sensor_maintenance_data.csv
Comma-separated, header row, UTF-8 (BOM).
`Fault Status` column contains `"Fault Detected"` / `"No Fault"` — used as ground-truth labels.

## Notes on file size
The full Berkeley dataset is ~75 MB (2.3M rows). The loader automatically:
- Clips to `max_T=2000` time steps
- Drops sensors with fewer than 2 valid readings
- Applies plausibility filters per channel

The full Air Quality dataset is ~380 KB (9358 rows) — loads fully in seconds.
