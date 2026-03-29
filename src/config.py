#Secreet Salt Key fot Hashing IMEI
SECRET SALT="renewcred_data_engineer_intern_2026"

#Critical Ranges for Parameters
SOC_RANGE=(0,100)
SOH_REANGE=(0,100)
CELL_VOLTAGE_RANGE=(2.5,4.2)

#Expected Null Columns for Data Cleaning
CRITICAL_NULL_COLUMNS = [
    "battery_soc_pct",
    "battery_voltage_v",
    "gps_lat",
    "gps_lon"
]
