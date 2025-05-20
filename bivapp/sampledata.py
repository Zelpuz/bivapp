import pandas as pd, numpy as np, os

dirname = os.path.dirname(__file__)


def ImportOpenairDataExample():
    """
    Example data from the R openair library.
    See https://www.rdocumentation.org/packages/openair/versions/2.18-2/topics/mydata.
    Gaseous pollutants are all in ppbv, except CO which is in ppmv.
    Particle-phase pollutants are in ug/m3.
    Wind speeds are in m/s.
    Wind directions are in degrees, with 0/360 being winds from the north.
    """
    df = pd.read_csv(dirname + "/sampledata/openair_test_data.csv")
    df = df.drop(columns=["Unnamed: 0"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_index()

    # Add some jitter to wind directions
    df["wd"] = df["wd"] + np.random.uniform(-10, 10, len(df.index))
    df["wd"] %= 360

    # Add some jitter to wind speeds
    df["ws"] = df["ws"] + np.random.uniform(-0.1, 0.1, len(df.index))
    df.loc[df["ws"] < 0, "ws"] = 0

    return df


def ImportTorontoNO2():
    """Imports test pollution data. The data is from 125 Resources Rd., Toronto, 2023-01-01 to 2023-12-31."""
    # Import and rename columns
    pol_df = pd.read_csv(dirname + "/sampledata/NO2_2023.csv", skiprows=7)
    pol_df.columns = [
        "pollutant",
        "naps_id",
        "city",
        "province",
        "lat",
        "lon",
        "date",
    ] + [i for i in range(1, 25)]

    # Choose only Toronto 125 Resources Rd. station
    pol_df = pol_df.loc[pol_df["naps_id"] == 60430, :]

    # Wide to long
    pol_df = pd.melt(
        pol_df,
        id_vars=["date"],
        value_vars=[i for i in range(1, 25)],
        var_name="hour",
        value_name="no2_ppbv",
    )

    # Form and set datetime index
    pol_df["datetime"] = pd.to_datetime(
        pol_df["date"] + " " + (pol_df["hour"] - 1).astype(str) + ":00:00"
    ) + pd.Timedelta("1h")
    pol_df = pol_df.drop(columns=["date", "hour"])
    pol_df = pol_df.set_index("datetime")
    pol_df = pol_df.sort_index()

    # Set missing values to np.nan
    pol_df["no2_ppbv"] = pol_df["no2_ppbv"].replace(-999, np.nan)

    return pol_df


def ImportTorontoMet():
    """Imports test meteorology. The data is from Toronto's Pearson Airport, 2023-01-01 to 2023-12-31."""
    # Import and rename columns
    met_df = pd.read_csv(dirname + "/sampledata/climate-hourly.csv")
    met_df = met_df[["UTC_DATE", "WIND_DIRECTION", "WIND_SPEED"]]
    met_df.columns = ["datetime", "wind_dir", "wind_speed"]

    # Convert from UTC to EST (pol data is in "local standard time")
    met_df["datetime"] = pd.to_datetime(met_df["datetime"])
    met_df["datetime"] = met_df["datetime"].dt.tz_localize("UTC").dt.tz_convert("EST")
    met_df["datetime"] = met_df["datetime"].dt.tz_localize(None)
    met_df = met_df.set_index("datetime")
    met_df = met_df.sort_index()

    # Drop the first row (last hour of prior year)
    met_df = met_df.iloc[1:, :]

    # Convert wind direction from tens of degrees to degrees
    met_df["wind_dir_deg"] = met_df["wind_dir"] * 10
    met_df = met_df.drop(columns=["wind_dir"])
    met_df["wind_dir_deg"] = met_df["wind_dir_deg"].replace(
        0, np.nan
    )  # zeros are calms

    # Add some jitter to wind direction
    met_df["wind_dir_deg"] = met_df["wind_dir_deg"] + np.random.uniform(
        -10, 10, len(met_df.index)
    )
    met_df["wind_dir_deg"] %= 360

    # Add some jitter to wind speed and convert from km/h to m/s
    met_df["wind_speed"] = met_df["wind_speed"] + np.random.uniform(
        -0.5, 0.5, len(met_df.index)
    )
    met_df.loc[met_df["wind_speed"] < 0, "wind_speed"] = (
        0  # the jittering could introduce negatives
    )
    met_df["wind_speed_m_s"] = met_df["wind_speed"] / 3.6
    met_df = met_df.drop(columns=["wind_speed"])

    return met_df


def LoadTorontoDataExample():
    pol_df = ImportTorontoNO2()
    met_df = ImportTorontoMet()
    df = pol_df.join(met_df)
    return df
