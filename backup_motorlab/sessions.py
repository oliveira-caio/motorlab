OLD_GBYK = [
    "bex_20230623_denoised",
    "ken_20230614_denoised",
    "ken_20230618_denoised",
]

GBYK = [
    "bex_20230621_spikes_sorted_SES",
    "bex_20230624_spikes_sorted_SES",
    "bex_20230629_spikes_sorted_SES",
    "bex_20230630_spikes_sorted_SES",
    "bex_20230701_spikes_sorted_SES",
    "bex_20230708_spikes_sorted_SES",
    "ken_20230614_spikes_sorted_SES",
    "ken_20230618_spikes_sorted_SES",
    "ken_20230622_spikes_sorted_SES",
    "ken_20230629_spikes_sorted_SES",
    "ken_20230630_spikes_sorted_SES",
    "ken_20230701_spikes_sorted_SES",
    "ken_20230703_spikes_sorted_SES",
]

PG = [
    "bex_20230221",
    "bex_20230222",
    "bex_20230223",
    "bex_20230224",
    "bex_20230225",
    "bex_20230226",
    "jon_20230125",
    "jon_20230126",
    "jon_20230127",
    "jon_20230130",
    "jon_20230131",
    "jon_20230202",
    "jon_20230203",
    "luk_20230126",
    "luk_20230127",
    "luk_20230130",
    "luk_20230131",
    "luk_20230202",
    "luk_20230203",
]


def load(experiment: str) -> list:
    """
    Get session list for a given experiment.

    Parameters
    ----------
    experiment : str
        Experiment name ('gbyk', 'old_gbyk', 'pg')

    Returns
    -------
    list
        List of session names
    """
    sessions_dict = {"gbyk": GBYK, "old_gbyk": OLD_GBYK, "pg": PG}
    if experiment not in sessions_dict:
        raise ValueError(f"Unknown experiment: {experiment}")
    return sessions_dict[experiment]
