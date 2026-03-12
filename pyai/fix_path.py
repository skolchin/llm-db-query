# FixPath utility function

from pathlib import Path
from ruamel.yaml import YAML
from tempfile import gettempdir

def fix_data_paths(profile_path: str | Path, root_path: Path) -> Path:
    """ Correct data file paths specified in BSL profile from relative to absolute ones.
    Return either original path or temporary one.
    """

    def adjust_path(orig: str) -> str:
        orig_path = Path(orig)
        if not orig_path.is_absolute():
            # assume the path is relative to projects' root
            return str((root_path / orig_path).absolute())
        return orig

    # Load profile to memory, update table filepaths to absolute,
    # then save to temporary location
    yaml = YAML()
    profile_path = Path(profile_path)
    profiles = yaml.load(profile_path.read_text())

    for profile in profiles.values():
        if "database" in profile:
            profile["database"] = adjust_path(profile["database"])
        if "tables" in profile:
            for t in profile["tables"]:
                profile["tables"][t] = adjust_path(profile["tables"][t])

    profile_path = Path(gettempdir()) / profile_path.name
    with open(profile_path, "wt") as fp:
        yaml.dump(profiles, fp)

    return profile_path
