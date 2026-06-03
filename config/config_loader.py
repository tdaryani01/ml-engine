import yaml

def load_production_config(config_path="config\\config.yaml"):
    """Safely parses production YAML files into standard dictionaries."""
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)